from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from app.domain.models import ParsedArticle, RawDocument
from app.processing.canonicalizer import normalize_url
from app.processing.cleaner import clean_text
from app.processing.deduplicator import hash_content
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchResponse:
    requested_url: str
    final_url: str
    status_code: int | None
    text: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 400


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    url: str


class BaseCrawler:
    source_name: str = ""
    category_link_selectors: tuple[str, ...] = (
        "nav a[href]",
        ".menu a[href]",
        ".main-nav a[href]",
        ".category a[href]",
        "a[href]",
    )
    excluded_category_path_patterns: tuple[str, ...] = (
        r"/rss",
        r"/video",
        r"/podcast",
        r"/photo",
        r"/login",
        r"/search",
        r"/tag/",
    )
    article_link_selectors: tuple[str, ...] = ("a[href]",)
    article_url_patterns: tuple[str, ...] = (r"\.html(?:$|\?)",)
    title_selectors: tuple[str, ...] = ("h1",)
    summary_selectors: tuple[str, ...] = ()
    content_selectors: tuple[str, ...] = ("article",)
    published_at_selectors: tuple[str, ...] = ()
    author_selectors: tuple[str, ...] = ()
    category_selectors: tuple[str, ...] = ()
    tag_selectors: tuple[str, ...] = ()
    min_content_chars: int = 120
    min_paragraph_chars: int = 20

    def __init__(self, source_config: Any) -> None:
        self.config = source_config
        self._seen_link_keys: set[str] = set()
        self._existing_link_keys: set[str] = set()
        self._existing_content_hashes: set[str] = set()
        self._seen_exact_document_keys: set[str] = set()
        self._existing_exact_document_keys: set[str] = set()

    def set_existing_keys(
        self,
        link_keys: set[str] | None = None,
        content_hashes: set[str] | None = None,
        exact_document_keys: set[str] | None = None,
    ) -> None:
        self._existing_link_keys = set(link_keys or set())
        self._existing_content_hashes = set(content_hashes or set())
        self._existing_exact_document_keys = set(exact_document_keys or set())

    def fetch_url(self, url: str) -> FetchResponse:
        timeout = int(getattr(self.config, "timeout_seconds", 15))
        retry_count = int(getattr(self.config, "retry_count", 2))
        user_agent = str(getattr(self.config, "user_agent", "databricks-news-ai-demo/1.0"))
        headers = {"User-Agent": user_agent}
        last_error: str | None = None

        for attempt in range(retry_count + 1):
            try:
                response = _fetch_with_requests(url, timeout, headers)
                logger.info("[CRAWL] HTTP status=%s source=%s url=%s", response.status_code, self.source_name, url)
                if response.status_code is not None and response.status_code >= 500 and attempt < retry_count:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                return response
            except Exception as exc:
                last_error = str(exc)
                if attempt < retry_count:
                    logger.warning(
                        "[CRAWL] Retry fetch source=%s url=%s attempt=%s error=%s",
                        self.source_name,
                        url,
                        attempt + 1,
                        last_error,
                    )
                    time.sleep(0.25 * (attempt + 1))
                    continue

        logger.warning("[CRAWL] Failed fetch source=%s url=%s error=%s", self.source_name, url, last_error)
        return FetchResponse(requested_url=url, final_url=url, status_code=None, text="", error=last_error)

    def discover_categories(self, source_home_url: str) -> list[Category]:
        if not bool(getattr(self.config, "discover_categories", True)):
            return self._configured_categories()

        response = self.fetch_url(source_home_url)
        if not response.ok or not response.text:
            logger.warning("[CRAWL] Category discovery failed source=%s url=%s", self.source_name, source_home_url)
            return self._configured_categories()

        soup = _soup(response.text)
        if soup is None:
            return self._configured_categories()

        categories: list[Category] = []
        seen_urls: set[str] = set()
        for selector in self.category_link_selectors:
            for node in soup.select(selector):
                href = node.get("href")
                if not href:
                    continue
                category_url = normalize_url(urljoin(response.final_url or source_home_url, str(href)))
                if not self.is_category_url(category_url):
                    continue
                if category_url in seen_urls:
                    continue
                category_name = clean_text(node.get_text(" ", strip=True)) or _category_name_from_url(category_url)
                seen_urls.add(category_url)
                categories.append(Category(name=category_name, url=category_url))

        if categories:
            logger.info("[CRAWL] Discovered %s categories source=%s", len(categories), self.source_name)
            return categories

        fallback_categories = self._configured_categories()
        logger.info("[CRAWL] Using %s configured categories source=%s", len(fallback_categories), self.source_name)
        return fallback_categories

    def is_category_url(self, url: str) -> bool:
        normalized = normalize_url(url)
        if not normalized:
            return False
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        if base_url and not normalized.startswith(base_url):
            return False
        if self.is_article_url(normalized):
            return False
        path = urlsplit(normalized).path.strip("/")
        if not path:
            return False
        return not any(re.search(pattern, f"/{path}") for pattern in self.excluded_category_path_patterns)

    def build_category_page_url(self, category_url: str, page: int) -> str:
        if page <= 1:
            return normalize_url(category_url)
        return f"{normalize_url(category_url)}?page={page}"

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        category_url = normalize_url(urljoin(str(getattr(self.config, "base_url", "")).rstrip("/") + "/", category_path))
        return [self.build_category_page_url(category_url, page_number)]

    def extract_article_links(self, category_page_url: str) -> list[str]:
        response = self.fetch_url(category_page_url)
        if not response.ok:
            raise RuntimeError(f"request_error status={response.status_code} url={category_page_url}")
        if not response.text:
            return []
        return self.extract_article_urls_from_html(response.final_url or category_page_url, response.text)

    def filter_new_links(self, article_links: list[str]) -> list[str]:
        new_links: list[str] = []
        for article_link in article_links:
            normalized = normalize_url(article_link)
            if not normalized:
                continue
            link_key = self.link_key(normalized)
            if link_key in self._seen_link_keys:
                continue
            self._seen_link_keys.add(link_key)
            new_links.append(normalized)
        return new_links

    def crawl_article_detail(
        self,
        article_url: str,
        category: Category | None = None,
        category_page: int | None = None,
    ) -> ParsedArticle | None:
        logger.info("[CRAWL] Fetch article source=%s url=%s", self.source_name, article_url)
        response = self.fetch_url(article_url)
        if not response.ok or not response.text:
            return None
        article = self.build_parsed_article(
            url=response.final_url or article_url,
            html=response.text,
            http_status=response.status_code,
        )
        if article is None:
            return None
        if category is not None:
            article.category = _source_category_path(category.url) or category.name
            article.category_url = category.url
            article.metadata["source_category_name"] = category.name
            article.metadata["source_category_url"] = category.url
        if category_page is not None:
            article.category_page = category_page
        return article

    def should_skip_exact_existing_document(self, canonical_url: str, checksum: str) -> bool:
        if not canonical_url or not checksum:
            return False
        exact_key = exact_document_identity_key(self.source_name, canonical_url, checksum)
        if exact_key in self._existing_exact_document_keys or exact_key in self._seen_exact_document_keys:
            return True
        if checksum in self._existing_content_hashes:
            logger.info(
                "[CRAWL] Keep article with same content hash because source or canonical_url is different source=%s canonical_url=%s",
                self.source_name,
                normalize_url(canonical_url),
            )
        return False

    def should_skip_content_hash(self, content_hash: str) -> bool:
        return False

    def mark_existing_link(self, article_url: str, canonical_url: str | None = None) -> None:
        self._existing_link_keys.add(self.link_key(article_url))
        if canonical_url:
            self._existing_link_keys.add(self.link_key(canonical_url))

    def mark_seen_document(self, canonical_url: str, checksum: str) -> None:
        if canonical_url and checksum:
            self._seen_exact_document_keys.add(exact_document_identity_key(self.source_name, canonical_url, checksum))

    def link_key(self, url: str) -> str:
        return raw_document_identity_key(self.source_name, url)

    def request_delay(self) -> None:
        delay = float(getattr(self.config, "request_delay_seconds", 0.0))
        if delay > 0:
            time.sleep(delay)

    def discover_article_urls(self, max_articles: int | None = None) -> list[str]:
        article_limit = _article_limit(max_articles)
        discovered_urls: list[str] = []
        seen_urls: set[str] = set()

        if bool(getattr(self.config, "crawl_homepage", True)):
            homepage_url = str(getattr(self.config, "homepage_url", "") or "")
            if homepage_url:
                self._discover_from_page(homepage_url, discovered_urls, seen_urls, article_limit)
                if _has_reached_article_limit(discovered_urls, article_limit):
                    logger.info("[CRAWL] Discovered %s urls source=%s", len(discovered_urls), self.source_name)
                    return discovered_urls

        max_pages = int(getattr(self.config, "max_pages_per_category", 1))
        for category_path in getattr(self.config, "category_paths", ()):
            stale_pages = 0
            page_number = 1
            while True:
                page_new_urls = 0
                for page_url in self.build_category_page_urls(str(category_path), page_number):
                    page_new_urls += self._discover_from_page(page_url, discovered_urls, seen_urls, article_limit)
                    if _has_reached_article_limit(discovered_urls, article_limit):
                        logger.info("[CRAWL] Discovered %s urls source=%s", len(discovered_urls), self.source_name)
                        return discovered_urls

                if max_pages > 0 and page_number >= max_pages:
                    break
                if max_pages <= 0:
                    stale_pages = stale_pages + 1 if page_new_urls == 0 else 0
                    if stale_pages >= 2:
                        break
                page_number += 1

        logger.info("[CRAWL] Discovered %s urls source=%s", len(discovered_urls), self.source_name)
        return discovered_urls

    def _discover_from_page(
        self,
        page_url: str,
        discovered_urls: list[str],
        seen_urls: set[str],
        article_limit: int | None,
    ) -> int:
        response = self.fetch_url(page_url)
        if not response.ok or not response.text:
            return 0

        new_count = 0
        for article_url in self.extract_article_urls_from_html(response.final_url or page_url, response.text):
            if article_url in seen_urls:
                continue
            seen_urls.add(article_url)
            discovered_urls.append(article_url)
            new_count += 1
            if _has_reached_article_limit(discovered_urls, article_limit):
                break
        return new_count

    def fetch_article_pages(self, urls: list[str], crawl_run_id: str) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for url in urls:
            logger.info("[CRAWL] Fetch article source=%s url=%s", self.source_name, url)
            response = self.fetch_url(url)
            if not response.ok or not response.text:
                continue
            article = self.build_parsed_article(
                url=response.final_url or url,
                html=response.text,
                http_status=response.status_code,
            )
            if article is None:
                continue
            documents.append(self.build_raw_document(article=article, crawl_run_id=crawl_run_id))
        return documents

    def extract_article_urls_from_html(self, page_url: str, html: str) -> list[str]:
        soup = _soup(html)
        if soup is None:
            return []

        urls: list[str] = []
        seen_urls: set[str] = set()
        selectors = self.article_link_selectors or ("a[href]",)
        for selector in selectors:
            for node in soup.select(selector):
                href = node.get("href")
                if not href:
                    continue
                absolute_url = normalize_url(urljoin(page_url, str(href)))
                if not absolute_url or not self.is_article_url(absolute_url):
                    continue
                if absolute_url in seen_urls:
                    continue
                seen_urls.add(absolute_url)
                urls.append(absolute_url)
        return urls

    def extract_paragraph_text(self, html: str) -> str:
        soup = _soup(html)
        if soup is None:
            return clean_text(html)

        root = self._select_first_node(soup, self.content_selectors) or soup
        paragraphs = [
            clean_text(paragraph.get_text(" ", strip=True))
            for paragraph in root.find_all("p")
        ]
        selected = [paragraph for paragraph in paragraphs if len(paragraph) >= self.min_paragraph_chars]
        if selected:
            return clean_text(" ".join(selected))
        return clean_text(root.get_text(" ", strip=True))

    def build_parsed_article(self, url: str, html: str, http_status: int | None) -> ParsedArticle | None:
        soup = _soup(html)
        if soup is None:
            return None

        canonical_url = self.extract_canonical_url(soup, url)
        title = self.extract_first_text(soup, self.title_selectors) or self.extract_meta_content(soup, "og:title")
        summary = self.extract_first_text(soup, self.summary_selectors) or self.extract_meta_content(
            soup,
            "description",
            name_attr="name",
        )
        content = self.extract_paragraph_text(html)

        if not title:
            logger.debug("[CRAWL] Skip article without title source=%s url=%s", self.source_name, url)
            return None
        if len(content) < self.min_content_chars:
            logger.debug(
                "[CRAWL] Skip short article source=%s url=%s content_len=%s",
                self.source_name,
                url,
                len(content),
            )
            return None

        return ParsedArticle(
            source_name=self.source_name,
            url=url,
            canonical_url=canonical_url,
            title=clean_text(title),
            summary=clean_text(summary) or None,
            content=content,
            category=self.extract_first_text(soup, self.category_selectors)
            or self.extract_meta_content(soup, "article:section"),
            published_at=self.extract_first_text(soup, self.published_at_selectors)
            or self.extract_meta_content(soup, "article:published_time"),
            author=self.extract_first_text(soup, self.author_selectors)
            or self.extract_meta_content(soup, "author", name_attr="name"),
            image=self.extract_meta_content(soup, "og:image"),
            crawled_at=utc_now(),
            content_hash=hash_content(content),
            tags=self.extract_tags(soup),
            raw_html=html,
            http_status=http_status,
            metadata={"parser": self.__class__.__name__},
        )

    def build_raw_document(self, article: ParsedArticle, crawl_run_id: str) -> RawDocument:
        now = utc_now()
        raw_payload = {
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
            "category": article.category,
            "source_category_name": article.metadata.get("source_category_name"),
            "source_category_url": article.metadata.get("source_category_url") or article.category_url,
            "published_at": article.published_at,
            "crawled_at": (article.crawled_at or now).isoformat(),
            "content_hash": article.content_hash or self.build_checksum(article),
            "author": article.author,
            "image": article.image,
            "tags": article.tags,
            "category_url": article.category_url,
            "category_page": article.category_page,
            "html": article.raw_html,
        }
        raw_payload_text = json.dumps(raw_payload, ensure_ascii=False)
        metadata = {
            **article.metadata,
            "source_name": article.source_name,
            "article_url": article.url,
            "category": article.category,
            "category_url": article.category_url,
            "source_category_name": article.metadata.get("source_category_name"),
            "source_category_url": article.metadata.get("source_category_url") or article.category_url,
            "category_page": article.category_page,
            "canonical_url": article.canonical_url,
            "title": article.title,
            "summary_raw": article.summary,
            "published_at": article.published_at,
            "crawled_at": (article.crawled_at or now).isoformat(),
            "content_hash": article.content_hash or self.build_checksum(article),
        }
        checksum = self.build_checksum(article)
        raw_document_id = self.build_raw_document_id(article)
        return RawDocument(
            raw_document_id=raw_document_id,
            source_name=article.source_name,
            crawl_run_id=crawl_run_id,
            url=article.url,
            canonical_url=article.canonical_url,
            page_type="article",
            fetch_status="success",
            http_status=article.http_status,
            fetched_at=now,
            payload_format="json",
            raw_payload=raw_payload_text,
            raw_text=article.content,
            checksum=checksum,
            metadata=json.dumps(metadata, ensure_ascii=False),
            created_at=now,
        )

    def build_category_urls(self, category_path: str, max_pages: int) -> list[str]:
        effective_max_pages = max(1, max_pages)
        urls: list[str] = []
        for page_number in range(1, effective_max_pages + 1):
            urls.extend(self.build_category_page_urls(category_path, page_number))
        return urls

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        if page_number <= 1:
            return [f"{base_url}/{path}"]
        return [f"{base_url}/{path}?page={page_number}"]

    def is_article_url(self, url: str) -> bool:
        normalized = normalize_url(url)
        return any(re.search(pattern, normalized) for pattern in self.article_url_patterns)

    def extract_first_text(self, soup: Any, selectors: tuple[str, ...]) -> str | None:
        node = self._select_first_node(soup, selectors)
        if node is None:
            return None
        text = clean_text(node.get_text(" ", strip=True))
        return text or None

    def extract_meta_content(self, soup: Any, key: str, name_attr: str = "property") -> str | None:
        node = soup.find("meta", attrs={name_attr: key})
        if node is None and name_attr != "name":
            node = soup.find("meta", attrs={"name": key})
        if node is None:
            return None
        content = clean_text(str(node.get("content") or ""))
        return content or None

    def extract_canonical_url(self, soup: Any, fallback_url: str) -> str:
        og_url = self.extract_meta_content(soup, "og:url")
        if og_url:
            return normalize_url(og_url)
        canonical = soup.find("link", rel=lambda rel: rel and "canonical" in rel)
        if canonical is not None:
            href = canonical.get("href")
            if href:
                return normalize_url(urljoin(fallback_url, str(href)))
        return normalize_url(fallback_url)

    def extract_tags(self, soup: Any) -> list[str]:
        tags: list[str] = []
        seen_tags: set[str] = set()
        for selector in self.tag_selectors:
            for node in soup.select(selector):
                tag = clean_text(node.get_text(" ", strip=True))
                if tag and tag not in seen_tags:
                    seen_tags.add(tag)
                    tags.append(tag)
        return tags

    def _discovery_page_urls(self) -> list[str]:
        page_urls: list[str] = []
        if bool(getattr(self.config, "crawl_homepage", True)):
            homepage_url = str(getattr(self.config, "homepage_url", "") or "")
            if homepage_url:
                page_urls.append(homepage_url)

        max_pages = max(1, int(getattr(self.config, "max_pages_per_category", 1)))
        for category_path in getattr(self.config, "category_paths", ()):
            page_urls.extend(self.build_category_urls(str(category_path), max_pages=max_pages))

        deduped: list[str] = []
        seen: set[str] = set()
        for page_url in page_urls:
            normalized = normalize_url(page_url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    def _configured_categories(self) -> list[Category]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        categories: list[Category] = []
        seen_urls: set[str] = set()
        for category_path in getattr(self.config, "category_paths", ()):
            category_url = normalize_url(urljoin(base_url + "/", str(category_path).strip("/")))
            if not category_url or category_url in seen_urls:
                continue
            seen_urls.add(category_url)
            categories.append(Category(name=_category_name_from_url(category_url), url=category_url))
        return categories

    @staticmethod
    def _select_first_node(soup: Any, selectors: tuple[str, ...]) -> Any | None:
        for selector in selectors:
            node = soup.select_one(selector)
            if node is not None:
                return node
        return None

    def build_checksum(self, article: ParsedArticle) -> str:
        return article.content_hash or hash_content(article.content)

    def build_raw_document_id(self, article: ParsedArticle) -> str:
        value = raw_document_identity_key(article.source_name, article.canonical_url)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _article_limit(max_articles: int | None) -> int | None:
    if max_articles is None or max_articles <= 0:
        return None
    return max_articles


def _has_reached_article_limit(discovered_urls: list[str], article_limit: int | None) -> bool:
    return article_limit is not None and len(discovered_urls) >= article_limit


def raw_document_identity_key(source_name: str, canonical_url: str) -> str:
    return f"{source_name}|{normalize_url(canonical_url)}"


def exact_document_identity_key(source_name: str, canonical_url: str, checksum: str) -> str:
    return f"{raw_document_identity_key(source_name, canonical_url)}|{checksum}"


def _category_name_from_url(category_url: str) -> str:
    path = urlsplit(category_url).path.strip("/")
    if not path:
        return category_url
    return path.rsplit("/", 1)[-1].replace("-", " ").replace(".chn", "").strip()


def _source_category_path(category_url: str) -> str:
    path = urlsplit(category_url).path.strip("/")
    return path or category_url


def _fetch_with_requests(url: str, timeout: int, headers: dict[str, str]) -> FetchResponse:
    try:
        import requests
    except ImportError:
        return _fetch_with_urllib(url, timeout, headers)

    response = requests.get(url, timeout=timeout, headers=headers)
    encoding = response.encoding or response.apparent_encoding or "utf-8"
    response.encoding = encoding
    return FetchResponse(
        requested_url=url,
        final_url=str(response.url),
        status_code=response.status_code,
        text=response.text,
        error=None,
    )


def _fetch_with_urllib(url: str, timeout: int, headers: dict[str, str]) -> FetchResponse:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        raw_body = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        return FetchResponse(
            requested_url=url,
            final_url=str(response.geturl()),
            status_code=getattr(response, "status", None),
            text=raw_body.decode(charset, errors="replace"),
            error=None,
        )


def _soup(html: str) -> Any | None:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 is required for HTML crawlers")
        return None
    return BeautifulSoup(html, "html.parser")
