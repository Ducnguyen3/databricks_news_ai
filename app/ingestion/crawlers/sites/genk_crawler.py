from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from app.ingestion.crawlers.base_crawler import BaseCrawler, Category, _soup
from app.processing.canonicalizer import normalize_url

GENK_ALLOWED_CATEGORY_PATHS = {
    "mobile.chn",
    "ai.chn",
    "tin-ict.chn",
    "internet.chn",
    "kham-pha.chn",
    "xem-mua-luon.chn",
    "xe.chn",
    "apps-games.chn",
    "do-choi-so.chn",
    "mobile/dien-thoai.chn",
    "mobile/may-tinh-bang.chn",
    "internet/digital-marketing.chn",
    "internet/media.chn",
    "kham-pha/lich-su.chn",
    "kham-pha/tri-thuc.chn",
    "tra-da-cong-nghe.chn",
    "tra-da-cong-nghe/tan-man.chn",
    "tra-da-cong-nghe/y-tuong-sang-tao.chn",
    "blockchain.chn",
    "blockchain/xu-huong.chn",
    "blockchain/cong-nghe.chn",
    "blockchain/nhan-vat.chn",
    "thu-thuat.chn",
    "song.chn",
    "nhom-chu-de/emagazine.chn",
    "gia-dung.chn",
}

_ARTICLE_ID_PATTERN = re.compile(r"-\d{10,}\.chn$")


class GenKCrawler(BaseCrawler):
    source_name = "genk"
    article_link_selectors = (
        "h3.knswli-title a[href]",
        ".knswli-right h3 a[href]",
        ".news-item a[href]",
        "h3 a[href]",
        "a[href]",
    )
    article_url_patterns = (
        r"^https://genk\.vn/.+-\d+\.chn(?:$|\?)",
        r"^https://genk\.vn/.+\.html(?:$|\?)",
    )
    title_selectors = ("h1.kbwc-title", "h1.knc-title", "h1")
    summary_selectors = (".knc-sapo", ".kbwc-sapo", ".sapo")
    content_selectors = (".knc-content", ".kbwc-content", ".news-content", ".detail-content")
    published_at_selectors = (".kbwcm-time", ".knc-date", ".time")
    author_selectors = (".kbwcm-author", ".author", "p.author")
    category_selectors = (".kbwcb-left a:last-child", ".breadcrumb a:last-child")
    tag_selectors = (".tags a", ".tag a")

    def __init__(self, source_config: object) -> None:
        super().__init__(source_config)
        self._active_category_url: str | None = None
        self._next_page_by_category_url: dict[str, str] = {}

    def build_category_page_url(self, category_url: str, page: int) -> str | None:
        category_key = normalize_url(category_url.rstrip("/"))
        self._active_category_url = category_key
        if page <= 1:
            return category_url.rstrip("/")
        return self._next_page_by_category_url.pop(category_key, None)

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        page_url = self.build_category_page_url(f"{base_url}/{path}", page_number)
        return [page_url] if page_url else []

    def discover_article_urls(self, max_articles: int | None = None) -> list[str]:
        article_limit = max_articles if max_articles and max_articles > 0 else None
        discovered_urls: list[str] = []
        seen_urls: set[str] = set()
        max_pages = max(1, int(getattr(self.config, "max_pages_per_category", 1)))

        if bool(getattr(self.config, "crawl_homepage", True)):
            homepage_url = str(getattr(self.config, "homepage_url", "") or "")
            if homepage_url:
                for article_url in self.extract_article_links(homepage_url):
                    if article_url not in seen_urls:
                        seen_urls.add(article_url)
                        discovered_urls.append(article_url)
                    if article_limit is not None and len(discovered_urls) >= article_limit:
                        return discovered_urls

        for category_path in getattr(self.config, "category_paths", ()):
            base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
            category_url = f"{base_url}/{str(category_path).strip('/')}"
            for page in range(1, max_pages + 1):
                page_url = self.build_category_page_url(category_url, page)
                if not page_url:
                    break
                for article_url in self.extract_article_links(page_url):
                    if article_url in seen_urls:
                        continue
                    seen_urls.add(article_url)
                    discovered_urls.append(article_url)
                    if article_limit is not None and len(discovered_urls) >= article_limit:
                        return discovered_urls
        return discovered_urls

    def is_category_url(self, url: str) -> bool:
        if not super().is_category_url(url):
            return False
        path = urlsplit(url).path.strip("/")
        return path in GENK_ALLOWED_CATEGORY_PATHS

    def is_article_url(self, url: str) -> bool:
        return is_valid_genk_article_url(url, GENK_ALLOWED_CATEGORY_PATHS)

    def extract_article_links(self, category_page_url: str) -> list[str]:
        response = self.fetch_url(category_page_url)
        if not response.ok:
            raise RuntimeError(f"request_error status={response.status_code} url={category_page_url}")
        if not response.text:
            self._store_next_page(category_page_url, next_genk_ajax_category_url(category_page_url))
            return []
        final_url = response.final_url or category_page_url
        next_url = extract_next_page_url_from_genk_html(final_url, response.text)
        if not next_url:
            next_url = extract_genk_ajax_category_url_from_html(final_url, response.text)
        if not next_url:
            next_url = next_genk_ajax_category_url(final_url)
        self._store_next_page(final_url, next_url)
        return self.extract_article_urls_from_html(final_url, response.text)

    def discover_categories(self, source_home_url: str) -> list[Category]:
        discovered = super().discover_categories(source_home_url)
        allowed_by_path = {urlsplit(category.url).path.strip("/"): category for category in discovered}
        configured = self._configured_categories()
        merged: list[Category] = []
        seen: set[str] = set()
        for category in [*configured, *allowed_by_path.values()]:
            path = urlsplit(category.url).path.strip("/")
            if path not in GENK_ALLOWED_CATEGORY_PATHS or path in seen:
                continue
            seen.add(path)
            merged.append(Category(name=path, url=category.url))
        return merged

    def _store_next_page(self, page_url: str, next_url: str | None) -> None:
        if self._active_category_url and next_url and normalize_url(next_url) != normalize_url(page_url):
            self._next_page_by_category_url[self._active_category_url] = normalize_url(next_url)


def build_genk_category_page_url(
    base_url: str,
    category_path: str,
    page: int,
    previous_html: str | None = None,
) -> str | None:
    category_url = f"{base_url.rstrip('/')}/{category_path.strip('/')}"
    if page <= 1:
        return category_url
    if previous_html:
        return extract_next_page_url_from_genk_html(category_url, previous_html) or extract_genk_ajax_category_url_from_html(
            category_url,
            previous_html,
        )
    return None


def extract_next_page_url_from_genk_html(page_url: str, html: str | None) -> str | None:
    if not html:
        return None
    soup = _soup(html)
    if soup is None:
        return None
    selectors = (
        "a[rel='next'][href]",
        ".pagination a.next[href]",
        ".pagination a[href]",
        ".page a[href]",
        "a.next[href]",
    )
    for selector in selectors:
        for node in soup.select(selector):
            href = str(node.get("href") or "").strip()
            text = " ".join(node.get_text(" ", strip=True).split()).casefold()
            if not href:
                continue
            if selector.endswith("a[href]") and text not in {"next", ">", "sau", "trang sau"}:
                continue
            next_url = normalize_url(urljoin(page_url, href))
            if next_url and next_url.startswith("https://genk.vn/"):
                return next_url
    return None


def extract_genk_ajax_category_url_from_html(page_url: str, html: str | None) -> str | None:
    if not html:
        return None
    soup = _soup(html)
    if soup is None:
        return None
    for node in soup.select("[data-api-url], [data-url], [data-href], a[href]"):
        for attr in ("data-api-url", "data-url", "data-href", "href"):
            value = str(node.get(attr) or "").strip()
            if not value or "ajax-cate" not in value:
                continue
            ajax_url = normalize_url(urljoin(page_url, value))
            if ajax_url and ajax_url.startswith("https://genk.vn/ajax-cate/"):
                return ajax_url
    match = re.search(r"(/ajax-cate/page-\d+-c\d+\.chn)", html)
    if match:
        ajax_url = normalize_url(urljoin(page_url, match.group(1)))
        if ajax_url and ajax_url.startswith("https://genk.vn/ajax-cate/"):
            return ajax_url
    return None


def next_genk_ajax_category_url(url: str) -> str | None:
    normalized = normalize_url(url)
    split = urlsplit(normalized)
    match = re.match(r"^/ajax-cate/page-(\d+)-c(\d+)\.chn$", split.path)
    if not match:
        return None
    try:
        next_page = int(match.group(1)) + 1
    except ValueError:
        return None
    category_id = match.group(2)
    return normalize_url(f"{split.scheme}://{split.netloc}/ajax-cate/page-{next_page}-c{category_id}.chn")


def is_valid_genk_article_url(url: str, category_paths: set[str] | frozenset[str] | None = None) -> bool:
    normalized = normalize_url(url)
    if not normalized.startswith("https://genk.vn/"):
        return False
    if normalized.startswith("javascript:"):
        return False
    path = urlsplit(normalized).path.strip("/")
    if not path or path in set(category_paths or GENK_ALLOWED_CATEGORY_PATHS):
        return False
    if path.startswith("rss/") or "/rss/" in path or path.startswith("video/") or "/video/" in path:
        return False
    return bool(_ARTICLE_ID_PATTERN.search(path))
