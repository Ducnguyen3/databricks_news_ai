from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from app.ingestion.crawlers.base_crawler import BaseCrawler, Category
from app.processing.canonicalizer import normalize_url

CAFEF_ALLOWED_CATEGORY_PATHS = {
    "xa-hoi.chn",
    "thi-truong-chung-khoan.chn",
    "bat-dong-san.chn",
    "doanh-nghiep.chn",
    "tai-chinh-ngan-hang.chn",
    "smart-money.chn",
    "tai-chinh-quoc-te.chn",
    "vi-mo-dau-tu.chn",
    "kinh-te-so.chn",
    "thi-truong.chn",
    "song.chn",
    "lifestyle.chn",
}


@dataclass(frozen=True, slots=True)
class CafeFAjaxState:
    zone_id: str | None
    zone_url: str | None
    cd_keys: tuple[str, ...]


class CafeFCrawler(BaseCrawler):
    source_name = "cafef"
    article_link_selectors = (
        "[role='article'] a[href]",
        ".category-page__top a[href]",
        ".category-page__list a[href]",
        "h3 a[href]",
        ".tlitem a[href]",
        ".box-category-item a[href]",
        ".avatar[href]",
        ".item a[href]",
    )
    article_url_patterns = (
        r"^https?://cafef\.vn/.+-\d{12,}\.chn(?:$|\?)",
        r"^https://cafef\.vn/.+\.html(?:$|\?)",
    )
    title_selectors = ("h1.title", "h1.titledetail", "h1")
    summary_selectors = ("h2.sapo", ".sapo")
    content_selectors = (".detail-content", ".fck_detail", ".contentdetail")
    published_at_selectors = (".time", ".pdate", ".date")
    author_selectors = (".author", "p.author")
    category_selectors = (".cat", ".breadcrumb a:last-child")
    tag_selectors = (".tags a", ".tag a")

    def build_category_page_url(self, category_url: str, page: int) -> str:
        if page <= 1:
            return category_url.rstrip("/")
        return f"{category_url.rstrip('/').removesuffix('.chn')}/trang-{page}.chn"

    def extract_article_links(self, category_page_url: str) -> list[str]:
        response = self.fetch_url(category_page_url)
        if not response.ok:
            ajax_links = self._extract_ajax_links_for_category_page(category_page_url)
            if ajax_links:
                return ajax_links
            raise RuntimeError(f"request_error status={response.status_code} url={category_page_url}")
        if not response.text:
            return []
        links = self.extract_article_urls_from_html(response.final_url or category_page_url, response.text)
        if links:
            return links
        ajax_links = self._extract_ajax_links_for_category_page(response.final_url or category_page_url)
        return ajax_links

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        return [self.build_category_page_url(f"{base_url}/{path}", page_number)]

    def is_category_url(self, url: str) -> bool:
        if not super().is_category_url(url):
            return False
        path = urlsplit(url).path.strip("/")
        return path in CAFEF_ALLOWED_CATEGORY_PATHS

    def discover_categories(self, source_home_url: str) -> list[Category]:
        discovered = super().discover_categories(source_home_url)
        allowed_by_path = {urlsplit(category.url).path.strip("/"): category for category in discovered}
        configured = self._configured_categories()
        merged: list[Category] = []
        seen: set[str] = set()
        for category in [*configured, *allowed_by_path.values()]:
            path = urlsplit(category.url).path.strip("/")
            if path not in CAFEF_ALLOWED_CATEGORY_PATHS or path in seen:
                continue
            seen.add(path)
            merged.append(Category(name=path, url=category.url))
        return merged

    def _extract_ajax_links_for_category_page(self, category_page_url: str) -> list[str]:
        page = extract_cafef_page_number(category_page_url)
        if page <= 1:
            return []
        category_url = cafef_original_category_url(category_page_url)
        category_response = self.fetch_url(category_url)
        if not category_response.ok or not category_response.text:
            return []
        state = extract_cafef_ajax_state(category_response.text)
        links: list[str] = []
        seen: set[str] = set()
        for ajax_url in build_cafef_ajax_candidate_urls(category_url, state, page):
            ajax_response = self.fetch_url(ajax_url)
            if not ajax_response.ok or not ajax_response.text:
                continue
            for article_url in extract_cafef_article_urls_from_ajax_payload(
                self,
                ajax_response.final_url or category_url,
                ajax_response.text,
            ):
                if article_url in seen:
                    continue
                seen.add(article_url)
                links.append(article_url)
            if links:
                return links
        return links


def extract_cafef_page_number(url: str) -> int:
    path = urlsplit(url).path
    match = re.search(r"/trang-(\d+)\.chn$", path)
    if not match:
        return 1
    try:
        return int(match.group(1))
    except ValueError:
        return 1


def cafef_original_category_url(category_page_url: str) -> str:
    split = urlsplit(category_page_url)
    path = re.sub(r"/trang-\d+\.chn$", ".chn", split.path)
    return normalize_url(f"{split.scheme}://{split.netloc}{path}")


def extract_cafef_ajax_state(html: str) -> CafeFAjaxState:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return CafeFAjaxState(zone_id=None, zone_url=None, cd_keys=())
    soup = BeautifulSoup(html, "html.parser")
    zone_id_node = soup.select_one("#hdZoneId")
    zone_url_node = soup.select_one("#hdZoneUrl")
    raw_cd_keys: list[str] = []
    seen: set[str] = set()
    for attr_name in ("data-cd-key", "data-key-cd"):
        for node in soup.select(f"[{attr_name}]"):
            value = str(node.get(attr_name) or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            raw_cd_keys.append(value)
    zone_id = str(zone_id_node.get("value") or "").strip() if zone_id_node is not None else None
    cd_keys = tuple(_cafef_relevant_news_cd_keys(raw_cd_keys, zone_id))
    return CafeFAjaxState(
        zone_id=zone_id,
        zone_url=str(zone_url_node.get("value") or "").strip() if zone_url_node is not None else None,
        cd_keys=cd_keys,
    )


def build_cafef_ajax_candidate_urls(category_url: str, state: CafeFAjaxState, page: int) -> list[str]:
    base = f"{urlsplit(category_url).scheme}://{urlsplit(category_url).netloc}"
    zone_id = state.zone_id or _zone_id_from_cd_keys(state.cd_keys)
    zone_url = state.zone_url or urlsplit(category_url).path.strip("/").removesuffix(".chn")
    candidates: list[str] = []
    if zone_id:
        candidates.extend(
            [
                f"{base}/ajax/list-news/{zone_id}/{page}.chn",
                f"{base}/ajax/list-news/{zone_id}.chn?page={page}",
                f"{base}/ajax/newsinzone/{zone_id}/{page}.chn",
                f"{base}/ajax/news-in-zone/{zone_id}/{page}.chn",
                f"{base}/ajax/list-news-by-zone/{zone_id}/{page}.chn",
                f"{base}/ajax/timeline/{zone_id}/{page}.chn",
                f"{base}/ajax/cate/{zone_id}/{page}.chn",
                f"{base}/timelinelist/{zone_id}/{page}.chn",
                f"{base}/timeline/{zone_id}/trang-{page}.chn",
                f"{base}/ajax/get-news-by-zone?zoneId={zone_id}&page={page}",
                f"{base}/ajax/get-news-by-zone?zoneid={zone_id}&page={page}",
                f"{base}/ajax/getlistnews?zoneId={zone_id}&page={page}",
                f"{base}/ajax/getlistnews?zoneid={zone_id}&page={page}",
                f"{base}/ajax/load-more?zoneId={zone_id}&page={page}",
                f"{base}/ajax/load-more?zoneid={zone_id}&page={page}",
                f"{base}/ajax/loadmore?zoneId={zone_id}&page={page}",
                f"{base}/ajax/loadmore?zoneid={zone_id}&page={page}",
                f"{base}/ajax/loadmore?zoneId={zone_id}&pageIndex={page}",
                f"{base}/ajax/loadmore?zoneid={zone_id}&pageindex={page}",
            ]
        )
    if zone_url:
        candidates.extend(
            [
                f"{base}/ajax/{zone_url}/trang-{page}.chn",
                f"{base}/ajax/list-news?zoneUrl={zone_url}&page={page}",
                f"{base}/ajax/list-news?zoneurl={zone_url}&page={page}",
            ]
        )
    for cd_key in state.cd_keys:
        candidates.extend(
            [
                f"{base}/ajax/load-more?key={cd_key}&page={page}",
                f"{base}/ajax/load-more?cdKey={cd_key}&page={page}",
                f"{base}/ajax/load-more?cdkey={cd_key}&page={page}",
                f"{base}/ajax/loadmore?key={cd_key}&page={page}",
                f"{base}/ajax/loadmore?cdKey={cd_key}&page={page}",
                f"{base}/ajax/loadmore?cdkey={cd_key}&page={page}",
                f"{base}/ajax/list-news?key={cd_key}&page={page}",
                f"{base}/ajax/list-news?cdKey={cd_key}&page={page}",
                f"{base}/ajax/list-news?cdkey={cd_key}&page={page}",
            ]
        )
    return _dedupe_urls(candidates)


def extract_cafef_article_urls_from_ajax_payload(crawler: CafeFCrawler, base_url: str, payload: str) -> list[str]:
    stripped = payload.strip()
    if not stripped:
        return []
    if not stripped.startswith(("{", "[")):
        links = crawler.extract_article_urls_from_html(base_url, payload)
        if links:
            return links
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return _extract_article_urls_from_json_value(crawler, base_url, parsed)


def _extract_article_urls_from_json_value(crawler: CafeFCrawler, base_url: str, value: object) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    def add_url(raw_url: object) -> None:
        if not isinstance(raw_url, str) or not raw_url.strip():
            return
        normalized = normalize_url(urljoin(base_url, raw_url))
        if not normalized or normalized in seen or not crawler.is_article_url(normalized):
            return
        seen.add(normalized)
        links.append(normalized)

    def walk(item: object) -> None:
        if isinstance(item, str):
            if "<" in item and ">" in item:
                for url in crawler.extract_article_urls_from_html(base_url, item):
                    add_url(url)
            add_url(item)
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, dict):
            for key in ("url", "href", "link", "newsUrl", "news_url", "share_url", "Url", "Href", "Link"):
                add_url(item.get(key))
            for child in item.values():
                walk(child)

    walk(value)
    return links


def _zone_id_from_cd_keys(cd_keys: tuple[str, ...]) -> str | None:
    for cd_key in cd_keys:
        match = re.search(r"zone(?:id)?(\d+)", cd_key, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _cafef_relevant_news_cd_keys(cd_keys: list[str], zone_id: str | None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for cd_key in cd_keys:
        normalized = cd_key.lower()
        if "newsinzone" not in normalized:
            continue
        if "newsinzoneisonhome" in normalized:
            continue
        if zone_id and f"zone{zone_id}" not in normalized and f"zoneid{zone_id}" not in normalized:
            continue
        if cd_key in seen:
            continue
        seen.add(cd_key)
        output.append(cd_key)
    return output


def _dedupe_urls(urls: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
