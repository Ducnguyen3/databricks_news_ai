from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from app.ingestion.crawlers.base_crawler import BaseCrawler, Category, _soup
from app.processing.canonicalizer import normalize_url

DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS = {
    "tin-moi-nhat",
    "chinh-tri-xa-hoi",
    "chinh-tri-xa-hoi/chinh-tri",
    "chinh-tri-xa-hoi/tam-diem",
    "chinh-tri-xa-hoi/mat-tran",
    "chinh-tri-xa-hoi/kinh-te",
    "chinh-tri-xa-hoi/xa-hoi",
    "vcci",
    "vcci/phat-trien-ben-vung",
    "vcci/tieng-noi-cua-hiep-hoi-doanh-nghiep",
    "vcci/doanh-nghiep-hang-dau-viet-nam",
    "vcci/dai-hoi-vcci-lan-thu-vii",
    "vcci/xuc-tien-dau-tu-thuong-mai",
    "vcci/tham-muu-chinh-sach",
    "doanh-nghiep",
    "doanh-nghiep/quan-tri",
    "doanh-nghiep/trach-nhiem-xa-hoi",
    "doanh-nghiep/chuyen-dong",
    "doanh-nghiep/giao-thuong",
    "doanh-nhan",
    "doanh-nhan/chuyen-lam-an",
    "doanh-nhan/ca-phe-doanh-nhan",
    "doanh-nhan/phong-cach-song",
    "doanh-nhan/suc-khoe",
    "doanh-nhan/khoa-hoc",
    "doanh-nhan/phong-thuy",
    "doanh-nhan/chat-luong-song",
    "khoi-nghiep",
    "khoi-nghiep/khoi-nghiep-quoc-gia",
    "khoi-nghiep/y-tuong-kinh-doanh",
    "khoi-nghiep/co-van-huan-luyen",
    "khoi-nghiep/so-tay-khoi-nghiep",
    "du-lich",
    "du-lich/trai-nghiem",
    "du-lich/hoat-dong-du-lich",
    "du-lich/hoi-nhap",
    "kinh-te-dia-phuong",
    "cong-nghe",
    "cong-nghe/kinh-te-so",
    "cong-nghe/ung-dung",
    "cong-nghe/chuyen-doi-so",
    "o-to-xe-may",
    "o-to-xe-may/dien-dan",
    "o-to-xe-may/thong-tin-thi-truong",
    "o-to-xe-may/san-pham",
    "o-to-xe-may/tu-van-ky-thuat",
    "doanh-nghiep-thi-truong",
    "doanh-nghiep-thi-truong/thong-tin-doanh-nghiep",
    "doanh-nghiep-thi-truong/san-pham-thi-truong",
    "ngan-hang-chung-khoan",
    "ngan-hang-chung-khoan/chung-khoan",
    "ngan-hang-chung-khoan/tin-dung-ngan-hang",
    "ngan-hang-chung-khoan/tai-chinh-doanh-nghiep",
    "ngan-hang-chung-khoan/thi-truong-vang",
    "ngan-hang-chung-khoan/dich-vu-tai-chinh",
    "ngan-hang-chung-khoan/tai-chinh-so",
    "ngan-hang-chung-khoan/chuyen-de",
    "bat-dong-san",
    "bat-dong-san/thi-truong",
    "bat-dong-san/doanh-nghiep-du-an",
    "bat-dong-san/chinh-sach-quy-hoach",
    "bat-dong-san/cafe-dia-oc",
    "bat-dong-san/tien-do-du-an",
    "quoc-te",
    "quoc-te/doi-ngoai",
    "quoc-te/kinh-te-the-gioi",
    "quoc-te/phan-tich-binh-luan",
    "phap-luat",
    "phap-luat/nghien-cuu-trao-doi",
    "phap-luat/ban-doc",
    "phap-luat/kien-nghi",
    "phap-luat/24h",
    "phap-luat/nhin-thang-noi-that",
    "phap-luat/chong-hang-gia",
    "phap-luat/ho-so",
    "phap-luat/phap-dinh",
}

_ARTICLE_ID_PATTERN = re.compile(r"-\d{6,}\.html$")
_EVENT_PAGE_PATTERN = re.compile(r"-event\d+\.html$")

DIENDANDOANHNGHIEP_LOAD_MORE_API_BY_PATH = {
    "phap-luat/phap-dinh": "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179492_434_0",
}


class DienDanDoanhNghiepCrawler(BaseCrawler):
    source_name = "diendandoanhnghiep"
    article_link_selectors = (
        ".b-grid a[href]",
        "h2.b-grid__title a[href]",
        "h3.b-grid__title a[href]",
        ".b-grid__title a[href]",
        ".news-item a[href]",
        ".article-item a[href]",
        ".list-news a[href]",
        ".latest-news a[href]",
        "h3 a[href]",
        "h2 a[href]",
        "a[href$='.html']",
        "a[href]",
    )
    article_url_patterns = (r"^https://diendandoanhnghiep\.vn/.+\.html(?:$|\?)",)
    title_selectors = ("h1.detail-title", "h1.article-title", "h1")
    summary_selectors = (".detail-sapo", ".article-sapo", ".sapo")
    content_selectors = (".detail-content", ".article-content", ".content-detail", "article")
    published_at_selectors = (".detail-date", ".article-date", ".time", ".date")
    author_selectors = (".author", "p.author")
    category_selectors = (".breadcrumb a:last-child", ".category a")
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
        next_page = self._next_page_by_category_url.pop(category_key, None)
        if next_page:
            return next_page
        category_path = _clean_path(urlsplit(category_key).path.strip("/"))
        return DIENDANDOANHNGHIEP_LOAD_MORE_API_BY_PATH.get(category_path)

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
        path = _clean_path(urlsplit(url).path.strip("/"))
        return path in DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS

    def is_article_url(self, url: str) -> bool:
        return is_valid_diendandoanhnghiep_article_url(url)

    def extract_article_links(self, category_page_url: str) -> list[str]:
        response = self.fetch_url(category_page_url)
        if not response.ok:
            raise RuntimeError(f"request_error status={response.status_code} url={category_page_url}")
        if not response.text:
            self._store_next_page(category_page_url, next_diendandoanhnghiep_load_more_api_url(category_page_url))
            return []
        final_url = response.final_url or category_page_url
        next_url = extract_next_page_url_from_diendandoanhnghiep_html(final_url, response.text)
        if not next_url:
            next_url = extract_load_more_url_from_onecms_html(final_url, response.text)
        if not next_url:
            next_url = next_diendandoanhnghiep_load_more_api_url(final_url)
        self._store_next_page(final_url, next_url)
        return self.extract_article_urls_from_html(final_url, response.text)

    def discover_categories(self, source_home_url: str) -> list[Category]:
        discovered = super().discover_categories(source_home_url)
        allowed_by_path = {
            _clean_path(urlsplit(category.url).path.strip("/")): category
            for category in discovered
        }
        configured = self._configured_categories()
        merged: list[Category] = []
        seen: set[str] = set()
        for category in [*configured, *allowed_by_path.values()]:
            path = _clean_path(urlsplit(category.url).path.strip("/"))
            if path not in DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS or path in seen:
                continue
            seen.add(path)
            merged.append(Category(name=path, url=category.url))
        return merged

    def _store_next_page(self, page_url: str, next_url: str | None) -> None:
        if self._active_category_url and next_url and normalize_url(next_url) != normalize_url(page_url):
            self._next_page_by_category_url[self._active_category_url] = normalize_url(next_url)


def build_diendandoanhnghiep_category_page_url(
    base_url: str,
    category_path: str,
    page: int,
    previous_html: str | None = None,
) -> str | None:
    category_url = f"{base_url.rstrip('/')}/{category_path.strip('/')}"
    if page <= 1:
        return category_url
    if previous_html:
        return extract_next_page_url_from_diendandoanhnghiep_html(category_url, previous_html) or (
            extract_load_more_url_from_onecms_html(category_url, previous_html)
        )
    return None


def extract_next_page_url_from_diendandoanhnghiep_html(page_url: str, html: str | None) -> str | None:
    if not html:
        return None
    soup = _soup(html)
    if soup is None:
        return None
    selectors = (
        "a[rel='next'][href]",
        ".pagination a.next[href]",
        ".pagination a[href]",
        ".paging a[href]",
        "a.next[href]",
    )
    for selector in selectors:
        for node in soup.select(selector):
            href = str(node.get("href") or "").strip()
            text = " ".join(node.get_text(" ", strip=True).split()).casefold()
            if not href:
                continue
            if selector.endswith("a[href]") and text not in {"next", ">", "sau", "trang sau", "xem them", "xem thêm"}:
                continue
            next_url = normalize_url(urljoin(page_url, href))
            if next_url and next_url.startswith("https://diendandoanhnghiep.vn/"):
                return next_url
    return None


def extract_load_more_url_from_onecms_html(page_url: str, html: str | None) -> str | None:
    if not html:
        return None
    soup = _soup(html)
    if soup is None:
        return None
    for node in soup.select("[data-api-url], [data-url], [data-href]"):
        for attr in ("data-api-url", "data-url", "data-href"):
            value = str(node.get(attr) or "").strip()
            if value and "/api/getMoreArticle/" in value:
                load_more_url = normalize_url(urljoin(page_url, value))
                if load_more_url and load_more_url.startswith("https://diendandoanhnghiep.vn/"):
                    return load_more_url
    html_text = str(html)
    match = re.search(r"(/api/getMoreArticle/[A-Za-z0-9_/-]+)", html_text)
    if match:
        load_more_url = normalize_url(urljoin(page_url, match.group(1)))
        if load_more_url and load_more_url.startswith("https://diendandoanhnghiep.vn/"):
            return load_more_url
    return None


def next_diendandoanhnghiep_load_more_api_url(url: str) -> str | None:
    normalized = normalize_url(url)
    split = urlsplit(normalized)
    match = re.match(r"^/api/getMoreArticle/(.+)_(\d+)$", split.path)
    if not match:
        return None
    prefix = match.group(1)
    try:
        next_index = int(match.group(2)) + 1
    except ValueError:
        return None
    return normalize_url(f"{split.scheme}://{split.netloc}/api/getMoreArticle/{prefix}_{next_index}")


def is_valid_diendandoanhnghiep_article_url(url: str) -> bool:
    normalized = normalize_url(url)
    if not normalized.startswith("https://diendandoanhnghiep.vn/"):
        return False
    path = _clean_path(urlsplit(normalized).path.strip("/"))
    if not path.endswith(".html"):
        return False
    if path.startswith("an-pham/") or path.startswith("search"):
        return False
    if _EVENT_PAGE_PATTERN.search(path):
        return False
    return bool(_ARTICLE_ID_PATTERN.search(path))


def _clean_path(path: str) -> str:
    return path.strip("/")
