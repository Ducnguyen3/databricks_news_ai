from __future__ import annotations

from urllib.parse import urlsplit

from app.ingestion.crawlers.base_crawler import BaseCrawler, Category

VNEXPRESS_ALLOWED_CATEGORY_PATHS = {
    "thoi-su",
    "thoi-su/chinh-tri",
    "thoi-su/huong-toi-ky-nguyen-moi",
    "thoi-su/dan-sinh",
    "thoi-su/lao-dong-viec-lam",
    "thoi-su/giao-thong",
    "thoi-su/quy-hy-vong",
    "the-gioi",
    "the-gioi/phan-tich",
    "the-gioi/tu-lieu",
    "the-gioi/quan-su",
    "the-gioi/cuoc-song-do-day",
    "the-gioi/nguoi-viet-5-chau",
    "the-gioi/bac-my",
    "kinh-doanh",
    "kinh-doanh/net-zero",
    "kinh-doanh/quoc-te",
    "kinh-doanh/doanh-nghiep",
    "kinh-doanh/chung-khoan",
    "kinh-doanh/ebank",
    "kinh-doanh/vi-mo",
    "kinh-doanh/tien-cua-toi",
    "kinh-doanh/hang-hoa",
    "kinh-doanh/kinh-te-vung",
    "kinh-doanh/doanh-nghiep-vuon-minh",
    "khoa-hoc-cong-nghe",
    "khoa-hoc-cong-nghe/bo-khoa-hoc-va-cong-nghe",
    "khoa-hoc-cong-nghe/chuyen-doi-so",
    "khoa-hoc-cong-nghe/doi-moi-sang-tao",
    "khoa-hoc-cong-nghe/ai",
    "khoa-hoc-cong-nghe/vu-tru",
    "khoa-hoc-cong-nghe/the-gioi-tu-nhien",
    "khoa-hoc-cong-nghe/thiet-bi",
    "khoa-hoc-cong-nghe/cua-so-tri-thuc",
    "khoa-hoc-cong-nghe/cuoc-thi-sang-kien-khoa-hoc",
    "goc-nhin",
    "goc-nhin/chinh-tri-chinh-sach",
    "goc-nhin/y-te-suc-khoe",
    "goc-nhin/kinh-doanh-quan-tri",
    "goc-nhin/giao-duc-tri-thuc",
    "goc-nhin/moi-truong",
    "goc-nhin/van-hoa-loi-song",
    "goc-nhin/tac-gia",
    "bat-dong-san",
    "bat-dong-san/chinh-sach",
    "bat-dong-san/thi-truong",
    "bat-dong-san/du-an",
    "bat-dong-san/khong-gian-song",
    "bat-dong-san/tu-van",
    "suc-khoe",
    "suc-khoe/tin-tuc",
    "suc-khoe/cac-benh",
    "suc-khoe/song-khoe",
    "suc-khoe/vaccine",
    "giai-tri",
    "giai-tri/gioi-sao",
    "giai-tri/sach",
    "giai-tri/phim",
    "giai-tri/nhac",
    "giai-tri/thoi-trang",
    "giai-tri/lam-dep",
    "giai-tri/san-khau-my-thuat",
    "the-thao",
    "the-thao/world-cup-2026",
    "bong-da",
    "the-thao/marathon",
    "the-thao/tennis",
    "the-thao/cac-mon-khac",
    "the-thao/hau-truong",
    "phap-luat",
    "phap-luat/ho-so-pha-an",
    "phap-luat/tu-van",
    "giao-duc",
    "giao-duc/tin-tuc",
    "giao-duc/tuyen-sinh",
    "giao-duc/chan-dung",
    "giao-duc/du-hoc",
    "giao-duc/thao-luan",
    "giao-duc/hoc-tieng-anh",
    "giao-duc/giao-duc-40",
    "doi-song",
    "doi-song/nhip-song",
    "doi-song/to-am",
    "doi-song/bai-hoc-song",
    "doi-song/cooking",
    "doi-song/tieu-dung",
    "oto-xe-may",
    "oto-xe-may/thi-truong",
    "oto-xe-may/xe-dien",
    "oto-xe-may/dien-dan",
    "oto-xe-may/v-car",
    "oto-xe-may/v-bike",
    "oto-xe-may/cam-lai",
    "du-lich",
    "du-lich/diem-den",
    "du-lich/am-thuc",
    "du-lich/dau-chan",
    "du-lich/tu-van",
    "du-lich/cam-nang",
    "y-kien",
    "y-kien/thoi-su",
    "y-kien/doi-song",
    "tam-su",
    "tam-su/hen-ho",
    "thu-gian",
    "thu-gian/cuoi",
    "thu-gian/do-vui",
    "thu-gian/chuyen-la",
    "thu-gian/crossword",
    "thu-gian/thu-cung",
    "thu-gian/tro-choi",
}

VNEXPRESS_STRUCTURED_PATHS = {
    "the-thao/du-lieu-bong-da",
}


class VnExpressCrawler(BaseCrawler):
    source_name = "vnexpress"
    article_link_selectors = (
        "article.item-news a[href]",
        "h3.title-news a[href]",
        ".title-news a[href]",
        "a[href]",
    )
    article_url_patterns = (r"^https://vnexpress\.net/.+-\d+\.html(?:$|\?)",)
    title_selectors = ("h1.title-detail", "h1")
    summary_selectors = ("p.description", ".description")
    content_selectors = ("article.fck_detail", ".fck_detail")
    published_at_selectors = (".date", ".time", ".article-date")
    author_selectors = (".author", "p.author", ".Normal strong")
    category_selectors = (".breadcrumb li:last-child a", ".breadcrumb a:last-child")
    tag_selectors = (".tags a", ".tag_item a")

    def build_category_page_url(self, category_url: str, page: int) -> str:
        if page <= 1:
            return category_url.rstrip("/")
        return f"{category_url.rstrip('/')}-p{page}"

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        return [self.build_category_page_url(f"{base_url}/{path}", page_number)]

    def is_category_url(self, url: str) -> bool:
        if not super().is_category_url(url):
            return False
        path = _clean_category_path(urlsplit(url).path.strip("/"))
        return path in VNEXPRESS_ALLOWED_CATEGORY_PATHS

    def discover_categories(self, source_home_url: str) -> list[Category]:
        discovered = super().discover_categories(source_home_url)
        allowed_by_path = {
            _clean_category_path(urlsplit(category.url).path.strip("/")): category
            for category in discovered
        }
        configured = self._configured_categories()
        merged: list[Category] = []
        seen: set[str] = set()
        for category in [*configured, *allowed_by_path.values()]:
            path = _clean_category_path(urlsplit(category.url).path.strip("/"))
            if path not in VNEXPRESS_ALLOWED_CATEGORY_PATHS or path in seen:
                continue
            seen.add(path)
            merged.append(Category(name=path, url=category.url))
        return merged


def _clean_category_path(path: str) -> str:
    return path.strip("/").removesuffix(".html")
