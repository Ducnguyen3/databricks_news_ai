from __future__ import annotations

from app.ingestion.crawlers.base_crawler import BaseCrawler


class DienDanDoanhNghiepCrawler(BaseCrawler):
    source_name = "diendandoanhnghiep"
    article_link_selectors = (
        ".news-item a[href]",
        ".article-item a[href]",
        "h3 a[href]",
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

    def build_category_page_url(self, category_url: str, page: int) -> str:
        if page <= 1:
            return category_url.rstrip("/")
        return f"{category_url.rstrip('/')}?page={page}"

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        if page_number <= 1:
            return [f"{base_url}/{path}"]
        return [
            f"{base_url}/{path}?page={page_number}",
            f"{base_url}/{path}?p={page_number}",
            f"{base_url}/{path}/trang-{page_number}",
            f"{base_url}/{path}/page/{page_number}",
        ]
