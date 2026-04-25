from __future__ import annotations

from app.ingestion.crawlers.base_crawler import BaseCrawler


class CafeFCrawler(BaseCrawler):
    source_name = "cafef"
    article_link_selectors = (
        "h3 a[href]",
        ".tlitem a[href]",
        ".box-category-item a[href]",
        "a[href]",
    )
    article_url_patterns = (
        r"^https://cafef\.vn/.+-\d+\.chn(?:$|\?)",
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

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        return [self.build_category_page_url(f"{base_url}/{path}", page_number)]
