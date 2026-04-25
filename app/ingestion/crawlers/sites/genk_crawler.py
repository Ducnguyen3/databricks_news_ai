from __future__ import annotations

from app.ingestion.crawlers.base_crawler import BaseCrawler


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

    def build_category_page_url(self, category_url: str, page: int) -> str:
        if page <= 1:
            return category_url.rstrip("/")
        return f"{category_url.rstrip('/').removesuffix('.chn')}/trang-{page}.chn"

    def build_category_page_urls(self, category_path: str, page_number: int) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        path = category_path.strip("/")
        return [self.build_category_page_url(f"{base_url}/{path}", page_number)]
