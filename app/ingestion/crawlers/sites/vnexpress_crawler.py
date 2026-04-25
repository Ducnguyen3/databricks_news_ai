from __future__ import annotations

from app.ingestion.crawlers.base_crawler import BaseCrawler


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
