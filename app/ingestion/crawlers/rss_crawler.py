from __future__ import annotations

from app.ingestion.crawlers.base_crawler import BaseCrawler


class RssCrawler(BaseCrawler):
    """Compatibility base for old RSS crawler imports.

    The crawl pipeline now uses HTML category discovery through BaseCrawler.
    """

