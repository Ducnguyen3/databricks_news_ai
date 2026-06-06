from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config import CrawlSettings
from app.ingestion.crawlers.sites.vnexpress_crawler import (
    VNEXPRESS_ALLOWED_CATEGORY_PATHS,
    VNEXPRESS_STRUCTURED_PATHS,
    VnExpressCrawler,
)
from app.ingestion.crawlers.sites.vnexpress_sports_structured_crawler import VnExpressSportsStructuredCrawler
from app.ingestion.services.source_management_service import SourceManagementService


def _settings() -> CrawlSettings:
    return CrawlSettings(
        sources=("vnexpress",),
        max_articles_per_source=20,
        crawl_mode="category_pagination",
        discover_categories=True,
        max_pages_per_category=3,
        stop_after_empty_pages=1,
        stop_after_duplicate_pages=1,
        request_delay_seconds=0.0,
        max_concurrent_requests=1,
        request_timeout_seconds=1,
        retry_count=0,
        user_agent="test",
    )


def _crawler_config() -> SimpleNamespace:
    return SimpleNamespace(base_url="https://vnexpress.net", timeout_seconds=1, retry_count=0, user_agent="test")


class VnExpressCrawlerTest(unittest.TestCase):
    def test_vnexpress_has_expected_article_category_paths(self) -> None:
        expected = {
            "thoi-su",
            "the-gioi/quan-su",
            "kinh-doanh/doanh-nghiep",
            "kinh-doanh/chung-khoan",
            "khoa-hoc-cong-nghe/ai",
            "bat-dong-san/du-an",
            "suc-khoe",
            "giai-tri/phim",
            "the-thao",
            "bong-da",
            "phap-luat",
            "giao-duc/tuyen-sinh",
            "doi-song/cooking",
            "oto-xe-may/xe-dien",
            "du-lich/am-thuc",
            "y-kien/thoi-su",
            "tam-su/hen-ho",
            "thu-gian/tro-choi",
        }
        config = SourceManagementService(_settings()).get_source_config("vnexpress")

        self.assertTrue(expected.issubset(set(config.category_paths)))
        self.assertTrue(set(config.category_paths).issubset(VNEXPRESS_ALLOWED_CATEGORY_PATHS))
        self.assertNotIn("the-thao/du-lieu-bong-da", config.category_paths)

    def test_vnexpress_has_sports_structured_path(self) -> None:
        config = SourceManagementService(_settings()).get_source_config("vnexpress")

        self.assertEqual(("the-thao/du-lieu-bong-da",), config.structured_paths)
        self.assertEqual({"the-thao/du-lieu-bong-da"}, VNEXPRESS_STRUCTURED_PATHS)

    def test_vnexpress_category_page_url_builder(self) -> None:
        crawler = VnExpressCrawler(_crawler_config())

        self.assertEqual(
            "https://vnexpress.net/the-gioi/quan-su",
            crawler.build_category_page_url("https://vnexpress.net/the-gioi/quan-su", 1),
        )
        self.assertEqual(
            "https://vnexpress.net/the-gioi/quan-su-p2",
            crawler.build_category_page_url("https://vnexpress.net/the-gioi/quan-su", 2),
        )

    def test_vnexpress_crawl_does_not_use_priority_page_limit(self) -> None:
        crawler = VnExpressCrawler(_crawler_config())

        self.assertEqual(3, len(crawler.build_category_urls("kinh-doanh/chung-khoan", max_pages=3)))
        self.assertEqual(3, len(crawler.build_category_urls("the-thao", max_pages=3)))

    def test_vnexpress_category_whitelist_excludes_structured_path(self) -> None:
        crawler = VnExpressCrawler(_crawler_config())

        self.assertTrue(crawler.is_category_url("https://vnexpress.net/the-thao"))
        self.assertFalse(crawler.is_category_url("https://vnexpress.net/the-thao/du-lieu-bong-da"))

    def test_structured_sports_crawler_builds_configured_urls(self) -> None:
        config = SourceManagementService(_settings()).get_source_config("vnexpress")

        urls = VnExpressSportsStructuredCrawler(config).build_structured_page_urls()

        self.assertEqual(["https://vnexpress.net/the-thao/du-lieu-bong-da"], urls)


if __name__ == "__main__":
    unittest.main()
