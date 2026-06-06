from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config import CrawlSettings
from app.domain.models import ParsedArticle
from app.ingestion.crawlers.base_crawler import FetchResponse
from app.ingestion.crawlers.sites.genk_crawler import (
    GENK_ALLOWED_CATEGORY_PATHS,
    GenKCrawler,
    build_genk_category_page_url,
    extract_genk_ajax_category_url_from_html,
    is_valid_genk_article_url,
    next_genk_ajax_category_url,
)
from app.utils.time import utc_now
from app.ingestion.services.source_management_service import SourceManagementService


def _settings(max_pages_per_category: int = 3) -> CrawlSettings:
    return CrawlSettings(
        sources=("genk",),
        max_articles_per_source=20,
        crawl_mode="category_pagination",
        discover_categories=True,
        max_pages_per_category=max_pages_per_category,
        stop_after_empty_pages=1,
        stop_after_duplicate_pages=1,
        request_delay_seconds=0.0,
        max_concurrent_requests=1,
        request_timeout_seconds=1,
        retry_count=0,
        user_agent="test",
    )


def _crawler_config() -> SimpleNamespace:
    return SimpleNamespace(base_url="https://genk.vn", timeout_seconds=1, retry_count=0, user_agent="test")


class FakeGenKCrawler(GenKCrawler):
    def __init__(self, responses):
        super().__init__(_crawler_config())
        self.responses = dict(responses)

    def fetch_url(self, url: str) -> FetchResponse:
        response = self.responses.get(url)
        if response is None:
            return FetchResponse(url, url, 404, "", error=None)
        return response


class GenKCrawlerTest(unittest.TestCase):
    def test_genk_has_expected_article_category_paths(self) -> None:
        expected = {
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

        genk_config = SourceManagementService(_settings()).get_source_config("genk")

        self.assertTrue(expected.issubset(set(genk_config.category_paths)))
        self.assertTrue(set(genk_config.category_paths).issubset(GENK_ALLOWED_CATEGORY_PATHS))

    def test_genk_category_page_url_builder_page_1(self) -> None:
        self.assertEqual(
            "https://genk.vn/ai.chn",
            build_genk_category_page_url("https://genk.vn", "ai.chn", 1),
        )
        self.assertEqual(
            "https://genk.vn/mobile/dien-thoai.chn",
            build_genk_category_page_url("https://genk.vn", "mobile/dien-thoai.chn", 1),
        )

    def test_genk_category_page_url_builder_requires_next_link_for_page_gt_1(self) -> None:
        self.assertIsNone(build_genk_category_page_url("https://genk.vn", "ai.chn", 2, previous_html=None))

    def test_genk_crawl_does_not_use_priority_page_limit(self) -> None:
        genk_config = SourceManagementService(_settings(max_pages_per_category=3)).get_source_config("genk")
        categories = ["ai.chn", "mobile.chn", "internet.chn", "blockchain.chn", "song.chn"]

        for category in categories:
            with self.subTest(category=category):
                self.assertIn(category, genk_config.category_paths)
                self.assertEqual(3, genk_config.max_pages_per_category)

    def test_genk_article_url_validation(self) -> None:
        category_paths = {"ai.chn", "mobile.chn", "mobile/dien-thoai.chn"}

        self.assertTrue(
            is_valid_genk_article_url(
                "https://genk.vn/meta-ai-de-dai-den-muc-ngo-nghech-165260602080717731.chn",
                category_paths,
            )
        )
        self.assertFalse(is_valid_genk_article_url("https://genk.vn/ai.chn", category_paths))
        self.assertFalse(is_valid_genk_article_url("https://genk.vn/mobile/dien-thoai.chn", category_paths))
        self.assertFalse(is_valid_genk_article_url("https://apps.apple.com/app/example", category_paths))

    def test_genk_category_whitelist_keeps_categories_out_of_articles(self) -> None:
        crawler = GenKCrawler(_crawler_config())

        self.assertTrue(crawler.is_category_url("https://genk.vn/ai.chn"))
        self.assertFalse(crawler.is_article_url("https://genk.vn/ai.chn"))
        self.assertTrue(
            crawler.is_article_url(
                "https://genk.vn/meta-ai-de-dai-den-muc-ngo-nghech-165260602080717731.chn"
            )
        )

    def test_genk_raw_document_keeps_source_and_category(self) -> None:
        crawler = GenKCrawler(_crawler_config())
        article = ParsedArticle(
            source_name="genk",
            url="https://genk.vn/meta-ai-de-dai-den-muc-ngo-nghech-165260602080717731.chn",
            canonical_url="https://genk.vn/meta-ai-de-dai-den-muc-ngo-nghech-165260602080717731.chn",
            title="Meta AI de dai den muc ngo nghech",
            summary=None,
            content="Content " * 40,
            category="ai.chn",
            published_at=None,
            author=None,
            image=None,
            category_url="https://genk.vn/ai.chn",
            category_page=1,
            crawled_at=utc_now(),
            content_hash="hash-g1",
            raw_html="<html></html>",
            http_status=200,
            metadata={
                "source_category_name": "ai.chn",
                "source_category_url": "https://genk.vn/ai.chn",
            },
        )

        document = crawler.build_raw_document(article, crawl_run_id="run-1")

        self.assertEqual("genk", document.source_name)
        self.assertTrue(document.raw_document_id)
        self.assertNotEqual("1", document.source_name)
        self.assertIn("genk", document.metadata)
        self.assertIn("ai.chn", document.raw_payload)

    def test_extract_genk_ajax_category_url_from_html(self) -> None:
        html = """
        <button class="view-more" data-api-url="/ajax-cate/page-4-c165253.chn"></button>
        """

        self.assertEqual(
            "https://genk.vn/ajax-cate/page-4-c165253.chn",
            extract_genk_ajax_category_url_from_html("https://genk.vn/ai.chn", html),
        )

    def test_next_genk_ajax_category_url_increments_page(self) -> None:
        self.assertEqual(
            "https://genk.vn/ajax-cate/page-5-c165253.chn",
            next_genk_ajax_category_url("https://genk.vn/ajax-cate/page-4-c165253.chn"),
        )

    def test_extract_article_links_from_genk_ajax_stores_next_page(self) -> None:
        url = "https://genk.vn/ajax-cate/page-4-c165253.chn"
        html = """
        <div class="news-item">
          <h3><a href="/ajax-loaded-genk-article-165260602080717731.chn">GenK article</a></h3>
        </div>
        """
        crawler = FakeGenKCrawler({url: FetchResponse(url, url, 200, html)})
        crawler._active_category_url = "https://genk.vn/ai.chn"

        links = crawler.extract_article_links(url)

        self.assertEqual(["https://genk.vn/ajax-loaded-genk-article-165260602080717731.chn"], links)
        self.assertEqual(
            "https://genk.vn/ajax-cate/page-5-c165253.chn",
            crawler._next_page_by_category_url["https://genk.vn/ai.chn"],
        )


if __name__ == "__main__":
    unittest.main()
