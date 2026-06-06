from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config import CrawlSettings
from app.ingestion.crawlers.base_crawler import FetchResponse
from app.ingestion.crawlers.sites.cafef_crawler import (
    CafeFAjaxState,
    CafeFCrawler,
    build_cafef_ajax_candidate_urls,
    cafef_original_category_url,
    extract_cafef_article_urls_from_ajax_payload,
    extract_cafef_ajax_state,
    extract_cafef_page_number,
)
from app.ingestion.services.source_management_service import SourceManagementService


def _config() -> SimpleNamespace:
    return SimpleNamespace(base_url="https://cafef.vn", timeout_seconds=1, retry_count=0, user_agent="test")


class FakeCafeFCrawler(CafeFCrawler):
    def __init__(self, responses):
        super().__init__(_config())
        self.responses = dict(responses)
        self.requested_urls: list[str] = []

    def fetch_url(self, url: str) -> FetchResponse:
        self.requested_urls.append(url)
        response = self.responses.get(url)
        if response is None:
            return FetchResponse(url, url, 404, "", error=None)
        return response


class CafeFCrawlerTest(unittest.TestCase):
    def test_cafef_has_expected_category_paths(self) -> None:
        expected = {
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
        settings = CrawlSettings(
            sources=("cafef",),
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

        cafef_config = SourceManagementService(settings).get_source_config("cafef")

        self.assertTrue(expected.issubset(set(cafef_config.category_paths)))

    def test_cafef_categories_use_same_configured_page_limit(self) -> None:
        crawler = CafeFCrawler(_config())
        categories = (
            "thi-truong-chung-khoan.chn",
            "bat-dong-san.chn",
            "doanh-nghiep.chn",
            "song.chn",
        )

        for category in categories:
            with self.subTest(category=category):
                urls = crawler.build_category_urls(category, max_pages=3)
                self.assertEqual(3, len(urls))

    def test_extracts_article_links_from_current_category_markup(self) -> None:
        html = """
        <section class="category-page__top">
            <a class="avatar img-resize" href="/co-phieu-ngan-hang-dang-re-vi-sao-tien-van-chua-vao-188260601221720539.chn">
                <img src="https://cafefcdn.com/news.jpg" alt="Cổ phiếu ngân hàng đang rẻ">
            </a>
            <h3>
                <a href="/tien-vao-chung-khoan-mat-hut-gia-tri-khop-lenh-tren-hose-xuong-thap-nhat-hon-1-nam-chuyen-gi-dang-xay-ra-188260601221925615.chn">
                    Tiền vào chứng khoán mất hút
                </a>
            </h3>
        </section>
        <div role="article" class="tlitem box-category-item" data-id="188260602224817762">
            <h3>
                <a href="/sau-doi-ten-sacombank-doi-them-mau-con-dau-188260602224817762.chn">
                    Sau đổi tên, SACOMBANK đổi thêm mẫu con dấu
                </a>
            </h3>
            <div class="tlitem-flex">
                <a class="avatar img-resize" href="/sau-doi-ten-sacombank-doi-them-mau-con-dau-188260602224817762.chn"></a>
            </div>
        </div>
        <ul>
            <li><a href="/su-kien/agm-awards-1083.chn">AGM Awards</a></li>
            <li><a href="/thi-truong-chung-khoan.chn">Chứng khoán</a></li>
        </ul>
        """

        urls = CafeFCrawler(_config()).extract_article_urls_from_html(
            "https://cafef.vn/thi-truong-chung-khoan.chn",
            html,
        )

        self.assertIn(
            "https://cafef.vn/co-phieu-ngan-hang-dang-re-vi-sao-tien-van-chua-vao-188260601221720539.chn",
            urls,
        )
        self.assertIn(
            "https://cafef.vn/tien-vao-chung-khoan-mat-hut-gia-tri-khop-lenh-tren-hose-xuong-thap-nhat-hon-1-nam-chuyen-gi-dang-xay-ra-188260601221925615.chn",
            urls,
        )
        self.assertIn(
            "https://cafef.vn/sau-doi-ten-sacombank-doi-them-mau-con-dau-188260602224817762.chn",
            urls,
        )
        self.assertNotIn("https://cafef.vn/su-kien/agm-awards-1083.chn", urls)
        self.assertNotIn("https://cafef.vn/thi-truong-chung-khoan.chn", urls)
        self.assertEqual(len(urls), len(set(urls)))

    def test_category_pagination_uses_cafef_trang_pattern(self) -> None:
        crawler = CafeFCrawler(_config())

        self.assertEqual(
            "https://cafef.vn/thi-truong-chung-khoan/trang-2.chn",
            crawler.build_category_page_url("https://cafef.vn/thi-truong-chung-khoan.chn", 2),
        )

    def test_extract_cafef_page_number(self) -> None:
        self.assertEqual(3, extract_cafef_page_number("https://cafef.vn/xa-hoi/trang-3.chn"))
        self.assertEqual(1, extract_cafef_page_number("https://cafef.vn/xa-hoi.chn"))

    def test_cafef_original_category_url_from_trang_url(self) -> None:
        self.assertEqual(
            "https://cafef.vn/xa-hoi.chn",
            cafef_original_category_url("https://cafef.vn/xa-hoi/trang-2.chn"),
        )

    def test_extract_ajax_state_from_rollup_html(self) -> None:
        html = """
        <div data-cd-key="siteid188:newsinzone:zone188112"></div>
        <div data-key-cd="siteid188:newsinzone:zone188113"></div>
        <div data-key-cd="siteid188:objectembedbox:zoneid0typeid6"></div>
        <div data-key-cd="siteid188:newsinzoneisonhome:zone0"></div>
        <input type="hidden" id="hdZoneId" value="188112" />
        <input type="hidden" id="hdZoneUrl" value="xa-hoi" />
        """

        state = extract_cafef_ajax_state(html)

        self.assertEqual("188112", state.zone_id)
        self.assertEqual("xa-hoi", state.zone_url)
        self.assertEqual(("siteid188:newsinzone:zone188112",), state.cd_keys)

    def test_build_cafef_ajax_candidate_urls(self) -> None:
        urls = build_cafef_ajax_candidate_urls(
            "https://cafef.vn/xa-hoi.chn",
            CafeFAjaxState(zone_id="188112", zone_url="xa-hoi", cd_keys=("siteid188:newsinzone:zone188112",)),
            2,
        )

        self.assertIn("https://cafef.vn/ajax/list-news/188112/2.chn", urls)
        self.assertIn("https://cafef.vn/timelinelist/188112/2.chn", urls)
        self.assertIn("https://cafef.vn/ajax/loadmore?pageIndex=2&zoneId=188112", urls)
        self.assertIn("https://cafef.vn/ajax/load-more?key=siteid188%3Anewsinzone%3Azone188112&page=2", urls)

    def test_extract_article_urls_from_ajax_json_html_payload(self) -> None:
        crawler = CafeFCrawler(_config())
        payload = {
            "html": """
            <div role="article">
                <a href="/json-html-article-188260602115647938.chn">Article</a>
            </div>
            """
        }

        urls = extract_cafef_article_urls_from_ajax_payload(crawler, "https://cafef.vn/xa-hoi.chn", __import__("json").dumps(payload))

        self.assertEqual(["https://cafef.vn/json-html-article-188260602115647938.chn"], urls)

    def test_extract_article_urls_from_ajax_json_list_payload(self) -> None:
        crawler = CafeFCrawler(_config())
        payload = {
            "data": [
                {"url": "/json-list-article-188260602115647938.chn"},
                {"href": "https://cafef.vn/json-list-article-188260602115647939.chn"},
            ]
        }

        urls = extract_cafef_article_urls_from_ajax_payload(crawler, "https://cafef.vn/xa-hoi.chn", __import__("json").dumps(payload))

        self.assertIn("https://cafef.vn/json-list-article-188260602115647938.chn", urls)
        self.assertIn("https://cafef.vn/json-list-article-188260602115647939.chn", urls)

    def test_extract_article_links_falls_back_to_cafef_ajax_rollup(self) -> None:
        category_url = "https://cafef.vn/xa-hoi.chn"
        page_2_url = "https://cafef.vn/xa-hoi/trang-2.chn"
        ajax_url = "https://cafef.vn/ajax/list-news/188112/2.chn"
        category_html = """
        <div data-cd-key="siteid188:newsinzone:zone188112"></div>
        <input type="hidden" id="hdZoneId" value="188112" />
        <input type="hidden" id="hdZoneUrl" value="xa-hoi" />
        """
        ajax_html = """
        <div role="article" class="tlitem box-category-item">
            <h3><a href="/ajax-loaded-article-188260602115647938.chn">Ajax loaded article</a></h3>
        </div>
        """
        crawler = FakeCafeFCrawler(
            {
                page_2_url: FetchResponse(page_2_url, page_2_url, 200, "<html></html>"),
                category_url: FetchResponse(category_url, category_url, 200, category_html),
                ajax_url: FetchResponse(ajax_url, ajax_url, 200, ajax_html),
            }
        )

        urls = crawler.extract_article_links(page_2_url)

        self.assertEqual(["https://cafef.vn/ajax-loaded-article-188260602115647938.chn"], urls)
        self.assertIn(ajax_url, crawler.requested_urls)


if __name__ == "__main__":
    unittest.main()
