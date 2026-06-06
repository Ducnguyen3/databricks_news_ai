from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config import CrawlSettings
from app.ingestion.crawlers.base_crawler import FetchResponse
from app.ingestion.crawlers.sites.diendandoanhnghiep_crawler import (
    DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS,
    DienDanDoanhNghiepCrawler,
    build_diendandoanhnghiep_category_page_url,
    extract_load_more_url_from_onecms_html,
    is_valid_diendandoanhnghiep_article_url,
    next_diendandoanhnghiep_load_more_api_url,
)
from app.ingestion.services.source_management_service import SourceManagementService


DIENDANDOANHNGHIEP_MENU_CATEGORY_PATHS = {
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


def _settings(max_pages_per_category: int = 3) -> CrawlSettings:
    return CrawlSettings(
        sources=("diendandoanhnghiep",),
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
    return SimpleNamespace(
        base_url="https://diendandoanhnghiep.vn",
        timeout_seconds=1,
        retry_count=0,
        user_agent="test",
    )


class FakeDienDanCrawler(DienDanDoanhNghiepCrawler):
    def __init__(self, responses):
        super().__init__(_crawler_config())
        self.responses = dict(responses)

    def fetch_url(self, url: str) -> FetchResponse:
        response = self.responses.get(url)
        if response is None:
            return FetchResponse(url, url, 404, "", error=None)
        return response


class DienDanDoanhNghiepCrawlerTest(unittest.TestCase):
    def test_diendandoanhnghiep_has_expected_article_category_paths(self) -> None:
        config = SourceManagementService(_settings()).get_source_config("diendandoanhnghiep")

        self.assertTrue(DIENDANDOANHNGHIEP_MENU_CATEGORY_PATHS.issubset(set(config.category_paths)))
        self.assertTrue(set(config.category_paths).issubset(DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS))

    def test_diendandoanhnghiep_menu_category_paths_are_whitelisted(self) -> None:
        config = SourceManagementService(_settings()).get_source_config("diendandoanhnghiep")

        for category_path in DIENDANDOANHNGHIEP_MENU_CATEGORY_PATHS:
            with self.subTest(category_path=category_path):
                self.assertIn(category_path, DIENDANDOANHNGHIEP_ALLOWED_CATEGORY_PATHS)
                self.assertIn(category_path, config.category_paths)

    def test_diendandoanhnghiep_has_no_special_paths_configured(self) -> None:
        config = SourceManagementService(_settings()).get_source_config("diendandoanhnghiep")

        self.assertEqual((), config.special_paths)

    def test_diendandoanhnghiep_crawl_does_not_use_priority_page_limit(self) -> None:
        config = SourceManagementService(_settings(max_pages_per_category=3)).get_source_config(
            "diendandoanhnghiep"
        )
        categories = [
            "chinh-tri-xa-hoi",
            "vcci",
            "doanh-nghiep",
            "khoi-nghiep",
            "ngan-hang-chung-khoan/chung-khoan",
            "bat-dong-san/thi-truong",
            "quoc-te/phan-tich-binh-luan",
            "phap-luat",
        ]

        for category in categories:
            with self.subTest(category=category):
                self.assertIn(category, config.category_paths)
                self.assertEqual(3, config.max_pages_per_category)

    def test_diendandoanhnghiep_category_page_url_builder_page_1(self) -> None:
        self.assertEqual(
            "https://diendandoanhnghiep.vn/chinh-tri-xa-hoi",
            build_diendandoanhnghiep_category_page_url(
                "https://diendandoanhnghiep.vn",
                "chinh-tri-xa-hoi",
                1,
            ),
        )
        self.assertEqual(
            "https://diendandoanhnghiep.vn/ngan-hang-chung-khoan/chung-khoan",
            build_diendandoanhnghiep_category_page_url(
                "https://diendandoanhnghiep.vn",
                "ngan-hang-chung-khoan/chung-khoan",
                1,
            ),
        )

    def test_diendandoanhnghiep_category_page_url_builder_requires_next_link_for_page_gt_1(self) -> None:
        self.assertIsNone(
            build_diendandoanhnghiep_category_page_url(
                "https://diendandoanhnghiep.vn",
                "chinh-tri-xa-hoi",
                2,
                previous_html=None,
            )
        )

    def test_diendandoanhnghiep_article_url_validation(self) -> None:
        self.assertTrue(
            is_valid_diendandoanhnghiep_article_url(
                "https://diendandoanhnghiep.vn/muc-tieu-cao-nhat-la-to-chuc-thanh-cong-mot-nam-apec-mang-dam-dau-an-viet-nam-10179807.html"
            )
        )
        self.assertFalse(is_valid_diendandoanhnghiep_article_url("https://diendandoanhnghiep.vn/chinh-tri-xa-hoi"))
        self.assertFalse(
            is_valid_diendandoanhnghiep_article_url(
                "https://diendandoanhnghiep.vn/bao-cao-kinh-te-tu-nhan-viet-nam-va-pci-2025-event223.html"
            )
        )
        self.assertFalse(
            is_valid_diendandoanhnghiep_article_url(
                "https://diendandoanhnghiep.vn/an-pham/an-pham-in-dien-dan-doanh-nghiep-so-43-29-05-2026-241.html"
            )
        )

    def test_diendandoanhnghiep_category_whitelist_keeps_special_paths_out(self) -> None:
        crawler = DienDanDoanhNghiepCrawler(_crawler_config())

        self.assertTrue(crawler.is_category_url("https://diendandoanhnghiep.vn/chinh-tri-xa-hoi"))
        self.assertFalse(crawler.is_category_url("https://diendandoanhnghiep.vn/tam-diem"))
        self.assertFalse(
            crawler.is_article_url(
                "https://diendandoanhnghiep.vn/bao-cao-kinh-te-tu-nhan-viet-nam-va-pci-2025-event223.html"
            )
        )

    def test_extract_get_more_article_api_from_html(self) -> None:
        html = """
        <button class="view-more" data-api-url="/api/getMoreArticle/channel_empty_10179233_434_0"></button>
        """

        self.assertEqual(
            "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179233_434_0",
            extract_load_more_url_from_onecms_html("https://diendandoanhnghiep.vn/doanh-nghiep", html),
        )

    def test_ignores_static_page_content_api_when_extracting_load_more(self) -> None:
        html = """
        <div class="c-menu static-page-content" data-api-url="/api/static-page-content"></div>
        """

        self.assertIsNone(
            extract_load_more_url_from_onecms_html("https://diendandoanhnghiep.vn/phap-luat/phap-dinh", html)
        )

    def test_phap_dinh_category_uses_known_get_more_endpoint_when_no_next_page_detected(self) -> None:
        crawler = DienDanDoanhNghiepCrawler(_crawler_config())

        self.assertEqual(
            "https://diendandoanhnghiep.vn/phap-luat/phap-dinh",
            crawler.build_category_page_url("https://diendandoanhnghiep.vn/phap-luat/phap-dinh", 1),
        )
        self.assertEqual(
            "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179492_434_0",
            crawler.build_category_page_url("https://diendandoanhnghiep.vn/phap-luat/phap-dinh", 2),
        )

    def test_next_get_more_article_api_url_increments_last_number(self) -> None:
        self.assertEqual(
            "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179233_434_1",
            next_diendandoanhnghiep_load_more_api_url(
                "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179233_434_0"
            ),
        )

    def test_extract_article_links_from_get_more_article_api_stores_next_page(self) -> None:
        url = "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179233_434_0"
        html = """
        <div class="b-grid">
          <h3 class="b-grid__title">
            <a href="/api-loaded-article-10179807.html">API loaded article</a>
          </h3>
        </div>
        """
        crawler = FakeDienDanCrawler({url: FetchResponse(url, url, 200, html)})
        crawler._active_category_url = "https://diendandoanhnghiep.vn/doanh-nghiep"

        links = crawler.extract_article_links(url)

        self.assertEqual(["https://diendandoanhnghiep.vn/api-loaded-article-10179807.html"], links)
        self.assertEqual(
            "https://diendandoanhnghiep.vn/api/getMoreArticle/channel_empty_10179233_434_1",
            crawler._next_page_by_category_url["https://diendandoanhnghiep.vn/doanh-nghiep"],
        )


if __name__ == "__main__":
    unittest.main()
