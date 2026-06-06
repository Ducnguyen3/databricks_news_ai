from __future__ import annotations

import unittest

from app.processing.taxonomy import NEWS_TOPICS, map_category_to_topic, normalize_topic


class TaxonomyTest(unittest.TestCase):
    def test_lifestyle_topic_id_is_consistent(self) -> None:
        self.assertIn("lifestyle_education_health_entertainment", NEWS_TOPICS)
        self.assertNotIn("life_education_health_entertainment", NEWS_TOPICS)

    def test_cafef_category_maps_to_expected_topic(self) -> None:
        cases = {
            "thi-truong-chung-khoan.chn": "economy_finance_stock",
            "tai-chinh-ngan-hang.chn": "economy_finance_stock",
            "smart-money.chn": "economy_finance_stock",
            "vi-mo-dau-tu.chn": "economy_finance_stock",
            "thi-truong.chn": "economy_finance_stock",
            "tai-chinh-quoc-te.chn": "economy_finance_stock",
            "doanh-nghiep.chn": "business_startup",
            "bat-dong-san.chn": "real_estate",
            "kinh-te-so.chn": "tech_ai_internet",
            "xa-hoi.chn": "politics_society",
            "song.chn": "lifestyle_education_health_entertainment",
            "lifestyle.chn": "lifestyle_education_health_entertainment",
        }

        for category, expected_topic in cases.items():
            with self.subTest(category=category):
                self.assertEqual(expected_topic, map_category_to_topic("cafef", category))

    def test_cafef_alias_does_not_override_other_sources(self) -> None:
        self.assertEqual("business_startup", map_category_to_topic("cafef", "doanh-nghiep.chn"))
        self.assertEqual("economy_finance_stock", map_category_to_topic("vnexpress", "kinh-doanh"))

    def test_vnexpress_category_maps_to_expected_topic(self) -> None:
        cases = {
            "thoi-su/chinh-tri": "politics_society",
            "the-gioi/quan-su": "world_geopolitics",
            "kinh-doanh": "economy_finance_stock",
            "kinh-doanh/doanh-nghiep": "business_startup",
            "kinh-doanh/chung-khoan": "economy_finance_stock",
            "khoa-hoc-cong-nghe/ai": "tech_ai_internet",
            "bat-dong-san/du-an": "real_estate",
            "suc-khoe": "lifestyle_education_health_entertainment",
            "giai-tri": "lifestyle_education_health_entertainment",
            "the-thao": "lifestyle_education_health_entertainment",
            "bong-da": "lifestyle_education_health_entertainment",
            "phap-luat": "politics_society",
            "giao-duc": "lifestyle_education_health_entertainment",
            "doi-song": "lifestyle_education_health_entertainment",
            "oto-xe-may/thi-truong": "economy_finance_stock",
            "oto-xe-may/xe-dien": "tech_ai_internet",
            "du-lich": "lifestyle_education_health_entertainment",
            "y-kien": "politics_society",
            "y-kien/doi-song": "lifestyle_education_health_entertainment",
            "tam-su": "lifestyle_education_health_entertainment",
            "thu-gian": "lifestyle_education_health_entertainment",
        }

        for category, expected_topic in cases.items():
            with self.subTest(category=category):
                self.assertEqual(expected_topic, map_category_to_topic("vnexpress", category))

    def test_vnexpress_alias_does_not_override_other_sources(self) -> None:
        self.assertEqual("business_startup", map_category_to_topic("vnexpress", "kinh-doanh/doanh-nghiep"))
        self.assertIsNone(map_category_to_topic("genk", "kinh-doanh/doanh-nghiep"))

    def test_genk_category_maps_to_expected_topic(self) -> None:
        cases = {
            "ai.chn": "tech_ai_internet",
            "tin-ict.chn": "tech_ai_internet",
            "mobile.chn": "tech_ai_internet",
            "mobile/dien-thoai.chn": "tech_ai_internet",
            "mobile/may-tinh-bang.chn": "tech_ai_internet",
            "internet.chn": "tech_ai_internet",
            "internet/digital-marketing.chn": "tech_ai_internet",
            "internet/media.chn": "tech_ai_internet",
            "kham-pha.chn": "tech_ai_internet",
            "kham-pha/lich-su.chn": "tech_ai_internet",
            "kham-pha/tri-thuc.chn": "tech_ai_internet",
            "tra-da-cong-nghe.chn": "tech_ai_internet",
            "tra-da-cong-nghe/tan-man.chn": "tech_ai_internet",
            "tra-da-cong-nghe/y-tuong-sang-tao.chn": "tech_ai_internet",
            "blockchain.chn": "tech_ai_internet",
            "blockchain/xu-huong.chn": "tech_ai_internet",
            "blockchain/cong-nghe.chn": "tech_ai_internet",
            "blockchain/nhan-vat.chn": "tech_ai_internet",
            "thu-thuat.chn": "tech_ai_internet",
            "apps-games.chn": "tech_ai_internet",
            "do-choi-so.chn": "tech_ai_internet",
            "xem-mua-luon.chn": "tech_ai_internet",
            "gia-dung.chn": "tech_ai_internet",
            "xe.chn": "tech_ai_internet",
            "song.chn": "lifestyle_education_health_entertainment",
            "nhom-chu-de/emagazine.chn": "tech_ai_internet",
        }

        for category, expected_topic in cases.items():
            with self.subTest(category=category):
                self.assertEqual(expected_topic, map_category_to_topic("genk", category))

    def test_genk_alias_does_not_override_other_sources(self) -> None:
        self.assertEqual("tech_ai_internet", map_category_to_topic("genk", "ai.chn"))
        self.assertEqual("business_startup", map_category_to_topic("cafef", "doanh-nghiep.chn"))
        self.assertEqual("economy_finance_stock", map_category_to_topic("vnexpress", "kinh-doanh/chung-khoan"))

    def test_diendandoanhnghiep_category_maps_to_expected_topic(self) -> None:
        cases = {
            "chinh-tri-xa-hoi": "politics_society",
            "chinh-tri-xa-hoi/chinh-tri": "politics_society",
            "chinh-tri-xa-hoi/tam-diem": "politics_society",
            "chinh-tri-xa-hoi/mat-tran": "politics_society",
            "chinh-tri-xa-hoi/kinh-te": "economy_finance_stock",
            "chinh-tri-xa-hoi/xa-hoi": "politics_society",
            "vcci": "business_startup",
            "vcci/tham-muu-chinh-sach": "politics_society",
            "doanh-nghiep": "business_startup",
            "doanh-nghiep/quan-tri": "business_startup",
            "khoi-nghiep": "business_startup",
            "khoi-nghiep/khoi-nghiep-quoc-gia": "business_startup",
            "ngan-hang-chung-khoan": "economy_finance_stock",
            "ngan-hang-chung-khoan/chung-khoan": "economy_finance_stock",
            "ngan-hang-chung-khoan/tin-dung-ngan-hang": "economy_finance_stock",
            "bat-dong-san": "real_estate",
            "bat-dong-san/thi-truong": "real_estate",
            "quoc-te": "world_geopolitics",
            "quoc-te/phan-tich-binh-luan": "world_geopolitics",
            "phap-luat": "politics_society",
            "phap-luat/kien-nghi": "politics_society",
            "o-to-xe-may": "lifestyle_education_health_entertainment",
            "o-to-xe-may/dien-dan": "lifestyle_education_health_entertainment",
            "o-to-xe-may/thong-tin-thi-truong": "economy_finance_stock",
            "o-to-xe-may/san-pham": "lifestyle_education_health_entertainment",
            "o-to-xe-may/tu-van-ky-thuat": "lifestyle_education_health_entertainment",
        }

        for category, expected_topic in cases.items():
            with self.subTest(category=category):
                self.assertEqual(expected_topic, map_category_to_topic("diendandoanhnghiep", category))

    def test_diendandoanhnghiep_alias_does_not_override_other_sources(self) -> None:
        self.assertEqual(
            "economy_finance_stock",
            map_category_to_topic("diendandoanhnghiep", "ngan-hang-chung-khoan/chung-khoan"),
        )
        self.assertEqual("business_startup", map_category_to_topic("cafef", "doanh-nghiep.chn"))
        self.assertEqual("tech_ai_internet", map_category_to_topic("genk", "ai.chn"))
        self.assertEqual("economy_finance_stock", map_category_to_topic("vnexpress", "kinh-doanh/chung-khoan"))

    def test_cafef_secondary_topic_rules(self) -> None:
        result = normalize_topic(source="cafef", source_category="tai-chinh-quoc-te.chn", title="Tai chinh quoc te")

        self.assertEqual("economy_finance_stock", result["primary_topic"])
        self.assertIn("world_geopolitics", result["secondary_topics"])

    def test_locked_vnexpress_sports_categories_keep_lifestyle_primary_topic(self) -> None:
        cases = [
            ("the-thao/tennis", "Tay vot My va Nga tranh ngoi so mot the gioi"),
            ("the-thao/cac-mon-khac", "Van dong vien Nga lap ky luc quoc te"),
            ("the-thao/hau-truong", "CLB Trung Quoc gay chu y tren dau truong the gioi"),
        ]

        for category, title in cases:
            with self.subTest(category=category):
                result = normalize_topic(source="vnexpress", source_category=category, title=title)
                self.assertEqual("lifestyle_education_health_entertainment", result["primary_topic"])

    def test_locked_vnexpress_thoi_su_categories_keep_politics_primary_topic(self) -> None:
        cases = [
            ("thoi-su/dan-sinh", "Nguoi dan gap kho vi nha dat va chung cu tang gia"),
            ("thoi-su/giao-thong", "Chuyen gia quoc te gop y ve giao thong do thi"),
            ("thoi-su/quy-hy-vong", "Cau chuyen doi song ve hoan canh kho khan"),
        ]

        for category, title in cases:
            with self.subTest(category=category):
                result = normalize_topic(source="vnexpress", source_category=category, title=title)
                self.assertEqual("politics_society", result["primary_topic"])

    def test_locked_vnexpress_science_and_entertainment_categories_keep_expected_primary_topic(self) -> None:
        cases = [
            ("khoa-hoc-cong-nghe/the-gioi-tu-nhien", "The gioi tu nhien co phat hien moi"),
            ("giai-tri/san-khau-my-thuat", "Nghe si My va Nga cung tham gia lien hoan quoc te"),
        ]
        expected = {
            "khoa-hoc-cong-nghe/the-gioi-tu-nhien": "tech_ai_internet",
            "giai-tri/san-khau-my-thuat": "lifestyle_education_health_entertainment",
        }

        for category, title in cases:
            with self.subTest(category=category):
                result = normalize_topic(source="vnexpress", source_category=category, title=title)
                self.assertEqual(expected[category], result["primary_topic"])

    def test_international_business_categories_keep_economy_primary_topic(self) -> None:
        cases = [
            ("vnexpress", "kinh-doanh/quoc-te"),
            ("cafef", "tai-chinh-quoc-te.chn"),
            ("cafef", "thi-truong.chn"),
        ]

        for source, category in cases:
            with self.subTest(source=source, category=category):
                result = normalize_topic(
                    source=source,
                    source_category=category,
                    title="Xung dot quoc te tac dong toi thi truong tai chinh",
                )
                self.assertEqual("economy_finance_stock", result["primary_topic"])

    def test_stock_category_maps_to_economy_finance_stock(self) -> None:
        result = normalize_topic(
            source="cafef",
            source_category="Chứng khoán",
            title="VN-Index tăng mạnh cuối phiên",
            summary="Cổ phiếu ngân hàng dẫn dắt thị trường",
            content="",
        )

        self.assertEqual("economy_finance_stock", result["primary_topic"])
        self.assertGreater(result["topic_confidence"], 0)

    def test_ai_article_maps_to_tech_ai_internet(self) -> None:
        result = normalize_topic(
            source="genk",
            source_category="AI",
            title="OpenAI ra mắt mô hình trí tuệ nhân tạo mới",
        )

        self.assertEqual("tech_ai_internet", result["primary_topic"])

    def test_real_estate_category_maps_to_real_estate(self) -> None:
        result = normalize_topic(
            source="cafef",
            source_category="Bất động sản",
            title="Giá chung cư tiếp tục tăng tại Hà Nội",
        )

        self.assertEqual("real_estate", result["primary_topic"])

    def test_world_geopolitics_article_maps_to_world_geopolitics(self) -> None:
        result = normalize_topic(
            source="vnexpress",
            source_category="Thế giới",
            title="Căng thẳng tại Ukraine và Trung Đông leo thang",
        )

        self.assertEqual("world_geopolitics", result["primary_topic"])

    def test_business_startup_article_maps_to_business_startup(self) -> None:
        result = normalize_topic(
            source="dddn",
            source_category="Doanh nghiệp",
            title="Startup Việt gọi vốn thành công, CEO công bố chiến lược mới",
        )

        self.assertEqual("business_startup", result["primary_topic"])

    def test_unclear_article_returns_general_news_without_error(self) -> None:
        result = normalize_topic(
            source="",
            source_category=None,
            title="Một ngày bình thường",
            summary=None,
            content=None,
        )

        self.assertIn("primary_topic", result)
        self.assertEqual("general_news", result["primary_topic"])

    def test_secondary_topics_do_not_include_primary(self) -> None:
        result = normalize_topic(
            source="cafef",
            source_category="Doanh nghiệp",
            title="Doanh nghiệp bất động sản công bố lợi nhuận tăng mạnh trên sàn chứng khoán",
        )

        self.assertIn(
            result["primary_topic"],
            ["business_startup", "economy_finance_stock", "real_estate"],
        )
        self.assertIsInstance(result["secondary_topics"], list)
        self.assertNotIn(result["primary_topic"], result["secondary_topics"])


if __name__ == "__main__":
    unittest.main()
