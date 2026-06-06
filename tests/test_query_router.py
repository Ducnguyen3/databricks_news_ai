from __future__ import annotations

import unittest
from datetime import date, timedelta

from app.local_ai.query_router import route_query


class QueryRouterTest(unittest.TestCase):
    def test_answer_mode_synthesis_ai_query(self) -> None:
        plan = route_query("tin AI gan day co gi")

        self.assertEqual("synthesis", plan["answer_mode"])
        self.assertEqual("tech_ai_internet", plan["primary_topic"])

    def test_answer_mode_citation_query(self) -> None:
        plan = route_query("so lieu cu the la bao nhieu")

        self.assertEqual("citation", plan["answer_mode"])

    def test_answer_mode_followup_query(self) -> None:
        plan = route_query("vu nay anh huong sao")

        self.assertEqual("followup", plan["answer_mode"])

    def test_absolute_date_query_routes_with_exact_date_filter(self) -> None:
        plan = route_query("tin tuc vao ngay 1 thang 6 co gi noi bat")
        selected = date(date.today().year, 6, 1)

        self.assertEqual("date", plan["time_range"])
        self.assertEqual(selected.isoformat(), plan["date_filter"]["exact_date"])
        self.assertEqual(selected.isoformat(), plan["date_filter"]["start_date"])
        self.assertEqual((selected + timedelta(days=1)).isoformat(), plan["date_filter"]["end_date"])
        self.assertTrue(plan["needs_recent"])

    def test_ai_latest_news(self) -> None:
        plan = route_query("tin AI mới nhất")

        self.assertIn(plan["intent"], {"latest_news", "topic_news"})
        self.assertEqual("tech_ai_internet", plan["primary_topic"])
        self.assertIn(plan["time_range"], {"today", "24h", "7d"})

    def test_world_today(self) -> None:
        plan = route_query("tình hình thế giới hôm nay")

        self.assertEqual("world_geopolitics", plan["primary_topic"])
        self.assertEqual("today", plan["time_range"])
        self.assertIn(plan["intent"], {"latest_news", "topic_news"})

    def test_stock_symbol_query(self) -> None:
        plan = route_query("HPG có gì mới")

        self.assertEqual("entity_news", plan["intent"])
        self.assertEqual("tai_chinh", plan["domain"])
        self.assertEqual("HPG", plan["ticker"])
        self.assertIn("HPG", plan["stock_symbols"])
        self.assertIn("HPG", plan["entities"])

    def test_real_estate_location_query(self) -> None:
        plan = route_query("bất động sản Hà Nội")

        self.assertEqual("real_estate", plan["primary_topic"])
        self.assertEqual("bat_dong_san", plan["domain"])
        self.assertIn("Hà Nội", plan["entities"])

    def test_image_query_about_ukraine_today(self) -> None:
        plan = route_query("ảnh về Ukraine hôm nay")

        self.assertEqual("media_lookup", plan["intent"])
        self.assertTrue(plan["need_images"])
        self.assertIn("Ukraine", plan["entities"])
        self.assertEqual("today", plan["time_range"])

    def test_lifestyle_query_uses_standard_topic_id(self) -> None:
        plan = route_query("tin giao duc suc khoe")

        self.assertEqual("lifestyle_education_health_entertainment", plan["primary_topic"])

    def test_routes_sports_schedule_to_structured_source(self) -> None:
        plan = route_query("lich thi dau hom nay")

        self.assertEqual("sports_schedule", plan["intent"])
        self.assertEqual("lifestyle_education_health_entertainment", plan["primary_topic"])
        self.assertEqual("structured_sports", plan["data_source"])

    def test_routes_sports_result_to_structured_source(self) -> None:
        plan = route_query("ket qua bong da hom qua")

        self.assertEqual("sports_result", plan["intent"])
        self.assertEqual("structured_sports", plan["data_source"])

    def test_question_pronoun_ai_does_not_route_to_tech_when_asking_c1_winner(self) -> None:
        plan = route_query("ai vo dich C1")

        self.assertEqual("sports_result", plan["intent"])
        self.assertEqual("lifestyle_education_health_entertainment", plan["primary_topic"])
        self.assertEqual("structured_sports", plan["data_source"])
        self.assertNotIn("AI", plan["entities"])

    def test_champions_league_winner_routes_to_sports(self) -> None:
        plan = route_query("doi nao vo dich Champions League?")

        self.assertEqual("sports_result", plan["intent"])
        self.assertEqual("lifestyle_education_health_entertainment", plan["primary_topic"])
        self.assertEqual("structured_sports", plan["data_source"])

    def test_routes_sports_standing_to_structured_source(self) -> None:
        plan = route_query("bang xep hang Ngoai hang Anh")

        self.assertEqual("sports_standing", plan["intent"])
        self.assertEqual("structured_sports", plan["data_source"])

    def test_routes_team_match_time_to_structured_schedule(self) -> None:
        plan = route_query("Man Utd da luc may gio")

        self.assertEqual("sports_schedule", plan["intent"])
        self.assertEqual("structured_sports", plan["data_source"])

    def test_routes_sports_news_to_article_rag(self) -> None:
        plan = route_query("tin the thao moi nhat")

        self.assertIn(plan["intent"], {"latest_news", "topic_news"})
        self.assertEqual("lifestyle_education_health_entertainment", plan["primary_topic"])
        self.assertEqual("article_rag", plan["data_source"])

    def test_routes_genk_related_queries_to_tech_topic(self) -> None:
        queries = (
            "tin AI moi nhat",
            "tin cong nghe moi nhat",
            "tin mobile moi nhat",
            "tin blockchain moi nhat",
            "thu thuat cong nghe",
            "do choi so co gi moi",
            "xe dien co cong nghe gi moi",
            "thiet bi gia dung thong minh moi nhat",
        )

        for query in queries:
            with self.subTest(query=query):
                plan = route_query(query)
                self.assertEqual("tech_ai_internet", plan["primary_topic"])
                self.assertEqual("article_rag", plan["data_source"])

    def test_routes_diendandoanhnghiep_related_queries(self) -> None:
        cases = {
            "tin chinh tri xa hoi moi nhat": "politics_society",
            "tin VCCI moi nhat": "business_startup",
            "tin doanh nghiep moi nhat": "business_startup",
            "tin khoi nghiep moi nhat": "business_startup",
            "tin chung khoan moi nhat": "economy_finance_stock",
            "tin ngan hang moi nhat": "economy_finance_stock",
            "tin bat dong san moi nhat": "real_estate",
            "tin quoc te moi nhat": "world_geopolitics",
            "tin phap luat moi nhat": "politics_society",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                plan = route_query(query)
                self.assertEqual(expected_topic, plan["primary_topic"])
                self.assertEqual("article_rag", plan["data_source"])

    def test_source_alias_vnepress_routes_to_vnexpress(self) -> None:
        plan = route_query("tom tat 1 bai bao hom nay tu nguon vnepress")

        self.assertEqual("vnexpress", plan["source"])
        self.assertEqual(["vnexpress"], plan["preferred_sources"])
        self.assertTrue(plan["requires_lexical"])

    def test_ai_is_not_treated_as_stock_ticker(self) -> None:
        plan = route_query("AI co gi moi")

        self.assertEqual("", plan["ticker"])
        self.assertNotIn("AI", plan["stock_symbols"])
        self.assertEqual("cong_nghe", plan["domain"])

    def test_ticker_whitelist_accepts_vfs_from_plain_uppercase_token(self) -> None:
        plan = route_query("VFS co tin gi moi")

        self.assertEqual("VFS", plan["ticker"])
        self.assertIn("VFS", plan["stock_symbols"])
        self.assertEqual("tai_chinh", plan["domain"])
        self.assertEqual("economy_finance_stock", plan["primary_topic"])

    def test_ticker_whitelist_accepts_bsr_stock_question(self) -> None:
        plan = route_query("BSR co con du dia tang truong khong")

        self.assertEqual("BSR", plan["ticker"])
        self.assertIn("BSR", plan["stock_symbols"])
        self.assertEqual("economy_finance_stock", plan["primary_topic"])
        self.assertEqual("tai_chinh", plan["domain"])

    def test_domain_mapping_for_required_topics(self) -> None:
        cases = {
            "tin cong nghe moi nhat": "cong_nghe",
            "tin chung khoan moi nhat": "tai_chinh",
            "tin bat dong san moi nhat": "bat_dong_san",
            "tin giao duc suc khoe": "doi_song",
            "tin chinh tri xa hoi moi nhat": "chinh_tri_xa_hoi",
            "tin quoc te moi nhat": "the_gioi",
            "tin startup moi nhat": "startup",
        }

        for query, expected_domain in cases.items():
            with self.subTest(query=query):
                self.assertEqual(expected_domain, route_query(query)["domain"])

    def test_need_images_is_not_always_true(self) -> None:
        self.assertFalse(route_query("tin AI moi nhat")["need_images"])
        self.assertFalse(route_query("tin doanh nghiep moi nhat")["need_images"])
        self.assertTrue(route_query("hinh anh ve Ukraine")["need_images"])

    def test_anh_huong_is_not_image_intent(self) -> None:
        policy = route_query("Co chinh sach moi nao anh huong den nguoi dan khong?")
        banking_real_estate = route_query("Ngan hang co anh huong gi den bat dong san?")

        self.assertNotEqual("media_lookup", policy["intent"])
        self.assertFalse(policy["need_images"])
        self.assertNotEqual("media_lookup", banking_real_estate["intent"])
        self.assertFalse(banking_real_estate["need_images"])

    def test_clear_image_intent_still_routes_to_media_lookup(self) -> None:
        plan = route_query("Co anh nao ve Ukraine khong?")

        self.assertEqual("media_lookup", plan["intent"])
        self.assertTrue(plan["need_images"])
        self.assertEqual("world_geopolitics", plan["primary_topic"])

    def test_techcombank_entity_alias_is_recognized(self) -> None:
        plan = route_query("Tin tuc ve ngan hang Techcombank")

        self.assertEqual("economy_finance_stock", plan["primary_topic"])
        self.assertIn("Techcombank", plan["entities"])

    def test_stock_market_overview_intent(self) -> None:
        queries = (
            "Gia co phieu hom nay the nao?",
            "Chung khoan hom nay the nao?",
            "Thi truong chung khoan hom nay",
            "VN-Index hom nay the nao?",
            "VN-Index tuan nay dien bien the nao?",
        )

        for query in queries:
            with self.subTest(query=query):
                plan = route_query(query)
                self.assertEqual("stock_market_overview", plan["intent"])
                self.assertEqual("synthesis", plan["answer_mode"])
                self.assertEqual("stock_market_overview", plan["sub_intent"])
                self.assertEqual("economy_finance_stock", plan["primary_topic"])

    def test_required_topic_routing_cases(self) -> None:
        cases = {
            "tong hop tin tuc tai chinh": "economy_finance_stock",
            "tong hop tin tuc chung khoan": "economy_finance_stock",
            "tin chung khoan moi nhat": "economy_finance_stock",
            "co phieu nao tang manh hom nay": "economy_finance_stock",
            "VN-Index hom nay the nao": "economy_finance_stock",
            "ngan hang va tin dung co gi moi": "economy_finance_stock",
            "van de phap ly trong hoat dong ngan hang": "economy_finance_stock",
            "lai suat va ty gia tuan nay": "economy_finance_stock",
            "tin AI moi nhat": "tech_ai_internet",
            "cong nghe hom nay co gi moi": "tech_ai_internet",
            "tong hop tin tuc internet": "tech_ai_internet",
            "Zalo co thong bao gi moi": "tech_ai_internet",
            "canh bao lua dao OTP qua mang": "tech_ai_internet",
            "OpenAI va Nvidia co tin gi moi": "tech_ai_internet",
            "tin thoi su hom nay": "politics_society",
            "cong an vua bat vu gi": "politics_society",
            "quoc hoi thong qua luat moi": "politics_society",
            "chinh phu co chinh sach moi nao": "politics_society",
            "cac vu an dang chu y gan day": "politics_society",
            "tinh hinh the gioi hom nay": "world_geopolitics",
            "chien su Ukraine moi nhat": "world_geopolitics",
            "My va Trung Quoc co cang thang gi": "world_geopolitics",
            "tin quoc te moi nhat": "world_geopolitics",
            "NATO va Nga co dien bien gi": "world_geopolitics",
            "tin doanh nghiep moi nhat": "business_startup",
            "startup Viet Nam goi von": "business_startup",
            "doanh nghiep nao co ket qua kinh doanh noi bat": "business_startup",
            "CEO cac cong ty cong nghe noi gi": "business_startup",
            "cac thuong vu M&A gan day": "business_startup",
            "tin bat dong san moi nhat": "real_estate",
            "gia nha Ha Noi tang the nao": "real_estate",
            "quy hoach khu cong nghiep moi": "real_estate",
            "du an nha o xa hoi co gi moi": "real_estate",
            "phap ly du an bat dong san": "real_estate",
            "tin giao duc moi nhat": "lifestyle_education_health_entertainment",
            "tuyen sinh dai hoc nam nay": "lifestyle_education_health_entertainment",
            "suc khoe va benh vien co tin gi": "lifestyle_education_health_entertainment",
            "showbiz Viet co gi moi": "lifestyle_education_health_entertainment",
            "du lich he nam nay": "lifestyle_education_health_entertainment",
            "bong da Viet Nam moi nhat": "lifestyle_education_health_entertainment",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                plan = route_query(query)
                self.assertEqual(expected_topic, plan["primary_topic"])


if __name__ == "__main__":
    unittest.main()
