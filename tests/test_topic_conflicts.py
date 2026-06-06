from __future__ import annotations

import unittest

from app.local_ai.query_router import route_query


class TopicConflictRoutingTest(unittest.TestCase):
    def test_generic_words_do_not_pull_domain_queries_to_politics(self) -> None:
        cases = {
            "tong hop tin tuc tai chinh": "economy_finance_stock",
            "tong hop tin tuc chung khoan tuan nay": "economy_finance_stock",
            "tinh hinh bat dong san hom nay": "real_estate",
            "tong hop tin cong nghe": "tech_ai_internet",
            "cap nhat tin quoc te": "world_geopolitics",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                self.assertEqual(expected_topic, route_query(query)["primary_topic"])

    def test_finance_beats_politics_when_domain_is_clear(self) -> None:
        cases = {
            "van de phap ly trong hoat dong ngan hang": "economy_finance_stock",
            "ngan hang noi tin dung cho nha o xa hoi": "economy_finance_stock",
            "lai suat cho vay bat dong san": "economy_finance_stock",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                self.assertEqual(expected_topic, route_query(query)["primary_topic"])

    def test_real_estate_beats_politics_when_domain_is_clear(self) -> None:
        cases = {
            "phap ly du an bat dong san": "real_estate",
            "quy hoach dat nen khu cong nghiep": "real_estate",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                self.assertEqual(expected_topic, route_query(query)["primary_topic"])

    def test_stock_beats_business(self) -> None:
        cases = {
            "doanh nghiep co co phieu tang tran": "economy_finance_stock",
            "ma co phieu HPG co gi moi": "economy_finance_stock",
        }

        for query, expected_topic in cases.items():
            with self.subTest(query=query):
                self.assertEqual(expected_topic, route_query(query)["primary_topic"])

    def test_business_or_tech_queries_do_not_fall_to_politics(self) -> None:
        topics = {
            route_query("doanh nghiep cong nghe ra san pham moi")["primary_topic"],
            route_query("startup goi von trong linh vuc AI")["primary_topic"],
        }

        self.assertNotIn("politics_society", topics)
        self.assertTrue(topics.issubset({"business_startup", "tech_ai_internet"}))


if __name__ == "__main__":
    unittest.main()
