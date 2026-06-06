from __future__ import annotations

import unittest

from app.local_ai.topic_profiles import TOPIC_PROFILES, get_topic_profile


MAIN_TOPICS = {
    "tech_ai_internet",
    "economy_finance_stock",
    "politics_society",
    "world_geopolitics",
    "business_startup",
    "real_estate",
    "lifestyle_education_health_entertainment",
}


class TopicProfilesTest(unittest.TestCase):
    def test_all_main_topics_have_profiles(self) -> None:
        self.assertTrue(MAIN_TOPICS.issubset(TOPIC_PROFILES))
        self.assertIn("general_news", TOPIC_PROFILES)

    def test_general_news_fallback(self) -> None:
        profile = get_topic_profile("unknown")

        self.assertEqual("general_news", profile.topic_id)
        self.assertEqual("Tin tong hop", profile.topic_name)

    def test_finance_profile_warns_against_certain_trading_advice(self) -> None:
        text = " ".join(get_topic_profile("economy_finance_stock").caution_rules).lower()

        self.assertIn("khuyen nghi mua/ban", text)
        self.assertIn("chac chan", text)

    def test_real_estate_profile_focuses_legal_planning_price_liquidity(self) -> None:
        text = " ".join(get_topic_profile("real_estate").focus_points).lower()

        for keyword in ("phap ly", "quy hoach", "gia", "thanh khoan"):
            self.assertIn(keyword, text)

    def test_world_profile_avoids_unconfirmed_claims(self) -> None:
        text = " ".join(get_topic_profile("world_geopolitics").caution_rules).lower()

        self.assertIn("chua xac nhan", text)


if __name__ == "__main__":
    unittest.main()
