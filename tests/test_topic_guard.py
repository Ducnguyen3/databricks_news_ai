from __future__ import annotations

import unittest

from app.local_ai.answer_validator import validate_answer_against_topic
from app.local_ai.topic_guard import filter_context_for_topic, validate_context_for_topic


def _chunk(text: str, topic: str) -> dict:
    return {
        "text": text,
        "metadata": {
            "title": text[:80],
            "primary_topic": topic,
            "topic": topic,
        },
    }


class TopicGuardTest(unittest.TestCase):
    def test_stock_query_drops_tomcat_context_and_keeps_stock_chunks(self) -> None:
        chunks = [
            _chunk("Tomcat da thay doi sessionTrackingMode trong catalina.properties.", "tech_ai_internet"),
            _chunk("VN-Index giam 13,64 diem, thanh khoan tren HoSE giam, khoi ngoai ban rong.", "economy_finance_stock"),
            _chunk("Co phieu ngan hang va chung khoan phan hoa, tu doanh mua rong tren HoSE.", "economy_finance_stock"),
        ]

        filtered = filter_context_for_topic("Tong hop tin tuc chung khoan tuan nay", "economy_finance_stock", chunks)

        self.assertTrue(filtered.result.allowed)
        self.assertEqual(2, filtered.kept_after_topic_filter)
        self.assertEqual(1, filtered.dropped_deny_keyword)
        self.assertNotIn("Tomcat", " ".join(chunk["text"] for chunk in filtered.kept))

    def test_answer_validator_blocks_tomcat_in_stock_answer(self) -> None:
        valid, violations = validate_answer_against_topic(
            answer="VN-Index giam diem, nen nang cap Tomcat len 7.0.26 de tranh loi sessionTrackingMode.",
            query="Tong hop tin tuc chung khoan tuan nay",
            topic="economy_finance_stock",
            sources=[{"topic": "economy_finance_stock"}],
        )

        self.assertFalse(valid)
        self.assertTrue(any("forbidden_technical_terms" in violation for violation in violations))

    def test_answer_validator_blocks_instruction_leak(self) -> None:
        valid, violations = validate_answer_against_topic(
            answer="Cau truc tra loi: Neu du lieu hien co thi tong hop theo expected va actual.",
            query="Tin AI gan day",
            topic="tech_ai_internet",
            sources=[{"topic": "tech_ai_internet"}],
        )

        self.assertFalse(valid)
        self.assertTrue(any("instruction_leak" in violation for violation in violations))

    def test_all_topic_guards_drop_obvious_cross_domain_noise(self) -> None:
        cases = [
            ("Tin AI gan day", "tech_ai_internet", "VN-Index va khoi ngoai ban rong tren HoSE", "economy_finance_stock"),
            ("Bat dong san Ha Noi co gi moi", "real_estate", "Tomcat cau hinh servlet trong catalina.properties", "tech_ai_internet"),
            ("Dien bien Ukraine hom nay", "world_geopolitics", "Can ho Ha Noi mo ban dot moi", "real_estate"),
            ("Tin phap luat hom nay", "politics_society", "Tomcat sessionTrackingMode bi loi", "tech_ai_internet"),
            ("Tin doanh nghiep moi nhat", "business_startup", "sessionTrackingMode trong Java server", "tech_ai_internet"),
            ("Tin suc khoe gan day", "lifestyle_education_health_entertainment", "VN-Index tang, khoi ngoai mua rong", "economy_finance_stock"),
            ("Tin chung khoan tuan nay", "economy_finance_stock", "Tomcat deployment web application", "tech_ai_internet"),
        ]

        for query, topic, noisy_text, noisy_topic in cases:
            with self.subTest(topic=topic):
                chunks = [
                    _chunk(noisy_text, noisy_topic),
                    _chunk(_allow_text(topic), topic),
                    _chunk(_allow_text(topic), topic),
                ]
                filtered = filter_context_for_topic(query, topic, chunks)
                kept_text = " ".join(chunk["text"] for chunk in filtered.kept).lower()
                self.assertTrue(filtered.result.allowed)
                self.assertNotIn(noisy_text.lower(), kept_text)

    def test_validate_context_returns_false_when_all_chunks_are_noise(self) -> None:
        result = validate_context_for_topic(
            "Tong hop tin tuc chung khoan tuan nay",
            "economy_finance_stock",
            [_chunk("Tomcat sessionTrackingMode catalina.properties", "tech_ai_internet")],
        )

        self.assertFalse(result.allowed)


def _allow_text(topic: str) -> str:
    return {
        "tech_ai_internet": "OpenAI va Nvidia cong bo cong nghe AI moi trong linh vuc cloud.",
        "economy_finance_stock": "VN-Index va co phieu ngan hang co thanh khoan tang, khoi ngoai mua rong.",
        "politics_society": "Chinh phu va Quoc hoi thao luan chinh sach dan sinh moi.",
        "world_geopolitics": "Ukraine, Nga va NATO co dien bien ngoai giao moi.",
        "business_startup": "Doanh nghiep startup cong bo doanh thu va chien luoc kinh doanh moi.",
        "real_estate": "Thi truong bat dong san, can ho va du an nha o xa hoi co tin moi.",
        "lifestyle_education_health_entertainment": "Suc khoe, giao duc va doi song co nhieu thong tin dang chu y.",
    }[topic]


if __name__ == "__main__":
    unittest.main()
