from __future__ import annotations

import unittest

from app.local_ai.processors import (
    BusinessStartupProcessor,
    DefaultNewsProcessor,
    EconomyFinanceProcessor,
    LifestyleProcessor,
    PoliticsSocietyProcessor,
    RealEstateProcessor,
    TechAIProcessor,
    WorldGeopoliticsProcessor,
    get_processor,
)


class ProcessorRegistryTest(unittest.TestCase):
    def test_known_topics_return_matching_processors(self) -> None:
        cases = {
            "tech_ai_internet": TechAIProcessor,
            "economy_finance_stock": EconomyFinanceProcessor,
            "politics_society": PoliticsSocietyProcessor,
            "world_geopolitics": WorldGeopoliticsProcessor,
            "business_startup": BusinessStartupProcessor,
            "real_estate": RealEstateProcessor,
            "lifestyle_education_health_entertainment": LifestyleProcessor,
        }

        for topic_id, processor_class in cases.items():
            with self.subTest(topic_id=topic_id):
                self.assertIsInstance(get_processor(topic_id), processor_class)

    def test_unknown_topics_fallback_to_default_processor(self) -> None:
        for topic_id in ("unknown_topic", None, ""):
            with self.subTest(topic_id=topic_id):
                self.assertIsInstance(get_processor(topic_id), DefaultNewsProcessor)


if __name__ == "__main__":
    unittest.main()
