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
)


def _route(topic: str) -> dict[str, object]:
    return {
        "intent": "topic_news",
        "primary_topic": topic,
        "entities": ["OpenAI"],
        "stock_symbols": ["FPT"],
        "time_range": "7d",
    }


def _chunk(topic: str) -> dict[str, object]:
    return {
        "chunk_id": "c1",
        "text": "OpenAI cong bo san pham moi trong tuan nay.",
        "metadata": {
            "article_id": "a1",
            "title": "Tin cong nghe",
            "source": "vnexpress",
            "url": "https://example.com/a1",
            "primary_topic": topic,
        },
    }


_SOURCES = [
    {
        "article_id": "a1",
        "title": "Tin cong nghe",
        "source": "vnexpress",
        "url": "https://example.com/a1",
    }
]
_IMAGES = [{"article_id": "a1", "image_url": "https://example.com/a1.jpg", "caption": "Anh bai viet"}]


class DomainProcessorsTest(unittest.TestCase):
    def test_processors_build_context_and_prompt(self) -> None:
        processors = [
            DefaultNewsProcessor(),
            TechAIProcessor(),
            EconomyFinanceProcessor(),
            PoliticsSocietyProcessor(),
            WorldGeopoliticsProcessor(),
            BusinessStartupProcessor(),
            RealEstateProcessor(),
            LifestyleProcessor(),
        ]

        for processor in processors:
            with self.subTest(processor=processor.__class__.__name__):
                context = processor.build_context(
                    query="Tin OpenAI moi nhat?",
                    route=_route(processor.topic_id),
                    retrieved_chunks=[_chunk(processor.topic_id)],
                    sources=_SOURCES,
                    images=_IMAGES,
                )
                prompt = processor.build_prompt(context)

                self.assertIsInstance(context, dict)
                self.assertEqual("Tin OpenAI moi nhat?", context["query"])
                self.assertIn("sources", context)
                self.assertIn("retrieved_chunks", context)
                self.assertIsInstance(prompt, str)
                self.assertIn("OpenAI cong bo san pham moi", prompt)
                self.assertIn("DOMAIN_INSTRUCTIONS", prompt)

    def test_economy_processor_has_finance_guardrails(self) -> None:
        processor = EconomyFinanceProcessor()
        context = processor.build_context(
            query="Co phieu FPT hom nay the nao?",
            route=_route(processor.topic_id),
            retrieved_chunks=[_chunk(processor.topic_id)],
            sources=_SOURCES,
            images=[],
        )

        prompt = processor.build_prompt(context)

        self.assertIn("Khong bia du lieu gia realtime", prompt)
        self.assertIn("chua co stock API", prompt)
        self.assertIn("Khong dua khuyen nghi mua/ban chac chan", prompt)

    def test_world_processor_separates_confirmed_claims_and_analysis(self) -> None:
        processor = WorldGeopoliticsProcessor()
        context = processor.build_context(
            query="Dien bien Ukraine hom nay?",
            route=_route(processor.topic_id),
            retrieved_chunks=[_chunk(processor.topic_id)],
            sources=_SOURCES,
            images=[],
        )

        prompt = processor.build_prompt(context)

        self.assertIn("su kien da xac nhan", prompt)
        self.assertIn("tuyen bo tu cac ben", prompt)
        self.assertIn("phan tich/nhan dinh", prompt)

    def test_lifestyle_processor_adds_health_guardrail_for_health_queries(self) -> None:
        processor = LifestyleProcessor()
        context = processor.build_context(
            query="Tin suc khoe ve thuoc moi?",
            route=_route(processor.topic_id),
            retrieved_chunks=[_chunk(processor.topic_id)],
            sources=_SOURCES,
            images=[],
        )

        prompt = processor.build_prompt(context)

        self.assertIn("khong chan doan", prompt)
        self.assertIn("khong thay the tu van y te", prompt)
        self.assertIn("chi tom tat theo nguon da retrieve", prompt)


if __name__ == "__main__":
    unittest.main()
