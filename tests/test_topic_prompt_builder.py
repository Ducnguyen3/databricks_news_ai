from __future__ import annotations

import unittest

from app.local_ai.prompt_builder import build_topic_rag_prompt
from app.local_ai.topic_profiles import get_topic_profile


class TopicPromptBuilderTest(unittest.TestCase):
    def test_prompt_uses_router_answer_mode_over_local_detection(self) -> None:
        prompt = build_topic_rag_prompt(
            question="tin AI moi nhat",
            context_blocks=[],
            topic_profile=get_topic_profile("tech_ai_internet"),
            query_plan={"intent": "topic_news", "answer_mode": "citation"},
        )

        self.assertIn("CHẾ ĐỘ TRẢ LỜI: TRÍCH DẪN / CHÍNH XÁC", prompt)
        self.assertIn("- answer_mode: citation", prompt)

    def test_tech_prompt_contains_role_focus_question_context_and_rules(self) -> None:
        profile = get_topic_profile("tech_ai_internet")
        prompt = build_topic_rag_prompt(
            question="tin AI moi nhat",
            context_blocks=[
                {
                    "text": "OpenAI cong bo san pham AI moi.",
                    "metadata": {
                        "article_id": "a1",
                        "title": "Tin AI",
                        "source": "vnexpress",
                        "url": "https://example.com/a1",
                        "primary_topic": "tech_ai_internet",
                    },
                }
            ],
            topic_profile=profile,
            query_plan={"intent": "topic_news", "entities": ["OpenAI"], "time_range": "7d"},
        )

        self.assertIn("Cong nghe - AI - Internet", prompt)
        self.assertIn("Chuyen gia phan tich cong nghe", prompt)
        self.assertIn("tin AI moi nhat", prompt)
        self.assertIn("OpenAI cong bo san pham AI moi.", prompt)
        self.assertIn("Khong bia", prompt)
        self.assertIn("databricks_news_ai", prompt)
        self.assertIn("CORE_RAG_RULES", prompt)
        self.assertIn("Không chèn ảnh Markdown", prompt)
        self.assertIn("tiếng Việt có dấu", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("CAUTION_RULES", prompt)

    def test_topic_prompt_includes_image_metadata(self) -> None:
        profile = get_topic_profile("tech_ai_internet")
        prompt = build_topic_rag_prompt(
            question="cho toi anh ve OpenAI",
            context_blocks=[
                {
                    "text": "OpenAI gioi thieu san pham moi.",
                    "metadata": {
                        "title": "OpenAI ra mat san pham",
                        "source": "genk",
                        "published_at": "2026-06-04",
                        "url": "https://example.com/openai",
                        "image_url": "https://example.com/openai.jpg",
                        "image_caption": "Anh minh hoa OpenAI",
                    },
                }
            ],
            topic_profile=profile,
            query_plan={"intent": "media_lookup", "need_images": True},
        )

        self.assertIn("https://example.com/openai.jpg", prompt)
        self.assertIn("Anh minh hoa OpenAI", prompt)

    def test_topic_prompt_reads_images_json_before_image_url(self) -> None:
        profile = get_topic_profile("tech_ai_internet")
        prompt = build_topic_rag_prompt(
            question="cho toi anh ve OpenAI",
            context_blocks=[
                {
                    "text": "OpenAI gioi thieu san pham moi.",
                    "metadata": {
                        "title": "OpenAI ra mat san pham",
                        "source": "genk",
                        "published_at": "2026-06-04",
                        "url": "https://example.com/openai",
                        "images_json": '[{"image_url": "https://example.com/from-json.jpg", "caption": "JSON image"}]',
                        "image_url": "https://example.com/fallback.jpg",
                    },
                }
            ],
            topic_profile=profile,
            query_plan={"intent": "media_lookup", "need_images": True},
        )

        self.assertIn("https://example.com/from-json.jpg", prompt)
        self.assertIn("JSON image", prompt)
        self.assertNotIn("https://example.com/fallback.jpg", prompt)

    def test_empty_context_prompt_contains_no_context_marker_and_no_info_rule(self) -> None:
        prompt = build_topic_rag_prompt(
            question="du lieu khong co",
            context_blocks=[],
            topic_profile=get_topic_profile("general_news"),
            query_plan={},
        )

        self.assertIn("[NO_CONTEXT]", prompt)
        self.assertIn("Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có thông tin đủ để trả lời câu hỏi này.", prompt)

    def test_context_groups_multiple_chunks_under_one_article_citation(self) -> None:
        prompt = build_topic_rag_prompt(
            question="tin AI moi nhat",
            context_blocks=[
                {
                    "text": "Chunk mot ve OpenAI.",
                    "score": 0.6,
                    "metadata": {
                        "article_id": "a1",
                        "title": "Tin AI",
                        "source": "vnexpress",
                        "url": "https://example.com/a1",
                        "primary_topic": "tech_ai_internet",
                    },
                },
                {
                    "text": "Chunk hai cung bai OpenAI.",
                    "score": 0.9,
                    "metadata": {
                        "article_id": "a1",
                        "title": "Tin AI",
                        "source": "vnexpress",
                        "url": "https://example.com/a1",
                        "primary_topic": "tech_ai_internet",
                    },
                },
            ],
            topic_profile=get_topic_profile("tech_ai_internet"),
            query_plan={"intent": "topic_news"},
        )

        self.assertEqual(1, prompt.count("Citation ID: [1]"))
        self.assertIn("[BÀI BÁO [1]]", prompt)
        self.assertIn("Chunk mot ve OpenAI.", prompt)
        self.assertIn("Chunk hai cung bai OpenAI.", prompt)

    def test_finance_prompt_contains_investment_caution(self) -> None:
        prompt = build_topic_rag_prompt(
            question="HPG co gi moi",
            context_blocks=[],
            topic_profile=get_topic_profile("economy_finance_stock"),
            query_plan={"stock_symbols": ["HPG"]},
        )

        self.assertIn("Khong dua khuyen nghi mua/ban", prompt)

    def test_real_estate_prompt_contains_required_focus(self) -> None:
        prompt = build_topic_rag_prompt(
            question="bat dong san Ha Noi co gi moi",
            context_blocks=[],
            topic_profile=get_topic_profile("real_estate"),
            query_plan={},
        )

        for keyword in ("Phap ly/quy hoach", "Gia va thanh khoan", "quy hoach"):
            self.assertIn(keyword, prompt)

    def test_world_prompt_contains_unconfirmed_claim_caution(self) -> None:
        prompt = build_topic_rag_prompt(
            question="tinh hinh Ukraine hom nay",
            context_blocks=[],
            topic_profile=get_topic_profile("world_geopolitics"),
            query_plan={},
        )

        self.assertIn("chua xac nhan", prompt)


if __name__ == "__main__":
    unittest.main()
