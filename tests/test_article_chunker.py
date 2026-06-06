from __future__ import annotations

import unittest

from app.local_ai.chunking.article_chunker import ArticleChunker
from app.local_ai.chunking.models import ParentArticle


def _article(content: str, summary: str | None = "Summary") -> ParentArticle:
    return ParentArticle(
        article_id="article-1",
        source="vnexpress",
        url="https://example.com/a",
        title="HPG và FPT tăng mạnh",
        summary=summary,
        content=content,
        published_at="2026-01-01T00:00:00",
        source_category="Chứng khoán",
        primary_topic="economy_finance_stock",
        primary_topic_name="Kinh tế - Tài chính - Chứng khoán",
        secondary_topics=["business_startup"],
        entities=[{"normalized_name": "HPG", "type": "stock_symbol"}],
        images=[{"image_url": "https://example.com/a.jpg"}],
    )


class ArticleChunkerTest(unittest.TestCase):
    def test_short_article_creates_one_or_two_chunks(self) -> None:
        chunks = ArticleChunker().chunk_article(_article("Lead paragraph.\n\nBody paragraph."))

        self.assertGreaterEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks), 2)

    def test_long_article_is_split_into_multiple_chunks(self) -> None:
        content = "\n\n".join(f"Đoạn {index} " + "nội dung " * 120 for index in range(20))
        chunks = ArticleChunker().chunk_article(_article(content))

        self.assertGreater(len(chunks), 1)

    def test_chunk_id_contains_article_and_block_id(self) -> None:
        chunk = ArticleChunker().chunk_article(_article("Lead.\n\nBody."))[0]

        self.assertIn("article-1", chunk.chunk_id)
        self.assertIn("::b", chunk.block_id)

    def test_embedding_text_contains_context(self) -> None:
        chunk = ArticleChunker().chunk_article(_article("Lead.\n\nBody."))[0]

        self.assertIn("Tiêu đề: HPG và FPT tăng mạnh", chunk.embedding_text)
        self.assertIn("Tóm tắt: Summary", chunk.embedding_text)
        self.assertIn("Chủ đề: Kinh tế - Tài chính - Chứng khoán", chunk.embedding_text)
        self.assertIn("Thực thể: HPG", chunk.embedding_text)
        self.assertIn("Nguồn: vnexpress", chunk.embedding_text)
        self.assertIn("Ngày đăng: 2026-01-01T00:00:00", chunk.embedding_text)
        self.assertIn("Nội dung:", chunk.embedding_text)

    def test_chunk_metadata_keeps_topic_source_entities_and_images(self) -> None:
        chunk = ArticleChunker().chunk_article(_article("Lead.\n\nBody."))[0]
        metadata = chunk.metadata

        self.assertEqual("economy_finance_stock", metadata["primary_topic"])
        self.assertEqual("vnexpress", metadata["source"])
        self.assertEqual(True, metadata["has_images"])
        self.assertEqual(1, metadata["image_count"])
        self.assertIn("https://example.com/a.jpg", metadata["images_json"])
        self.assertIn("HPG", metadata["entities_json"])

    def test_chunker_handles_missing_optional_fields(self) -> None:
        chunks = ArticleChunker().chunk_article(_article("Lead only.", summary=None))

        self.assertGreaterEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
