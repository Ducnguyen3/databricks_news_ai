from __future__ import annotations

import unittest

from app.local_ai.chunking.metadata_builder import ChunkMetadataBuilder
from app.local_ai.chunking.models import ArticleBlock, ParentArticle


class ChunkMetadataTest(unittest.TestCase):
    def test_metadata_contains_required_filter_fields(self) -> None:
        article = ParentArticle(
            article_id="a1",
            source="cafef",
            url="https://example.com/a",
            canonical_url="https://example.com/canonical-a",
            title="Title",
            summary=None,
            content="Content",
            published_at="2026-01-01",
            source_category="thi-truong-chung-khoan.chn",
            source_category_name="Chung khoan",
            source_category_url="https://cafef.vn/thi-truong-chung-khoan.chn",
            primary_topic="economy_finance_stock",
            primary_topic_name="Kinh te - Tai chinh - Chung khoan",
            topic_confidence=0.9,
            secondary_topics=["business_startup"],
            entities=[{"normalized_name": "HPG", "type": "stock_symbol"}],
            images=[{"image_url": "https://example.com/a.jpg"}],
        )
        block = ArticleBlock("a1::b0", "a1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "chunk-1", 0, "lead")

        self.assertEqual("a1", metadata["article_id"])
        self.assertEqual("cafef", metadata["source"])
        self.assertEqual("https://example.com/canonical-a", metadata["canonical_url"])
        self.assertEqual("2026-01-01", metadata["published_at"])
        self.assertEqual("thi-truong-chung-khoan.chn", metadata["source_category"])
        self.assertEqual("Chung khoan", metadata["source_category_name"])
        self.assertEqual("https://cafef.vn/thi-truong-chung-khoan.chn", metadata["source_category_url"])
        self.assertEqual("economy_finance_stock", metadata["primary_topic"])
        self.assertEqual("Kinh te - Tai chinh - Chung khoan", metadata["primary_topic_name"])
        self.assertEqual(0.9, metadata["topic_confidence"])
        self.assertIn("HPG", metadata["entity_names"])
        self.assertEqual(True, metadata["has_images"])
        self.assertEqual("a1", metadata["parent_id"])

    def test_metadata_contains_index_sync_fields(self) -> None:
        article = ParentArticle(
            article_id="a1",
            source="genk",
            url="https://example.com/a",
            title="Title",
            summary=None,
            content="Content",
            published_at=None,
            source_category="ai.chn",
            primary_topic="tech_ai_internet",
            primary_topic_name="Cong nghe - AI - Internet",
            secondary_topics=[],
            entities=[],
            images=[],
            metadata={"content_hash": "hash-a1"},
        )
        block = ArticleBlock("a1::b0", "a1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder(
            index_version="test-index",
            embedding_model="test-model",
            chunking_version="test-chunking",
        ).build_metadata(article, block, "chunk-1", 0, "lead")

        self.assertIn("indexed_at", metadata)
        self.assertEqual("test-index", metadata["index_version"])
        self.assertEqual("test-model", metadata["embedding_model"])
        self.assertEqual("test-chunking", metadata["chunking_version"])
        self.assertEqual("hash-a1", metadata["content_hash"])
        self.assertIn("canonical_url", metadata)

    def test_metadata_serializes_lists_and_dicts_for_chroma(self) -> None:
        article = ParentArticle(
            article_id="a1",
            source="cafef",
            url="https://example.com/a",
            title="Title",
            summary=None,
            content="Content",
            published_at=None,
            source_category=None,
            primary_topic=None,
            primary_topic_name=None,
            secondary_topics=["x"],
            entities=[{"normalized_name": "HPG", "type": "stock_symbol"}],
            images=[],
        )
        block = ArticleBlock("a1::b0", "a1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "chunk-1", 0, "lead")

        self.assertIsInstance(metadata["secondary_topics_json"], str)
        self.assertIsInstance(metadata["entities_json"], str)
        self.assertIsInstance(metadata["entity_names"], str)
        self.assertIsInstance(metadata["entity_types"], str)
        self.assertEqual("a1", metadata["article_id"])
        self.assertEqual("[]", metadata["images_json"])
        self.assertIsInstance(metadata["has_images"], bool)
        self.assertIsInstance(metadata["image_count"], int)
        for value in metadata.values():
            self.assertNotIsInstance(value, (dict, list))

    def test_chunk_metadata_contains_source_category_and_topic_for_genk(self) -> None:
        article = ParentArticle(
            article_id="g1",
            source="genk",
            url="https://genk.vn/meta-ai-de-dai-den-muc-ngo-nghech-165260602080717731.chn",
            title="Meta AI de dai den muc ngo nghech",
            summary=None,
            content="Content",
            published_at="2026-06-02",
            source_category="ai.chn",
            source_category_name="AI",
            source_category_url="https://genk.vn/ai.chn",
            primary_topic="tech_ai_internet",
            primary_topic_name="Cong nghe - AI - Internet",
            topic_confidence=0.9,
            secondary_topics=["business_startup"],
            entities=[],
            images=[],
        )
        block = ArticleBlock("g1::b0", "g1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "chunk-g1", 0, "lead")

        self.assertEqual("genk", metadata["source"])
        self.assertEqual("ai.chn", metadata["source_category"])
        self.assertEqual("AI", metadata["source_category_name"])
        self.assertEqual("https://genk.vn/ai.chn", metadata["source_category_url"])
        self.assertEqual("tech_ai_internet", metadata["primary_topic"])
        self.assertEqual("Cong nghe - AI - Internet", metadata["primary_topic_name"])

    def test_chunk_metadata_contains_source_category_and_topic_for_diendandoanhnghiep(self) -> None:
        article = ParentArticle(
            article_id="d1",
            source="diendandoanhnghiep",
            url="https://diendandoanhnghiep.vn/muc-tieu-cao-nhat-la-to-chuc-thanh-cong-mot-nam-apec-mang-dam-dau-an-viet-nam-10179807.html",
            title="Muc tieu cao nhat la to chuc thanh cong mot nam APEC",
            summary=None,
            content="Content",
            published_at="2026-06-02",
            source_category="chinh-tri-xa-hoi",
            source_category_name="Chinh tri - Xa hoi",
            source_category_url="https://diendandoanhnghiep.vn/chinh-tri-xa-hoi",
            primary_topic="politics_society",
            primary_topic_name="Thoi su - Chinh tri - Xa hoi",
            topic_confidence=0.9,
            secondary_topics=[],
            entities=[],
            images=[],
        )
        block = ArticleBlock("d1::b0", "d1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "chunk-d1", 0, "lead")

        self.assertEqual("diendandoanhnghiep", metadata["source"])
        self.assertEqual("chinh-tri-xa-hoi", metadata["source_category"])
        self.assertEqual("Chinh tri - Xa hoi", metadata["source_category_name"])
        self.assertEqual("https://diendandoanhnghiep.vn/chinh-tri-xa-hoi", metadata["source_category_url"])
        self.assertEqual("politics_society", metadata["primary_topic"])
        self.assertEqual("Thoi su - Chinh tri - Xa hoi", metadata["primary_topic_name"])


if __name__ == "__main__":
    unittest.main()
