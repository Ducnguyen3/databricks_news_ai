from __future__ import annotations

import json
import unittest

from app.local_ai.chunking.article_chunker import parent_article_from_mapping
from app.local_ai.chunking.metadata_builder import ChunkMetadataBuilder
from app.local_ai.chunking.models import ArticleBlock
from app.local_ai.image_enrichment import enrich_articles_with_images


class ImageMetadataEnrichmentTest(unittest.TestCase):
    def test_article_with_two_images_is_enriched(self) -> None:
        articles = [{"article_id": "a1", "canonical_url": "https://example.com/a"}]
        images = [
            {"article_id": "a1", "image_url": "https://example.com/1.jpg", "caption": "One", "position": 0},
            {"article_id": "a1", "image_url": "https://example.com/2.jpg", "caption": "Two", "position": 1},
        ]

        enriched = enrich_articles_with_images(articles, images)
        image_items = json.loads(enriched[0]["images_json"])

        self.assertTrue(enriched[0]["has_images"])
        self.assertEqual(2, enriched[0]["image_count"])
        self.assertEqual(2, len(enriched[0]["images"]))
        self.assertEqual(2, len(image_items))
        self.assertEqual("One", image_items[0]["caption"])

    def test_article_without_images_gets_empty_metadata(self) -> None:
        enriched = enrich_articles_with_images([{"article_id": "a1", "canonical_url": "https://example.com/a"}], [])

        self.assertEqual("[]", enriched[0]["images_json"])
        self.assertEqual([], enriched[0]["images"])
        self.assertFalse(enriched[0]["has_images"])
        self.assertEqual(0, enriched[0]["image_count"])

    def test_chunk_metadata_serializes_images_for_chroma(self) -> None:
        article = parent_article_from_mapping(
            {
                "article_id": "a1",
                "source": "vnexpress",
                "url": "https://example.com/a",
                "title": "Title",
                "content": "Content",
                "images_json": json.dumps([{"image_url": "https://example.com/1.jpg"}]),
            }
        )
        block = ArticleBlock("a1::b0", "a1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "a1::b0::c0", 0, "lead")

        self.assertIsInstance(metadata["images_json"], str)
        self.assertTrue(metadata["has_images"])
        self.assertEqual(1, metadata["image_count"])


if __name__ == "__main__":
    unittest.main()
