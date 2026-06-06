from __future__ import annotations

import unittest

from app.local_ai.chunking.models import ArticleChunk
from app.local_ai.vector_store import ChromaVectorStore


class _Collection:
    def __init__(self) -> None:
        self.payload = {}

    def upsert(self, **kwargs) -> None:
        self.payload = kwargs


class VectorStoreMetadataTest(unittest.TestCase):
    def test_upsert_chunks_preserves_image_metadata(self) -> None:
        vector_store = ChromaVectorStore.__new__(ChromaVectorStore)
        vector_store._collection = _Collection()
        vector_store._collection_name = "news_articles"
        vector_store._persist_directory = "data/chroma"
        chunk = ArticleChunk(
            chunk_id="a1::b0::c0",
            article_id="a1",
            block_id="a1::b0",
            chunk_index=0,
            chunk_text="chunk text",
            embedding_text="embedding text",
            token_count=2,
            metadata={
                "article_id": "a1",
                "images_json": '[{"image_url":"https://example.com/a.jpg"}]',
                "has_images": True,
                "image_count": 1,
            },
        )

        vector_store.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])

        metadata = vector_store._collection.payload["metadatas"][0]
        self.assertEqual('[{"image_url":"https://example.com/a.jpg"}]', metadata["images_json"])
        self.assertTrue(metadata["has_images"])
        self.assertEqual(1, metadata["image_count"])


if __name__ == "__main__":
    unittest.main()
