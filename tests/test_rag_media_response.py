from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from app.local_ai.rag_service import RAGService


class _EmbeddingModel:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class _VectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query_embedding, top_k=5):
        return self._results[:top_k]


class _EmptyImageRepository:
    def fetch_article_images(self, article_ids):
        return []


def _result(has_images: bool = True):
    return {
        "chunk_id": "chunk-1",
        "text": "Ukraine hom nay co dien bien moi.",
        "score": 0.92,
        "metadata": {
            "article_id": "article-ukraine",
            "title": "Anh ve Ukraine",
            "source": "vnexpress",
            "url": "https://example.com/ukraine",
            "primary_topic": "world_geopolitics",
            "entity_names": "Ukraine",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "has_images": has_images,
            "images_json": json.dumps(
                [
                    {
                        "article_id": "article-ukraine",
                        "image_url": "https://example.com/ukraine.jpg",
                        "caption": "Ukraine",
                        "credit": "Reuters",
                        "is_representative": True,
                    }
                ]
                if has_images
                else []
            ),
        },
    }


class RagMediaResponseTest(unittest.TestCase):
    def test_media_lookup_response_includes_images_from_chunk_metadata(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_result(has_images=True)]),
            ollama_client=None,
        )

        response = service.answer_structured("anh ve Ukraine hom nay", top_k=3)

        self.assertEqual("media_lookup", response["intent"])
        self.assertTrue(response["query_plan"]["need_images"])
        self.assertIn("images", response)
        self.assertEqual(1, len(response["images"]))
        self.assertEqual("article-ukraine", response["images"][0]["article_id"])
        self.assertEqual("https://example.com/ukraine.jpg", response["images"][0]["image_url"])
        self.assertEqual("Anh ve Ukraine", response["images"][0]["article_title"])
        self.assertEqual("https://example.com/ukraine", response["images"][0]["article_url"])

    def test_media_lookup_without_images_does_not_crash(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_result(has_images=False)]),
            ollama_client=None,
            image_repository=_EmptyImageRepository(),
        )

        response = service.answer_structured("anh ve Ukraine hom nay", top_k=3)

        self.assertEqual([], response["images"])
        self.assertIn("answer", response)
        self.assertIn("chưa có metadata ảnh phù hợp", response["answer"])
        self.assertNotIn("không đủ dữ liệu", response["answer"])

    def test_media_lookup_returns_available_images_when_less_than_limit(self) -> None:
        result = _result(has_images=True)
        result["metadata"]["images_json"] = json.dumps(
            [
                {
                    "article_id": "article-ukraine",
                    "image_url": "https://example.com/ukraine-1.jpg",
                    "caption": "Ukraine",
                },
                {
                    "article_id": "article-ukraine",
                    "image_url": "https://example.com/ukraine-2.jpg",
                    "caption": "Ukraine",
                },
            ]
        )
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([result]),
            ollama_client=None,
        )

        response = service.answer_structured("anh ve Ukraine hom nay", top_k=3)

        self.assertEqual(2, len(response["images"]))
        self.assertIn("Tìm thấy 2 ảnh", response["answer"])


if __name__ == "__main__":
    unittest.main()
