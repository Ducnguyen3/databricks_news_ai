from __future__ import annotations

import unittest

from app.local_ai.retriever import MetadataFilteringRetriever


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class _VectorStore:
    def __init__(self) -> None:
        self.requested_top_k: list[int] = []

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, object]]:
        self.requested_top_k.append(top_k)
        return [
            {
                "chunk_id": f"chunk-{index}",
                "text": "Tin AI moi nhat",
                "score": 0.8,
                "metadata": {
                    "article_id": f"article-{index}",
                    "primary_topic": "tech_ai_internet",
                    "source": "genk",
                    "published_at": "2026-06-05T00:00:00+00:00",
                    "entity_names": "",
                },
            }
            for index in range(min(top_k, 3))
        ]


class RetrieverRerankFlowTest(unittest.TestCase):
    def test_retriever_uses_wide_candidates_before_rerank(self) -> None:
        vector_store = _VectorStore()
        retriever = MetadataFilteringRetriever(vector_store, _EmbeddingModel(), retrieval_mode="vector")

        retriever.retrieve("tin AI", {"primary_topic": "tech_ai_internet", "time_range": "all"}, top_n=5, top_k=2)

        self.assertGreaterEqual(vector_store.requested_top_k[-1], 50)

    def test_time_sensitive_query_uses_even_wider_candidates(self) -> None:
        vector_store = _VectorStore()
        retriever = MetadataFilteringRetriever(vector_store, _EmbeddingModel(), retrieval_mode="vector")

        retriever.retrieve(
            "tin AI moi nhat hom nay",
            {"intent": "latest_news", "primary_topic": "tech_ai_internet", "time_range": "today"},
            top_n=5,
            top_k=2,
        )

        self.assertGreaterEqual(vector_store.requested_top_k[-1], 80)


if __name__ == "__main__":
    unittest.main()
