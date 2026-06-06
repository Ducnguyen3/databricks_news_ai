from __future__ import annotations

import unittest

from app.local_ai.query_router import route_query
from app.local_ai.rag_service import RAGService, _resolve_followup_route


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _VectorStore:
    def __init__(self) -> None:
        self.search_calls = 0
        self.get_chunks_calls: list[str] = []
        self._chunks = {
            "article-1": [_chunk("article-1", "Article 1", "Noi dung bai mot.")],
            "article-2": [_chunk("article-2", "Article 2", "Noi dung bai hai can tom tat.")],
            "article-3": [_chunk("article-3", "Article 3", "Noi dung bai ba.")],
        }

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        self.search_calls += 1
        return []

    def get_chunks_by_article_id(self, article_id: str, limit: int | None = None) -> list[dict]:
        self.get_chunks_calls.append(article_id)
        chunks = list(self._chunks.get(article_id, []))
        return chunks[:limit] if limit else chunks


def _chunk(article_id: str, title: str, text: str) -> dict:
    return {
        "chunk_id": f"{article_id}:0",
        "text": text,
        "score": 0.9,
        "metadata": {
            "chunk_id": f"{article_id}:0",
            "article_id": article_id,
            "title": title,
            "source": "vnexpress",
            "url": f"https://example.com/{article_id}",
            "published_at": "2026-01-01T00:00:00+00:00",
            "primary_topic": "general_news",
            "chunk_index": 0,
        },
    }


def _context() -> dict:
    return {
        "previous_answer": "Cau tra loi truoc dai hon can rut gon.",
        "previous_sources": [
            {"citation_id": 1, "article_id": "article-1", "title": "Article 1", "source": "vnexpress", "url": "https://example.com/article-1"},
            {"citation_id": 2, "article_id": "article-2", "title": "Article 2", "source": "vnexpress", "url": "https://example.com/article-2"},
            {"citation_id": 3, "article_id": "article-3", "title": "Article 3", "source": "vnexpress", "url": "https://example.com/article-3"},
        ],
    }


def _ai_context() -> dict:
    return {
        "previous_query_plan": {
            "intent": "topic_news",
            "answer_mode": "synthesis",
            "primary_topic": "tech_ai_internet",
            "domain": "cong_nghe",
            "entities": ["OpenAI"],
            "stock_symbols": [],
            "time_range": "7d",
            "topic_confidence": 1.0,
            "normalized_query": "tin ai gan day co gi",
        },
        "previous_sources": [
            {"citation_id": 1, "article_id": "ai-1", "title": "Tin AI", "topic": "tech_ai_internet", "source": "genk"}
        ],
    }


class RagServiceContextTest(unittest.TestCase):
    def test_followup_route_inherits_ai_topic_and_standalone_query(self) -> None:
        plan = route_query("vu nay anh huong sao")
        resolved = _resolve_followup_route("vu nay anh huong sao", plan, _ai_context())

        self.assertEqual("followup", resolved["answer_mode"])
        self.assertEqual("tech_ai_internet", resolved["primary_topic"])
        self.assertEqual("cong_nghe", resolved["domain"])
        self.assertIn("OpenAI", resolved["entities"])
        self.assertIn("tin ai", resolved["standalone_query"])
        self.assertFalse(resolved.get("needs_clarification", False))

    def test_followup_without_context_needs_clarification(self) -> None:
        plan = route_query("vu nay anh huong sao")
        resolved = _resolve_followup_route("vu nay anh huong sao", plan, None)

        self.assertTrue(resolved["needs_clarification"])
        self.assertEqual("", resolved["standalone_query"])

    def test_followup_citation_summary_uses_referenced_article_without_search(self) -> None:
        vector_store = _VectorStore()
        service = RAGService(embedding_model=_EmbeddingModel(), vector_store=vector_store, ollama_client=None)

        response = service.answer_structured("tom tat bai [2]", current_context=_context())

        self.assertEqual("article_summary", response["intent"])
        self.assertEqual(["article-2"], vector_store.get_chunks_calls)
        self.assertEqual(0, vector_store.search_calls)
        self.assertEqual("article-2", response["sources"][0]["article_id"])

    def test_ambiguous_this_article_asks_for_clarification(self) -> None:
        vector_store = _VectorStore()
        service = RAGService(embedding_model=_EmbeddingModel(), vector_store=vector_store, ollama_client=None)

        response = service.answer_structured("tom tat bai nay", current_context=_context())

        self.assertEqual("followup_article_summary", response["intent"])
        self.assertIn("chọn nguồn", response["answer"])
        self.assertEqual([], vector_store.get_chunks_calls)
        self.assertEqual(0, vector_store.search_calls)

    def test_simplify_uses_previous_answer_without_search(self) -> None:
        vector_store = _VectorStore()
        service = RAGService(embedding_model=_EmbeddingModel(), vector_store=vector_store, ollama_client=None)

        response = service.answer_structured("tom tat ngan hon", current_context=_context())

        self.assertEqual("followup_simplify", response["intent"])
        self.assertIn("Cau tra loi truoc", response["answer"])
        self.assertEqual([], response["query_plan"]["entities"])
        self.assertEqual([], response["query_plan"]["stock_symbols"])
        self.assertEqual(0, vector_store.search_calls)


if __name__ == "__main__":
    unittest.main()
