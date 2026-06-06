from __future__ import annotations

import unittest
from dataclasses import replace

from app.config import load_settings
from app.local_ai.prompt_builder import NO_INFO_FALLBACK
from app.local_ai.rag_service import RAGService


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _VectorStore:
    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        return self._chunks[:top_k]

    def get_chunks_by_article_id(self, article_id: str, limit: int | None = None) -> list[dict]:
        chunks = [chunk for chunk in self._chunks if chunk["metadata"]["article_id"] == article_id]
        return chunks[:limit] if limit else chunks


class _EvasiveOllama:
    def generate(self, prompt: str) -> str:
        return "Câu hỏi không được chỉ rõ trong đoạn văn bạn cung cấp. Vui lòng cung cấp thêm thông tin."

    def runtime_config(self) -> dict[str, str]:
        return {"model": "test-evasive"}


def _chunk(article_id: str, title: str, text: str, score: float = 0.8) -> dict:
    return {
        "chunk_id": f"{article_id}:0",
        "text": text,
        "score": score,
        "metadata": {
            "chunk_id": f"{article_id}:0",
            "article_id": article_id,
            "title": title,
            "source": "cafef",
            "url": f"https://example.com/{article_id}",
            "published_at": "2026-06-01T00:00:00+00:00",
            "primary_topic": "economy_finance_stock",
            "topic": "economy_finance_stock",
            "entity_names": "BSR,DST,VN-Index",
            "chunk_index": 0,
        },
    }


def _ai_chunk(article_id: str, title: str, text: str, score: float = 0.8) -> dict:
    chunk = _chunk(article_id, title, text, score=score)
    chunk["metadata"].update(
        {
            "source": "genk",
            "primary_topic": "tech_ai_internet",
            "topic": "tech_ai_internet",
            "entity_names": "AI,OpenAI",
        }
    )
    return chunk


class RagNoAnswerGuardTest(unittest.TestCase):
    def test_stock_evidence_context_does_not_return_no_answer(self) -> None:
        settings = replace(
            load_settings().local_ai,
            rag_retrieval_mode="vector",
            rag_broad_retrieve_top_n=4,
            rag_top_k=3,
            rag_min_score=0.35,
            rag_max_chunks_per_article=1,
        )
        chunks = [
            _chunk(
                "stock-1",
                "Doanh nghiep trang doanh thu nhieu quy, co phieu van tim tran 8 phien lien tiep",
                "Co phieu van tim tran 8 phien lien tiep va duoc nha dau tu chu y.",
            ),
            _chunk(
                "stock-2",
                "BSR duoc danh gia con nhieu du dia tang truong",
                "BSR duoc nhac den voi nhieu du dia tang truong trong cac bai phan tich.",
                score=0.76,
            ),
            _chunk(
                "stock-3",
                "Co phieu tang gia ap dao, VN-Index van giam 7 phien lien tiep",
                "Nhieu co phieu tang gia ap dao trong khi VN-Index van giam.",
                score=0.74,
            ),
        ]
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore(chunks),
            ollama_client=None,
            settings=settings,
        )

        response = service.answer_structured(
            "co phieu nao dang tang",
            top_k=3,
            debug_retrieval=True,
            debug_prompt=True,
        )

        self.assertNotEqual(NO_INFO_FALLBACK, response["answer"])
        self.assertNotIn("chua co thong tin du", str(response["answer"]).lower())
        self.assertTrue(response["sources"])
        self.assertIn("debug_trace", response["query_plan"])
        generation = response["query_plan"]["debug_trace"]["generation"]
        self.assertFalse(generation["no_answer_triggered"])
        self.assertTrue(generation["extractive_answer_used"])

    def test_evasive_llm_answer_with_sources_falls_back_to_extractive_answer(self) -> None:
        settings = replace(
            load_settings().local_ai,
            rag_retrieval_mode="vector",
            rag_broad_retrieve_top_n=4,
            rag_top_k=3,
            rag_min_score=0.35,
            rag_max_chunks_per_article=1,
        )
        chunks = [
            _ai_chunk(
                "ai-1",
                "OpenAI va cac cong ty cong nghe day manh ung dung AI",
                "OpenAI va cac doanh nghiep cong nghe dang day manh ung dung AI trong san pham moi.",
            )
        ]
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore(chunks),
            ollama_client=_EvasiveOllama(),
            settings=settings,
        )

        response = service.answer_structured("Tin AI gan day co gi noi bat?", top_k=3, debug_retrieval=True)

        self.assertNotIn("Câu hỏi không được chỉ rõ", str(response["answer"]))
        self.assertIn("OpenAI", str(response["answer"]))
        generation = response["query_plan"]["debug_trace"]["generation"]
        self.assertTrue(generation["extractive_answer_used"])
        self.assertEqual("llm_evasive_despite_retrieved_sources", generation["answer_validator_reason"])


if __name__ == "__main__":
    unittest.main()
