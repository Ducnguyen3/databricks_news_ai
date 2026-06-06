from __future__ import annotations

import unittest
from dataclasses import replace

from app.config import load_settings
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


class _ContaminatedOllama:
    def generate(self, prompt: str) -> str:
        return (
            "Tomcat sessionTrackingMode trong catalina.properties co the anh huong den web application. "
            "Cau truc tra loi nhu sau: Expected stock summary."
        )

    def runtime_config(self) -> dict[str, str]:
        return {"model": "test-contaminated"}


def _stock_chunk(article_id: str, text: str, score: float = 0.82) -> dict:
    return {
        "chunk_id": f"{article_id}:0",
        "text": text,
        "score": score,
        "metadata": {
            "chunk_id": f"{article_id}:0",
            "article_id": article_id,
            "title": "VN-Index va thanh khoan thi truong chung khoan",
            "source": "cafef",
            "url": f"https://example.com/{article_id}",
            "published_at": "2026-06-05T00:00:00+00:00",
            "primary_topic": "economy_finance_stock",
            "topic": "economy_finance_stock",
            "entity_names": "VN-Index,HoSE",
            "chunk_index": 0,
        },
    }


def _tech_noise_chunk() -> dict:
    chunk = _stock_chunk("noise", "Tomcat sessionTrackingMode trong catalina.properties.", score=0.99)
    chunk["metadata"].update(
        {
            "primary_topic": "tech_ai_internet",
            "topic": "tech_ai_internet",
            "title": "Tomcat sessionTrackingMode",
            "source": "genk",
            "entity_names": "Tomcat",
        }
    )
    return chunk


class RagTopicContaminationRegressionTest(unittest.TestCase):
    def test_stock_weekly_summary_must_not_contain_tomcat(self) -> None:
        settings = replace(
            load_settings().local_ai,
            rag_retrieval_mode="vector",
            rag_broad_retrieve_top_n=6,
            rag_top_k=4,
            rag_min_score=0.35,
            rag_max_chunks_per_article=1,
        )
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore(
                [
                    _tech_noise_chunk(),
                    _stock_chunk("stock-1", "VN-Index giam diem, thanh khoan tren HoSE thu hep, khoi ngoai ban rong."),
                    _stock_chunk("stock-2", "Co phieu ngan hang phan hoa, mot so ma chung khoan tang tot.", score=0.8),
                ]
            ),
            ollama_client=_ContaminatedOllama(),
            settings=settings,
        )

        result = service.answer_structured("Tong hop tin tuc chung khoan tuan nay", top_k=3, debug_retrieval=True)
        answer = str(result["answer"]).lower()

        self.assertNotIn("tomcat", answer)
        self.assertNotIn("sessiontrackingmode", answer)
        self.assertNotIn("catalina.properties", answer)
        self.assertNotIn("cau truc tra loi", answer)
        self.assertNotIn("neu du lieu hien co", answer)
        self.assertTrue(answer.strip())


if __name__ == "__main__":
    unittest.main()
