from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.local_ai.query_router import route_query
from app.local_ai.reranker import NewsReranker
from app.local_ai.retriever import MetadataFilteringRetriever


def _chunk(
    chunk_id: str,
    title: str,
    topic: str,
    published_at: str,
    score: float = 0.8,
    text: str = "",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": text or title,
        "score": score,
        "metadata": {
            "article_id": chunk_id,
            "title": title,
            "primary_topic": topic,
            "source": "vnexpress",
            "published_at": published_at,
            "entity_names": "",
        },
    }


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class _VectorStore:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._chunks = chunks

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._chunks[:top_k]


class StockTopicRetrievalTest(unittest.TestCase):
    def test_stock_topic_queries_route_to_finance_topic(self) -> None:
        queries = (
            "tổng hợp tin tức chứng khoán tuần này",
            "tin chứng khoán mới nhất",
            "thị trường cổ phiếu hôm nay",
            "VN-Index và khối ngoại có gì mới",
            "thanh khoản khớp lệnh trên HOSE",
            "mã cổ phiếu tăng trần giảm sàn sắc tím",
            "tự doanh và UPCOM tuần này",
        )

        for query in queries:
            with self.subTest(query=query):
                plan = route_query(query)
                self.assertEqual("economy_finance_stock", plan["primary_topic"])
                self.assertTrue(plan["explicit_topic"])

    def test_reranker_prefers_strong_stock_keywords_over_wrong_topic(self) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _chunk("zalo", "Người dân chú ý nhắn tin qua Zalo", "politics_society", now.isoformat(), score=0.95),
            _chunk("vneid", "Công an hướng dẫn định danh điện tử VNeID", "politics_society", now.isoformat(), score=0.93),
            _chunk("stock-1", "Cổ phiếu ngân hàng tím trần, thanh khoản khớp lệnh tăng", "economy_finance_stock", (now - timedelta(days=2)).isoformat(), score=0.75),
            _chunk("stock-2", "VN-Index tăng nhờ khối ngoại mua ròng bluechip", "economy_finance_stock", (now - timedelta(days=3)).isoformat(), score=0.72),
        ]

        reranked = NewsReranker().rerank(
            "tổng hợp tin tức chứng khoán tuần này",
            results,
            query_context={
                "intent": "topic_news",
                "primary_topic": "economy_finance_stock",
                "explicit_topic": True,
                "topic_confidence": 1.0,
                "time_range": "7d",
            },
        )

        self.assertEqual("stock-1", reranked[0]["chunk_id"])
        self.assertEqual("stock-2", reranked[1]["chunk_id"])
        self.assertGreater(reranked[0]["keyword_score"], 0.0)
        self.assertGreater(reranked[2]["topic_penalty"], 0.0)

    def test_wrong_topic_not_in_top_three_when_enough_stock_results_exist(self) -> None:
        now = datetime.now(timezone.utc)
        chunks = [
            _chunk("zalo", "Người dân chú ý nhắn tin qua Zalo", "politics_society", now.isoformat(), score=0.99),
            _chunk("vneid", "Công an hướng dẫn VNeID", "politics_society", now.isoformat(), score=0.98),
            _chunk("stock-1", "Cổ phiếu tăng trần trên HOSE", "economy_finance_stock", (now - timedelta(days=1)).isoformat(), score=0.70),
            _chunk("stock-2", "VN-Index tăng điểm, thanh khoản cải thiện", "economy_finance_stock", (now - timedelta(days=2)).isoformat(), score=0.69),
            _chunk("stock-3", "Khối ngoại mua ròng bluechip", "economy_finance_stock", (now - timedelta(days=3)).isoformat(), score=0.68),
            _chunk("stock-4", "Tự doanh gom cổ phiếu ngân hàng", "economy_finance_stock", (now - timedelta(days=4)).isoformat(), score=0.67),
        ]
        retriever = MetadataFilteringRetriever(_VectorStore(chunks), _EmbeddingModel(), retrieval_mode="vector")

        results = retriever.retrieve(
            "tin chứng khoán mới nhất",
            {
                "intent": "latest_news",
                "primary_topic": "economy_finance_stock",
                "explicit_topic": True,
                "topic_confidence": 1.0,
                "time_range": "all",
            },
            top_n=6,
            top_k=3,
        )

        self.assertEqual(3, len(results))
        self.assertTrue(all(result["metadata"]["primary_topic"] == "economy_finance_stock" for result in results))

    def test_recency_cannot_beat_clear_topic_match(self) -> None:
        now = datetime.now(timezone.utc)
        reranked = NewsReranker().rerank(
            "thị trường cổ phiếu hôm nay",
            [
                _chunk("new-wrong", "Công an cảnh báo lừa đảo qua VNeID", "politics_society", now.isoformat(), score=0.99),
                _chunk("old-stock", "Cổ phiếu bluechip tăng, VN-Index giữ sắc xanh", "economy_finance_stock", (now - timedelta(days=9)).isoformat(), score=0.65),
            ],
            query_context={
                "intent": "stock_market_overview",
                "sub_intent": "stock_market_overview",
                "primary_topic": "economy_finance_stock",
                "explicit_topic": True,
                "topic_confidence": 1.0,
                "time_range": "today",
            },
        )

        self.assertEqual("old-stock", reranked[0]["chunk_id"])
        self.assertGreater(reranked[1]["topic_penalty"], reranked[0]["topic_penalty"])


if __name__ == "__main__":
    unittest.main()
