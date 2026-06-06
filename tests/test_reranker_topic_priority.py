from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.local_ai.reranker import NewsReranker


def _chunk(chunk_id: str, title: str, topic: str, published_at: str, score: float = 0.85) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": title,
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


class RerankerTopicPriorityTest(unittest.TestCase):
    def test_finance_topic_beats_newer_wrong_topic(self) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _chunk(
                "finance-old",
                "Co phieu ngan hang tang manh, thanh khoan cai thien",
                "economy_finance_stock",
                (now - timedelta(days=8)).isoformat(),
                score=0.70,
            ),
            _chunk(
                "politics-new",
                "Cong an canh bao lua dao qua Zalo va OTP",
                "politics_society",
                now.isoformat(),
                score=0.98,
            ),
        ]

        reranked = NewsReranker().rerank(
            "tong hop tin tuc tai chinh",
            results,
            query_context={
                "intent": "topic_news",
                "primary_topic": "economy_finance_stock",
                "topic_confidence": 1.0,
                "explicit_topic": True,
                "time_range": "7d",
            },
        )

        self.assertEqual("finance-old", reranked[0]["chunk_id"])
        self.assertGreater(reranked[1]["topic_penalty"], 0.8)

    def test_stock_query_prefers_stock_candidates_and_penalizes_wrong_topics(self) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _chunk(
                "stock-1",
                "VN-Index tang diem, co phieu ngan hang hut tien",
                "economy_finance_stock",
                now.isoformat(),
                score=0.75,
            ),
            _chunk(
                "stock-2",
                "Khoi ngoai mua rong bluechip, thanh khoan tang",
                "economy_finance_stock",
                (now - timedelta(days=1)).isoformat(),
                score=0.72,
            ),
            _chunk(
                "stock-3",
                "Co phieu chung khoan tang tran tren HOSE",
                "economy_finance_stock",
                (now - timedelta(days=2)).isoformat(),
                score=0.70,
            ),
            _chunk(
                "politics",
                "Nguoi dan chu y tai khoan dinh danh dien tu",
                "politics_society",
                now.isoformat(),
                score=0.99,
            ),
            _chunk(
                "tech",
                "Better List cong nghe buoc vao chang nong",
                "tech_ai_internet",
                now.isoformat(),
                score=0.97,
            ),
        ]

        reranked = NewsReranker().rerank(
            "tin chung khoan moi nhat",
            results,
            query_context={
                "intent": "latest_news",
                "primary_topic": "economy_finance_stock",
                "topic_confidence": 1.0,
                "explicit_topic": True,
                "time_range": "7d",
            },
        )

        self.assertEqual("stock-1", reranked[0]["chunk_id"])
        self.assertTrue(all(item["metadata"]["primary_topic"] == "economy_finance_stock" for item in reranked[:3]))

    def test_ai_synthesis_prefers_tech_over_newer_wrong_topics(self) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _chunk(
                "tech-ai",
                "OpenAI gioi thieu mo hinh AI moi cho doanh nghiep",
                "tech_ai_internet",
                (now - timedelta(days=3)).isoformat(),
                score=0.72,
            ),
            _chunk(
                "real-estate-new",
                "Gia chung cu Ha Noi tiep tuc tang trong thang nay",
                "real_estate",
                now.isoformat(),
                score=0.98,
            ),
            _chunk(
                "stock-new",
                "VN-Index tang diem nho co phieu ngan hang",
                "economy_finance_stock",
                now.isoformat(),
                score=0.96,
            ),
        ]

        reranked = NewsReranker().rerank(
            "tin AI gan day co gi",
            results,
            query_context={
                "intent": "topic_news",
                "answer_mode": "synthesis",
                "primary_topic": "tech_ai_internet",
                "topic_confidence": 1.0,
                "explicit_topic": True,
                "time_range": "7d",
            },
        )

        self.assertEqual("tech-ai", reranked[0]["chunk_id"])
        self.assertGreater(reranked[1]["topic_penalty"], 0.8)
        self.assertGreater(reranked[0]["weights_used"]["topic"], reranked[0]["weights_used"]["recency"])
        self.assertIn("score_breakdown", reranked[0])
        self.assertEqual(reranked[0]["final_score"], reranked[0]["score_breakdown"]["total_score"])
        self.assertIn("reason", reranked[0]["score_breakdown"])


if __name__ == "__main__":
    unittest.main()
