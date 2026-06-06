from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.local_ai.reranker import NewsReranker, compute_recency_score


def _chunk(chunk_id: str, text: str, score: float, topic: str, published_at: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "metadata": {
            "article_id": chunk_id,
            "primary_topic": topic,
            "entity_names": "",
            "source": "vnexpress",
            "published_at": published_at,
        },
    }


class RerankerRecencyTest(unittest.TestCase):
    def test_recency_score_decays_and_bad_date_does_not_crash(self) -> None:
        now = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
        today = compute_recency_score(now.isoformat(), now=now, half_life_days=7.0)
        seven_days_old = compute_recency_score((now - timedelta(days=7)).isoformat(), now=now, half_life_days=7.0)

        self.assertGreater(today, seven_days_old)
        self.assertGreater(today, 0.95)
        self.assertEqual(0.0, compute_recency_score("not-a-date", now=now))

    def test_latest_news_prefers_relevant_recent_article_over_old_or_wrong_topic(self) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _chunk(
                "old-relevant",
                "OpenAI cong bo tinh nang AI moi cho doanh nghiep.",
                0.95,
                "tech_ai_internet",
                (now - timedelta(days=10)).isoformat(),
            ),
            _chunk(
                "recent-relevant",
                "Cac cong ty AI ra mat san pham moi trong hom nay.",
                0.78,
                "tech_ai_internet",
                now.isoformat(),
            ),
            _chunk(
                "recent-wrong-topic",
                "Gia chung cu va bat dong san tang trong hom nay.",
                0.92,
                "real_estate",
                now.isoformat(),
            ),
        ]

        reranked = NewsReranker().rerank(
            "tin AI moi nhat hom nay",
            results,
            query_context={"intent": "latest_news", "primary_topic": "tech_ai_internet", "time_range": "today"},
        )

        self.assertEqual("recent-relevant", reranked[0]["chunk_id"])
        self.assertNotEqual("recent-wrong-topic", reranked[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
