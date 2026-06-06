from __future__ import annotations

import unittest

from app.local_ai.reranker import NewsReranker


class NewsRerankerTest(unittest.TestCase):
    def test_topic_match_beats_slightly_higher_vector_score(self) -> None:
        results = [
            {
                "chunk_id": "wrong-topic",
                "text": "Giá chung cư tăng",
                "score": 0.82,
                "metadata": {
                    "article_id": "a-real-estate",
                    "primary_topic": "real_estate",
                    "entity_names": "",
                    "published_at": "2026-01-01T00:00:00+00:00",
                    "source": "vnexpress",
                },
            },
            {
                "chunk_id": "right-topic",
                "text": "OpenAI ra mắt mô hình AI mới",
                "score": 0.80,
                "metadata": {
                    "article_id": "a-ai",
                    "primary_topic": "tech_ai_internet",
                    "entity_names": "OpenAI",
                    "published_at": "2026-01-01T00:00:00+00:00",
                    "source": "vnexpress",
                },
            },
        ]

        reranked = NewsReranker().rerank(
            "tin AI mới nhất",
            results,
            query_context={"primary_topic": "tech_ai_internet"},
        )

        self.assertEqual("right-topic", reranked[0]["chunk_id"])

    def test_entity_match_boosts_matching_chunk(self) -> None:
        results = [
            {
                "chunk_id": "generic",
                "text": "Thị trường chứng khoán tăng",
                "score": 0.81,
                "metadata": {"primary_topic": "economy_finance_stock", "entity_names": "VIC", "source": "cafef"},
            },
            {
                "chunk_id": "hpg",
                "text": "HPG tăng mạnh",
                "score": 0.80,
                "metadata": {"primary_topic": "economy_finance_stock", "entity_names": "HPG", "source": "cafef"},
            },
        ]

        reranked = NewsReranker().rerank(
            "HPG có gì mới",
            results,
            query_context={"primary_topic": "economy_finance_stock", "entities": ["HPG"], "stock_symbols": ["HPG"]},
        )

        self.assertEqual("hpg", reranked[0]["chunk_id"])

    def test_stock_market_overview_prefers_index_article_over_single_stock_noise(self) -> None:
        results = [
            {
                "chunk_id": "single-stock",
                "text": "AAN giai trinh co phieu tang tran va ke hoach chia co tuc.",
                "score": 0.86,
                "metadata": {
                    "primary_topic": "economy_finance_stock",
                    "entity_names": "AAN",
                    "source": "cafef",
                    "title": "Co phieu AAN tang tran",
                    "published_at": "2026-06-05T00:00:00+00:00",
                },
            },
            {
                "chunk_id": "market-overview",
                "text": "VN-Index tang diem, thanh khoan cai thien, khoi ngoai mua rong tren HoSE va nhom nganh ngan hang dan dat.",
                "score": 0.78,
                "metadata": {
                    "primary_topic": "economy_finance_stock",
                    "entity_names": "VNINDEX,HNX,UPCOM",
                    "source": "cafef",
                    "title": "Thi truong chung khoan hom nay",
                    "published_at": "2026-06-05T00:00:00+00:00",
                },
            },
        ]

        reranked = NewsReranker().rerank(
            "Gia co phieu hom nay the nao?",
            results,
            query_context={"intent": "stock_market_overview", "primary_topic": "economy_finance_stock"},
        )

        self.assertEqual("market-overview", reranked[0]["chunk_id"])
        self.assertGreater(reranked[0]["stock_overview_score"], 0)
        self.assertGreater(reranked[1]["stock_single_name_penalty"], 0)

    def test_techcombank_primary_article_beats_transfer_noise_for_generic_entity_query(self) -> None:
        results = [
            {
                "chunk_id": "transfer-noise",
                "text": "Mot nguoi chuyen nham tien vao tai khoan Techcombank cua nguoi la.",
                "score": 0.84,
                "metadata": {
                    "primary_topic": "economy_finance_stock",
                    "entity_names": "Techcombank",
                    "source": "vnexpress",
                    "title": "Chuyen nham tien qua ngan hang",
                    "published_at": "2026-06-05T00:00:00+00:00",
                },
            },
            {
                "chunk_id": "techcombank-analysis",
                "text": "Techcombank mo rong he sinh thai khach hang, nang luc cong nghe va chien luoc ngan hang so.",
                "score": 0.78,
                "metadata": {
                    "primary_topic": "economy_finance_stock",
                    "entity_names": "Techcombank",
                    "source": "cafef",
                    "title": "Techcombank va loi the ngan hang so",
                    "published_at": "2026-06-05T00:00:00+00:00",
                },
            },
        ]

        reranked = NewsReranker().rerank(
            "Tin tuc ve ngan hang Techcombank",
            results,
            query_context={"primary_topic": "economy_finance_stock", "entities": ["Techcombank"]},
        )

        self.assertEqual("techcombank-analysis", reranked[0]["chunk_id"])
        self.assertGreater(reranked[1]["peripheral_entity_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
