from __future__ import annotations

import unittest

from app.local_ai.query_router import route_query
from app.local_ai.retriever import MetadataFilteringRetriever, diversify_results, filter_results_by_metadata
from app.local_ai.topic_guard import filter_context_for_topic


def _result(
    article_id: str,
    topic: str,
    entity_names: str = "",
    has_images: bool = False,
    published_at: str = "2026-01-01T00:00:00+00:00",
    title: str = "",
):
    return {
        "chunk_id": f"{article_id}:0",
        "text": "content",
        "score": 0.8,
        "metadata": {
            "article_id": article_id,
            "title": title,
            "primary_topic": topic,
            "secondary_topics_json": "[]",
            "entity_names": entity_names,
            "entities_json": "[]",
            "published_at": published_at,
            "source": "vnexpress",
            "has_images": has_images,
        },
    }


class _FakeEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._results[:top_k]


class RagMetadataFilterTest(unittest.TestCase):
    def test_ai_topic_does_not_return_real_estate_when_ai_exists(self) -> None:
        results = [
            _result("ai-1", "tech_ai_internet"),
            _result("real-estate-1", "real_estate"),
        ]

        filtered = filter_results_by_metadata(results, {"primary_topic": "tech_ai_internet", "time_range": "all"})

        self.assertEqual(["ai-1"], [item["metadata"]["article_id"] for item in filtered])

    def test_explicit_topic_accepts_secondary_topic_match(self) -> None:
        secondary_match = _result("life-ai", "lifestyle_education_health_entertainment")
        secondary_match["metadata"]["secondary_topics_json"] = '["tech_ai_internet"]'
        results = [
            _result("ai-primary", "tech_ai_internet"),
            secondary_match,
        ]

        filtered = filter_results_by_metadata(
            results,
            {
                "primary_topic": "tech_ai_internet",
                "explicit_topic": True,
                "topic_confidence": 1.0,
                "time_range": "all",
            },
        )

        self.assertEqual(["ai-primary", "life-ai"], [item["metadata"]["article_id"] for item in filtered])

    def test_hpg_query_prefers_entity_match(self) -> None:
        results = [
            _result("hpg-1", "economy_finance_stock", entity_names="HPG,VNINDEX"),
            _result("finance-1", "economy_finance_stock", entity_names="VIC"),
        ]

        filtered = filter_results_by_metadata(
            results,
            {"primary_topic": "economy_finance_stock", "entities": ["HPG"], "stock_symbols": ["HPG"], "time_range": "all"},
        )

        self.assertEqual(["hpg-1"], [item["metadata"]["article_id"] for item in filtered])

    def test_hpg_query_matches_title_when_entity_metadata_missing(self) -> None:
        results = [
            _result("hpg-title", "economy_finance_stock", title="HPG co tin moi ve giao dich co phieu"),
            _result("finance-1", "economy_finance_stock", title="Thi truong chung khoan tang diem"),
        ]

        filtered = filter_results_by_metadata(
            results,
            {"primary_topic": "economy_finance_stock", "entities": ["HPG"], "stock_symbols": ["HPG"], "time_range": "all"},
        )

        self.assertEqual(["hpg-title"], [item["metadata"]["article_id"] for item in filtered])

    def test_fpt_query_matches_title_when_entity_metadata_missing(self) -> None:
        results = [
            _result("fpt-title", "economy_finance_stock", title="FPT cong bo ke hoach kinh doanh moi"),
            _result("finance-1", "economy_finance_stock", title="Thi truong chung khoan tang diem"),
        ]

        filtered = filter_results_by_metadata(
            results,
            {"primary_topic": "economy_finance_stock", "entities": ["FPT"], "stock_symbols": ["FPT"], "time_range": "all"},
        )

        self.assertEqual(["fpt-title"], [item["metadata"]["article_id"] for item in filtered])

    def test_stock_overview_without_symbol_does_not_require_entity_match(self) -> None:
        plan = route_query("gia co phieu nao dang giam")
        self.assertEqual("economy_finance_stock", plan["primary_topic"])
        self.assertEqual([], plan["stock_symbols"])
        results = [
            _result("stock-market", "economy_finance_stock", title="Co phieu ngan hang giam, VN-Index mat diem"),
            _result("tech-1", "tech_ai_internet", title="Cong nghe AI moi"),
        ]

        filtered = filter_results_by_metadata(results, plan)

        self.assertEqual(["stock-market"], [item["metadata"]["article_id"] for item in filtered])

    def test_stock_topic_guard_keeps_exchange_index_context(self) -> None:
        chunks = [
            _result("vnindex", "business_startup", title="VN-Index tang nho thanh khoan tren HOSE cai thien"),
        ]

        guarded = filter_context_for_topic("VN-Index va HOSE co gi moi", "economy_finance_stock", chunks)

        self.assertEqual(["vnindex"], [item["metadata"]["article_id"] for item in guarded.kept])
        self.assertEqual(0, guarded.dropped_wrong_topic)

    def test_retriever_falls_back_when_date_filter_removes_entity_match(self) -> None:
        results = [
            _result(
                "hpg-old",
                "economy_finance_stock",
                title="HPG co tin moi ve loi nhuan",
                published_at="2026-01-01T00:00:00+00:00",
            ),
        ]
        retriever = MetadataFilteringRetriever(_FakeVectorStore(results), _FakeEmbeddingModel(), retrieval_mode="vector")
        query_plan = {
            "primary_topic": "economy_finance_stock",
            "explicit_topic": True,
            "topic_confidence": 1.0,
            "entities": ["HPG"],
            "stock_symbols": ["HPG"],
            "time_range": "today",
            "debug_scores": True,
        }

        retrieved = retriever.retrieve("HPG co gi moi", query_plan, top_n=5, top_k=3)

        self.assertEqual(["hpg-old"], [item["metadata"]["article_id"] for item in retrieved])
        debug = query_plan["_retrieval_debug"]
        self.assertTrue(debug["fallback_used"])
        self.assertEqual("metadata_filter_removed_all_candidates", debug["fallback_reason"])
        self.assertEqual("relaxed_entity_filter", debug["fallback_strategy"])

    def test_vnindex_query_matches_hyphenated_entity_metadata(self) -> None:
        results = [
            _result("vnindex-1", "economy_finance_stock", entity_names="VN-Index,HoSE"),
            _result("finance-1", "economy_finance_stock", entity_names="HNX"),
        ]

        filtered = filter_results_by_metadata(
            results,
            {"primary_topic": "economy_finance_stock", "entities": ["VNINDEX"], "stock_symbols": ["VNINDEX"], "time_range": "all"},
        )

        self.assertEqual(["vnindex-1"], [item["metadata"]["article_id"] for item in filtered])

    def test_need_images_filters_to_chunks_with_images(self) -> None:
        results = [
            _result("image-1", "world_geopolitics", entity_names="Ukraine", has_images=True),
            _result("no-image-1", "world_geopolitics", entity_names="Ukraine", has_images=False),
        ]

        filtered = filter_results_by_metadata(
            results,
            {"primary_topic": "world_geopolitics", "entities": ["Ukraine"], "need_images": True, "time_range": "all"},
        )

        self.assertEqual(["image-1"], [item["metadata"]["article_id"] for item in filtered])

    def test_exact_date_filter_keeps_only_requested_day(self) -> None:
        results = [
            _result("june-1", "tech_ai_internet", published_at="2026-06-01T09:00:00+07:00"),
            _result("june-2", "tech_ai_internet", published_at="2026-06-02T00:00:00+07:00"),
            _result("june-3", "tech_ai_internet", published_at="2026-06-03"),
        ]

        filtered = filter_results_by_metadata(
            results,
            {
                "primary_topic": "tech_ai_internet",
                "time_range": "date",
                "date_filter": {
                    "type": "exact_date",
                    "exact_date": "2026-06-01",
                    "start_date": "2026-06-01",
                    "end_date": "2026-06-02",
                },
            },
        )

        self.assertEqual(["june-1"], [item["metadata"]["article_id"] for item in filtered])

    def test_diversify_results_limits_single_source_dominance(self) -> None:
        results = [
            _result("vn-1", "tech_ai_internet"),
            _result("vn-2", "tech_ai_internet"),
            _result("vn-3", "tech_ai_internet"),
            _result("genk-1", "tech_ai_internet"),
        ]
        for index, result in enumerate(results):
            result["score"] = 1.0 - index * 0.01
            result["metadata"]["source"] = "vnexpress" if result["metadata"]["article_id"].startswith("vn") else "genk"

        diversified = diversify_results(results, top_k=4)
        sources = [item["metadata"]["source"] for item in diversified]

        self.assertLessEqual(sources.count("vnexpress"), 2)
        self.assertIn("genk", sources)

    def test_retriever_does_not_fallback_outside_selected_source(self) -> None:
        results = [
            _result("vn-1", "tech_ai_internet"),
            _result("genk-1", "tech_ai_internet"),
        ]
        results[0]["metadata"]["source"] = "vnexpress"
        results[1]["metadata"]["source"] = "genk"
        retriever = MetadataFilteringRetriever(_FakeVectorStore(results), _FakeEmbeddingModel(), retrieval_mode="vector")

        retrieved = retriever.retrieve(
            "tin cong nghe",
            {"primary_topic": "tech_ai_internet", "preferred_sources": ["cafef"], "time_range": "all"},
            top_n=10,
            top_k=3,
        )

        self.assertEqual([], retrieved)

    def test_retriever_keeps_only_selected_source(self) -> None:
        results = [
            _result("vn-1", "tech_ai_internet"),
            _result("genk-1", "tech_ai_internet"),
        ]
        results[0]["metadata"]["source"] = "vnexpress"
        results[1]["metadata"]["source"] = "genk"
        retriever = MetadataFilteringRetriever(_FakeVectorStore(results), _FakeEmbeddingModel(), retrieval_mode="vector")

        retrieved = retriever.retrieve(
            "tin cong nghe",
            {"primary_topic": "tech_ai_internet", "preferred_sources": ["genk"], "time_range": "all"},
            top_n=10,
            top_k=3,
        )

        self.assertEqual(["genk"], [item["metadata"]["source"] for item in retrieved])


if __name__ == "__main__":
    unittest.main()
