from __future__ import annotations

import unittest

from app.local_ai.hybrid_search import HybridSearchEngine, diversify_hybrid_results, merge_candidates, weights_for_query


def _candidate(chunk_id: str, score: float = 0.5, **metadata):
    base = {
        "article_id": chunk_id.split(":")[0],
        "source": "cafef",
        "title": "HPG co tin moi",
        "primary_topic": "economy_finance_stock",
        "has_images": False,
    }
    base.update(metadata)
    return {"chunk_id": chunk_id, "text": "HPG co phieu tang", "score": score, "metadata": base}


def _bm25(chunk_id: str, score: float = 2.0, **metadata):
    item = _candidate(chunk_id, **metadata)
    item["bm25_score"] = score
    return item


class HybridSearchTest(unittest.TestCase):
    def test_merge_vector_and_bm25_by_chunk_id(self) -> None:
        results = merge_candidates([_candidate("a1:0", 0.8)], [_bm25("a2:0", 2.0)], "HPG", {})

        self.assertEqual(2, len(results))

    def test_overlapping_chunk_has_both_sources(self) -> None:
        results = merge_candidates([_candidate("a1:0", 0.8)], [_bm25("a1:0", 2.0)], "HPG", {})

        self.assertIn("vector", results[0]["retrieval_sources"])
        self.assertIn("bm25", results[0]["retrieval_sources"])

    def test_normalize_score_handles_equal_scores(self) -> None:
        results = merge_candidates([_candidate("a1:0", 0.5), _candidate("a2:0", 0.5)], [], "HPG", {})

        self.assertTrue(all("hybrid_score" in result for result in results))

    def test_requires_lexical_increases_bm25_weight(self) -> None:
        vector_weight, bm25_weight = weights_for_query("HPG co gi moi", {"requires_lexical": True})

        self.assertGreater(bm25_weight, vector_weight)

    def test_entity_exact_match_gets_bonus(self) -> None:
        results = merge_candidates(
            [_candidate("a1:0", 0.2, entity_names="HPG")],
            [_bm25("a1:0", 2.0, entity_names="HPG")],
            "HPG co gi moi",
            {"requires_lexical": True, "lexical_terms": ["HPG"]},
        )

        self.assertGreater(results[0]["hybrid_score"], 0.5)

    def test_source_match_gets_bonus(self) -> None:
        results = merge_candidates(
            [_candidate("a1:0", 0.2, source="cafef")],
            [],
            "tin tu CafeF",
            {"preferred_sources": ["cafef"]},
        )

        self.assertGreater(results[0]["hybrid_score"], 0.0)

    def test_image_requirement_boosts_has_images(self) -> None:
        with_image = merge_candidates([_candidate("a1:0", 0.2, has_images=True)], [], "anh Ukraine", {"need_images": True})
        without_image = merge_candidates([_candidate("a1:0", 0.2, has_images=False)], [], "anh Ukraine", {"need_images": True})

        self.assertGreater(with_image[0]["hybrid_score"], without_image[0]["hybrid_score"])

    def test_diversification_limits_chunks_per_article(self) -> None:
        results = [_candidate("a1:0"), _candidate("a1:1"), _candidate("a1:2"), _candidate("a2:0")]

        diversified = diversify_hybrid_results(results, top_k=4, max_chunks_per_article=2)

        self.assertEqual(3, len(diversified))
        self.assertEqual(2, sum(1 for item in diversified if item["metadata"]["article_id"] == "a1"))

    def test_diversification_keeps_multiple_sources(self) -> None:
        results = [
            _candidate("a1:0", source="cafef"),
            _candidate("a2:0", source="cafef"),
            _candidate("a3:0", source="cafef"),
            _candidate("a4:0", source="genk"),
        ]

        diversified = diversify_hybrid_results(results, top_k=4, max_articles_per_source=2, requires_multi_source=True)

        self.assertIn("genk", {item["metadata"]["source"] for item in diversified})

    def test_result_has_required_fields(self) -> None:
        result = merge_candidates([_candidate("a1:0", 0.8)], [_bm25("a1:0", 2.0)], "HPG", {})[0]

        for field in ("chunk_id", "document", "metadata", "vector_score", "bm25_score", "hybrid_score", "retrieval_sources"):
            self.assertIn(field, result)

    def test_hybrid_engine_rebuilds_bm25_when_vector_store_count_changes(self) -> None:
        vector_store = _MutableVectorStore([_plain_chunk("a1:0", "tin khac", title="Tin khac")])
        engine = HybridSearchEngine.from_vector_store(vector_store, _EmbeddingModel())
        vector_store.chunks.append(_plain_chunk("a2:0", "HPG co phieu tang", title="HPG co tin moi", entity_names="HPG"))

        results = engine.search(
            "HPG co gi moi",
            {
                "requires_lexical": True,
                "lexical_terms": ["HPG"],
                "stock_symbols": ["HPG"],
                "primary_topic": "economy_finance_stock",
            },
            top_n=5,
        )

        self.assertIn("a2:0", [result["chunk_id"] for result in results])
        self.assertIn("bm25", next(result for result in results if result["chunk_id"] == "a2:0")["retrieval_sources"])


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class _MutableVectorStore:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.chunks = chunks

    def get_all_chunks(self):
        return list(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def search(self, query_embedding: list[float], top_k: int):
        return self.chunks[:1]


def _plain_chunk(chunk_id: str, text: str, **metadata):
    base = {
        "article_id": chunk_id.split(":")[0],
        "source": "cafef",
        "title": "",
        "primary_topic": "economy_finance_stock",
        "has_images": False,
    }
    base.update(metadata)
    return {"chunk_id": chunk_id, "text": text, "score": 0.2, "metadata": base}


if __name__ == "__main__":
    unittest.main()
