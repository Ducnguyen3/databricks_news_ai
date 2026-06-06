from __future__ import annotations

import unittest

from app.local_ai.bm25_index import BM25ChunkIndex, tokenize


def _chunk(chunk_id: str, text: str, **metadata):
    base = {
        "article_id": chunk_id,
        "source": "cafef",
        "title": "",
        "primary_topic": "economy_finance_stock",
        "primary_topic_name": "Finance",
        "entities_json": "[]",
    }
    base.update(metadata)
    return {"chunk_id": chunk_id, "text": text, "metadata": base}


class BM25IndexTest(unittest.TestCase):
    def test_tokenize_vietnamese_basic(self) -> None:
        tokens = tokenize("lãi suất ngân hàng")

        self.assertIn("lai", tokens)
        self.assertIn("suat", tokens)
        self.assertIn("ngan", tokens)

    def test_tokenize_keeps_uppercase_ticker(self) -> None:
        tokens = tokenize("HPG có gì mới")

        self.assertIn("HPG", tokens)
        self.assertIn("hpg", tokens)

    def test_search_matches_exact_keyword(self) -> None:
        index = BM25ChunkIndex([_chunk("c1", "Co phieu HPG tang manh"), _chunk("c2", "Tin bat dong san")])

        results = index.search("HPG", top_k=5)

        self.assertEqual("c1", results[0]["chunk_id"])
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_scores_are_descending(self) -> None:
        index = BM25ChunkIndex([_chunk("c1", "HPG HPG HPG"), _chunk("c2", "HPG")])

        results = index.search("HPG", top_k=5)

        self.assertGreaterEqual(results[0]["bm25_score"], results[1]["bm25_score"])

    def test_source_filter(self) -> None:
        index = BM25ChunkIndex([_chunk("c1", "HPG", source="cafef"), _chunk("c2", "HPG", source="genk")])

        results = index.search("HPG", filters={"source": "genk"})

        self.assertEqual(["genk"], [result["metadata"]["source"] for result in results])

    def test_topic_filter(self) -> None:
        index = BM25ChunkIndex([_chunk("c1", "HPG", primary_topic="finance"), _chunk("c2", "HPG", primary_topic="tech")])

        results = index.search("HPG", filters={"topic": "tech"})

        self.assertEqual(["tech"], [result["metadata"]["primary_topic"] for result in results])

    def test_empty_index_does_not_crash(self) -> None:
        self.assertEqual([], BM25ChunkIndex([]).search("HPG"))

    def test_no_match_returns_empty(self) -> None:
        index = BM25ChunkIndex([_chunk("c1", "Tin bat dong san")])

        self.assertEqual([], index.search("OpenAI"))


if __name__ == "__main__":
    unittest.main()
