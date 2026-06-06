from __future__ import annotations

import unittest

from app.local_ai.vector_store import _search_results


class VectorStorePaddingFilterTest(unittest.TestCase):
    def test_search_results_filters_system_flush_padding(self) -> None:
        response = {
            "ids": [["padding", "real"]],
            "documents": [["system flush padding document", "real document"]],
            "metadatas": [
                [
                    {"is_system_flush": True, "is_padding": True, "chunk_id": "padding"},
                    {"article_id": "article-1", "chunk_id": "real"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        results = _search_results(response)

        self.assertEqual(1, len(results))
        self.assertEqual("real", results[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
