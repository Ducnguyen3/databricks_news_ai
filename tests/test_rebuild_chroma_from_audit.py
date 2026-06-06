from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.rebuild_chroma_from_audit import audit_row_to_article, load_audit_articles, padding_needed_for_flush


class RebuildChromaFromAuditTest(unittest.TestCase):
    def test_padding_needed_for_tail_flush(self) -> None:
        self.assertEqual(17, padding_needed_for_flush(8431, 64))
        self.assertEqual(0, padding_needed_for_flush(8448, 64))

    def test_audit_row_maps_to_internal_article_schema(self) -> None:
        article = audit_row_to_article(
            {
                "article_id": "a1",
                "title": "Title",
                "cleaned_content": "Content",
                "source_name": "cafef",
                "topic": "economy_finance_stock",
                "published_at": "2026-06-05T00:00:00",
            }
        )

        self.assertEqual("Content", article["content"])
        self.assertEqual("cafef", article["source"])
        self.assertEqual("economy_finance_stock", article["primary_topic"])
        self.assertTrue(article["content_hash"])

    def test_load_audit_articles_supports_topic_filter(self) -> None:
        rows = [
            {
                "article_id": "a1",
                "title": "Stock",
                "cleaned_content": "Stock content",
                "source_name": "cafef",
                "topic": "economy_finance_stock",
                "published_at": "2026-06-05T00:00:00",
            },
            {
                "article_id": "a2",
                "title": "AI",
                "cleaned_content": "AI content",
                "source_name": "genk",
                "topic": "tech_ai_internet",
                "published_at": "2026-06-05T00:00:00",
            },
        ]
        tmpdir = Path("data/test_tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / "valid_articles_for_rebuild_test.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        articles = list(load_audit_articles(path, topic="tech_ai_internet"))

        self.assertEqual(1, len(articles))
        self.assertEqual("a2", articles[0]["article_id"])


if __name__ == "__main__":
    unittest.main()
