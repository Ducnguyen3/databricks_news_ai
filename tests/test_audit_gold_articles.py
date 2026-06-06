from __future__ import annotations

import unittest

from scripts.audit_gold_articles import REQUIRED_FIELDS, build_query, validate_article


class AuditGoldArticlesTest(unittest.TestCase):
    def test_build_query_aliases_gold_columns_to_required_fields(self) -> None:
        query = build_query(table="main.news_ai.articles_clean", limit=10)

        self.assertIn("content AS cleaned_content", query)
        self.assertIn("source AS source_name", query)
        self.assertIn("primary_topic AS topic", query)
        self.assertIn("FROM main.news_ai.articles_clean", query)
        self.assertIn("LIMIT 10", query)

    def test_validate_article_requires_all_fields(self) -> None:
        article = {field: "ok" for field in REQUIRED_FIELDS}
        self.assertEqual([], validate_article(article, seen_article_ids=set()))

    def test_validate_article_reports_empty_required_fields(self) -> None:
        article = {field: "ok" for field in REQUIRED_FIELDS}
        article["cleaned_content"] = "  "
        article["published_at"] = None

        reasons = validate_article(article, seen_article_ids=set())

        self.assertIn("empty_field:cleaned_content", reasons)
        self.assertIn("empty_field:published_at", reasons)

    def test_validate_article_reports_duplicate_article_id(self) -> None:
        seen = {"article-1"}
        article = {field: "ok" for field in REQUIRED_FIELDS}
        article["article_id"] = "article-1"

        self.assertIn("duplicate_article_id", validate_article(article, seen_article_ids=seen))


if __name__ == "__main__":
    unittest.main()
