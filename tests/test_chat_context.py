from __future__ import annotations

import unittest

from app.local_ai.rag_service import detect_follow_up_intent, resolve_referenced_source


def _context(source_count: int = 3) -> dict:
    return {
        "previous_sources": [
            {
                "citation_id": index,
                "article_id": f"article-{index}",
                "title": f"Article {index}",
                "source": "vnexpress",
                "url": f"https://example.com/{index}",
            }
            for index in range(1, source_count + 1)
        ]
    }


class ChatContextTest(unittest.TestCase):
    def test_parse_bracket_citation(self) -> None:
        resolved = resolve_referenced_source("tom tat bai [2]", _context())

        self.assertEqual("article-2", resolved.source["article_id"])
        self.assertFalse(resolved.needs_clarification)

    def test_parse_source_number(self) -> None:
        resolved = resolve_referenced_source("nguon so 3 la gi", _context())

        self.assertEqual("article-3", resolved.source["article_id"])

    def test_parse_article_number(self) -> None:
        resolved = resolve_referenced_source("tom tat bai bao 2", _context())

        self.assertEqual("article-2", resolved.source["article_id"])

    def test_explicit_citation_does_not_win_when_question_mentions_different_article(self) -> None:
        resolved = resolve_referenced_source("Khoai Lang Thang dang vuong vao rac roi gi? tom tat ve bai bao 1", _context())

        self.assertIsNone(resolved.source)
        self.assertFalse(resolved.needs_clarification)

    def test_explicit_citation_can_match_question_terms(self) -> None:
        context = _context()
        context["previous_sources"][0]["title"] = "Khoai Lang Thang dang vuong vao rac roi gi"

        resolved = resolve_referenced_source("Khoai Lang Thang dang vuong vao rac roi gi? tom tat ve bai bao 1", context)

        self.assertEqual("article-1", resolved.source["article_id"])

    def test_selected_article_id_wins_without_explicit_citation(self) -> None:
        context = _context()
        context["selected_article_id"] = "article-2"

        resolved = resolve_referenced_source("bai nay noi gi", context)

        self.assertEqual("article-2", resolved.source["article_id"])

    def test_single_source_can_resolve_this_article(self) -> None:
        resolved = resolve_referenced_source("tom tat bai nay", _context(source_count=1))

        self.assertEqual("article-1", resolved.source["article_id"])

    def test_ambiguous_this_article_needs_clarification(self) -> None:
        resolved = resolve_referenced_source("tom tat bai nay", _context(source_count=3))

        self.assertIsNone(resolved.source)
        self.assertTrue(resolved.needs_clarification)
        self.assertIn("[1]", resolved.message)

    def test_follow_up_intents(self) -> None:
        self.assertEqual("followup_article_summary", detect_follow_up_intent("tom tat bai [2]"))
        self.assertEqual("followup_article_summary", detect_follow_up_intent("tom tat bai nay"))
        self.assertEqual("followup_simplify", detect_follow_up_intent("tom tat ngan hon"))
        self.assertEqual("followup_media_lookup", detect_follow_up_intent("cho toi xem anh cua bai do"))


if __name__ == "__main__":
    unittest.main()
