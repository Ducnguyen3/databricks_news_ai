from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.local_ai.rag_quality_eval import (
    QualityEvalResult,
    build_report,
    load_quality_queries,
    print_json_report,
    print_text_report,
    score_images,
    score_intent,
    score_multi_source,
    score_response,
    score_schema,
    score_sources,
    topic_matches,
)


def _response(**overrides):
    response = {
        "answer": "Day la cau tra loi co du thong tin tu cac nguon da truy hoi.",
        "intent": "latest_news",
        "topic": "tech_ai_internet",
        "query_plan": {"entities": ["HPG"], "stock_symbols": ["HPG"], "need_images": False},
        "sources": [
            {
                "article_id": "a1",
                "title": "HPG co tin moi",
                "url": "https://example.com/a1",
                "source": "cafef",
                "published_at": "2026-06-01T00:00:00+00:00",
            },
            {
                "article_id": "a2",
                "title": "Tin AI moi",
                "url": "https://example.com/a2",
                "source": "genk",
                "published_at": "2026-06-01T00:00:00+00:00",
            },
        ],
        "images": [{"article_id": "a1", "image_url": "https://example.com/a.jpg"}],
        "related_articles": [{"article_id": "a1", "title": "HPG co tin moi"}],
    }
    response.update(overrides)
    return response


class RagQualityEvalTest(unittest.TestCase):
    def test_load_quality_queries_reads_json(self) -> None:
        queries = load_quality_queries()

        self.assertGreater(len(queries), 0)
        self.assertIn("id", queries[0])
        self.assertIn("query", queries[0])

    def test_load_quality_queries_validates_id_and_query(self) -> None:
        with self.assertRaises(ValueError):
            load_quality_queries(Path("tests/fixtures/rag_quality_invalid_queries.json"))

    def test_score_schema_ok_when_required_fields_exist(self) -> None:
        status, message = score_schema(_response())

        self.assertEqual("OK", status)
        self.assertIsNone(message)

    def test_score_schema_fail_when_missing_field(self) -> None:
        response = _response()
        response.pop("sources")

        status, message = score_schema(response)

        self.assertEqual("FAIL", status)
        self.assertIn("missing fields", str(message))

    def test_score_intent_ok_and_warn(self) -> None:
        self.assertEqual(("OK", None), score_intent({"expected_intent": "latest_news"}, _response()))

        status, message = score_intent({"expected_intent": "entity_news"}, _response())

        self.assertEqual("WARN", status)
        self.assertIn("wrong_intent", str(message))

    def test_topic_matches_soft_aliases(self) -> None:
        self.assertTrue(topic_matches("world", "quoc_te_dia_chinh_tri_the_gioi"))
        self.assertTrue(topic_matches("finance", "kinh_te_tai_chinh_chung_khoan"))
        self.assertTrue(topic_matches("ai", "cong_nghe_ai_internet"))

    def test_score_sources_warn_when_sources_empty(self) -> None:
        status, message = score_sources({"min_sources": 1}, _response(sources=[]))

        self.assertEqual("WARN", status)
        self.assertIn("empty_sources", str(message))

    def test_score_multi_source_warn_for_one_source(self) -> None:
        response = _response(sources=[{"source": "cafef"}])

        status, message = score_multi_source({"requires_multi_source": True}, response)

        self.assertEqual("WARN", status)
        self.assertIn("multi_source_weak", str(message))

    def test_score_images_warn_when_required_but_empty(self) -> None:
        status, message = score_images({"requires_images": True, "min_images": 1}, _response(images=[]))

        self.assertEqual("WARN", status)
        self.assertIn("images_missing", str(message))

    def test_score_response_builds_warn_for_missing_images(self) -> None:
        result = score_response(
            {"id": "ukraine_images", "query": "anh Ukraine", "requires_images": True, "min_images": 1},
            _response(images=[]),
        )

        self.assertEqual("WARN", result["status"])
        self.assertEqual("WARN", result["scores"]["images"])

    def test_build_summary_counts_statuses(self) -> None:
        results = [
            QualityEvalResult(id="ok", query="q1", status="OK", expected={}, scores={}, actual={}),
            QualityEvalResult(
                id="warn",
                query="q2",
                status="WARN",
                expected={},
                scores={},
                actual={},
                warnings=["multi_source_weak: one source"],
            ),
            QualityEvalResult(id="fail", query="q3", status="FAIL", expected={}, scores={}, actual={}, errors=["runtime"]),
        ]

        report = build_report(results, duration_seconds=1.2)

        self.assertEqual(3, report["summary"]["total"])
        self.assertEqual(1, report["summary"]["ok"])
        self.assertEqual(1, report["summary"]["warn"])
        self.assertEqual(1, report["summary"]["fail"])
        self.assertEqual(1, report["issues"]["multi_source_weak"])

    def test_json_report_is_parseable(self) -> None:
        report = build_report(
            [QualityEvalResult(id="ok", query="q1", status="OK", expected={}, scores={}, actual={})],
            duration_seconds=0.1,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_json_report(report)

        parsed = json.loads(output.getvalue())
        self.assertEqual(1, parsed["summary"]["total"])

    def test_text_report_contains_required_sections(self) -> None:
        report = build_report(
            [
                QualityEvalResult(
                    id="ok",
                    query="tin AI moi nhat",
                    status="OK",
                    expected={"expected_intent": "latest_news"},
                    scores={"schema": "OK"},
                    actual={"intent": "latest_news", "topic": "tech_ai_internet"},
                )
            ],
            duration_seconds=0.1,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_text_report(report)

        text = output.getvalue()
        self.assertIn("RAG QUALITY EVALUATION", text)
        self.assertIn("Summary", text)
        self.assertIn("Common issues", text)


if __name__ == "__main__":
    unittest.main()
