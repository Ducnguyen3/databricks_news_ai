from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from app.local_ai.rag_smoke_test import (
    QuerySmokeResult,
    get_default_queries,
    parse_args,
    print_json_report,
    print_text_report,
    validate_response,
)


def _response(**overrides):
    response = {
        "answer": "Tra loi",
        "intent": "topic_news",
        "topic": "tech_ai_internet",
        "query_plan": {"need_images": False},
        "sources": [{"title": "Title", "url": "https://example.com", "source": "genk", "published_at": "2026-01-01"}],
        "images": [],
        "related_articles": [{"title": "Related"}],
    }
    response.update(overrides)
    return response


class RagSmokeTestCliTest(unittest.TestCase):
    def test_default_queries_not_empty(self) -> None:
        self.assertGreater(len(get_default_queries()), 0)

    def test_parse_args_with_one_query(self) -> None:
        args = parse_args(["--query", "HPG co gi moi"])

        self.assertEqual(["HPG co gi moi"], args.query)

    def test_parse_args_with_multiple_queries(self) -> None:
        args = parse_args(["--query", "HPG co gi moi", "--query", "tin AI moi nhat"])

        self.assertEqual(["HPG co gi moi", "tin AI moi nhat"], args.query)

    def test_validate_response_ok_when_schema_is_complete(self) -> None:
        validation = validate_response(_response())

        self.assertEqual("OK", validation["status"])

    def test_validate_response_fail_when_missing_field(self) -> None:
        response = _response()
        response.pop("sources")

        validation = validate_response(response)

        self.assertEqual("FAIL", validation["status"])
        self.assertIn("missing fields", validation["errors"][0])

    def test_validate_response_fail_when_list_fields_have_wrong_type(self) -> None:
        validation = validate_response(_response(sources={}))

        self.assertEqual("FAIL", validation["status"])
        self.assertIn("sources must be list", validation["errors"][0])

    def test_validate_response_warn_when_sources_empty(self) -> None:
        validation = validate_response(_response(sources=[]))

        self.assertEqual("WARN", validation["status"])
        self.assertIn("sources is empty", validation["warnings"])

    def test_strict_mode_fails_on_warning(self) -> None:
        validation = validate_response(_response(sources=[]), strict=True)

        self.assertEqual("FAIL", validation["status"])

    def test_print_text_report_contains_required_lines(self) -> None:
        result = QuerySmokeResult(
            query="HPG co gi moi",
            status="OK",
            duration_seconds=0.1,
            response=_response(),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_text_report([result])

        text = output.getvalue()
        self.assertIn("RAG STRUCTURED SMOKE TEST", text)
        self.assertIn("Query:", text)
        self.assertIn("Intent:", text)
        self.assertIn("Topic:", text)
        self.assertIn("Sources:", text)
        self.assertIn("Images:", text)
        self.assertIn("Related articles:", text)
        self.assertIn("Summary:", text)

    def test_json_output_is_parseable(self) -> None:
        result = QuerySmokeResult(
            query="HPG co gi moi",
            status="OK",
            duration_seconds=0.1,
            response=_response(),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_json_report([result])

        parsed = json.loads(output.getvalue())
        self.assertEqual("HPG co gi moi", parsed[0]["query"])
        self.assertEqual("OK", parsed[0]["status"])


if __name__ == "__main__":
    unittest.main()
