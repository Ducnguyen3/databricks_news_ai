from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from app.local_ai.chunk_quality_audit import (
    audit_chunks,
    build_status,
    compute_article_grouping,
    compute_boundary_warnings,
    compute_duplicate_stats,
    compute_image_consistency,
    compute_json_validity,
    compute_length_stats,
    compute_metadata_completeness,
    parse_json_list,
    print_json_report,
    print_text_report,
)


def _chunk(chunk_id: str, text: str, **metadata):
    base_metadata = {
        "article_id": "a1",
        "source": "cafef",
        "title": "Title",
        "url": "https://example.com",
        "published_at": "2026-06-01T00:00:00+00:00",
        "primary_topic": "economy_finance_stock",
        "primary_topic_name": "Finance",
        "entities_json": "[]",
        "images_json": "[]",
        "secondary_topics_json": "[]",
        "has_images": False,
        "image_count": 0,
        "content_hash": "hash",
        "indexed_at": "2026-06-01T00:00:00+00:00",
        "embedding_model": "model",
        "chunking_version": "v1",
        "index_version": "v1",
    }
    base_metadata.update(metadata)
    return {"chunk_id": chunk_id, "text": text, "metadata": base_metadata}


class ChunkQualityAuditTest(unittest.TestCase):
    def test_compute_length_stats(self) -> None:
        chunks = [_chunk("c1", "a" * 50), _chunk("c2", "b" * 150), _chunk("c3", "c" * 3000)]

        stats = compute_length_stats(chunks, min_chars=100, max_chars=2500)

        self.assertEqual(50, stats["min_chars"])
        self.assertEqual(3000, stats["max_chars"])
        self.assertEqual(1, stats["too_short_count"])
        self.assertEqual(1, stats["too_long_count"])

    def test_metadata_completeness_detects_missing_field(self) -> None:
        chunks = [_chunk("c1", "valid text", article_id="")]

        metadata = compute_metadata_completeness(chunks)

        self.assertEqual(1, metadata["missing"]["article_id"]["missing_count"])

    def test_json_validity_parses_entities_and_images(self) -> None:
        chunks = [
            _chunk("c1", "valid text", entities_json='["HPG"]', images_json='[{"url":"x"}]'),
            _chunk("c2", "valid text", entities_json="{bad", images_json="[]"),
        ]

        validity = compute_json_validity(chunks)

        self.assertEqual(1, validity["entities_json"]["invalid_count"])
        self.assertEqual(0, validity["images_json"]["invalid_count"])
        self.assertEqual(["HPG"], parse_json_list('["HPG"]'))

    def test_image_consistency_warning(self) -> None:
        chunks = [_chunk("c1", "valid text", has_images=True, image_count=0, images_json="[]")]

        consistency = compute_image_consistency(chunks)

        self.assertEqual(1, consistency["issue_count"])
        self.assertEqual("has_images_without_count", consistency["issues"][0]["issue"])

    def test_duplicate_detection(self) -> None:
        chunks = [_chunk("c1", "same text"), _chunk("c2", "same text"), _chunk("c3", "other text")]

        duplicates = compute_duplicate_stats(chunks)

        self.assertEqual(1, duplicates["exact_duplicate_chunks"])
        self.assertGreater(duplicates["duplicate_ratio"], 0)

    def test_article_grouping_stats(self) -> None:
        chunks = [
            _chunk("c1", "text", article_id="a1"),
            _chunk("c2", "text", article_id="a1"),
            _chunk("c3", "text", article_id="a2"),
        ]

        grouping = compute_article_grouping(chunks)

        self.assertEqual(1, grouping["min_chunks_per_article"])
        self.assertEqual(2, grouping["max_chunks_per_article"])
        self.assertEqual(1, grouping["articles_with_one_chunk"])

    def test_boundary_warning_detects_cut_chunk(self) -> None:
        chunks = [_chunk("c1", "Noi dung dang bi cat va"), _chunk("c2", "nay la mot doan ngan")]

        boundary = compute_boundary_warnings(chunks, min_chars=100)

        self.assertEqual(1, boundary["possibly_cut_chunks"])
        self.assertEqual(1, boundary["starts_with_weak_pronoun"])

    def test_build_status_ok_warn_fail(self) -> None:
        ok_report = audit_chunks(
            [_chunk("c1", "a" * 300), _chunk("c2", "b" * 300, article_id="a2")],
            chroma_path="data/chroma",
            collection_name="news_articles",
        )
        self.assertEqual("OK", ok_report["status"])

        warn_status, reasons = build_status(
            {"too_short_ratio": 0.11, "too_long_ratio": 0, "too_short_count": 1, "too_long_count": 0},
            compute_metadata_completeness([_chunk("c1", "text")]),
            {"duplicate_ratio": 0},
            {"total_chunks": 10},
        )
        self.assertEqual("WARN", warn_status)
        self.assertTrue(reasons)

        fail_status, fail_reasons = build_status(
            {"too_short_ratio": 0, "too_long_ratio": 0},
            compute_metadata_completeness([]),
            {"duplicate_ratio": 0},
            {"total_chunks": 0},
        )
        self.assertEqual("FAIL", fail_status)
        self.assertTrue(fail_reasons)

    def test_json_report_is_parseable(self) -> None:
        report = audit_chunks([_chunk("c1", "a" * 300)], chroma_path="data/chroma", collection_name="news_articles")
        output = io.StringIO()

        with redirect_stdout(output):
            print_json_report(report)

        parsed = json.loads(output.getvalue())
        self.assertIn("overview", parsed)

    def test_text_report_contains_required_sections(self) -> None:
        report = audit_chunks([_chunk("c1", "a" * 300)], chroma_path="data/chroma", collection_name="news_articles")
        output = io.StringIO()

        with redirect_stdout(output):
            print_text_report(report)

        text = output.getvalue()
        self.assertIn("CHUNK QUALITY AUDIT", text)
        self.assertIn("Overview", text)
        self.assertIn("Length", text)
        self.assertIn("Metadata", text)
        self.assertIn("Duplicates", text)
        self.assertIn("Status", text)


if __name__ == "__main__":
    unittest.main()
