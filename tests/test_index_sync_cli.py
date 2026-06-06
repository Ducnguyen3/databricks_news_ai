from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout

from app.local_ai.index_sync import parse_args, print_index_stats, stats_as_dict
from app.local_ai.pipeline import IndexingResult


def _stats() -> IndexingResult:
    return IndexingResult(
        rebuild_mode="incremental",
        articles_loaded=100,
        articles_indexed=15,
        articles_skipped=80,
        articles_reindexed=5,
        articles_failed=0,
        chunks_created=320,
        chunks_upserted=320,
        index_size=12345,
        articles_with_images=42,
        chunks_with_images=120,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        chunking_version="semantic-recursive-v1",
        index_version="local-index-v1",
        chroma_path="data/chroma",
        collection_name="news_articles",
        gold_table="main.news_ai.articles_clean",
        duration_seconds=10.52,
    )


class IndexSyncCliTest(unittest.TestCase):
    def test_parse_args_rebuild_mode_full(self) -> None:
        args = parse_args(["--rebuild_mode", "full"])

        self.assertEqual("full", args.rebuild_mode)

    def test_parse_args_rebuild_mode_incremental(self) -> None:
        args = parse_args(
            [
                "--rebuild_mode",
                "incremental",
                "--limit",
                "100",
                "--source",
                "cafef",
                "--since_days",
                "14",
                "--allow_partial_index",
            ]
        )

        self.assertEqual("incremental", args.rebuild_mode)
        self.assertEqual(100, args.limit)
        self.assertEqual("cafef", args.source)
        self.assertEqual(14, args.since_days)
        self.assertTrue(args.allow_partial_index)

    def test_reject_invalid_rebuild_mode(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--rebuild_mode", "partial"])

    def test_stats_dict_contains_required_fields(self) -> None:
        data = stats_as_dict(_stats())

        required = {
            "rebuild_mode",
            "articles_loaded",
            "articles_skipped",
            "articles_indexed",
            "articles_reindexed",
            "articles_failed",
            "chunks_generated",
            "chunks_upserted",
            "articles_with_images",
            "chunks_with_images",
            "embedding_model",
            "chunking_version",
            "index_version",
            "chroma_path",
            "collection_name",
            "index_size",
            "duration_seconds",
        }
        self.assertTrue(required.issubset(data))

    def test_print_output_contains_required_lines(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_index_stats(_stats())

        text = output.getvalue()
        self.assertIn("CHROMA INDEX SYNC", text)
        self.assertIn("Mode:", text)
        self.assertIn("Articles loaded:", text)
        self.assertIn("Articles skipped:", text)
        self.assertIn("Chunks upserted:", text)
        self.assertIn("Status: OK", text)


if __name__ == "__main__":
    unittest.main()
