from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EndToEndRunbookTest(unittest.TestCase):
    def test_runbook_documents_required_commands(self) -> None:
        runbook = ROOT / "docs" / "end_to_end_runbook.md"

        text = runbook.read_text(encoding="utf-8")

        self.assertIn("main.news_ai.news_raw_documents", text)
        self.assertIn("main.news_ai.news_articles", text)
        self.assertIn("main.news_ai.articles_clean", text)
        self.assertIn("python -m app.local_ai.index_sync --rebuild_mode full", text)
        self.assertIn("python -m app.local_ai.index_sync --rebuild_mode incremental", text)
        self.assertIn("python -m app.local_ai.chunk_quality_audit", text)
        self.assertIn("python -m app.local_ai.rag_smoke_test", text)
        self.assertIn("python -m app.local_ai.rag_quality_eval", text)
        self.assertIn("RAG_RETRIEVAL_MODE", text)
        self.assertIn("data/rag_quality_hybrid.json", text)
        self.assertIn("tin AI moi nhat", text)
        self.assertIn("Troubleshooting", text)

    def test_end_to_end_script_exists_and_runs_index_sync(self) -> None:
        script = ROOT / "scripts" / "run_end_to_end_local.ps1"

        text = script.read_text(encoding="utf-8")

        self.assertIn("app.local_ai.index_sync", text)
        self.assertIn("app.local_ai.chunk_quality_audit", text)
        self.assertIn("app.local_ai.rag_smoke_test", text)
        self.assertIn("-RunDatabricksJobs", text)
        self.assertIn("RunSmokeTest", text)
        self.assertIn("RunChunkAudit", text)
        self.assertIn("unittest", text)


if __name__ == "__main__":
    unittest.main()
