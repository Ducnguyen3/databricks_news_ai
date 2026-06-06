from __future__ import annotations

import unittest
from uuid import uuid4

from app.local_ai.chunking.models import ArticleChunk
from app.local_ai.index_manifest import IndexManifest, article_content_hash, chunk_hash


def _article(article_id: str = "a1", content_hash: str = "h1") -> dict[str, object]:
    return {
        "article_id": article_id,
        "title": "Title",
        "summary_raw": "Summary",
        "content": "Content",
        "content_hash": content_hash,
    }


def _chunk(chunk_id: str = "a1:0", text: str = "Chunk text") -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        article_id="a1",
        block_id="b1",
        chunk_index=0,
        chunk_text=text,
        embedding_text=text,
        token_count=2,
        metadata={},
    )


class IndexManifestTest(unittest.TestCase):
    def _manifest_path(self) -> str:
        return f"file:test_index_manifest_{uuid4().hex}?mode=memory&cache=shared"

    def test_manifest_skips_unchanged_active_article_with_same_versions(self) -> None:
        manifest = IndexManifest(self._manifest_path())
        manifest.upsert_article(_article(), [_chunk()], embedding_model="model-a")

        self.assertTrue(manifest.should_skip_article(_article(), embedding_model="model-a"))

    def test_manifest_does_not_skip_when_embedding_model_changes(self) -> None:
        manifest = IndexManifest(self._manifest_path())
        manifest.upsert_article(_article(), [_chunk()], embedding_model="model-a")

        self.assertFalse(manifest.should_skip_article(_article(), embedding_model="model-b"))

    def test_manifest_records_chunk_ids_and_hashes(self) -> None:
        manifest = IndexManifest(self._manifest_path())
        chunk = _chunk()
        manifest.upsert_article(_article(), [chunk], embedding_model="model-a")

        record = manifest.get_article("a1")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual("h1", record.content_hash)
        self.assertEqual(("a1:0",), record.chunk_ids)
        self.assertEqual((chunk_hash(chunk),), record.chunk_hashes)
        self.assertEqual("active", record.status)

    def test_content_hash_fallback_is_stable_for_whitespace(self) -> None:
        first = _article(content_hash="")
        second = _article(content_hash="")
        second["content"] = "Content\n\n"

        self.assertEqual(article_content_hash(first), article_content_hash(second))


if __name__ == "__main__":
    unittest.main()
