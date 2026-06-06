from __future__ import annotations

import unittest
from typing import Any
from uuid import uuid4

from app.config import load_settings
from app.local_ai.index_manifest import IndexManifest
from app.local_ai.pipeline import index_articles


class _EmbeddingModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.model_name = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class _ArticleClient:
    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self._articles = articles

    def fetch_articles(self, limit: int | None = None, since: str | None = None) -> list[dict[str, Any]]:
        return self._articles[:limit] if limit else list(self._articles)

    def fetch_article_images(self, article_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return []


class _VectorStore:
    def __init__(self, indexed_hashes: dict[str, str] | None = None) -> None:
        self.indexed_hashes = indexed_hashes or {}
        self.reset_called = False
        self.deleted_article_ids: list[str] = []
        self.upserted_chunks = []

    def reset_collection(self) -> None:
        self.reset_called = True

    def get_indexed_article_hashes(self) -> dict[str, str]:
        return dict(self.indexed_hashes)

    def delete_article_ids(self, article_ids: list[str]) -> None:
        self.deleted_article_ids.extend(article_ids)

    def upsert_chunks(self, chunks, embeddings) -> None:
        self.upserted_chunks.extend(chunks)

    def count(self) -> int:
        return len(self.upserted_chunks)


class _StorageHealth:
    def __init__(self, is_healthy: bool) -> None:
        self.is_healthy = is_healthy

    def summary(self) -> str:
        return "sqlite_embeddings=3, hnsw_ids=2, missing_in_hnsw=1, extra_in_hnsw=0, embeddings_queue=1"


class _UnhealthyVectorStore(_VectorStore):
    def storage_health(self) -> _StorageHealth:
        return _StorageHealth(False)


def _article(article_id: str, content_hash: str) -> dict[str, Any]:
    return {
        "article_id": article_id,
        "source": "genk",
        "url": f"https://example.com/{article_id}",
        "title": f"Title {article_id}",
        "summary_raw": "Summary",
        "content": "Paragraph one.\n\n" + ("Content " * 120),
        "category": "ai.chn",
        "published_at": "2026-01-01T00:00:00+00:00",
        "content_hash": content_hash,
        "primary_topic": "tech_ai_internet",
        "primary_topic_name": "Cong nghe - AI - Internet",
        "secondary_topics_json": "[]",
        "entities_json": "[]",
    }


class IndexSyncTest(unittest.TestCase):
    def _manifest_path(self) -> str:
        return f"file:test_index_sync_manifest_{uuid4().hex}?mode=memory&cache=shared"

    def test_full_rebuild_resets_and_indexes_all_articles(self) -> None:
        vector_store = _VectorStore()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1"), _article("a2", "h2")]),
            rebuild_mode="full",
        )

        self.assertTrue(vector_store.reset_called)
        self.assertEqual(2, result.articles_loaded)
        self.assertEqual(2, result.articles_indexed)
        self.assertEqual(0, result.articles_skipped)
        self.assertGreater(result.chunks_upserted, 0)

    def test_incremental_rebuild_indexes_new_or_changed_articles_only(self) -> None:
        vector_store = _VectorStore(indexed_hashes={"a1": "h1", "a2": "old"})

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1"), _article("a2", "h2"), _article("a3", "h3")]),
            rebuild_mode="incremental",
        )

        self.assertFalse(vector_store.reset_called)
        self.assertEqual(["a2"], vector_store.deleted_article_ids)
        self.assertEqual(3, result.articles_loaded)
        self.assertEqual(2, result.articles_indexed)
        self.assertEqual(1, result.articles_skipped)
        indexed_article_ids = {chunk.article_id for chunk in vector_store.upserted_chunks}
        self.assertEqual({"a2", "a3"}, indexed_article_ids)

    def test_incremental_manifest_skips_unchanged_article_without_chunking_or_embedding(self) -> None:
        manifest_path = self._manifest_path()
        manifest = IndexManifest(manifest_path)
        article = _article("a1", "h1")
        manifest.upsert_article(article, [], embedding_model="test-embedding-model")
        vector_store = _VectorStore()
        embedding_model = _EmbeddingModel()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=embedding_model,
            vector_store=vector_store,
            article_client=_ArticleClient([article]),
            rebuild_mode="incremental",
            manifest_path=str(manifest_path),
        )

        self.assertEqual(1, result.articles_loaded)
        self.assertEqual(0, result.articles_indexed)
        self.assertEqual(1, result.articles_skipped)
        self.assertEqual([], embedding_model.batch_sizes)
        self.assertEqual([], vector_store.upserted_chunks)

    def test_incremental_manifest_model_mismatch_reindexes_even_when_chroma_hash_matches(self) -> None:
        manifest_path = self._manifest_path()
        manifest = IndexManifest(manifest_path)
        article = _article("a1", "h1")
        manifest.upsert_article(article, [], embedding_model="old-model")
        vector_store = _VectorStore(indexed_hashes={"a1": "h1"})

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([article]),
            rebuild_mode="incremental",
            manifest_path=str(manifest_path),
        )

        self.assertEqual(1, result.articles_indexed)
        self.assertGreater(result.chunks_upserted, 0)

    def test_incremental_stops_source_after_five_consecutive_duplicates(self) -> None:
        duplicate_genk = [_article(f"genk-{index}", f"h{index}") for index in range(1, 6)]
        for article in duplicate_genk:
            article["source"] = "genk"
        late_genk_new = _article("genk-new-after-duplicates", "new")
        late_genk_new["source"] = "genk"
        cafef_new = _article("cafef-new", "cafef-new")
        cafef_new["source"] = "cafef"
        indexed_hashes = {str(article["article_id"]): str(article["content_hash"]) for article in duplicate_genk}
        vector_store = _VectorStore(indexed_hashes=indexed_hashes)

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([*duplicate_genk, late_genk_new, cafef_new]),
            rebuild_mode="incremental",
        )

        self.assertEqual(("genk",), result.duplicate_stopped_sources)
        self.assertEqual(5, result.duplicate_stop_threshold)
        self.assertEqual({"cafef-new"}, {chunk.article_id for chunk in vector_store.upserted_chunks})

    def test_dry_run_full_does_not_reset_or_upsert(self) -> None:
        vector_store = _VectorStore()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1")]),
            rebuild_mode="full",
            dry_run=True,
        )

        self.assertFalse(vector_store.reset_called)
        self.assertEqual(1, result.articles_indexed)
        self.assertEqual(0, result.chunks_upserted)
        self.assertEqual([], vector_store.upserted_chunks)

    def test_source_filter_limits_articles_to_index(self) -> None:
        cafef = _article("cafef-1", "h1")
        cafef["source"] = "cafef"
        genk = _article("genk-1", "h2")
        genk["source"] = "genk"
        vector_store = _VectorStore()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([cafef, genk]),
            rebuild_mode="full",
            source="cafef",
        )

        self.assertEqual(1, result.articles_loaded)
        self.assertEqual(1, result.articles_indexed)
        self.assertEqual({"cafef-1"}, {chunk.article_id for chunk in vector_store.upserted_chunks})

    def test_indexes_embeddings_in_batches(self) -> None:
        vector_store = _VectorStore()
        embedding_model = _EmbeddingModel()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=embedding_model,
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1")]),
            rebuild_mode="full",
            embedding_batch_size=1,
        )

        self.assertGreater(result.chunks_upserted, 1)
        self.assertTrue(all(batch_size <= 1 for batch_size in embedding_model.batch_sizes))
        self.assertEqual(result.chunks_upserted, sum(embedding_model.batch_sizes))

    def test_fails_when_chroma_storage_health_is_unhealthy_after_upsert(self) -> None:
        vector_store = _UnhealthyVectorStore()

        with self.assertRaisesRegex(RuntimeError, "Chroma storage health check failed"):
            index_articles(
                settings=load_settings().local_ai,
                embedding_model=_EmbeddingModel(),
                vector_store=vector_store,
                article_client=_ArticleClient([_article("a1", "h1")]),
                rebuild_mode="full",
            )

        self.assertGreater(len(vector_store.upserted_chunks), 0)

    def test_allow_partial_index_returns_warning_result_instead_of_failing(self) -> None:
        vector_store = _UnhealthyVectorStore()

        result = index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1")]),
            rebuild_mode="full",
            allow_partial_index=True,
        )

        self.assertTrue(result.partial_index)
        self.assertIn("missing_in_hnsw=1", result.health_summary)
        self.assertGreater(result.chunks_upserted, 0)

    def test_incremental_fails_preflight_when_existing_chroma_storage_is_unhealthy(self) -> None:
        vector_store = _UnhealthyVectorStore(indexed_hashes={"a1": "h1"})

        with self.assertRaisesRegex(RuntimeError, "before incremental rebuild"):
            index_articles(
                settings=load_settings().local_ai,
                embedding_model=_EmbeddingModel(),
                vector_store=vector_store,
                article_client=_ArticleClient([_article("a1", "h1")]),
                rebuild_mode="incremental",
            )

        self.assertEqual([], vector_store.upserted_chunks)

    def test_chunk_metadata_uses_current_embedding_model_name(self) -> None:
        vector_store = _VectorStore()

        index_articles(
            settings=load_settings().local_ai,
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            article_client=_ArticleClient([_article("a1", "h1")]),
            rebuild_mode="full",
        )

        self.assertGreater(len(vector_store.upserted_chunks), 0)
        self.assertEqual("test-embedding-model", vector_store.upserted_chunks[0].metadata["embedding_model"])


if __name__ == "__main__":
    unittest.main()
