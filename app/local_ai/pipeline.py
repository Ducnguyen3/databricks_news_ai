from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from tempfile import gettempdir
from time import perf_counter
from uuid import uuid4

from app.config import LocalAiSettings
from app.local_ai.chunker import ArticleChunker
from app.local_ai.databricks_client import DatabricksArticleClient
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.image_enrichment import enrich_articles_with_images
from app.local_ai.index_manifest import CHUNKING_VERSION, INDEX_VERSION, IndexManifest, article_content_hash, chunk_hash, default_manifest_path
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.rag_service import RAGService
from app.local_ai.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    rebuild_mode: str
    articles_loaded: int
    articles_indexed: int
    articles_skipped: int
    articles_reindexed: int
    articles_failed: int
    chunks_created: int
    chunks_upserted: int
    index_size: int
    articles_with_images: int = 0
    chunks_with_images: int = 0
    embedding_model: str = ""
    chunking_version: str = "semantic-recursive-v1"
    index_version: str = "local-index-v1"
    chroma_path: str = ""
    collection_name: str = ""
    gold_table: str = ""
    duration_seconds: float = 0.0
    dry_run: bool = False
    source_filter: str | None = None
    topic_filter: str | None = None
    partial_index: bool = False
    health_summary: str = ""
    duplicate_stop_threshold: int = 0
    duplicate_stopped_sources: tuple[str, ...] = ()
    manifest_path: str = ""

    @property
    def chunks_generated(self) -> int:
        return self.chunks_created


def create_embedding_model(settings: LocalAiSettings) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(settings.embedding_model_name)


def create_vector_store(settings: LocalAiSettings) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


def create_rag_service(
    settings: LocalAiSettings,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> RAGService:
    return RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ollama_client=OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        ),
        settings=settings,
    )


def index_articles(
    settings: LocalAiSettings,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    limit: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_batch_size: int | None = None,
    reset_index: bool = False,
    rebuild_mode: str = "full",
    article_client: DatabricksArticleClient | None = None,
    source: str | None = None,
    topic: str | None = None,
    since: str | None = None,
    dry_run: bool = False,
    allow_partial_index: bool = False,
    duplicate_stop_threshold: int = 5,
    manifest_path: str | None = None,
) -> IndexingResult:
    started_at = perf_counter()
    normalized_mode = rebuild_mode.strip().lower()
    if normalized_mode not in {"full", "incremental"}:
        raise ValueError("rebuild_mode must be 'full' or 'incremental'")
    if reset_index:
        normalized_mode = "full"
    if normalized_mode == "full" and not dry_run:
        vector_store.reset_collection()

    embedding_model_name = getattr(embedding_model, "model_name", "")
    resolved_manifest_path = str(manifest_path or _default_manifest_path(vector_store))
    manifest = IndexManifest(resolved_manifest_path)
    article_client = article_client or DatabricksArticleClient()
    articles = _fetch_articles(article_client, limit=limit, since=since, order_by_source=normalized_mode == "incremental")
    articles = _filter_articles(articles, source=source, topic=topic)
    if not articles:
        raise RuntimeError("No articles returned from Databricks table.")
    loaded_count = len(articles)
    reindexed_count = 0
    selection = _IncrementalSelection(selected_articles=articles, reindexed_count=0)
    if normalized_mode == "incremental":
        if not dry_run:
            _verify_vector_store_health(
                vector_store,
                context="before incremental rebuild",
                allow_partial_index=allow_partial_index,
            )
        selection = _select_incremental_articles(
            vector_store,
            articles,
            manifest=manifest,
            embedding_model=embedding_model_name,
            delete_changed=not dry_run,
            duplicate_stop_threshold=duplicate_stop_threshold,
        )
        articles = selection.selected_articles
        reindexed_count = selection.reindexed_count
        if not articles:
            result = IndexingResult(
                rebuild_mode=normalized_mode,
                articles_loaded=loaded_count,
                articles_indexed=0,
                articles_skipped=loaded_count,
                articles_reindexed=0,
                articles_failed=0,
                chunks_created=0,
                chunks_upserted=0,
                index_size=vector_store.count(),
                embedding_model=embedding_model_name,
                chroma_path=str(getattr(vector_store, "persist_directory", "")),
                collection_name=str(getattr(vector_store, "collection_name", "")),
                gold_table=str(getattr(getattr(article_client, "_config", None), "articles_table", "")),
                duration_seconds=round(perf_counter() - started_at, 4),
                dry_run=dry_run,
                source_filter=source,
                topic_filter=topic,
                duplicate_stop_threshold=max(0, int(duplicate_stop_threshold or 0)),
                duplicate_stopped_sources=selection.stopped_sources,
                manifest_path=resolved_manifest_path,
            )
            manifest.close()
            return result
    article_images = article_client.fetch_article_images([str(article.get("article_id") or "") for article in articles])
    articles = enrich_articles_with_images(articles, article_images)
    articles_with_images = sum(1 for article in articles if article.get("has_images") or article.get("image_count"))

    chunker = ArticleChunker(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        embedding_model_name=getattr(embedding_model, "model_name", ""),
    )
    chunks = chunker.chunk_articles(articles)
    if not chunks:
        raise RuntimeError("No chunks were created from loaded articles.")
    _annotate_chunk_hashes(chunks)
    chunks_with_images = sum(1 for chunk in chunks if _truthy(chunk.metadata.get("has_images")))

    if dry_run:
        chunks_upserted = 0
        partial_index = False
        health_summary = ""
    else:
        chunks_upserted = _embed_and_upsert_batches(
            chunks=chunks,
            embedding_model=embedding_model,
            vector_store=vector_store,
            batch_size=embedding_batch_size or settings.embedding_batch_size,
        )
        _record_manifest_articles(
            manifest=manifest,
            articles=articles,
            chunks=chunks,
            embedding_model=embedding_model_name,
        )
        health_ok, health_summary = _verify_vector_store_health(
            vector_store,
            context="after upsert",
            allow_partial_index=allow_partial_index,
        )
        partial_index = not health_ok
    result = IndexingResult(
        rebuild_mode=normalized_mode,
        articles_loaded=loaded_count,
        articles_indexed=len(articles),
        articles_skipped=loaded_count - len(articles),
        articles_reindexed=reindexed_count,
        articles_failed=0,
        chunks_created=len(chunks),
        chunks_upserted=chunks_upserted,
        index_size=vector_store.count(),
        articles_with_images=articles_with_images,
        chunks_with_images=chunks_with_images,
        embedding_model=embedding_model_name,
        chroma_path=str(getattr(vector_store, "persist_directory", "")),
        collection_name=str(getattr(vector_store, "collection_name", "")),
        gold_table=str(getattr(getattr(article_client, "_config", None), "articles_table", "")),
        duration_seconds=round(perf_counter() - started_at, 4),
        dry_run=dry_run,
        source_filter=source,
        topic_filter=topic,
        partial_index=partial_index,
        health_summary=health_summary,
        duplicate_stop_threshold=max(0, int(duplicate_stop_threshold or 0)) if normalized_mode == "incremental" else 0,
        duplicate_stopped_sources=selection.stopped_sources if normalized_mode == "incremental" else (),
        manifest_path=resolved_manifest_path,
    )
    manifest.close()
    return result


@dataclass(frozen=True, slots=True)
class _IncrementalSelection:
    selected_articles: list[dict[str, object]]
    reindexed_count: int
    stopped_sources: tuple[str, ...] = ()


def _select_incremental_articles(
    vector_store: ChromaVectorStore,
    articles: list[dict[str, object]],
    manifest: IndexManifest | None = None,
    embedding_model: str = "",
    delete_changed: bool = True,
    duplicate_stop_threshold: int = 5,
) -> _IncrementalSelection:
    indexed_hashes = _indexed_article_hashes(vector_store, manifest=manifest, embedding_model=embedding_model)
    if indexed_hashes is None:
        return _IncrementalSelection(selected_articles=articles, reindexed_count=0)
    selected: list[dict[str, object]] = []
    changed_article_ids: list[str] = []
    duplicate_streak_by_source: dict[str, int] = {}
    stopped_sources: set[str] = set()
    threshold = max(0, int(duplicate_stop_threshold or 0))
    for article in articles:
        source = str(article.get("source") or "__unknown__")
        if source in stopped_sources:
            continue
        article_id = str(article.get("article_id") or "")
        if not article_id:
            continue
        content_hash = str(article.get("content_hash") or "")
        if not content_hash:
            content_hash = article_content_hash(article)
        is_unchanged = article_id in indexed_hashes and indexed_hashes.get(article_id, "") == content_hash
        if is_unchanged:
            if threshold > 0:
                duplicate_streak_by_source[source] = duplicate_streak_by_source.get(source, 0) + 1
                if duplicate_streak_by_source[source] >= threshold:
                    stopped_sources.add(source)
            continue
        duplicate_streak_by_source[source] = 0
        selected.append(article)
        if article_id in indexed_hashes:
            changed_article_ids.append(article_id)
    delete_article_ids = getattr(vector_store, "delete_article_ids", None)
    if delete_changed and changed_article_ids and delete_article_ids is not None:
        delete_article_ids(changed_article_ids)
    if delete_changed and changed_article_ids and manifest is not None:
        manifest.mark_inactive(changed_article_ids)
    return _IncrementalSelection(
        selected_articles=selected,
        reindexed_count=len(changed_article_ids),
        stopped_sources=tuple(sorted(stopped_sources)),
    )


def _fetch_articles(
    article_client: DatabricksArticleClient,
    limit: int | None,
    since: str | None,
    order_by_source: bool,
) -> list[dict[str, object]]:
    fetch_from_gold = getattr(article_client, "fetch_articles_from_gold", None)
    fetch = fetch_from_gold or article_client.fetch_articles
    try:
        return fetch(limit=limit, since=since, order_by_source=order_by_source)
    except TypeError:
        return fetch(limit=limit, since=since)


def _default_manifest_path(vector_store: ChromaVectorStore) -> Path:
    persist_directory = str(getattr(vector_store, "persist_directory", "") or "")
    if persist_directory:
        return default_manifest_path(persist_directory)
    return Path(gettempdir()) / f"databricks_news_ai_index_manifest_{uuid4().hex}.sqlite3"


def _indexed_article_hashes(
    vector_store: ChromaVectorStore,
    manifest: IndexManifest | None,
    embedding_model: str,
) -> dict[str, str] | None:
    if manifest is not None:
        hashes = manifest.get_article_hashes(
            embedding_model=embedding_model,
            chunking_version=CHUNKING_VERSION,
            index_version=INDEX_VERSION,
        )
        if hashes or manifest.has_records():
            return hashes
    get_hashes = getattr(vector_store, "get_indexed_article_hashes", None)
    if get_hashes is None:
        return None
    return dict(get_hashes())


def _annotate_chunk_hashes(chunks: list[object]) -> None:
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if isinstance(metadata, dict):
            metadata["chunk_hash"] = chunk_hash(chunk)


def _record_manifest_articles(
    manifest: IndexManifest,
    articles: list[dict[str, object]],
    chunks: list[object],
    embedding_model: str,
) -> None:
    chunks_by_article_id: dict[str, list[object]] = defaultdict(list)
    for chunk in chunks:
        article_id = str(getattr(chunk, "article_id", "") or "")
        if article_id:
            chunks_by_article_id[article_id].append(chunk)
    for article in articles:
        article_id = str(article.get("article_id") or "")
        article_chunks = chunks_by_article_id.get(article_id, [])
        if article_chunks:
            # TODO: When an embedding cache is introduced, reuse embeddings for unchanged
            # chunk_hash values instead of embedding every changed article chunk again.
            manifest.upsert_article(
                article,
                article_chunks,
                embedding_model=embedding_model,
                chunking_version=CHUNKING_VERSION,
                index_version=INDEX_VERSION,
            )


def _filter_articles(
    articles: list[dict[str, object]],
    source: str | None = None,
    topic: str | None = None,
) -> list[dict[str, object]]:
    normalized_source = (source or "").strip()
    normalized_topic = _normalize_topic_filter(topic)
    filtered: list[dict[str, object]] = []
    for article in articles:
        if normalized_source and str(article.get("source") or "") != normalized_source:
            continue
        if normalized_topic and str(article.get("primary_topic") or "") != normalized_topic:
            continue
        filtered.append(article)
    return filtered


def _normalize_topic_filter(topic: str | None) -> str | None:
    if not topic or not topic.strip():
        return None
    value = topic.strip()
    aliases = {
        "kinh_te_tai_chinh_chung_khoan": "economy_finance_stock",
        "cong_nghe_ai_internet": "tech_ai_internet",
        "thoi_su_chinh_tri_xa_hoi": "politics_society",
        "quoc_te_dia_chinh_tri_the_gioi": "world_geopolitics",
        "doanh_nghiep_khoi_nghiep": "business_startup",
        "bat_dong_san": "real_estate",
        "doi_song_giao_duc_suc_khoe_giai_tri": "lifestyle_education_health_entertainment",
    }
    return aliases.get(value, value)


def _embed_and_upsert_batches(
    chunks: list[object],
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    batch_size: int,
) -> int:
    safe_batch_size = max(1, int(batch_size))
    total = len(chunks)
    upserted = 0
    logger.info("Embedding and upserting %s chunks in batches of %s", total, safe_batch_size)
    for start in range(0, total, safe_batch_size):
        batch = chunks[start : start + safe_batch_size]
        embeddings = embedding_model.embed_texts([chunk.embedding_text for chunk in batch])
        vector_store.upsert_chunks(batch, embeddings)
        upserted += len(batch)
        logger.info("Indexed Chroma batch %s-%s/%s", start + 1, min(start + len(batch), total), total)
    return upserted


def _verify_vector_store_health(
    vector_store: ChromaVectorStore,
    context: str,
    allow_partial_index: bool = False,
) -> tuple[bool, str]:
    storage_health = getattr(vector_store, "storage_health", None)
    if storage_health is None:
        return True, ""
    health = storage_health()
    summary = health.summary()
    if health.is_healthy:
        logger.info("Chroma storage health OK %s: %s", context, summary)
        return True, summary
    if allow_partial_index:
        logger.warning("Chroma storage health is partial %s: %s", context, summary)
        return False, summary
    raise RuntimeError(
        f"Chroma storage health check failed {context}; "
        f"{summary}. Rebuild into a clean Chroma directory before using this index, "
        "or rerun with --allow_partial_index if you accept using only the vectors already in HNSW."
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
