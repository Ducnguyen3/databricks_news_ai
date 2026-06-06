from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import pickle
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Any, Iterable, Sequence
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.chunker import ArticleChunker
from app.local_ai.chunking.models import ArticleChunk
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.pipeline import _annotate_chunk_hashes
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging


DEFAULT_INPUT = "data/gold_audit/valid_articles.jsonl"
PADDING_DOCUMENT = "system flush padding document"
logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_stdout()
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)

    summary = rebuild_chroma_from_audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings().local_ai
    parser = argparse.ArgumentParser(description="Rebuild local Chroma from audit valid_articles.jsonl.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to valid_articles.jsonl from audit_gold_articles.py.")
    parser.add_argument("--chroma-path", default="data/chroma_from_audit")
    parser.add_argument("--collection", default=settings.chroma_collection_name)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--since-days", type=int, default=None, help="Only load articles published in the last N days.")
    parser.add_argument("--source", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--embedding-batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--embedding-model", default=settings.embedding_model_name)
    parser.add_argument("--no-reset", action="store_true", help="Append/upsert into existing collection instead of resetting it first.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-hnsw", action="store_true", default=True, help="Wait for Chroma HNSW files to flush after upsert.")
    parser.add_argument("--no-wait-for-hnsw", dest="wait_for_hnsw", action="store_false", help="Do not wait for Chroma HNSW flush.")
    parser.add_argument("--hnsw-timeout-seconds", type=int, default=180)
    parser.add_argument("--hnsw-check-interval-seconds", type=int, default=5)
    parser.add_argument("--hnsw-batch-size", type=int, default=64)
    parser.add_argument("--hnsw-sync-threshold", type=int, default=64)
    parser.add_argument("--auto-tail-flush", action="store_true", default=True)
    parser.add_argument("--no-auto-tail-flush", dest="auto_tail_flush", action="store_false")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def rebuild_chroma_from_audit(args: argparse.Namespace) -> dict[str, Any]:
    started_at = perf_counter()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Audit input not found: {input_path}")

    articles = list(
        load_audit_articles(
            input_path,
            limit=args.limit,
            source=args.source,
            topic=args.topic,
            since_days=args.since_days,
        )
    )
    if not articles:
        raise RuntimeError("No valid audit articles matched the requested filters.")

    embedding_model_name = str(args.embedding_model)
    chunker = ArticleChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model_name=embedding_model_name,
    )
    chunks = chunker.chunk_articles(articles)
    if not chunks:
        raise RuntimeError("No chunks were created from audit articles.")
    _annotate_chunk_hashes(chunks)

    chunks_upserted = 0
    real_chunks_upserted = 0
    padding_chunks_added = 0
    total_chunks_with_padding = 0
    health_after_auto_tail_flush: dict[str, Any] = {}
    index_size = 0
    health_summary = ""
    hnsw_wait_seconds = 0.0
    hnsw_healthy = False
    health_before_wait: dict[str, Any] = {}
    health_after_wait: dict[str, Any] = {}
    if not args.dry_run:
        embedding_model = LocalEmbeddingModel(embedding_model_name)
        if not args.no_reset:
            reset_chroma_directory(args.chroma_path)
        vector_store = ChromaVectorStore(
            persist_directory=args.chroma_path,
            collection_name=args.collection,
            hnsw_batch_size=args.hnsw_batch_size,
            hnsw_sync_threshold=args.hnsw_sync_threshold,
        )
        if not args.no_reset:
            vector_store.reset_collection()
        chunks_upserted = embed_and_upsert_batches(
            chunks=chunks,
            embedding_model=embedding_model,
            vector_store=vector_store,
            batch_size=args.embedding_batch_size,
        )
        real_chunks_upserted = chunks_upserted
        total_chunks_with_padding = real_chunks_upserted
        if args.auto_tail_flush:
            padding_chunks = build_flush_padding_chunks(
                real_chunks_upserted=real_chunks_upserted,
                sync_threshold=args.hnsw_sync_threshold,
            )
            padding_chunks_added = len(padding_chunks)
            if padding_chunks:
                padding_embedding = embedding_model.embed_texts([PADDING_DOCUMENT])[0]
                vector_store.upsert_chunks(padding_chunks, [padding_embedding for _ in padding_chunks])
                chunks_upserted += len(padding_chunks)
                total_chunks_with_padding = chunks_upserted
                try:
                    vector_store.search(padding_embedding, top_k=1)
                except Exception:
                    logger.debug("Padding query smoke before close failed", exc_info=True)
            health_after_auto_tail_flush = read_chroma_hnsw_health(
                args.chroma_path,
                args.collection,
                expected_count=total_chunks_with_padding,
            )
        index_size = vector_store.count()
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        health_before_wait = read_chroma_hnsw_health(args.chroma_path, args.collection, expected_count=total_chunks_with_padding or chunks_upserted)
        health_after_wait = dict(health_before_wait)
        hnsw_healthy = bool(health_after_wait.get("healthy"))
        if args.wait_for_hnsw:
            del vector_store
            gc.collect()
            wait_result = wait_for_chroma_hnsw_flush(
                chroma_path=args.chroma_path,
                collection_name=args.collection,
                expected_count=total_chunks_with_padding or chunks_upserted,
                timeout_seconds=args.hnsw_timeout_seconds,
                interval_seconds=args.hnsw_check_interval_seconds,
            )
            hnsw_wait_seconds = float(wait_result["wait_seconds"])
            hnsw_healthy = bool(wait_result["healthy"])
            health_after_wait = dict(wait_result["health_after_wait"])
        health_summary = health_after_wait.get("summary", "")

    summary = {
        "input": str(input_path),
        "chroma_path": str(args.chroma_path),
        "collection": str(args.collection),
        "dry_run": bool(args.dry_run),
        "reset_collection": not bool(args.no_reset),
        "articles_loaded": len(articles),
        "chunks_created": len(chunks),
        "chunks_upserted": chunks_upserted,
        "auto_tail_flush_enabled": bool(args.auto_tail_flush and not args.dry_run),
        "real_chunks_upserted": real_chunks_upserted,
        "padding_chunks_added": padding_chunks_added,
        "total_chunks_with_padding": total_chunks_with_padding or chunks_upserted,
        "health_after_auto_tail_flush": health_after_auto_tail_flush,
        "index_size": index_size,
        "embedding_model": embedding_model_name,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "source_filter": args.source,
        "topic_filter": args.topic,
        "since_days": args.since_days,
        "topic_counts": dict(Counter(str(article.get("primary_topic") or "") for article in articles)),
        "source_counts": dict(Counter(str(article.get("source") or "") for article in articles)),
        "health_summary": health_summary,
        "hnsw_wait_enabled": bool(args.wait_for_hnsw and not args.dry_run),
        "hnsw_batch_size": int(args.hnsw_batch_size),
        "hnsw_sync_threshold": int(args.hnsw_sync_threshold),
        "hnsw_collection_metadata": {
            "hnsw:space": "cosine",
            "hnsw:batch_size": int(args.hnsw_batch_size),
            "hnsw:sync_threshold": int(args.hnsw_sync_threshold),
        },
        "hnsw_wait_seconds": round(hnsw_wait_seconds, 3),
        "hnsw_healthy": hnsw_healthy,
        "health_before_wait": health_before_wait,
        "health_after_wait": health_after_wait,
        "status": "DRY_RUN" if args.dry_run else ("OK" if hnsw_healthy else "MISMATCH"),
        "status_reason": ""
        if args.dry_run or hnsw_healthy
        else "SQLite has all embeddings but HNSW has not flushed all vectors.",
        "duration_seconds": round(perf_counter() - started_at, 4),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = Path(args.summary_path) if args.summary_path else Path(args.chroma_path) / "audit_rebuild_summary.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
        summary["summary_path"] = str(summary_path)
    print_rebuild_status(summary)
    return summary


def load_audit_articles(
    path: Path,
    *,
    limit: int | None = None,
    source: str | None = None,
    topic: str | None = None,
    since_days: int | None = None,
) -> Iterable[dict[str, Any]]:
    count = 0
    cutoff = published_cutoff(since_days)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            article = audit_row_to_article(row)
            if source and str(article.get("source") or "") != source:
                continue
            if topic and str(article.get("primary_topic") or "") != topic:
                continue
            if cutoff is not None:
                published_at = parse_datetime(article.get("published_at"))
                if published_at is None or published_at < cutoff:
                    continue
            yield article
            count += 1
            if limit is not None and count >= int(limit):
                break


def audit_row_to_article(row: dict[str, Any]) -> dict[str, Any]:
    article_id = str(row.get("article_id") or "").strip()
    title = str(row.get("title") or "").strip()
    content = str(row.get("cleaned_content") or "").strip()
    source = str(row.get("source_name") or "").strip()
    topic = str(row.get("topic") or "").strip()
    published_at = row.get("published_at")
    content_hash = sha256_text(content)
    return {
        "article_id": article_id,
        "source": source,
        "source_name": source,
        "url": str(row.get("url") or ""),
        "canonical_url": str(row.get("canonical_url") or row.get("url") or ""),
        "title": title,
        "summary_raw": str(row.get("summary_raw") or row.get("summary") or ""),
        "content": content,
        "cleaned_content": content,
        "category": str(row.get("category") or ""),
        "published_at": published_at,
        "crawled_at": row.get("crawled_at") or published_at,
        "content_hash": content_hash,
        "dedup_group_id": str(row.get("dedup_group_id") or article_id),
        "is_duplicate": False,
        "primary_topic": topic,
        "topic": topic,
        "primary_topic_name": str(row.get("primary_topic_name") or topic),
        "topic_confidence": float(row.get("topic_confidence") or 1.0),
        "secondary_topics_json": str(row.get("secondary_topics_json") or "[]"),
        "entities_json": str(row.get("entities_json") or "[]"),
    }


def embed_and_upsert_batches(
    *,
    chunks: list[Any],
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    batch_size: int,
) -> int:
    upserted = 0
    size = max(1, int(batch_size or 1))
    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        embeddings = embedding_model.embed_texts([chunk.text for chunk in batch])
        vector_store.upsert_chunks(batch, embeddings)
        upserted += len(batch)
        print(f"Indexed Chroma batch {start + 1}-{start + len(batch)}/{len(chunks)}")
    return upserted


def padding_needed_for_flush(real_chunks_upserted: int, sync_threshold: int) -> int:
    threshold = max(1, int(sync_threshold or 1))
    remainder = int(real_chunks_upserted or 0) % threshold
    return 0 if remainder == 0 else threshold - remainder


def build_flush_padding_chunks(real_chunks_upserted: int, sync_threshold: int, run_id: str | None = None) -> list[ArticleChunk]:
    padding_needed = padding_needed_for_flush(real_chunks_upserted, sync_threshold)
    if padding_needed <= 0:
        return []
    resolved_run_id = run_id or uuid4().hex
    indexed_at = datetime.now(timezone.utc).isoformat()
    chunks: list[ArticleChunk] = []
    for index in range(padding_needed):
        chunk_id = f"__flush_padding__::{resolved_run_id}::{index + 1}"
        metadata = {
            "is_system_flush": True,
            "is_padding": True,
            "source": "__system__",
            "article_id": "__flush_padding__",
            "chunk_id": chunk_id,
            "primary_topic": "__padding__",
            "topic": "__padding__",
            "topic_category": "__padding__",
            "title": "system flush padding",
            "published_at": indexed_at,
            "indexed_at": indexed_at,
            "chunk_index": index,
        }
        chunks.append(
            ArticleChunk(
                chunk_id=chunk_id,
                article_id="__flush_padding__",
                block_id=f"__flush_padding__::{resolved_run_id}",
                chunk_index=index,
                chunk_text=PADDING_DOCUMENT,
                embedding_text=PADDING_DOCUMENT,
                token_count=len(PADDING_DOCUMENT.split()),
                metadata=metadata,
            )
        )
    return chunks


def reset_chroma_directory(chroma_path: str) -> None:
    path = Path(chroma_path)
    if not path.exists():
        return
    resolved = path.resolve()
    workspace = ROOT.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Refusing to reset Chroma directory outside workspace: {resolved}")
    if resolved == workspace or resolved == workspace / "data":
        raise ValueError(f"Refusing to reset unsafe Chroma directory: {resolved}")
    shutil.rmtree(resolved)
    logger.info("Removed existing Chroma directory before rebuild: %s", resolved)


def wait_for_chroma_hnsw_flush(
    chroma_path: str,
    collection_name: str,
    expected_count: int,
    timeout_seconds: int = 180,
    interval_seconds: int = 5,
) -> dict[str, Any]:
    started_at = perf_counter()
    timeout = max(1, int(timeout_seconds or 1))
    interval = max(1, int(interval_seconds or 1))
    attempts = 0
    last_health = read_chroma_hnsw_health(chroma_path, collection_name, expected_count=expected_count)
    last_health["query_smoke_ok"] = refresh_chroma_client(chroma_path, collection_name)
    last_health["healthy"] = is_hnsw_health_healthy(last_health, expected_count)
    while True:
        attempts += 1
        if last_health.get("healthy"):
            return {
                "healthy": True,
                "wait_seconds": perf_counter() - started_at,
                "attempts": attempts,
                "health_after_wait": last_health,
            }
        elapsed = perf_counter() - started_at
        if elapsed >= timeout:
            logger.warning(
                "SQLite has all embeddings but HNSW has not flushed all vectors. health=%s",
                last_health,
            )
            return {
                "healthy": False,
                "wait_seconds": elapsed,
                "attempts": attempts,
                "health_after_wait": last_health,
            }
        sleep(interval)
        last_health = read_chroma_hnsw_health(chroma_path, collection_name, expected_count=expected_count)
        last_health["query_smoke_ok"] = refresh_chroma_client(chroma_path, collection_name)
        last_health["healthy"] = is_hnsw_health_healthy(last_health, expected_count)


def refresh_chroma_client(chroma_path: str, collection_name: str) -> bool:
    try:
        store = ChromaVectorStore(persist_directory=chroma_path, collection_name=collection_name)
        smoke = getattr(store, "_query_smoke_test", None)
        query_smoke_ok = bool(smoke()) if callable(smoke) else bool(store.count())
        close = getattr(store, "close", None)
        if callable(close):
            close()
        del store
        return query_smoke_ok
    except Exception as exc:
        logger.debug("Chroma refresh during HNSW wait failed: %s", exc)
        return False
    finally:
        gc.collect()


def read_chroma_hnsw_health(chroma_path: str, collection_name: str, expected_count: int) -> dict[str, Any]:
    chroma_dir = Path(chroma_path)
    db_path = chroma_dir / "chroma.sqlite3"
    if not db_path.exists():
        return {
            "healthy": False,
            "summary": f"missing Chroma SQLite file: {db_path}",
            "collection_count": 0,
            "sqlite_embeddings": 0,
            "hnsw_ids": 0,
            "missing_in_hnsw": expected_count,
            "extra_in_hnsw": 0,
            "embeddings_queue": 0,
            "query_smoke_ok": False,
        }
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            health = _read_chroma_hnsw_health(conn, chroma_dir, collection_name, expected_count)
        finally:
            conn.close()
    except Exception as exc:
        return {
            "healthy": False,
            "summary": f"healthcheck_failed: {type(exc).__name__}: {exc}",
            "collection_count": 0,
            "sqlite_embeddings": 0,
            "hnsw_ids": 0,
            "missing_in_hnsw": expected_count,
            "extra_in_hnsw": 0,
            "embeddings_queue": 0,
            "query_smoke_ok": False,
        }
    health["healthy"] = is_hnsw_health_healthy(health, expected_count)
    health["summary"] = (
        f"collection_count={health['collection_count']}, sqlite_embeddings={health['sqlite_embeddings']}, "
        f"hnsw_ids={health['hnsw_ids']}, missing_in_hnsw={health['missing_in_hnsw']}, "
        f"extra_in_hnsw={health['extra_in_hnsw']}, embeddings_queue={health['embeddings_queue']}, "
        f"query_smoke_ok={health['query_smoke_ok']}"
    )
    return health


def is_hnsw_health_healthy(health: dict[str, Any], expected_count: int) -> bool:
    return (
        health.get("collection_count") == expected_count
        and health.get("sqlite_embeddings") == expected_count
        and health.get("hnsw_ids") == expected_count
        and health.get("missing_in_hnsw") == 0
        and health.get("embeddings_queue") == 0
    )


def _read_chroma_hnsw_health(
    conn: sqlite3.Connection,
    chroma_dir: Path,
    collection_name: str,
    expected_count: int,
) -> dict[str, Any]:
    collection = conn.execute("SELECT id FROM collections WHERE name=?", (collection_name,)).fetchone()
    if collection is None:
        return {
            "collection_count": 0,
            "sqlite_embeddings": 0,
            "hnsw_ids": 0,
            "missing_in_hnsw": expected_count,
            "extra_in_hnsw": 0,
            "embeddings_queue": count_table(conn, "embeddings_queue"),
            "query_smoke_ok": False,
        }
    collection_id = str(collection["id"])
    sqlite_ids = {str(row[0]) for row in conn.execute("SELECT embedding_id FROM embeddings")}
    hnsw_ids: set[str] = set()
    for row in conn.execute("SELECT id FROM segments WHERE collection=? AND UPPER(scope)='VECTOR'", (collection_id,)):
        ids = read_hnsw_ids(chroma_dir / str(row[0]))
        if ids is not None:
            hnsw_ids.update(ids)
    return {
        "collection_count": len(sqlite_ids),
        "sqlite_embeddings": len(sqlite_ids),
        "hnsw_ids": len(hnsw_ids),
        "missing_in_hnsw": len(sqlite_ids - hnsw_ids),
        "extra_in_hnsw": len(hnsw_ids - sqlite_ids),
        "embeddings_queue": len(sqlite_ids - hnsw_ids),
        "embeddings_queue_raw": count_table(conn, "embeddings_queue"),
        "query_smoke_ok": False,
    }


def read_hnsw_ids(folder: Path) -> set[str] | None:
    try:
        with (folder / "index_metadata.pickle").open("rb") as handle:
            metadata = pickle.load(handle)
    except Exception:
        return None
    id_to_label = metadata.get("id_to_label") if isinstance(metadata, dict) else None
    return {str(value) for value in id_to_label or ()}


def count_table(conn: sqlite3.Connection, table_name: str) -> int:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table_name,)).fetchone() is None:
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def print_rebuild_status(summary: dict[str, Any]) -> None:
    status = str(summary.get("status") or "")
    if status == "DRY_RUN":
        return
    if status == "OK":
        print("Status OK: Chroma SQLite and HNSW are in sync.")
    elif status == "MISMATCH":
        print("Status MISMATCH: SQLite has all embeddings but HNSW has not flushed all vectors.")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def published_cutoff(since_days: int | None) -> datetime | None:
    if since_days is None:
        return None
    days = int(since_days)
    if days <= 0:
        raise ValueError("--since-days must be positive")
    return datetime.now(timezone.utc) - timedelta(days=days)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
