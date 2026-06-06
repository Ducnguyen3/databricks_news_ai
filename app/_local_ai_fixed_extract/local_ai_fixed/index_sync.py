from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import date, timedelta
from typing import Sequence

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.pipeline import IndexingResult, index_articles
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)
    stats = run_index_sync(args)
    print_index_stats(stats)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Sync Databricks Gold articles_clean into local Chroma index.")
    parser.add_argument("--rebuild_mode", choices=("full", "incremental"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--since", default=os.getenv("DATABRICKS_ARTICLES_SINCE"))
    parser.add_argument("--since_days", type=int, default=_env_int("DATABRICKS_ARTICLES_SINCE_DAYS", 0))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_partial_index", action="store_true")
    parser.add_argument(
        "--duplicate_stop_threshold",
        type=int,
        default=_env_int("INDEX_SYNC_DUPLICATE_STOP_THRESHOLD", 5),
        help="Incremental sync stops scanning a source after this many consecutive unchanged articles. Use 0 to disable.",
    )
    parser.add_argument("--chroma_path", default=settings.local_ai.chroma_persist_dir)
    parser.add_argument("--collection_name", default=settings.local_ai.chroma_collection_name)
    parser.add_argument("--manifest_path", default=os.getenv("INDEX_MANIFEST_PATH"))
    parser.add_argument("--chunk_size", type=int, default=settings.local_ai.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=settings.local_ai.chunk_overlap)
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=_env_int("EMBEDDING_BATCH_SIZE", settings.local_ai.embedding_batch_size),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run_index_sync(args: argparse.Namespace) -> IndexingResult:
    settings = load_settings()
    embedding_model = LocalEmbeddingModel(settings.local_ai.embedding_model_name)
    vector_store = ChromaVectorStore(
        persist_directory=args.chroma_path,
        collection_name=args.collection_name,
    )
    return index_articles(
        settings=settings.local_ai,
        embedding_model=embedding_model,
        vector_store=vector_store,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        rebuild_mode=args.rebuild_mode,
        source=args.source,
        topic=args.topic,
        since=args.since or _since_days(args.since_days),
        dry_run=bool(args.dry_run),
        allow_partial_index=bool(args.allow_partial_index),
        duplicate_stop_threshold=args.duplicate_stop_threshold,
        manifest_path=args.manifest_path,
    )


def stats_as_dict(stats: IndexingResult) -> dict[str, object]:
    data = asdict(stats)
    data["chunks_generated"] = stats.chunks_generated
    return data


def print_index_stats(stats: IndexingResult) -> None:
    data = stats_as_dict(stats)
    print("========================================")
    print("CHROMA INDEX SYNC")
    print("========================================")
    print(f"Mode: {data.get('rebuild_mode')}")
    print(f"Gold table: {data.get('gold_table') or 'unknown'}")
    print(f"Chroma path: {data.get('chroma_path') or 'unknown'}")
    print(f"Collection: {data.get('collection_name') or 'unknown'}")
    print(f"Embedding model: {data.get('embedding_model') or 'unknown'}")
    print(f"Chunking version: {data.get('chunking_version') or 'unknown'}")
    print(f"Index version: {data.get('index_version') or 'unknown'}")
    if data.get("manifest_path"):
        print(f"Manifest path: {data.get('manifest_path')}")
    if data.get("source_filter"):
        print(f"Source filter: {data.get('source_filter')}")
    if data.get("topic_filter"):
        print(f"Topic filter: {data.get('topic_filter')}")
    if data.get("duplicate_stop_threshold"):
        print(f"Duplicate stop threshold: {data.get('duplicate_stop_threshold')}")
    if data.get("duplicate_stopped_sources"):
        print(f"Duplicate stopped sources: {', '.join(data.get('duplicate_stopped_sources') or [])}")
    print("")
    print(f"Articles loaded: {data.get('articles_loaded')}")
    print(f"Articles skipped: {data.get('articles_skipped')}")
    print(f"Articles indexed: {data.get('articles_indexed')}")
    print(f"Articles reindexed: {data.get('articles_reindexed')}")
    print(f"Articles failed: {data.get('articles_failed')}")
    print("")
    print(f"Chunks generated: {data.get('chunks_generated')}")
    print(f"Chunks upserted: {data.get('chunks_upserted')}")
    print(f"Articles with images: {data.get('articles_with_images')}")
    print(f"Chunks with images: {data.get('chunks_with_images')}")
    print("")
    print(f"Index size: {data.get('index_size')}")
    print(f"Duration: {data.get('duration_seconds')}s")
    print(f"Dry run: {data.get('dry_run')}")
    if data.get("partial_index"):
        print("Status: WARN_PARTIAL_INDEX")
        print(f"Health: {data.get('health_summary')}")
    else:
        print("Status: OK")
    print("========================================")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _since_days(days: int | None) -> str | None:
    if not days:
        return None
    if days <= 0:
        raise ValueError("since_days must be positive")
    return (date.today() - timedelta(days=days)).isoformat()


if __name__ == "__main__":
    main()
