from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from typing import Sequence

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.index_sync import print_index_stats
from app.local_ai.pipeline import index_articles
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)

    settings = load_settings()
    embedding_model = LocalEmbeddingModel(args.embedding_model or settings.local_ai.embedding_model_name)
    vector_store = ChromaVectorStore(
        persist_directory=args.chroma_path or settings.local_ai.chroma_persist_dir,
        collection_name=args.collection_name or settings.local_ai.chroma_collection_name,
    )
    stats = index_articles(
        settings=settings.local_ai,
        embedding_model=embedding_model,
        vector_store=vector_store,
        limit=args.limit,
        chunk_size=args.chunk_size or settings.local_ai.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.local_ai.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size or settings.local_ai.embedding_batch_size,
        rebuild_mode="full" if args.rebuild_full else args.rebuild_mode,
        source=args.source,
        topic=args.topic,
        since=args.since or _since_days(args.since_days),
        dry_run=args.dry_run,
        allow_partial_index=args.allow_partial_index,
    )
    print_index_stats(stats)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Rebuild local Chroma index from Databricks Gold articles_clean.")
    parser.add_argument("--rebuild_mode", choices=("full", "incremental"), default=os.getenv("REBUILD_MODE", "full"))
    parser.add_argument("--rebuild_full", action="store_true", default=_env_bool("REBUILD_FULL", False))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--since", default=os.getenv("DATABRICKS_ARTICLES_SINCE"))
    parser.add_argument("--since_days", type=int, default=_env_int("DATABRICKS_ARTICLES_SINCE_DAYS", 0))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_partial_index", action="store_true")
    parser.add_argument("--chroma_path", default=os.getenv("CHROMA_PERSIST_DIR", settings.local_ai.chroma_persist_dir))
    parser.add_argument(
        "--collection_name",
        default=os.getenv(
            "CHROMA_COLLECTION",
            os.getenv("CHROMA_COLLECTION_NAME", settings.local_ai.chroma_collection_name),
        ),
    )
    parser.add_argument("--chunk_size", type=int, default=settings.local_ai.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=settings.local_ai.chunk_overlap)
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=_env_int("EMBEDDING_BATCH_SIZE", settings.local_ai.embedding_batch_size),
    )
    parser.add_argument(
        "--embedding_model",
        default=os.getenv(
            "EMBEDDING_MODEL_NAME",
            os.getenv("LOCAL_EMBEDDING_MODEL", settings.local_ai.embedding_model_name),
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


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
