from __future__ import annotations

import argparse
import sys
from typing import Any


def main() -> int:
    configure_stdout()
    args = parse_args()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=args.chroma_dir)
        collection = client.get_collection(args.collection)
        count = collection.count()
        print(f"CHROMA_OK collection={args.collection} count={count}")
        response = collection.get(limit=3, include=["documents", "metadatas"])
        print_samples(response)
        return 0
    except Exception as exc:
        message = str(exc)
        code = "HNSW_LOAD_ERROR" if "hnsw" in message.lower() or "error loading hnsw index" in message.lower() else "CHROMA_UNAVAILABLE"
        print(f"{code}: {message}")
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Chroma collection and verify count/get works.")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default="news_articles")
    return parser.parse_args()


def print_samples(response: dict[str, Any]) -> None:
    ids = response.get("ids", [])
    documents = response.get("documents", [])
    metadatas = response.get("metadatas", [])
    for index, chunk_id in enumerate(ids[:3], start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) and isinstance(metadatas[index - 1], dict) else {}
        document = documents[index - 1] if index - 1 < len(documents) else ""
        print(f"--- sample {index} ---")
        print(f"chunk_id: {chunk_id}")
        print(f"title: {metadata.get('title') or ''}")
        print(f"source: {metadata.get('source') or ''}")
        print(f"topic: {metadata.get('primary_topic') or ''}")
        print(f"has_images: {metadata.get('has_images')}")
        print(f"image_count: {metadata.get('image_count')}")
        print(f"document_preview: {' '.join(str(document or '').split())[:240]}")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
