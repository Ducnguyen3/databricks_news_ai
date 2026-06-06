from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_HNSW_FILES = (
    "header.bin",
    "data_level0.bin",
    "length.bin",
    "link_lists.bin",
    "index_metadata.pickle",
)


def main() -> None:
    configure_stdout()
    args = parse_args()
    chroma_dir = Path(args.chroma_dir)
    db_path = chroma_dir / "chroma.sqlite3"
    print_header("CHROMA HEALTHCHECK")
    print(f"Chroma path: {chroma_dir}")
    print(f"Collection: {args.collection}")
    if not db_path.exists():
        print(f"ERROR: missing SQLite file: {db_path}")
        return

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        inspect_sqlite(conn, chroma_dir, args.collection, args.sample_missing)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Chroma SQLite/HNSW health without modifying files.")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default="news_articles")
    parser.add_argument("--sample-missing", type=int, default=10)
    return parser.parse_args()


def inspect_sqlite(conn: sqlite3.Connection, chroma_dir: Path, collection_name: str, sample_missing: int) -> None:
    collection = conn.execute("SELECT * FROM collections WHERE name=?", (collection_name,)).fetchone()
    print_header("Collections")
    if collection is None:
        print(f"ERROR: collection not found: {collection_name}")
        return
    print(dict(collection))

    print_header("Counts")
    print(f"SQLite embeddings: {count_table(conn, 'embeddings')}")
    print(f"embedding_metadata: {count_table(conn, 'embedding_metadata')}")
    print(f"embedding_metadata_array: {count_table(conn, 'embedding_metadata_array')}")
    print(f"embeddings_queue_raw: {count_table(conn, 'embeddings_queue')}")

    segments = [dict(row) for row in conn.execute("SELECT id, type, scope, collection FROM segments ORDER BY scope, type")]
    print_header("Segments in SQLite")
    for segment in segments:
        print(segment)

    referenced = {str(segment["id"]) for segment in segments}
    vector_segments = [segment for segment in segments if str(segment["scope"]).upper() == "VECTOR"]
    folders = [path for path in chroma_dir.iterdir() if path.is_dir()]
    print_header("Folders in data/chroma")
    for folder in sorted(folders):
        state = "REFERENCED_BY_SQLITE" if folder.name in referenced else "ORPHAN_FOLDER"
        print(f"{folder.name} => {state}")

    print_header("Referenced segment folders missing on disk")
    missing_folders = [segment["id"] for segment in segments if segment["scope"] == "VECTOR" and not (chroma_dir / segment["id"]).exists()]
    print("\n".join(missing_folders) if missing_folders else "None")

    sqlite_ids = {row[0] for row in conn.execute("SELECT embedding_id FROM embeddings")}
    active_hnsw_ids: set[str] = set()
    for segment in vector_segments:
        folder = chroma_dir / str(segment["id"])
        print_header(f"HNSW folder: {folder.name}")
        if not folder.exists():
            print("ERROR: folder missing")
            continue
        for filename in REQUIRED_HNSW_FILES:
            path = folder / filename
            status = "OK" if path.exists() else "MISSING"
            size = path.stat().st_size if path.exists() else 0
            print(f"{filename}: {status} size={size}")
        ids = read_hnsw_ids(folder)
        if ids is None:
            print("HNSW metadata: unreadable")
            continue
        active_hnsw_ids = ids
        print(f"HNSW id_to_label count: {len(ids)}")
        print(f"SQLite ids count: {len(sqlite_ids)}")
        print(f"missing_in_hnsw: {len(sqlite_ids - ids)}")
        print(f"extra_in_hnsw: {len(ids - sqlite_ids)}")
        print(f"embeddings_queue_pending: {len(sqlite_ids - ids)}")
        print_missing_samples(conn, sqlite_ids - ids, sample_missing)

    print_header("Distributions")
    print_counter("source", metadata_counter(conn, "source"))
    print_counter("primary_topic", metadata_counter(conn, "primary_topic"))
    print_counter("embedding_model", metadata_counter(conn, "embedding_model"))
    print_counter("chunking_version", metadata_counter(conn, "chunking_version"))
    print_counter("index_version", metadata_counter(conn, "index_version"))
    print(f"chunks_with_images: {count_bool_metadata(conn, 'has_images')}")

    if active_hnsw_ids:
        print_header("Summary")
        print(f"embeddings_queue_pending: {len(sqlite_ids - active_hnsw_ids)}")
        print(f"Status: {'OK' if len(sqlite_ids - active_hnsw_ids) == 0 else 'MISMATCH'}")


def print_missing_samples(conn: sqlite3.Connection, missing_ids: set[str], limit: int) -> None:
    print_header("Missing chunk samples")
    if not missing_ids:
        print("None")
        return
    ordered = sorted(
        missing_ids,
        key=lambda embedding_id: conn.execute("SELECT id FROM embeddings WHERE embedding_id=?", (embedding_id,)).fetchone()[0],
    )
    for embedding_id in ordered[:limit]:
        row = conn.execute("SELECT id FROM embeddings WHERE embedding_id=?", (embedding_id,)).fetchone()
        if row is None:
            continue
        metadata = metadata_for_id(conn, int(row[0]))
        print(
            json.dumps(
                {
                    "sqlite_id": row[0],
                    "chunk_id": embedding_id,
                    "article_id": metadata.get("article_id"),
                    "title": metadata.get("title"),
                    "source": metadata.get("source"),
                    "topic": metadata.get("primary_topic"),
                    "indexed_at": metadata.get("indexed_at"),
                },
                ensure_ascii=False,
            )
        )


def metadata_for_id(conn: sqlite3.Connection, sqlite_id: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in conn.execute(
        "SELECT key, string_value, int_value, float_value, bool_value FROM embedding_metadata WHERE id=?",
        (sqlite_id,),
    ):
        output[row[0]] = first_not_none(row[1], row[2], row[3], row[4])
    return output


def read_hnsw_ids(folder: Path) -> set[str] | None:
    try:
        with (folder / "index_metadata.pickle").open("rb") as handle:
            metadata = pickle.load(handle)
    except Exception as exc:
        print(f"index_metadata.pickle read error: {exc}")
        return None
    id_to_label = metadata.get("id_to_label") if isinstance(metadata, dict) else None
    return set(id_to_label or {})


def metadata_counter(conn: sqlite3.Connection, key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in conn.execute("SELECT string_value FROM embedding_metadata WHERE key=?", (key,)):
        counter[str(row[0] or "")] += 1
    return counter


def count_bool_metadata(conn: sqlite3.Connection, key: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM embedding_metadata WHERE key=? AND bool_value=1", (key,)).fetchone()[0])


def count_table(conn: sqlite3.Connection, table_name: str) -> int | str:
    if not table_exists(conn, table_name):
        return "missing"
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table_name,)).fetchone() is not None


def print_counter(name: str, counter: Counter[str]) -> None:
    print(f"{name}: {counter.most_common(20)}")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
