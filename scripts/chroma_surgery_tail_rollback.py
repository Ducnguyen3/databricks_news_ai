from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


COUNT_TABLES = (
    "embeddings",
    "embedding_metadata",
    "embedding_metadata_array",
    "embeddings_queue",
    "embedding_fulltext_search",
    "embedding_fulltext_search_content",
    "embedding_fulltext_search_docsize",
    "max_seq_id",
)


def main() -> int:
    configure_stdout()
    args = parse_args()
    chroma_dir = Path(args.chroma_dir)
    db_path = chroma_dir / "chroma.sqlite3"
    print_header("CHROMA SURGERY TAIL ROLLBACK")
    print(f"Chroma dir: {chroma_dir}")
    print(f"SQLite: {db_path}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Rollback id range: {args.start_id}..{args.end_id}")
    print(f"Max safe seq id: {args.max_safe_seq_id}")

    if not db_path.exists():
        print(f"ERROR: missing SQLite file: {db_path}")
        return 2
    if args.apply and normalize_path(chroma_dir) == normalize_path(Path("data/chroma")):
        print("ERROR: refusing --apply on data/chroma. Copy to data/chroma_surgery_test first.")
        return 2

    if args.apply:
        backup_path = backup_sqlite(db_path)
        print(f"Backup created: {backup_path}")
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        inspect_and_maybe_apply(conn, args)
    finally:
        conn.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback Chroma SQLite tail rows to match an existing HNSW segment.")
    parser.add_argument("--chroma-dir", default="data/chroma_surgery_test")
    parser.add_argument("--start-id", type=int, default=36673)
    parser.add_argument("--end-id", type=int, default=37410)
    parser.add_argument("--max-safe-seq-id", type=int, default=36928)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    return parser.parse_args()


def inspect_and_maybe_apply(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    print_header("Before counts")
    before = table_counts(conn)
    print_counts(before)

    planned = planned_deletes(conn, args.start_id, args.end_id, args.max_safe_seq_id)
    print_header("Planned deletes")
    print_counts(planned)
    print_max_seq_id(conn)

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to modify the copy.")
        return

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        apply_deletes(conn, args.start_id, args.end_id, args.max_safe_seq_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print_header("After counts")
    after = table_counts(conn)
    print_counts(after)
    print_max_seq_id(conn)

    print_header("Integrity check")
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    print(integrity)
    if args.vacuum:
        print_header("VACUUM")
        conn.execute("VACUUM")
        print("VACUUM complete")


def planned_deletes(conn: sqlite3.Connection, start_id: int, end_id: int, max_safe_seq_id: int) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    counts["embeddings"] = count_where(conn, "embeddings", "id BETWEEN ? AND ?", (start_id, end_id))
    counts["embedding_metadata"] = count_where(conn, "embedding_metadata", "id BETWEEN ? AND ?", (start_id, end_id))
    counts["embedding_metadata_array"] = count_where(conn, "embedding_metadata_array", "id BETWEEN ? AND ?", (start_id, end_id))
    counts["embeddings_queue"] = count_where(conn, "embeddings_queue", "seq_id > ?", (max_safe_seq_id,))
    counts["embedding_fulltext_search"] = count_fts_rows(conn, start_id, end_id)
    counts["embedding_fulltext_search_content"] = count_where(
        conn,
        "embedding_fulltext_search_content",
        "id BETWEEN ? AND ?",
        (start_id, end_id),
    )
    counts["embedding_fulltext_search_docsize"] = count_where(
        conn,
        "embedding_fulltext_search_docsize",
        "id BETWEEN ? AND ?",
        (start_id, end_id),
    )
    return counts


def apply_deletes(conn: sqlite3.Connection, start_id: int, end_id: int, max_safe_seq_id: int) -> None:
    delete_fts_rows(conn, start_id, end_id)
    execute_if_table(conn, "embedding_metadata_array", "DELETE FROM embedding_metadata_array WHERE id BETWEEN ? AND ?", (start_id, end_id))
    execute_if_table(conn, "embedding_metadata", "DELETE FROM embedding_metadata WHERE id BETWEEN ? AND ?", (start_id, end_id))
    execute_if_table(conn, "embeddings", "DELETE FROM embeddings WHERE id BETWEEN ? AND ?", (start_id, end_id))
    execute_if_table(conn, "embeddings_queue", "DELETE FROM embeddings_queue WHERE seq_id > ?", (max_safe_seq_id,))
    if table_exists(conn, "max_seq_id"):
        conn.execute("UPDATE max_seq_id SET seq_id=? WHERE seq_id > ?", (max_safe_seq_id, max_safe_seq_id))


def delete_fts_rows(conn: sqlite3.Connection, start_id: int, end_id: int) -> None:
    if table_exists(conn, "embedding_fulltext_search"):
        conn.execute("DELETE FROM embedding_fulltext_search WHERE rowid BETWEEN ? AND ?", (start_id, end_id))


def count_fts_rows(conn: sqlite3.Connection, start_id: int, end_id: int) -> int | str:
    if not table_exists(conn, "embedding_fulltext_search"):
        return "missing"
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM embedding_fulltext_search WHERE rowid BETWEEN ? AND ?",
            (start_id, end_id),
        ).fetchone()[0]
    )


def table_counts(conn: sqlite3.Connection) -> dict[str, int | str]:
    return {table_name: count_table(conn, table_name) for table_name in COUNT_TABLES}


def count_table(conn: sqlite3.Connection, table_name: str) -> int | str:
    if not table_exists(conn, table_name):
        return "missing"
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def count_where(conn: sqlite3.Connection, table_name: str, where_sql: str, params: tuple[Any, ...]) -> int | str:
    if not table_exists(conn, table_name):
        return "missing"
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_sql}", params).fetchone()[0])


def execute_if_table(conn: sqlite3.Connection, table_name: str, sql: str, params: tuple[Any, ...]) -> None:
    if table_exists(conn, table_name):
        conn.execute(sql, params)


def print_max_seq_id(conn: sqlite3.Connection) -> None:
    print_header("max_seq_id")
    if not table_exists(conn, "max_seq_id"):
        print("missing")
        return
    for row in conn.execute("SELECT segment_id, seq_id FROM max_seq_id ORDER BY segment_id"):
        print(dict(row))


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table_name,)).fetchone() is not None


def backup_sqlite(db_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.backup_{timestamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def print_counts(counts: dict[str, int | str]) -> None:
    for table_name, count in counts.items():
        print(f"{table_name}: {count}")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_path(path: Path) -> Path:
    return path.resolve()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
