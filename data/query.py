import sqlite3
import pickle
from pathlib import Path
from collections import defaultdict

CHROMA_DIR = Path("data/chroma")
DB_PATH = CHROMA_DIR / "chroma.sqlite3"
HNSW_SEGMENT_ID = "792fc08a-52f7-44bb-8c67-712399c5afdb"
HNSW_META_PATH = CHROMA_DIR / HNSW_SEGMENT_ID / "index_metadata.pickle"


def print_table_info(conn, table):
    print(f"\n=== PRAGMA table_info({table}) ===")
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        print(row)


def load_hnsw_metadata():
    if not HNSW_META_PATH.exists():
        raise FileNotFoundError(f"Không thấy file: {HNSW_META_PATH}")

    with open(HNSW_META_PATH, "rb") as f:
        meta = pickle.load(f)

    print("\n=== HNSW metadata object ===")
    print("Type:", type(meta))

    if isinstance(meta, dict):
        print("Keys:", meta.keys())
        return meta

    print("Attrs:", dir(meta))
    return meta


def get_hnsw_embedding_ids(hnsw_meta):
    """
    Chroma version khác nhau có thể lưu mapping khác nhau.
    Hàm này cố lấy danh sách embedding_id/string id đang có trong HNSW metadata.
    """

    candidates = []

    if isinstance(hnsw_meta, dict):
        for key in [
            "id_to_label",
            "label_to_id",
            "id_to_seq_id",
            "seq_id_to_id",
            "id_to_index",
            "index_to_id",
        ]:
            if key in hnsw_meta:
                candidates.append((key, hnsw_meta[key]))
    else:
        for key in [
            "id_to_label",
            "label_to_id",
            "id_to_seq_id",
            "seq_id_to_id",
            "id_to_index",
            "index_to_id",
        ]:
            if hasattr(hnsw_meta, key):
                candidates.append((key, getattr(hnsw_meta, key)))

    print("\n=== HNSW mapping candidates ===")
    for key, value in candidates:
        try:
            print(key, type(value), len(value))
        except Exception:
            print(key, type(value))

    # Ưu tiên mapping có key là string embedding_id/chunk_id
    for key, mapping in candidates:
        if isinstance(mapping, dict):
            keys = list(mapping.keys())
            values = list(mapping.values())

            string_keys = [x for x in keys[:20] if isinstance(x, str)]
            string_values = [x for x in values[:20] if isinstance(x, str)]

            if string_keys:
                print(f"\nUsing HNSW mapping keys from: {key}")
                return set(mapping.keys())

            if string_values:
                print(f"\nUsing HNSW mapping values from: {key}")
                return set(mapping.values())

    raise RuntimeError(
        "Không tìm được mapping string id trong index_metadata.pickle. "
        "Hãy in output Keys/Attrs để kiểm tra format Chroma version này."
    )


def get_sqlite_embedding_rows(conn):
    print_table_info(conn, "embeddings")
    print_table_info(conn, "embedding_metadata")

    # Lấy column của bảng embeddings
    cols = [r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()]
    print("\nEmbeddings columns:", cols)

    # Chroma thường có id dạng int và embedding_id dạng string
    id_col = None
    for candidate in ["embedding_id", "id"]:
        if candidate in cols:
            id_col = candidate
            break

    if id_col is None:
        raise RuntimeError("Không tìm thấy cột id/embedding_id trong bảng embeddings")

    print("Using SQLite embedding id column:", id_col)

    rows = conn.execute(f"""
        SELECT
            e.id AS sqlite_row_id,
            {id_col} AS embedding_identifier
        FROM embeddings e
        ORDER BY e.id
    """).fetchall()

    return rows


def get_metadata_by_sqlite_ids(conn, sqlite_ids):
    if not sqlite_ids:
        return {}

    placeholders = ",".join("?" for _ in sqlite_ids)

    rows = conn.execute(f"""
        SELECT
            id,
            key,
            string_value,
            int_value,
            float_value,
            bool_value
        FROM embedding_metadata
        WHERE id IN ({placeholders})
        ORDER BY id
    """, sqlite_ids).fetchall()

    grouped = defaultdict(dict)

    for row in rows:
        row_id, key, string_value, int_value, float_value, bool_value = row

        value = string_value
        if value is None:
            value = int_value
        if value is None:
            value = float_value
        if value is None:
            value = bool_value

        grouped[row_id][key] = value

    return grouped


def main():
    conn = sqlite3.connect(DB_PATH)

    hnsw_meta = load_hnsw_metadata()
    hnsw_ids = get_hnsw_embedding_ids(hnsw_meta)

    sqlite_rows = get_sqlite_embedding_rows(conn)

    print("\n=== Counts ===")
    print("SQLite embeddings:", len(sqlite_rows))
    print("HNSW ids:", len(hnsw_ids))

    missing = []

    for sqlite_row_id, embedding_identifier in sqlite_rows:
        # So sánh cả int id và string id để tránh lệch format
        if embedding_identifier not in hnsw_ids and str(embedding_identifier) not in hnsw_ids:
            missing.append((sqlite_row_id, embedding_identifier))

    print("Missing in HNSW:", len(missing))

    print("\n=== First 50 missing rows ===")
    for item in missing[:50]:
        print(item)

    missing_sqlite_ids = [x[0] for x in missing]
    metadata = get_metadata_by_sqlite_ids(conn, missing_sqlite_ids[:200])

    print("\n=== Missing chunk details sample ===")
    for sqlite_id in missing_sqlite_ids[:50]:
        m = metadata.get(sqlite_id, {})
        print("-" * 80)
        print("sqlite_id:", sqlite_id)
        print("chunk_id:", m.get("chunk_id"))
        print("article_id:", m.get("article_id"))
        print("title:", m.get("title"))
        print("source:", m.get("source"))
        print("topic:", m.get("primary_topic"))
        print("published_at:", m.get("published_at"))
        print("indexed_at:", m.get("indexed_at"))

    print("\n=== Missing indexed_at distribution ===")
    rows = conn.execute(f"""
        SELECT string_value AS indexed_at, COUNT(*) AS total
        FROM embedding_metadata
        WHERE key = 'indexed_at'
          AND id IN ({",".join("?" for _ in missing_sqlite_ids)})
        GROUP BY string_value
        ORDER BY indexed_at
        LIMIT 100
    """, missing_sqlite_ids).fetchall() if missing_sqlite_ids else []

    for row in rows:
        print(row)

    conn.close()


if __name__ == "__main__":
    main()