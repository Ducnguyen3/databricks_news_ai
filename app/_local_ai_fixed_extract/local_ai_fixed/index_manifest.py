from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.local_ai.chunking.models import ArticleChunk

CHUNKING_VERSION = os.getenv("CHUNKING_VERSION", "semantic-recursive-v1")
INDEX_VERSION = os.getenv("INDEX_VERSION", "local-index-v1")


@dataclass(frozen=True, slots=True)
class ArticleManifestRecord:
    article_id: str
    content_hash: str
    chunk_ids: tuple[str, ...]
    chunk_hashes: tuple[str, ...]
    embedding_model: str
    chunking_version: str
    index_version: str
    last_indexed_at: str
    status: str

    @property
    def is_active(self) -> bool:
        return self.status == "active"


class IndexManifest:
    def __init__(self, path: str | Path) -> None:
        self._path_value = str(path)
        self._is_uri = self._path_value.startswith("file:")
        self._is_memory = self._path_value == ":memory:" or (self._is_uri and "mode=memory" in self._path_value)
        self._path = Path(path) if not self._is_uri and self._path_value != ":memory:" else Path(self._path_value)
        if not self._is_memory:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._persistent_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(self._path_value, uri=self._is_uri)
            self._persistent_conn.row_factory = sqlite3.Row
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def get_article(self, article_id: str) -> ArticleManifestRecord | None:
        if not article_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT article_id, content_hash, chunk_ids_json, chunk_hashes_json,
                       embedding_model, chunking_version, index_version, last_indexed_at, status
                FROM article_index_state
                WHERE article_id = ?
                """,
                (article_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def get_article_hashes(
        self,
        embedding_model: str,
        chunking_version: str = CHUNKING_VERSION,
        index_version: str = INDEX_VERSION,
    ) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT article_id, content_hash
                FROM article_index_state
                WHERE status = 'active'
                  AND embedding_model = ?
                  AND chunking_version = ?
                  AND index_version = ?
                """,
                (embedding_model, chunking_version, index_version),
            ).fetchall()
        return {str(row["article_id"]): str(row["content_hash"] or "") for row in rows}

    def has_records(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM article_index_state LIMIT 1").fetchone()
        return row is not None

    def should_skip_article(
        self,
        article: dict[str, Any],
        embedding_model: str,
        chunking_version: str = CHUNKING_VERSION,
        index_version: str = INDEX_VERSION,
    ) -> bool:
        article_id = str(article.get("article_id") or "")
        if not article_id:
            return False
        record = self.get_article(article_id)
        if record is None or not record.is_active:
            return False
        return (
            record.content_hash == article_content_hash(article)
            and record.embedding_model == embedding_model
            and record.chunking_version == chunking_version
            and record.index_version == index_version
        )

    def upsert_article(
        self,
        article: dict[str, Any],
        chunks: list[ArticleChunk],
        embedding_model: str,
        chunking_version: str = CHUNKING_VERSION,
        index_version: str = INDEX_VERSION,
        status: str = "active",
    ) -> None:
        article_id = str(article.get("article_id") or "")
        if not article_id:
            return
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_hashes = [chunk_hash(chunk, chunking_version=chunking_version) for chunk in chunks]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO article_index_state (
                    article_id, content_hash, chunk_ids_json, chunk_hashes_json,
                    embedding_model, chunking_version, index_version, last_indexed_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    chunk_ids_json = excluded.chunk_ids_json,
                    chunk_hashes_json = excluded.chunk_hashes_json,
                    embedding_model = excluded.embedding_model,
                    chunking_version = excluded.chunking_version,
                    index_version = excluded.index_version,
                    last_indexed_at = excluded.last_indexed_at,
                    status = excluded.status
                """,
                (
                    article_id,
                    article_content_hash(article),
                    json.dumps(chunk_ids, ensure_ascii=False),
                    json.dumps(chunk_hashes, ensure_ascii=False),
                    embedding_model,
                    chunking_version,
                    index_version,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                ),
            )

    def mark_inactive(self, article_ids: list[str]) -> None:
        values = [str(article_id) for article_id in article_ids if str(article_id)]
        if not values:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE article_index_state SET status = 'inactive', last_indexed_at = ? WHERE article_id = ?",
                [(datetime.now(timezone.utc).isoformat(), article_id) for article_id in values],
            )

    def close(self) -> None:
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def __del__(self) -> None:
        self.close()

    def chunk_hash_exists(
        self,
        chunk_hash_value: str,
        embedding_model: str,
        chunking_version: str = CHUNKING_VERSION,
        index_version: str = INDEX_VERSION,
    ) -> bool:
        if not chunk_hash_value:
            return False
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM article_index_state
                WHERE status = 'active'
                  AND embedding_model = ?
                  AND chunking_version = ?
                  AND index_version = ?
                  AND chunk_hashes_json LIKE ?
                LIMIT 1
                """,
                (embedding_model, chunking_version, index_version, f"%{chunk_hash_value}%"),
            ).fetchone()
        return row is not None

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS article_index_state (
                    article_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL,
                    chunk_hashes_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    last_indexed_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
            return
        conn = sqlite3.connect(self._path_value, uri=self._is_uri)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def default_manifest_path(chroma_path: str | Path) -> Path:
    return Path(chroma_path) / "index_manifest.sqlite3"


def article_content_hash(article: dict[str, Any]) -> str:
    existing = str(article.get("content_hash") or "").strip()
    if existing:
        return existing
    text = " ".join(
        str(article.get(key) or "")
        for key in ("title", "summary_raw", "summary", "content")
    )
    return sha256_text(_normalize_hash_text(text))


def chunk_hash(chunk: ArticleChunk, chunking_version: str = CHUNKING_VERSION) -> str:
    return sha256_text(f"{_normalize_hash_text(chunk.text)}\n{chunking_version}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_hash_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(normalized.strip().split())


def _record_from_row(row: sqlite3.Row) -> ArticleManifestRecord:
    return ArticleManifestRecord(
        article_id=str(row["article_id"]),
        content_hash=str(row["content_hash"] or ""),
        chunk_ids=tuple(_json_list(row["chunk_ids_json"])),
        chunk_hashes=tuple(_json_list(row["chunk_hashes_json"])),
        embedding_model=str(row["embedding_model"] or ""),
        chunking_version=str(row["chunking_version"] or ""),
        index_version=str(row["index_version"] or ""),
        last_indexed_at=str(row["last_indexed_at"] or ""),
        status=str(row["status"] or ""),
    )


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []
