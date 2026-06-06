from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.chunker import ArticleChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChromaStorageHealth:
    collection_count: int
    sqlite_embeddings_count: int
    hnsw_ids_count: int
    missing_in_hnsw: int
    extra_in_hnsw: int
    embeddings_queue_count: int
    query_smoke_ok: bool = False
    missing_vector_folders: tuple[str, ...] = ()
    unreadable_hnsw_folders: tuple[str, ...] = ()

    @property
    def is_healthy(self) -> bool:
        persisted_hnsw_ok = (
            self.sqlite_embeddings_count == self.hnsw_ids_count
            and self.missing_in_hnsw == 0
            and self.extra_in_hnsw == 0
            and not self.missing_vector_folders
            and not self.unreadable_hnsw_folders
        )
        unpersisted_small_index_ok = (
            self.query_smoke_ok
            and self.collection_count == self.sqlite_embeddings_count
            and self.sqlite_embeddings_count > 0
            and self.hnsw_ids_count == 0
            and not self.missing_vector_folders
        )
        return persisted_hnsw_ok or unpersisted_small_index_ok

    def summary(self) -> str:
        return (
            f"collection_count={self.collection_count}, sqlite_embeddings={self.sqlite_embeddings_count}, "
            f"hnsw_ids={self.hnsw_ids_count}, "
            f"missing_in_hnsw={self.missing_in_hnsw}, extra_in_hnsw={self.extra_in_hnsw}, "
            f"embeddings_queue={self.embeddings_queue_count}, query_smoke_ok={self.query_smoke_ok}"
        )


class ChromaUnavailableError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        distance_space: str = "cosine",
        hnsw_batch_size: int | None = None,
        hnsw_sync_threshold: int | None = None,
    ) -> None:
        load_dotenv()
        settings = load_settings()
        resolved_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", settings.local_ai.chroma_persist_dir)
        resolved_collection = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME",
            settings.local_ai.chroma_collection_name,
        )
        Path(resolved_directory).mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.config import Settings

        self._persist_directory = resolved_directory
        self._collection_name = resolved_collection
        self._distance_space = distance_space
        self._hnsw_batch_size = int(hnsw_batch_size or os.getenv("CHROMA_HNSW_BATCH_SIZE") or 64)
        self._hnsw_sync_threshold = int(hnsw_sync_threshold or os.getenv("CHROMA_HNSW_SYNC_THRESHOLD") or self._hnsw_batch_size)
        try:
            self._client = chromadb.PersistentClient(
                path=resolved_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._get_or_create_collection()
        except Exception as exc:
            raise _chroma_error(exc) from exc

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_directory(self) -> str:
        return self._persist_directory

    @property
    def hnsw_metadata(self) -> dict[str, int | str]:
        return self._collection_metadata()

    def upsert_chunks(self, chunks: list[ArticleChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            logger.info("No chunks to upsert into Chroma")
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[dict(chunk.metadata) for chunk in chunks],
        )
        logger.info(
            "Upserted %s chunks into Chroma collection=%s path=%s",
            len(chunks),
            self._collection_name,
            self._persist_directory,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if not query_embedding:
            return []

        try:
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise _chroma_error(exc) from exc
        return _search_results(response)

    def get_chunks_by_article_id(self, article_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not article_id:
            return []
        try:
            response = self._collection.get(
                where={"article_id": article_id},
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise _chroma_error(exc) from exc
        results = _get_results(response)
        return sorted(results, key=lambda item: _chunk_index(item))

    def get_chunks_by_url(self, url: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not url:
            return []
        results = self._get_chunks_by_metadata("url", url, limit=limit)
        if results:
            return sorted(results, key=lambda item: _chunk_index(item))
        results = self._get_chunks_by_metadata("canonical_url", url, limit=limit)
        return sorted(results, key=lambda item: _chunk_index(item))

    def get_chunks_by_topic_category(self, topic_category: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not topic_category:
            return []
        return self._get_chunks_by_metadata("topic_category", topic_category, limit=limit)

    def get_indexed_article_hashes(self) -> dict[str, str]:
        try:
            response = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise _chroma_error(exc) from exc
        metadatas = response.get("metadatas", [])
        indexed: dict[str, str] = {}
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            article_id = str(metadata.get("article_id") or "")
            if not article_id or article_id in indexed:
                continue
            indexed[article_id] = str(metadata.get("content_hash") or "")
        return indexed

    def delete_article_ids(self, article_ids: list[str]) -> None:
        for article_id in _dedupe_values([str(value) for value in article_ids if str(value)]):
            self._collection.delete(where={"article_id": article_id})
        if article_ids:
            logger.info("Deleted chunks for %s articles from Chroma collection=%s", len(set(article_ids)), self._collection_name)

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("Deleted Chroma collection=%s", self._collection_name)
        except Exception:
            logger.info("Chroma collection=%s did not exist; creating a new one", self._collection_name)
        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise _chroma_error(exc) from exc

    def close(self) -> None:
        system = getattr(getattr(self, "_client", None), "_system", None)
        stop = getattr(system, "stop", None)
        if callable(stop):
            try:
                stop()
                logger.info("Stopped Chroma client system collection=%s path=%s", self._collection_name, self._persist_directory)
            except Exception:
                logger.warning("Could not stop Chroma client system cleanly", exc_info=True)

    def storage_health(self) -> ChromaStorageHealth:
        db_path = Path(self._persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            raise ChromaUnavailableError("CHROMA_SQLITE_MISSING", f"missing Chroma SQLite file: {db_path}")
        try:
            conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                health = _read_storage_health(conn, Path(self._persist_directory), self._collection_name, self.count())
                if not health.is_healthy and health.hnsw_ids_count == 0 and health.unreadable_hnsw_folders:
                    health = _with_query_smoke_result(health, self._query_smoke_test())
                return health
            finally:
                conn.close()
        except ChromaUnavailableError:
            raise
        except Exception as exc:
            raise ChromaUnavailableError("CHROMA_HEALTHCHECK_FAILED", str(exc)) from exc

    def get_all_chunks(self, limit: int | None = None) -> list[dict[str, Any]]:
        try:
            response = self._collection.get(
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise _chroma_error(exc) from exc
        return _get_results(response)

    def _get_chunks_by_metadata(self, field: str, value: str, limit: int | None = None) -> list[dict[str, Any]]:
        try:
            response = self._collection.get(
                where={field: value},
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise _chroma_error(exc) from exc
        return _get_results(response)

    def _get_or_create_collection(self) -> Any:
        try:
            collection = self._client.get_collection(name=self._collection_name)
            logger.info(
                "Using existing Chroma collection=%s path=%s metadata=%s",
                self._collection_name,
                self._persist_directory,
                getattr(collection, "metadata", None) or {},
            )
            return collection
        except Exception as exc:
            if not _is_collection_not_found(exc):
                raise _chroma_error(exc) from exc
        metadata = self._collection_metadata()
        try:
            collection = self._client.create_collection(name=self._collection_name, metadata=metadata)
            logger.info(
                "Created Chroma collection=%s path=%s metadata=%s",
                self._collection_name,
                self._persist_directory,
                metadata,
            )
            return collection
        except Exception as exc:
            raise _chroma_error(exc) from exc

    def _collection_metadata(self) -> dict[str, int | str]:
        return {
            "hnsw:space": self._distance_space,
            "hnsw:batch_size": self._hnsw_batch_size,
            "hnsw:sync_threshold": self._hnsw_sync_threshold,
        }

    def _query_smoke_test(self) -> bool:
        try:
            metadata = getattr(self._collection, "metadata", None) or {}
            dimension = int(metadata.get("dimension") or 0)
        except Exception:
            dimension = 0
        if dimension <= 0:
            dimension = _collection_dimension(Path(self._persist_directory), self._collection_name)
        if dimension <= 0:
            return False
        try:
            response = self._collection.query(
                query_embeddings=[[0.0] * dimension],
                n_results=1,
                include=["distances"],
            )
        except Exception:
            return False
        ids = response.get("ids", [[]])
        return bool(ids and ids[0])


def _search_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if _is_system_flush_metadata(metadata):
            continue
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        score = None if distance is None else 1.0 / (1.0 + float(distance))
        results.append(
            {
                "chunk_id": str(chunk_id),
                "text": str(document or ""),
                "distance": None if distance is None else float(distance),
                "score": score,
                "metadata": dict(metadata or {}),
            }
        )
    return results


def _get_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    ids = response.get("ids", [])
    documents = response.get("documents", [])
    metadatas = response.get("metadatas", [])

    results: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if _is_system_flush_metadata(metadata):
            continue
        document = documents[index] if index < len(documents) else ""
        results.append(
            {
                "chunk_id": str(chunk_id),
                "text": str(document or ""),
                "distance": None,
                "score": None,
                "metadata": dict(metadata or {}),
            }
        )
    return results


def _chunk_index(item: dict[str, Any]) -> int:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    try:
        return int(metadata.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _is_system_flush_metadata(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    return _truthy(metadata.get("is_system_flush")) or _truthy(metadata.get("is_padding"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _dedupe_values(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _read_storage_health(
    conn: sqlite3.Connection,
    chroma_dir: Path,
    collection_name: str,
    collection_count: int,
) -> ChromaStorageHealth:
    collection = conn.execute("SELECT id, dimension FROM collections WHERE name=?", (collection_name,)).fetchone()
    if collection is None:
        raise ChromaUnavailableError("CHROMA_COLLECTION_MISSING", f"collection not found: {collection_name}")
    collection_id = str(collection["id"])
    sqlite_ids = {str(row[0]) for row in conn.execute("SELECT embedding_id FROM embeddings")}
    vector_segments = [
        dict(row)
        for row in conn.execute(
            "SELECT id, scope FROM segments WHERE collection=? AND UPPER(scope)='VECTOR'",
            (collection_id,),
        )
    ]
    missing_folders: list[str] = []
    unreadable_folders: list[str] = []
    hnsw_ids: set[str] = set()
    for segment in vector_segments:
        segment_id = str(segment["id"])
        folder = chroma_dir / segment_id
        if not folder.exists():
            missing_folders.append(segment_id)
            continue
        ids = _read_hnsw_ids(folder)
        if ids is None:
            unreadable_folders.append(segment_id)
            continue
        hnsw_ids.update(ids)
    return ChromaStorageHealth(
        collection_count=collection_count,
        sqlite_embeddings_count=len(sqlite_ids),
        hnsw_ids_count=len(hnsw_ids),
        missing_in_hnsw=len(sqlite_ids - hnsw_ids),
        extra_in_hnsw=len(hnsw_ids - sqlite_ids),
        embeddings_queue_count=_count_table(conn, "embeddings_queue"),
        missing_vector_folders=tuple(missing_folders),
        unreadable_hnsw_folders=tuple(unreadable_folders),
    )


def _with_query_smoke_result(health: ChromaStorageHealth, query_smoke_ok: bool) -> ChromaStorageHealth:
    return ChromaStorageHealth(
        collection_count=health.collection_count,
        sqlite_embeddings_count=health.sqlite_embeddings_count,
        hnsw_ids_count=health.hnsw_ids_count,
        missing_in_hnsw=health.missing_in_hnsw,
        extra_in_hnsw=health.extra_in_hnsw,
        embeddings_queue_count=health.embeddings_queue_count,
        query_smoke_ok=query_smoke_ok,
        missing_vector_folders=health.missing_vector_folders,
        unreadable_hnsw_folders=health.unreadable_hnsw_folders,
    )


def _collection_dimension(chroma_dir: Path, collection_name: str) -> int:
    db_path = chroma_dir / "chroma.sqlite3"
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            row = conn.execute("SELECT dimension FROM collections WHERE name=?", (collection_name,)).fetchone()
        finally:
            conn.close()
    except Exception:
        return 0
    if row is None:
        return 0
    try:
        return int(row[0] or 0)
    except (TypeError, ValueError):
        return 0


def _read_hnsw_ids(folder: Path) -> set[str] | None:
    try:
        with (folder / "index_metadata.pickle").open("rb") as handle:
            metadata = pickle.load(handle)
    except Exception:
        return None
    id_to_label = metadata.get("id_to_label") if isinstance(metadata, dict) else None
    return {str(value) for value in id_to_label or ()}


def _count_table(conn: sqlite3.Connection, table_name: str) -> int:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table_name,)).fetchone() is None:
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _is_collection_not_found(exc: Exception) -> bool:
    text = str(exc).casefold()
    return "does not exist" in text or "not found" in text or "collection" in text and "not" in text


def _chroma_error(exc: Exception) -> ChromaUnavailableError:
    message = str(exc)
    lowered = message.lower()
    if "hnsw" in lowered or "error loading hnsw index" in lowered:
        return ChromaUnavailableError("HNSW_LOAD_ERROR", message)
    return ChromaUnavailableError("CHROMA_UNAVAILABLE", message)
