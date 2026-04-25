from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.chunker import ArticleChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        distance_space: str = "cosine",
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
        self._client = chromadb.PersistentClient(
            path=resolved_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()

    @property
    def collection_name(self) -> str:
        return self._collection_name

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

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(top_k)),
            include=["documents", "metadatas", "distances"],
        )
        return _search_results(response)

    def get_chunks_by_article_id(self, article_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not article_id:
            return []
        response = self._collection.get(
            where={"article_id": article_id},
            limit=limit,
            include=["documents", "metadatas"],
        )
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

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("Deleted Chroma collection=%s", self._collection_name)
        except Exception:
            logger.info("Chroma collection=%s did not exist; creating a new one", self._collection_name)
        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        return int(self._collection.count())

    def _get_chunks_by_metadata(self, field: str, value: str, limit: int | None = None) -> list[dict[str, Any]]:
        response = self._collection.get(
            where={field: value},
            limit=limit,
            include=["documents", "metadatas"],
        )
        return _get_results(response)

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_space},
        )


def _search_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
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
