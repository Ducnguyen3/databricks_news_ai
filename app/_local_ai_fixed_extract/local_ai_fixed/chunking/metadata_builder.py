from __future__ import annotations

import json
import os
from typing import Any

from app.config import load_settings
from app.utils.time import utc_now

from app.local_ai.chunking.models import ArticleBlock, ParentArticle


class ChunkMetadataBuilder:
    def __init__(
        self,
        index_version: str | None = None,
        embedding_model: str | None = None,
        chunking_version: str | None = None,
    ) -> None:
        settings = load_settings()
        self._index_version = index_version or os.getenv("INDEX_VERSION", "local-index-v1")
        self._embedding_model = embedding_model or os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            settings.local_ai.embedding_model_name,
        )
        self._chunking_version = chunking_version or os.getenv("CHUNKING_VERSION", "semantic-recursive-v1")

    def build_metadata(
        self,
        article: ParentArticle,
        block: ArticleBlock,
        chunk_id: str,
        chunk_index: int,
        chunk_type: str,
    ) -> dict[str, Any]:
        entity_names = _entity_names(article.entities)
        entity_types = _entity_types(article.entities)
        return {
            "chunk_id": chunk_id,
            "article_id": article.article_id,
            "block_id": block.block_id,
            "chunk_index": chunk_index,
            "block_index": block.block_index,
            "source": article.source,
            "url": article.url,
            "canonical_url": article.canonical_url or str(article.metadata.get("canonical_url") or article.url or ""),
            "title": article.title,
            "summary": article.summary or "",
            "content_hash": str(article.metadata.get("content_hash") or ""),
            "published_at": article.published_at or "",
            "source_category": article.source_category or "",
            "source_category_name": article.source_category_name or "",
            "source_category_url": article.source_category_url or "",
            "category": article.source_category or "",
            "primary_topic": article.primary_topic or "",
            "primary_topic_name": article.primary_topic_name or "",
            "topic_confidence": float(article.topic_confidence or 0.0),
            "topic_category": article.primary_topic or "",
            "secondary_topics_json": json.dumps(article.secondary_topics, ensure_ascii=False),
            "entities_json": json.dumps(article.entities, ensure_ascii=False),
            "entity_names": ",".join(entity_names),
            "entity_types": ",".join(entity_types),
            "has_images": bool(article.images),
            "image_count": len(article.images),
            "images_json": json.dumps(article.images, ensure_ascii=False),
            "chunk_type": chunk_type,
            "parent_id": article.article_id,
            "section_title": str(block.metadata.get("section_title") or ""),
            "block_type": block.block_type,
            "language": article.language,
            "indexed_at": utc_now().isoformat(),
            "index_version": self._index_version,
            "embedding_model": self._embedding_model,
            "chunking_version": self._chunking_version,
            "chunk_text": "",
        }


def _entity_names(entities: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for entity in entities:
        name = str(entity.get("normalized_name") or entity.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _entity_types(entities: list[dict[str, Any]]) -> list[str]:
    types: list[str] = []
    for entity in entities:
        entity_type = str(entity.get("type") or "").strip()
        if entity_type and entity_type not in types:
            types.append(entity_type)
    return types
