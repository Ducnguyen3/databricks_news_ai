from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from app.local_ai.chunking.metadata_builder import ChunkMetadataBuilder
from app.local_ai.chunking.models import ArticleBlock, ArticleChunk, ParentArticle
from app.local_ai.chunking.recursive_splitter import RecursiveSplitter
from app.local_ai.chunking.semantic_blocker import SemanticBlocker


class ArticleChunker:
    def __init__(
        self,
        blocker: SemanticBlocker | None = None,
        splitter: RecursiveSplitter | None = None,
        metadata_builder: ChunkMetadataBuilder | None = None,
    ) -> None:
        self._blocker = blocker or SemanticBlocker()
        self._splitter = splitter or RecursiveSplitter()
        self._metadata_builder = metadata_builder or ChunkMetadataBuilder()

    def chunk_article(self, article: ParentArticle) -> list[ArticleChunk]:
        chunks: list[ArticleChunk] = []
        for block in self._blocker.build_blocks(article):
            for local_index, chunk_text in enumerate(self._splitter.split_block(block)):
                chunk_id = f"{_sanitize_id(article.article_id)}::b{block.block_index}::c{local_index}"
                metadata = self._metadata_builder.build_metadata(
                    article=article,
                    block=block,
                    chunk_id=chunk_id,
                    chunk_index=len(chunks),
                    chunk_type=block.block_type,
                )
                metadata["chunk_text"] = chunk_text
                chunks.append(
                    ArticleChunk(
                        chunk_id=chunk_id,
                        article_id=article.article_id,
                        block_id=block.block_id,
                        chunk_index=len(chunks),
                        chunk_text=chunk_text,
                        embedding_text=build_embedding_text(article, chunk_text),
                        token_count=max(1, len(chunk_text.split())),
                        metadata=metadata,
                    )
                )
        return chunks


def build_embedding_text(article: ParentArticle, chunk_text: str) -> str:
    entity_names = [
        str(entity.get("normalized_name") or entity.get("name") or "")
        for entity in article.entities
        if entity.get("normalized_name") or entity.get("name")
    ]
    parts = [
        f"Tiêu đề: {article.title}",
        f"Tóm tắt: {article.summary or ''}",
        f"Chủ đề: {article.primary_topic_name or article.primary_topic or ''}",
        f"Thực thể: {', '.join(entity_names)}",
        f"Nguồn: {article.source}",
        f"Ngày đăng: {article.published_at or ''}",
        f"Nội dung:\n{chunk_text}",
    ]
    return "\n".join(parts)


def parent_article_from_mapping(article: Mapping[str, Any]) -> ParentArticle:
    return ParentArticle(
        article_id=str(article.get("article_id") or ""),
        source=str(article.get("source") or ""),
        url=str(article.get("url") or ""),
        canonical_url=str(article.get("canonical_url") or article.get("url") or ""),
        title=str(article.get("title") or ""),
        summary=_optional_str(article.get("summary_raw") or article.get("summary")),
        content=str(article.get("content") or ""),
        published_at=_optional_str(article.get("published_at")),
        source_category=_optional_str(article.get("category")),
        source_category_name=_optional_str(article.get("source_category_name")),
        source_category_url=_optional_str(article.get("source_category_url")),
        primary_topic=_optional_str(article.get("primary_topic")),
        primary_topic_name=_optional_str(article.get("primary_topic_name")),
        topic_confidence=_float_or_none(article.get("topic_confidence")),
        secondary_topics=_json_list(article.get("secondary_topics_json")),
        entities=_json_list(article.get("entities_json")),
        images=_json_list(article.get("images_json")),
        language="vi",
        metadata=dict(article),
    )


def group_chunks_by_article(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        article_id = str(metadata.get("article_id") or result.get("article_id") or "")
        if article_id:
            grouped[article_id].append(result)
    return dict(grouped)


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "None", "nan", "NaT"} else text


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
