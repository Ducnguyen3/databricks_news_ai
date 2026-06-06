from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParentArticle:
    article_id: str
    source: str
    url: str
    title: str
    summary: str | None
    content: str
    published_at: str | None
    source_category: str | None
    canonical_url: str = ""
    source_category_name: str | None = None
    source_category_url: str | None = None
    primary_topic: str | None = None
    primary_topic_name: str | None = None
    topic_confidence: float | None = None
    secondary_topics: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    language: str = "vi"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArticleBlock:
    block_id: str
    article_id: str
    block_index: int
    block_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArticleChunk:
    chunk_id: str
    article_id: str
    block_id: str
    chunk_index: int
    chunk_text: str
    embedding_text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.chunk_text
