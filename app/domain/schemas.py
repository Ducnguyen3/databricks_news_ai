from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class ArticleRow(TypedDict, total=False):
    article_id: str
    source: str
    url: str
    canonical_url: str
    title: str
    summary_raw: str | None
    content: str
    category: str | None
    published_at: Any
    crawled_at: Any
    content_hash: str
    dedup_group_id: str
    is_duplicate: bool
    created_at: Any
    updated_at: Any


@dataclass(frozen=True, slots=True)
class ArticleChunk:
    chunk_id: str
    article_id: str
    title: str
    url: str
    source: str
    published_at: str | None
    chunk_text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: ArticleChunk
    score: float

