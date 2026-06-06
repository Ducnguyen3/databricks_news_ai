from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from app.local_ai.chunking.article_chunker import ArticleChunker as SemanticArticleChunker
from app.local_ai.chunking.article_chunker import parent_article_from_mapping
from app.local_ai.chunking.metadata_builder import ChunkMetadataBuilder
from app.local_ai.chunking.models import ArticleChunk
from app.local_ai.chunking.recursive_splitter import RecursiveSplitter, split_sentences
from app.local_ai.chunking.semantic_blocker import split_paragraphs

logger = logging.getLogger(__name__)


class ArticleChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        splitter = RecursiveSplitter(
            target_chunk_chars=chunk_size or 2000,
            max_chunk_chars=max(chunk_size or 3000, chunk_size or 3000),
            overlap_chars=chunk_overlap or 250,
        )
        metadata_builder = ChunkMetadataBuilder(embedding_model=embedding_model_name) if embedding_model_name else None
        self._chunker = SemanticArticleChunker(splitter=splitter, metadata_builder=metadata_builder)

    def chunk_article(self, article: Mapping[str, Any]) -> list[ArticleChunk]:
        parent = parent_article_from_mapping(article)
        if not parent.article_id or not parent.content.strip():
            return []
        return self._chunker.chunk_article(parent)

    def chunk_articles(self, articles: list[dict[str, Any]] | Iterable[Mapping[str, Any]]) -> list[ArticleChunk]:
        rows = _article_rows(articles)
        chunks: list[ArticleChunk] = []
        for article in rows:
            chunks.extend(self.chunk_article(article))
        logger.info("Created %s semantic chunks from %s articles", len(chunks), len(rows))
        return chunks


def chunk_articles(
    articles: Any,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ArticleChunk]:
    return ArticleChunker(chunk_size=chunk_size, chunk_overlap=overlap).chunk_articles(articles)


def chunk_text_with_overlap(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    block = type("_Block", (), {"text": text})()
    return RecursiveSplitter(
        target_chunk_chars=chunk_size,
        max_chunk_chars=max(chunk_size, chunk_size + overlap),
        min_chunk_chars=80,
        overlap_chars=overlap,
    ).split_block(block)


def infer_topic_category(article: Mapping[str, Any]) -> str:
    return str(article.get("primary_topic") or article.get("topic_category") or "")


def normalize_topic_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _article_rows(articles: Any) -> list[Mapping[str, Any]]:
    if hasattr(articles, "to_dict"):
        return articles.to_dict("records")
    if isinstance(articles, Iterable):
        return list(articles)
    raise TypeError("articles must be a pandas DataFrame or an iterable of mappings")


__all__ = [
    "ArticleChunk",
    "ArticleChunker",
    "chunk_articles",
    "chunk_text_with_overlap",
    "infer_topic_category",
    "normalize_topic_text",
    "split_paragraphs",
    "split_sentences",
]
