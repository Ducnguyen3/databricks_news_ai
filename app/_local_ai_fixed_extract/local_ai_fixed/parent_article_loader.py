from __future__ import annotations

import json
from typing import Any


class ParentArticleLoader:
    def __init__(self, vector_store: Any) -> None:
        self._vector_store = vector_store

    def load_parent_articles(self, article_ids: list[str], chunks_per_article: int | None = None) -> list[dict[str, Any]]:
        get_chunks = getattr(self._vector_store, "get_chunks_by_article_id", None)
        if get_chunks is None:
            return []
        parents: list[dict[str, Any]] = []
        for article_id in article_ids:
            chunks = get_chunks(article_id, limit=chunks_per_article)
            if not chunks:
                continue
            parents.append(parent_article_from_chunks(chunks))
        return parents


def parent_article_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = chunks[0]
    metadata = first.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "article_id": str(metadata.get("article_id") or ""),
        "title": str(metadata.get("title") or ""),
        "summary": "",
        "content": " ".join(str(chunk.get("text") or "") for chunk in chunks),
        "source": str(metadata.get("source") or ""),
        "source_url": str(metadata.get("url") or ""),
        "url": str(metadata.get("url") or ""),
        "published_at": str(metadata.get("published_at") or ""),
        "primary_topic": str(metadata.get("primary_topic") or ""),
        "secondary_topics": _json_list(metadata.get("secondary_topics_json")),
        "entities": _json_list(metadata.get("entities_json")),
        "images": _dedupe_images(_json_list(metadata.get("images_json"))),
        "chunks": chunks,
    }


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


def _dedupe_images(images: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for image in images:
        if not isinstance(image, dict):
            continue
        image_url = str(image.get("image_url") or "")
        if not image_url or image_url in seen:
            continue
        seen.add(image_url)
        deduped.append(image)
    return deduped
