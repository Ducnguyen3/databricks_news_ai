from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.local_ai.media_retriever import is_valid_image
from app.local_ai.query_router import domain_from_topic
from app.local_ai.reranker import safe_float


@dataclass(slots=True)
class RetrievedArticle:
    article_id: str
    title: str
    summary: str
    content: str
    selected_context: str
    source_name: str
    canonical_url: str
    published_at: str
    topic: str
    entities: list[Any] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    matched_chunks: list[dict[str, Any]] = field(default_factory=list)
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieved_articles(
    chunks: list[dict[str, Any]],
    parent_articles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parent_by_id = {
        str(parent.get("article_id") or ""): parent
        for parent in parent_articles or []
        if str(parent.get("article_id") or "")
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    fallback_index = 0
    for chunk in chunks:
        metadata = _metadata(chunk)
        article_id = str(metadata.get("article_id") or "")
        if not article_id:
            fallback_index += 1
            article_id = f"chunk:{chunk.get('chunk_id') or fallback_index}"
        grouped.setdefault(article_id, []).append(chunk)

    articles: list[RetrievedArticle] = []
    for article_id, matched_chunks in grouped.items():
        parent = parent_by_id.get(article_id, {})
        metadata = _metadata(matched_chunks[0]) if matched_chunks else {}
        selected_context = _merge_chunk_texts(matched_chunks)
        content = str(parent.get("content") or selected_context)
        images = _dedupe_images([*_json_list(metadata.get("images_json")), *_fallback_images(metadata), *list(parent.get("images") or [])])
        entities = parent.get("entities")
        if not isinstance(entities, list):
            entities = _json_list(metadata.get("entities_json"))
        articles.append(
            RetrievedArticle(
                article_id=article_id,
                title=str(parent.get("title") or metadata.get("title") or ""),
                summary=str(parent.get("summary") or metadata.get("summary") or metadata.get("summary_raw") or ""),
                content=content,
                selected_context=selected_context,
                source_name=str(parent.get("source") or metadata.get("source") or ""),
                canonical_url=str(
                    parent.get("canonical_url")
                    or parent.get("source_url")
                    or parent.get("url")
                    or metadata.get("canonical_url")
                    or metadata.get("url")
                    or ""
                ),
                published_at=str(parent.get("published_at") or metadata.get("published_at") or ""),
                topic=str(parent.get("primary_topic") or metadata.get("primary_topic") or ""),
                entities=entities,
                images=images,
                matched_chunks=matched_chunks,
                relevance_score=max((_score(chunk) for chunk in matched_chunks), default=0.0),
            )
        )

    articles.sort(key=lambda article: (article.relevance_score, article.published_at), reverse=True)
    return [article.to_dict() for article in articles]


def build_sources_from_retrieved_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in articles:
        article_id = str(article.get("article_id") or "")
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        topic = str(article.get("topic") or "")
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "id": len(sources) + 1,
                "article_id": article_id,
                "title": str(article.get("title") or ""),
                "url": str(article.get("canonical_url") or ""),
                "source": str(article.get("source_name") or ""),
                "source_name": str(article.get("source_name") or ""),
                "published_at": str(article.get("published_at") or ""),
                "primary_topic": str(article.get("topic") or ""),
                "topic": topic,
                "domain": domain_from_topic(topic),
                "score": safe_float(article.get("relevance_score")),
                "snippet": _source_snippet(article),
            }
        )
    return sources


def build_related_articles_from_retrieved_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in articles:
        article_id = str(article.get("article_id") or "")
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        related.append(
            {
                "article_id": article_id,
                "title": str(article.get("title") or ""),
                "url": str(article.get("canonical_url") or ""),
                "source": str(article.get("source_name") or ""),
                "published_at": str(article.get("published_at") or ""),
                "primary_topic": str(article.get("topic") or ""),
            }
        )
    return related


def images_by_article_id(articles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(article.get("article_id") or ""): list(article.get("images") or [])
        for article in articles
        if str(article.get("article_id") or "")
    }


def article_metadata_by_id(articles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for article in articles:
        article_id = str(article.get("article_id") or "")
        if not article_id:
            continue
        metadata[article_id] = {
            "article_id": article_id,
            "citation_id": article.get("citation_id") or "",
            "title": str(article.get("title") or ""),
            "source": str(article.get("source_name") or ""),
            "url": str(article.get("canonical_url") or ""),
            "published_at": str(article.get("published_at") or ""),
            "primary_topic": str(article.get("topic") or ""),
        }
    return metadata


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _score(chunk: dict[str, Any]) -> float:
    return max(
        safe_float(chunk.get("final_score")),
        safe_float(chunk.get("score")),
        safe_float(chunk.get("vector_score")),
        safe_float(chunk.get("hybrid_score")),
    )


def _merge_chunk_texts(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = " ".join(str(chunk.get("text") or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " ".join(parts)


def _source_snippet(article: dict[str, Any], max_chars: int = 240) -> str:
    text = " ".join(str(article.get("selected_context") or article.get("summary") or article.get("content") or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


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


def _dedupe_images(images: list[Any]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for image in images:
        if not isinstance(image, dict):
            continue
        normalized = dict(image)
        image_url = _image_url_from_record(normalized)
        if image_url:
            normalized["image_url"] = image_url
            normalized.setdefault("url", image_url)
        if not image_url or image_url in seen or not is_valid_image(normalized):
            continue
        seen.add(image_url)
        deduped.append(normalized)
    return deduped


def _fallback_images(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    image_url = _image_url_from_record(metadata, include_url=False)
    if not image_url:
        return []
    return [
        {
            "article_id": str(metadata.get("article_id") or ""),
            "image_url": image_url,
            "url": image_url,
            "caption": str(metadata.get("image_caption") or metadata.get("caption") or ""),
            "credit": str(metadata.get("credit") or ""),
            "source": str(metadata.get("source") or ""),
            "article_title": str(metadata.get("title") or ""),
            "article_url": str(metadata.get("url") or ""),
        }
    ]


def _image_url_from_record(record: dict[str, Any], include_url: bool = True) -> str:
    keys = ["image_url", "src", "thumbnail_url", "thumbnailUrl", "thumb_url", "thumb", "thumbnail", "content_url", "contentUrl"]
    if include_url:
        keys.insert(1, "url")
    for key in keys:
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""
