from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def enrich_articles_with_images(
    articles: list[dict[str, Any]],
    images: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    images_by_article_id = _group_images(images, "article_id")
    images_by_canonical_url = _group_images(images, "canonical_url")
    enriched: list[dict[str, Any]] = []

    for article in articles:
        row = dict(article)
        article_id = str(row.get("article_id") or "")
        canonical_url = str(row.get("canonical_url") or "")
        article_images = images_by_article_id.get(article_id, [])
        if not article_images and canonical_url:
            article_images = images_by_canonical_url.get(canonical_url, [])
        deduped_images = _dedupe_images(article_images)
        row["images"] = list(deduped_images)
        row["images_json"] = json.dumps(deduped_images, ensure_ascii=False)
        row["has_images"] = bool(deduped_images)
        row["image_count"] = len(deduped_images)
        enriched.append(row)
    return enriched


def _group_images(images: list[Mapping[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        value = str(image.get(key) or "")
        if not value:
            continue
        grouped.setdefault(value, []).append(_normalize_image(image))
    for rows in grouped.values():
        rows.sort(key=lambda item: (not bool(item.get("is_representative")), int(item.get("position") or 0)))
    return grouped


def _normalize_image(image: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "article_id": str(image.get("article_id") or ""),
        "image_url": str(image.get("image_url") or ""),
        "caption": str(image.get("caption") or ""),
        "alt": str(image.get("alt") or image.get("alt_text") or ""),
        "alt_text": str(image.get("alt_text") or image.get("alt") or ""),
        "credit": str(image.get("credit") or ""),
        "source": str(image.get("source") or ""),
        "canonical_url": str(image.get("canonical_url") or ""),
        "position": int(image.get("position") or 0),
        "width": _int_or_none(image.get("width")),
        "height": _int_or_none(image.get("height")),
        "is_representative": bool(image.get("is_representative")),
    }


def _dedupe_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for image in images:
        image_url = str(image.get("image_url") or "")
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        deduped.append(image)
    return deduped


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
