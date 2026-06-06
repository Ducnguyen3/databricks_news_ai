from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_INVALID_IMAGE_KEYWORDS = (
    "logo",
    "banner",
    "avatar",
    "ad",
    "ads",
    "advertisement",
    "icon",
    "tracking",
    "pixel",
    "sprite",
    "share",
    "button",
    "placeholder",
)


class MediaRetriever:
    def __init__(
        self,
        image_repository: Any | None = None,
        metadata_images_by_article_id: dict[str, list[dict[str, Any]]] | None = None,
        article_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.image_repository = image_repository
        self._metadata_images_by_article_id = metadata_images_by_article_id or {}
        self._article_metadata_by_id = article_metadata_by_id or {}

    @classmethod
    def from_retrieval_results(
        cls,
        results: list[dict[str, Any]],
        image_repository: Any | None = None,
    ) -> "MediaRetriever":
        images_by_article: dict[str, list[dict[str, Any]]] = {}
        article_metadata_by_id: dict[str, dict[str, Any]] = {}
        for result in results:
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            article_id = str(metadata.get("article_id") or "")
            if not article_id:
                continue
            article_metadata_by_id.setdefault(article_id, dict(metadata))
            images_by_article.setdefault(article_id, [])
            images_by_article[article_id].extend(_json_list(metadata.get("images_json")))
            metadata_image_url = _metadata_image_url_from_record(metadata)
            if metadata_image_url:
                images_by_article[article_id].append(
                    {
                        "article_id": article_id,
                        "image_url": metadata_image_url,
                        "caption": metadata.get("image_caption") or metadata.get("caption") or "",
                        "credit": metadata.get("image_credit") or "",
                        "is_representative": metadata.get("is_representative") or True,
                    }
                )
        return cls(
            image_repository=image_repository,
            metadata_images_by_article_id=images_by_article,
            article_metadata_by_id=article_metadata_by_id,
        )

    def get_images_for_articles(
        self,
        article_ids: list[str],
        limit_per_article: int = 1,
        max_images: int | None = None,
        query_terms: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if not article_ids:
            return []
        normalized_article_ids = _dedupe_values([str(article_id) for article_id in article_ids if str(article_id)])
        images_by_article = {article_id: list(self._metadata_images_by_article_id.get(article_id, [])) for article_id in normalized_article_ids}
        missing_article_ids = [article_id for article_id in normalized_article_ids if not images_by_article.get(article_id)]
        if missing_article_ids and self.image_repository is not None:
            try:
                repository_images = self.image_repository.fetch_article_images(missing_article_ids)
                for image in repository_images:
                    article_id = str(image.get("article_id") or "")
                    if article_id:
                        images_by_article.setdefault(article_id, []).append(dict(image))
            except Exception as exc:
                logger.warning("Could not load images from image repository: %s", exc)

        output: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        max_output = max(1, int(max_images)) if max_images is not None else None
        normalized_query_terms = [_normalize_text(term) for term in query_terms or [] if str(term).strip()]
        for article_id in normalized_article_ids:
            article_images = [
                _normalize_image(image, self._article_metadata_by_id.get(article_id, {}))
                for image in images_by_article.get(article_id, [])
            ]
            valid_images = [image for image in article_images if _is_valid_image(image)]
            valid_images.sort(key=lambda image: _image_priority(image, normalized_query_terms))
            per_article_count = 0
            for image in valid_images:
                image_url = str(image.get("image_url") or "")
                if image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                output.append(image)
                per_article_count += 1
                if max_output is not None and len(output) >= max_output:
                    return output
                if per_article_count >= max(1, int(limit_per_article)):
                    break
        return output


def _normalize_image(image: Mapping[str, Any], article_metadata: Mapping[str, Any]) -> dict[str, Any]:
    article_id = str(image.get("article_id") or article_metadata.get("article_id") or "")
    image_url = _image_url_from_record(image)
    return {
        "article_id": article_id,
        "citation_id": article_metadata.get("citation_id") or image.get("citation_id") or "",
        "image_url": image_url,
        "url": image_url,
        "caption": str(image.get("caption") or image.get("alt_text") or image.get("alt") or ""),
        "credit": str(image.get("credit") or ""),
        "is_representative": _truthy(image.get("is_representative")),
        "source": str(image.get("source") or article_metadata.get("source") or ""),
        "article_title": str(image.get("article_title") or image.get("title") or article_metadata.get("title") or ""),
        "article_url": str(
            image.get("article_url")
            or image.get("canonical_url")
            or image.get("source_url")
            or article_metadata.get("url")
            or article_metadata.get("canonical_url")
            or ""
        ),
        "published_at": str(image.get("published_at") or article_metadata.get("published_at") or ""),
        "topic": str(image.get("topic") or article_metadata.get("primary_topic") or article_metadata.get("topic") or ""),
        "score": _float_or_none(image.get("score") or article_metadata.get("score")) or 0.0,
        "width": _int_or_none(image.get("width")),
        "height": _int_or_none(image.get("height")),
        "position": _int_or_none(image.get("position")) or 0,
        "type": "original",
    }


def _is_valid_image(image: Mapping[str, Any]) -> bool:
    image_url = str(image.get("image_url") or "").strip()
    if not image_url or image_url.startswith("data:"):
        return False
    lowered = image_url.lower()
    if lowered.endswith(".svg") or ".svg?" in lowered:
        return False
    if any(keyword in lowered for keyword in _INVALID_IMAGE_KEYWORDS):
        return False
    width = _int_or_none(image.get("width"))
    height = _int_or_none(image.get("height"))
    if width is not None and height is not None and width <= 2 and height <= 2:
        return False
    return True


def is_valid_image(image: Mapping[str, Any]) -> bool:
    return _is_valid_image(image)


def _image_url_from_record(image: Mapping[str, Any]) -> str:
    for key in ("image_url", "url", "src", "thumbnail_url", "thumbnailUrl", "thumb_url", "thumb", "thumbnail", "content_url", "contentUrl"):
        value = str(image.get(key) or "").strip()
        if value:
            return value
    return ""


def _metadata_image_url_from_record(metadata: Mapping[str, Any]) -> str:
    for key in ("image_url", "thumbnail_url", "thumbnailUrl", "thumb_url", "thumb", "thumbnail", "content_url", "contentUrl"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _image_priority(image: Mapping[str, Any], query_terms: list[str] | tuple[str, ...] = ()) -> tuple[int, int, int, int]:
    representative_rank = 0 if _truthy(image.get("is_representative")) else 1
    caption = str(image.get("caption") or "")
    caption_norm = _normalize_text(caption)
    query_rank = 0 if query_terms and any(term and term in caption_norm for term in query_terms) else 1
    caption_rank = 0 if caption.strip() else 1
    position_rank = _int_or_none(image.get("position")) or 0
    return query_rank, representative_rank, caption_rank, position_rank


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _dedupe_values(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")
