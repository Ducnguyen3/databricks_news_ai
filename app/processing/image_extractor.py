from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from app.domain.models import Article, ArticleImage
from app.processing.cleaner import clean_text, strip_html_tags
from app.utils.time import utc_now

_INVALID_IMAGE_KEYWORDS = (
    "logo",
    "icon",
    "avatar",
    "banner",
    "ads",
    "advertisement",
    "tracking",
    "pixel",
    "sprite",
    "social",
    "share",
    "button",
    "placeholder",
)
_SOURCE_ATTRS = (
    "data-src",
    "data-srcset",
    "data-original",
    "data-original-src",
    "data-original-set",
    "data-lazy-src",
    "data-lazy-srcset",
    "data-url",
    "data-medium",
    "data-thumb",
    "src",
)
_CAPTION_CLASS_RE = re.compile(r"(caption|photocms_caption|image|figcaption)", re.IGNORECASE)
_CREDIT_CLASS_RE = re.compile(r"(credit|author|source|photo-author|photographer)", re.IGNORECASE)


def extract_article_images(raw_document: Mapping[str, Any], article: Article) -> list[ArticleImage]:
    html = _raw_html(raw_document)
    if not html:
        return []

    soup = _soup(html)
    if soup is None:
        return []

    base_url = article.canonical_url or article.url
    image_candidates = _image_nodes(soup)
    images: list[ArticleImage] = []
    seen_urls: set[str] = set()

    for node in image_candidates:
        image_url = _extract_image_url(node, base_url)
        if not image_url or image_url in seen_urls:
            continue
        if _is_invalid_image(node, image_url):
            continue
        seen_urls.add(image_url)
        position = len(images)
        images.append(
            ArticleImage(
                id=_image_id(article.article_id, image_url, position),
                article_id=article.article_id,
                source=article.source,
                canonical_url=article.canonical_url,
                image_url=image_url,
                caption=_extract_caption(node),
                alt_text=clean_text(str(node.get("alt") or "")) or None,
                credit=_extract_credit(node),
                position=position,
                width=_int_attr(node, "width") or _int_attr(node, "data-width"),
                height=_int_attr(node, "height") or _int_attr(node, "data-height"),
                image_type="body",
                is_representative=False,
                created_at=utc_now(),
            )
        )

    if not images:
        images.extend(_document_images(soup, article, seen_urls))
    if not images:
        images.extend(_payload_images(raw_document, article, seen_urls))

    representative_index = _representative_index(images)
    if representative_index is None:
        return images
    for index, image in enumerate(images):
        image.is_representative = index == representative_index
    return images


def _raw_html(raw_document: Mapping[str, Any]) -> str:
    payload = raw_document.get("raw_payload") or raw_document.get("payload")
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(payload, dict):
        parsed = payload
    else:
        parsed = {}
    html = parsed.get("html") if isinstance(parsed, dict) else None
    return html if isinstance(html, str) else ""


def _image_nodes(soup: Any) -> list[Any]:
    nodes: list[Any] = []
    seen: set[int] = set()
    for selector in ("article img", ".article-content img", ".content-detail img", ".detail-content img", ".post-content img", "figure img", "img"):
        for node in soup.select(selector):
            identity = id(node)
            if identity not in seen:
                seen.add(identity)
                nodes.append(node)
    return nodes


def _extract_image_url(node: Any, base_url: str) -> str:
    for value in _candidate_image_values(node):
        if _is_placeholder_image_value(value):
            continue
        return urljoin(base_url, value)
    return ""


def _candidate_image_values(node: Any) -> list[str]:
    values: list[str] = []
    for attr in _SOURCE_ATTRS:
        raw_value = str(node.get(attr) or "").strip()
        value = _first_srcset_url(raw_value) if "srcset" in attr else raw_value
        if value:
            values.append(value)
    srcset_value = _first_srcset_url(str(node.get("srcset") or ""))
    if srcset_value:
        values.append(srcset_value)
    parent = node.parent
    if parent is not None:
        for source in parent.find_all("source"):
            value = _first_srcset_url(str(source.get("srcset") or "")) or str(source.get("src") or "").strip()
            if value:
                values.append(value)
    return values


def _is_placeholder_image_value(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized or normalized.startswith("data:"):
        return True
    return any(keyword in normalized for keyword in ("loading", "placeholder", "blank.gif", "pixel.gif"))


def _payload_images(raw_document: Mapping[str, Any], article: Article, seen_urls: set[str]) -> list[ArticleImage]:
    payload = raw_document.get("raw_payload") or raw_document.get("payload")
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(payload, dict):
        parsed = payload
    else:
        parsed = {}

    image_values = _payload_image_values(parsed.get("image") if isinstance(parsed, dict) else None)
    images: list[ArticleImage] = []
    for raw_url in image_values:
        image_url = urljoin(article.canonical_url or article.url, raw_url)
        if not image_url or image_url in seen_urls or _is_placeholder_image_value(image_url):
            continue
        seen_urls.add(image_url)
        position = len(images)
        images.append(
            ArticleImage(
                id=_image_id(article.article_id, image_url, position),
                article_id=article.article_id,
                source=article.source,
                canonical_url=article.canonical_url,
                image_url=image_url,
                position=position,
                image_type="representative",
                is_representative=False,
                created_at=utc_now(),
            )
        )
    return images


def _document_images(soup: Any, article: Article, seen_urls: set[str]) -> list[ArticleImage]:
    base_url = article.canonical_url or article.url
    values: list[str] = []
    selectors = (
        "meta[property='og:image']",
        "meta[property='og:image:url']",
        "meta[name='twitter:image']",
        "meta[name='twitter:image:src']",
        "meta[itemprop='thumbnailUrl']",
        "meta[itemprop='contentUrl']",
        "link[rel='image_src']",
    )
    for selector in selectors:
        for node in soup.select(selector):
            value = str(node.get("content") or node.get("href") or "").strip()
            if value:
                values.append(value)

    for script in soup.select("script[type='application/ld+json']"):
        values.extend(_json_ld_image_values(script.get_text(" ", strip=True)))

    html_text = str(soup)
    for pattern in (
        r'"(?:image|imageUrl|thumbnailUrl)"\s*:\s*"([^"]+)"',
        r"'(?:image|imageUrl|thumbnailUrl)'\s*:\s*'([^']+)'",
    ):
        values.extend(match.group(1) for match in re.finditer(pattern, html_text))

    return _image_records_from_values(values, article, base_url, seen_urls, image_type="metadata")


def _json_ld_image_values(text: str) -> list[str]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _extract_image_values_from_json(data)


def _extract_image_values_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_extract_image_values_from_json(item))
        return output
    if not isinstance(value, dict):
        return []

    output: list[str] = []
    for key in ("image", "imageUrl", "thumbnailUrl", "url", "contentUrl"):
        if key in value:
            output.extend(_extract_image_values_from_json(value[key]))
    return output


def _payload_image_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_payload_image_values(item))
        return output
    if isinstance(value, dict):
        for key in ("url", "src", "image_url", "thumb", "thumbnail"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return [item]
    return []


def _image_records_from_values(
    values: list[str],
    article: Article,
    base_url: str,
    seen_urls: set[str],
    image_type: str,
) -> list[ArticleImage]:
    images: list[ArticleImage] = []
    for raw_url in values:
        if not isinstance(raw_url, str):
            continue
        image_url = urljoin(base_url, raw_url.strip())
        if not image_url or image_url in seen_urls or _is_placeholder_image_value(image_url):
            continue
        if any(keyword in image_url.lower() for keyword in _INVALID_IMAGE_KEYWORDS):
            continue
        seen_urls.add(image_url)
        position = len(images)
        images.append(
            ArticleImage(
                id=_image_id(article.article_id, image_url, position),
                article_id=article.article_id,
                source=article.source,
                canonical_url=article.canonical_url,
                image_url=image_url,
                position=position,
                image_type=image_type,
                is_representative=False,
                created_at=utc_now(),
            )
        )
    return images


def _first_srcset_url(srcset: str) -> str:
    if not srcset.strip():
        return ""
    first = srcset.split(",", maxsplit=1)[0].strip()
    return first.split()[0].strip() if first else ""


def _is_invalid_image(node: Any, image_url: str) -> bool:
    haystack = " ".join(
        [
            image_url,
            str(node.get("class") or ""),
            str(node.get("id") or ""),
            str(node.get("alt") or ""),
        ]
    ).lower()
    if any(keyword in haystack for keyword in _INVALID_IMAGE_KEYWORDS):
        return True
    width = _int_attr(node, "width") or _int_attr(node, "data-width")
    height = _int_attr(node, "height") or _int_attr(node, "data-height")
    return bool(width is not None and height is not None and width <= 2 and height <= 2)


def _extract_caption(node: Any) -> str | None:
    figure = _closest(node, "figure")
    if figure is not None:
        figcaption = figure.find("figcaption")
        if figcaption is not None:
            text = clean_text(figcaption.get_text(" ", strip=True))
            if text:
                return text
        caption = figure.find(class_=_CAPTION_CLASS_RE)
        if caption is not None:
            text = clean_text(caption.get_text(" ", strip=True))
            if text:
                return text

    parent = node.parent
    if parent is not None:
        caption = parent.find(class_=_CAPTION_CLASS_RE)
        if caption is not None:
            text = clean_text(caption.get_text(" ", strip=True))
            if text:
                return text
        next_node = parent.find_next_sibling()
        if next_node is not None and _CAPTION_CLASS_RE.search(" ".join(str(item) for item in next_node.get("class", []))):
            text = clean_text(next_node.get_text(" ", strip=True))
            if text:
                return text
    return None


def _extract_credit(node: Any) -> str | None:
    figure = _closest(node, "figure") or node.parent
    if figure is None:
        return None
    credit = figure.find(class_=_CREDIT_CLASS_RE)
    if credit is None:
        return None
    text = clean_text(strip_html_tags(credit.get_text(" ", strip=True)))
    return text or None


def _representative_index(images: list[ArticleImage]) -> int | None:
    if not images:
        return None

    def score(image: ArticleImage) -> tuple[int, int, int]:
        has_caption = 1 if image.caption else 0
        has_reasonable_size = 1 if _has_reasonable_size(image) else 0
        return has_caption, has_reasonable_size, -image.position

    return max(range(len(images)), key=lambda index: score(images[index]))


def _has_reasonable_size(image: ArticleImage) -> bool:
    if image.width is None or image.height is None:
        return False
    return image.width >= 200 and image.height >= 120


def _int_attr(node: Any, attr: str) -> int | None:
    value = node.get(attr)
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return int(match.group(0))


def _closest(node: Any, tag_name: str) -> Any | None:
    current = node
    while current is not None:
        if getattr(current, "name", None) == tag_name:
            return current
        current = getattr(current, "parent", None)
    return None


def _image_id(article_id: str, image_url: str, position: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article_id}|{image_url}|{position}"))


def _soup(html: str) -> Any | None:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    return BeautifulSoup(html, "html.parser")
