from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.domain.models import Article
from app.processing.canonicalizer import normalize_url
from app.processing.cleaner import clean_text, strip_html_tags
from app.processing.deduplicator import hash_content
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

_SOURCE_SELECTORS = {
    "vnexpress": (".fck_detail", "article.fck_detail", ".Normal"),
    "cafef": (".detail-content", ".newscontent", "#mainContent"),
    "genk": (".knc-content", ".detail-content", ".content"),
}

_COMMON_SELECTORS = (
    "article",
    ".article-content",
    ".content-detail",
    ".detail-content",
    ".post-content",
)


def parse_raw_document(raw_document: Mapping[str, Any]) -> Article | None:
    payload = _read_payload(raw_document.get("raw_payload") or raw_document.get("payload"))
    rss = payload.get("rss", {}) if isinstance(payload.get("rss"), dict) else {}
    html = payload.get("html", "") if isinstance(payload.get("html"), str) else ""

    source = str(raw_document.get("source_name") or raw_document.get("source") or "")
    url = str(raw_document.get("url") or rss.get("link") or "")
    canonical_url = normalize_url(str(raw_document.get("canonical_url") or url))
    title = clean_text(str(payload.get("title") or rss.get("title") or _extract_title(html) or ""))
    summary_raw = clean_text(strip_html_tags(str(payload.get("summary") or rss.get("summary") or ""))) or None
    content = clean_text(str(payload.get("content") or "")) or clean_text(_extract_content(html, source)) or clean_text(summary_raw)

    if not canonical_url or not title or not content:
        logger.warning("Skipping raw document because required fields are missing: %s", raw_document.get("raw_id"))
        return None

    now = utc_now()
    content_hash = hash_content(content)
    article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_url or content_hash))
    return Article(
        article_id=article_id,
        source=source,
        url=url,
        canonical_url=canonical_url,
        title=title,
        summary_raw=summary_raw,
        content=content,
        category=clean_text(str(payload.get("category") or rss.get("category") or "")) or None,
        published_at=parse_datetime(str(payload.get("published_at") or rss.get("published_raw") or "")),
        crawled_at=_as_datetime(raw_document.get("fetched_at")) or now,
        content_hash=content_hash,
        dedup_group_id=article_id,
        is_duplicate=False,
        created_at=now,
        updated_at=now,
        raw_id=str(raw_document.get("raw_document_id") or raw_document.get("raw_id") or ""),
    )


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _read_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Invalid raw payload JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_title(html: str) -> str:
    soup = _soup(html)
    if soup is not None and soup.title is not None:
        return soup.title.get_text(" ", strip=True)
    return ""


def _extract_content(html: str, source: str) -> str:
    if not html:
        return ""

    soup = _soup(html)
    if soup is None:
        return strip_html_tags(html)

    for tag_name in ("script", "style", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    selectors = _SOURCE_SELECTORS.get(source, ()) + _COMMON_SELECTORS
    for selector in selectors:
        node = soup.select_one(selector)
        if node is None:
            continue
        paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in node.find_all(["p", "h2"])]
        text = clean_text(" ".join(part for part in paragraphs if part))
        if len(text) >= 200:
            return text

    paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    return clean_text(" ".join(part for part in paragraphs if part))


def _soup(html: str) -> Any | None:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    return BeautifulSoup(html, "html.parser")


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        return parse_datetime(value)
    return None
