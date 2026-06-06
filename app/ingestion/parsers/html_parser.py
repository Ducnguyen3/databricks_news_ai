from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from app.domain.models import Article
from app.processing.canonicalizer import normalize_url
from app.processing.cleaner import clean_text, strip_html_tags
from app.processing.deduplicator import hash_content
from app.processing.entity_extractor import extract_entities
from app.processing.taxonomy import normalize_topic
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

_PUBLISHED_AT_SELECTORS = {
    "vnexpress": (".date", ".time", ".article-date", "time"),
    "genk": (".kbwcm-time", ".knc-date", ".detail-time", ".news-date", ".time", "time"),
    "diendandoanhnghiep": (
        ".detail-date",
        ".article-date",
        ".detail__time",
        ".b-maincontent__time",
        ".block-sc-publish-time",
        ".sc-longform-header-date",
        ".time-public",
        ".date-public",
        ".time",
        ".date",
        "time",
    ),
    "cafef": (".time", ".pdate", ".date", "time"),
}

_COMMON_PUBLISHED_AT_SELECTORS = (
    "time",
    ".date",
    ".time",
    ".article-date",
    ".detail-date",
    ".pdate",
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
    article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{canonical_url or content_hash}"))
    category = clean_text(str(payload.get("category") or rss.get("category") or "")) or None
    source_category_name = clean_text(str(payload.get("source_category_name") or payload.get("category_name") or "")) or None
    source_category_url = normalize_url(str(payload.get("source_category_url") or payload.get("category_url") or ""))
    topic = normalize_topic(
        source=source,
        source_category=category,
        title=title,
        summary=summary_raw,
        content=content,
    )
    entity_result = extract_entities(
        title=title,
        summary=summary_raw,
        content=content,
        source_category=category,
    )
    published_at_raw = (
        payload.get("published_at")
        or rss.get("published_raw")
        or _extract_published_at(html, source)
    )
    return Article(
        article_id=article_id,
        source=source,
        url=url,
        canonical_url=canonical_url,
        title=title,
        summary_raw=summary_raw,
        content=content,
        category=category,
        source_category_name=source_category_name,
        source_category_url=source_category_url or None,
        published_at=parse_datetime(str(published_at_raw or "")),
        crawled_at=_as_datetime(raw_document.get("fetched_at")) or now,
        content_hash=content_hash,
        dedup_group_id=article_id,
        is_duplicate=False,
        created_at=now,
        updated_at=now,
        raw_id=str(raw_document.get("raw_document_id") or raw_document.get("raw_id") or ""),
        primary_topic=str(topic["primary_topic"]),
        primary_topic_name=str(topic["primary_topic_name"]),
        topic_confidence=float(topic["topic_confidence"]),
        secondary_topics_json=json.dumps(topic["secondary_topics"], ensure_ascii=False),
        entities_json=json.dumps(entity_result["entities"], ensure_ascii=False),
    )


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = _parse_standard_datetime(text) or _parse_vietnamese_datetime(text)
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_standard_datetime(text: str) -> datetime | None:
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None


def _parse_vietnamese_datetime(text: str) -> datetime | None:
    normalized = clean_text(strip_html_tags(text))
    if not normalized:
        return None

    timezone_offset = _timezone_offset(normalized)
    separator = r"(?:\s|,|\||-|–|—|lúc|luc)+"
    patterns = (
        rf"(?P<day>\d{{1,2}})[/-](?P<month>\d{{1,2}})[/-](?P<year>\d{{4}}){separator}(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})(?::(?P<second>\d{{2}}))?\s*(?P<ampm>AM|PM|SA|CH)?",
        rf"(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})(?::(?P<second>\d{{2}}))?\s*(?P<ampm>AM|PM|SA|CH)?{separator}(?P<day>\d{{1,2}})[/-](?P<month>\d{{1,2}})[/-](?P<year>\d{{4}})",
        rf"(?P<month>\d{{1,2}})[/-](?P<day>\d{{1,2}})[/-](?P<year>\d{{4}}){separator}(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})(?::(?P<second>\d{{2}}))?\s*(?P<ampm>AM|PM)?",
        rf"(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})(?::(?P<second>\d{{2}}))?\s*(?P<ampm>AM|PM)?{separator}(?P<month>\d{{1,2}})[/-](?P<day>\d{{1,2}})[/-](?P<year>\d{{4}})",
        r"(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        values = match.groupdict(default="0")
        try:
            parsed = datetime(
                int(values["year"]),
                int(values["month"]),
                int(values["day"]),
                _hour_24(int(values.get("hour") or 0), values.get("ampm") or ""),
                int(values.get("minute") or 0),
                int(values.get("second") or 0),
            )
        except ValueError:
            continue
        if timezone_offset is not None:
            parsed = parsed.replace(tzinfo=timezone(timezone_offset))
        return parsed
    return None


def _hour_24(hour: int, ampm: str) -> int:
    marker = ampm.strip().casefold()
    if marker in {"pm", "ch"} and hour < 12:
        return hour + 12
    if marker in {"am", "sa"} and hour == 12:
        return 0
    return hour


def _extract_datetime_text(text: str) -> str:
    normalized = clean_text(strip_html_tags(text))
    if not normalized:
        return ""
    patterns = (
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}(?:\s|,|\||-|–|—|lúc|luc)+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|SA|CH)?(?:\s*\(?GMT\s*[+-]\s*\d{1,2}:?\d{0,2}\)?)?",
        r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|SA|CH)?(?:\s|,|\||-|–|—|lúc|luc)+\d{1,2}[/-]\d{1,2}[/-]\d{4}(?:\s*\(?GMT\s*[+-]\s*\d{1,2}:?\d{0,2}\)?)?",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _timezone_offset(text: str) -> timedelta | None:
    match = re.search(r"GMT\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text, re.IGNORECASE)
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3) or 0)
    return sign * timedelta(hours=hours, minutes=minutes)


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


def _extract_published_at(html: str, source: str) -> str:
    if not html:
        return ""
    soup = _soup(html)
    if soup is None:
        return ""

    for attrs in (
        {"property": "article:published_time"},
        {"name": "article:published_time"},
        {"property": "article:modified_time"},
        {"name": "article:modified_time"},
        {"name": "pubdate"},
        {"name": "publishdate"},
        {"name": "datePublished"},
        {"name": "parsely-pub-date"},
        {"name": "date"},
        {"itemprop": "datePublished"},
        {"itemprop": "dateModified"},
    ):
        node = soup.find("meta", attrs=attrs)
        if node is not None:
            value = clean_text(str(node.get("content") or ""))
            if value:
                return value

    for node in soup.select("time[datetime], time[content]"):
        value = clean_text(str(node.get("datetime") or node.get("content") or ""))
        if value:
            return value

    selectors = _PUBLISHED_AT_SELECTORS.get(source, ()) + _COMMON_PUBLISHED_AT_SELECTORS
    for selector in selectors:
        node = soup.select_one(selector)
        if node is None:
            continue
        value = clean_text(str(node.get("datetime") or node.get("content") or node.get_text(" ", strip=True) or ""))
        if value:
            return value
    return _extract_datetime_text(soup.get_text(" ", strip=True))


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
