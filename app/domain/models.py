from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RawDocument:
    raw_document_id: str
    source_name: str
    crawl_run_id: str
    url: str
    canonical_url: str
    page_type: str
    fetch_status: str
    http_status: int | None
    fetched_at: datetime
    payload_format: str
    raw_payload: str
    raw_text: str
    checksum: str
    metadata: str
    created_at: datetime

    @property
    def raw_id(self) -> str:
        return self.raw_document_id

    @property
    def source(self) -> str:
        return self.source_name

    @property
    def payload(self) -> str:
        return self.raw_payload

    @property
    def payload_type(self) -> str:
        return self.payload_format

    @property
    def status_code(self) -> int | None:
        return self.http_status

    @property
    def metadata_json(self) -> str:
        return self.metadata


@dataclass(slots=True)
class ParsedArticle:
    source_name: str
    url: str
    canonical_url: str
    title: str
    summary: str | None
    content: str
    category: str | None
    published_at: str | None
    author: str | None
    image: str | None
    category_url: str | None = None
    category_page: int | None = None
    crawled_at: datetime | None = None
    content_hash: str | None = None
    tags: list[str] = field(default_factory=list)
    raw_html: str = ""
    http_status: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Article:
    article_id: str
    source: str
    url: str
    canonical_url: str
    title: str
    summary_raw: str | None
    content: str
    category: str | None
    published_at: datetime | None
    crawled_at: datetime
    content_hash: str
    dedup_group_id: str
    is_duplicate: bool
    created_at: datetime
    updated_at: datetime
    raw_id: str | None = None
