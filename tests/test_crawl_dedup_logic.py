from __future__ import annotations

import unittest
import json
from datetime import datetime
from types import SimpleNamespace

from app.config import TableNames
from app.domain.models import RawDocument
from app.ingestion.crawlers.base_crawler import BaseCrawler, exact_document_identity_key, raw_document_identity_key
from app.ingestion.parsers.html_parser import parse_raw_document
from app.repositories.raw_document_repository import RawDocumentRepository


class _Crawler(BaseCrawler):
    source_name = "vnexpress"


class _Row:
    def __init__(self, **values):
        self._values = values

    def asDict(self, recursive: bool = True):
        return dict(self._values)


class _DataFrame:
    def createOrReplaceTempView(self, name: str) -> None:
        self.view_name = name


class _Spark:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []
        self.sql_calls: list[str] = []

    def sql(self, query: str):
        self.sql_calls.append(query)
        return SimpleNamespace(collect=lambda: self.rows)

    def createDataFrame(self, rows, schema):
        return _DataFrame()


def _config():
    return SimpleNamespace(timeout_seconds=1, retry_count=0, user_agent="test", request_delay_seconds=0)


def _raw_document(source: str, canonical_url: str, url: str, checksum: str) -> RawDocument:
    now = datetime(2026, 1, 1)
    return RawDocument(
        raw_document_id=f"{source}-{canonical_url}",
        source_name=source,
        crawl_run_id="run-1",
        url=url,
        canonical_url=canonical_url,
        page_type="article",
        fetch_status="success",
        http_status=200,
        fetched_at=now,
        payload_format="json",
        raw_payload="{}",
        raw_text="content",
        checksum=checksum,
        metadata="{}",
        created_at=now,
    )


class CrawlDedupLogicTest(unittest.TestCase):
    def test_exact_existing_same_source_canonical_and_checksum_is_skipped(self) -> None:
        crawler = _Crawler(_config())
        crawler.set_existing_keys(
            exact_document_keys={
                exact_document_identity_key("vnexpress", "https://example.com/a", "checksum-x")
            }
        )

        self.assertTrue(crawler.should_skip_exact_existing_document("https://example.com/a", "checksum-x"))

    def test_same_checksum_different_source_is_not_a_raw_merge_match(self) -> None:
        spark = _Spark()
        repository = RawDocumentRepository(spark, TableNames())

        repository.upsert(
            [
                _raw_document("vnexpress", "https://vnexpress.net/a", "https://vnexpress.net/a", "checksum-x"),
                _raw_document("cafef", "https://cafef.vn/b", "https://cafef.vn/b", "checksum-x"),
            ]
        )

        merge_sql = "\n".join(spark.sql_calls)
        self.assertIn("target.source_name = source.source_name", merge_sql)
        self.assertIn("target.canonical_url = source.canonical_url", merge_sql)
        self.assertNotIn("target.checksum = source.checksum", merge_sql)

    def test_same_source_different_canonical_same_checksum_is_kept(self) -> None:
        crawler = _Crawler(_config())
        crawler.set_existing_keys(
            link_keys={raw_document_identity_key("vnexpress", "https://example.com/a")},
            content_hashes={"checksum-x"},
            exact_document_keys={
                exact_document_identity_key("vnexpress", "https://example.com/a", "checksum-x")
            },
        )

        self.assertFalse(crawler.should_skip_exact_existing_document("https://example.com/b", "checksum-x"))

    def test_same_source_same_canonical_different_checksum_is_kept_for_update(self) -> None:
        crawler = _Crawler(_config())
        crawler.set_existing_keys(
            content_hashes={"checksum-x"},
            exact_document_keys={
                exact_document_identity_key("vnexpress", "https://example.com/a", "checksum-x")
            },
        )

        self.assertFalse(crawler.should_skip_exact_existing_document("https://example.com/a", "checksum-y"))

    def test_existing_article_keys_include_exact_documents_without_global_checksum_key(self) -> None:
        spark = _Spark(
            rows=[
                _Row(url="https://example.com/a?utm_source=x", canonical_url="https://example.com/a", checksum="checksum-x")
            ]
        )
        repository = RawDocumentRepository(spark, TableNames())

        keys = repository.get_existing_article_keys("vnexpress")

        self.assertIn(raw_document_identity_key("vnexpress", "https://example.com/a"), keys.link_keys)
        self.assertIn(
            exact_document_identity_key("vnexpress", "https://example.com/a", "checksum-x"),
            keys.exact_document_keys,
        )
        self.assertNotIn("checksum-x", keys.link_keys)

    def test_parsed_article_id_keeps_same_canonical_url_from_different_sources_distinct(self) -> None:
        payload = json.dumps(
            {
                "title": "Shared story",
                "content": "This is a long enough article body for parsing and downstream dedup testing.",
            }
        )
        first = parse_raw_document(
            {
                "raw_document_id": "raw-1",
                "source_name": "vnexpress",
                "url": "https://example.com/a",
                "canonical_url": "https://example.com/a",
                "raw_payload": payload,
                "fetched_at": datetime(2026, 1, 1),
            }
        )
        second = parse_raw_document(
            {
                "raw_document_id": "raw-2",
                "source_name": "cafef",
                "url": "https://example.com/a",
                "canonical_url": "https://example.com/a",
                "raw_payload": payload,
                "fetched_at": datetime(2026, 1, 1),
            }
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.article_id, second.article_id)


if __name__ == "__main__":
    unittest.main()
