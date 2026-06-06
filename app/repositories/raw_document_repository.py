from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from app.config import TableNames
from app.databricks.delta_tables import raw_documents_schema
from app.domain.models import RawDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExistingArticleKeys:
    link_keys: set[str]
    content_hashes: set[str]
    exact_document_keys: set[str]


class RawDocumentRepository:
    def __init__(self, spark: Any, tables: TableNames) -> None:
        self._spark = spark
        self._tables = tables

    def upsert(self, documents: list[RawDocument]) -> int:
        if not documents:
            logger.info("[CRAWL] Saved 0 documents")
            return 0

        schema = raw_documents_schema()
        rows = [asdict(document) for document in documents]
        updates = _records_to_dataframe(self._spark, rows, schema)
        view_name = _temp_view_name("raw_documents_updates")
        updates.createOrReplaceTempView(view_name)
        self._spark.sql(
            f"""
            MERGE INTO {self._tables.raw_documents_fqn} AS target
            USING {view_name} AS source
            ON target.source_name = source.source_name
               AND (
                   target.canonical_url = source.canonical_url
                   OR target.url = source.url
               )
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
        logger.info("[CRAWL] Saved %s documents table=%s", len(documents), self._tables.raw_documents_fqn)
        return len(documents)

    def insert_batch(self, documents: list[RawDocument]) -> int:
        return self.upsert(documents)

    def read_all(self) -> list[dict[str, Any]]:
        rows = self._spark.table(self._tables.raw_documents_fqn).collect()
        return [row.asDict(recursive=True) for row in rows]

    def get_existing_article_keys(self, source_name: str) -> ExistingArticleKeys:
        from app.ingestion.crawlers.base_crawler import exact_document_identity_key, raw_document_identity_key

        rows = self._spark.sql(
            f"""
            SELECT url, canonical_url, checksum
            FROM {self._tables.raw_documents_fqn}
            WHERE source_name = '{source_name.replace("'", "''")}'
            """
        ).collect()

        link_keys: set[str] = set()
        content_hashes: set[str] = set()
        exact_document_keys: set[str] = set()
        for row in rows:
            row_dict = row.asDict(recursive=True)
            url = row_dict.get("url")
            canonical_url = row_dict.get("canonical_url")
            checksum = row_dict.get("checksum")
            if url:
                link_keys.add(raw_document_identity_key(source_name, str(url)))
            if canonical_url:
                link_keys.add(raw_document_identity_key(source_name, str(canonical_url)))
            if checksum:
                content_hashes.add(str(checksum))
            if canonical_url and checksum:
                exact_document_keys.add(exact_document_identity_key(source_name, str(canonical_url), str(checksum)))
        logger.info(
            "[CRAWL] Loaded existing keys source=%s links=%s hashes=%s exact_documents=%s",
            source_name,
            len(link_keys),
            len(content_hashes),
            len(exact_document_keys),
        )
        return ExistingArticleKeys(
            link_keys=link_keys,
            content_hashes=content_hashes,
            exact_document_keys=exact_document_keys,
        )


def _records_to_dataframe(spark: Any, records: list[dict[str, Any]], schema: Any) -> Any:
    field_names = [field.name for field in schema.fields]
    ordered_rows = [tuple(record.get(field_name) for field_name in field_names) for record in records]
    return spark.createDataFrame(ordered_rows, schema=schema)


def _temp_view_name(prefix: str) -> str:
    return f"_{prefix}_{uuid.uuid4().hex}"


RawDocumentsRepository = RawDocumentRepository
