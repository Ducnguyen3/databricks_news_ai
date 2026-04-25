from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any

from app.config import TableNames
from app.databricks.delta_tables import articles_clean_schema, news_articles_schema
from app.domain.models import Article
from app.repositories.raw_document_repository import RawDocumentRepository, RawDocumentsRepository

logger = logging.getLogger(__name__)


class NewsArticlesRepository:
    def __init__(self, spark: Any, tables: TableNames) -> None:
        self._spark = spark
        self._tables = tables

    def upsert(self, articles: list[Article]) -> None:
        if not articles:
            logger.info("No parsed articles to write")
            return
        schema = news_articles_schema()
        rows = [asdict(article) for article in articles]
        updates = _records_to_dataframe(self._spark, rows, schema)
        view_name = _temp_view_name("news_articles_updates")
        updates.createOrReplaceTempView(view_name)
        self._spark.sql(
            f"""
            MERGE INTO {self._tables.news_articles_fqn} AS target
            USING {view_name} AS source
            ON target.article_id = source.article_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )
        logger.info("Upserted %s articles into %s", len(articles), self._tables.news_articles_fqn)


class ArticlesCleanRepository:
    def __init__(self, spark: Any, tables: TableNames) -> None:
        self._spark = spark
        self._tables = tables

    def rebuild_from_news_articles(self) -> None:
        self._spark.sql(
            f"""
            INSERT OVERWRITE {self._tables.articles_clean_fqn}
            SELECT
                article_id,
                source,
                url,
                canonical_url,
                title,
                summary_raw,
                content,
                category,
                published_at,
                crawled_at,
                content_hash,
                dedup_group_id,
                is_duplicate,
                created_at,
                updated_at
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            CASE
                                WHEN canonical_url IS NOT NULL AND canonical_url <> '' THEN canonical_url
                                ELSE content_hash
                            END
                        ORDER BY COALESCE(published_at, crawled_at) DESC, updated_at DESC
                    ) AS rn
                FROM {self._tables.news_articles_fqn}
                WHERE is_duplicate = false
                  AND content IS NOT NULL
                  AND LENGTH(TRIM(content)) > 0
            )
            WHERE rn = 1
            """
        )
        count = self._spark.table(self._tables.articles_clean_fqn).count()
        logger.info("Rebuilt %s with %s rows", self._tables.articles_clean_fqn, count)


def clean_article_rows(articles: list[Article]) -> list[dict[str, Any]]:
    schema_fields = {field.name for field in articles_clean_schema().fields}
    return [{key: value for key, value in asdict(article).items() if key in schema_fields} for article in articles]


def _records_to_dataframe(spark: Any, records: list[dict[str, Any]], schema: Any) -> Any:
    field_names = [field.name for field in schema.fields]
    ordered_rows = [tuple(record.get(field_name) for field_name in field_names) for record in records]
    return spark.createDataFrame(ordered_rows, schema=schema)


def _temp_view_name(prefix: str) -> str:
    return f"_{prefix}_{uuid.uuid4().hex}"
