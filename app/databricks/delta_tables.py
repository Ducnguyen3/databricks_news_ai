from __future__ import annotations

from typing import Any

from app.config import TableNames


def raw_documents_schema() -> Any:
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

    return StructType(
        [
            StructField("raw_document_id", StringType(), False),
            StructField("source_name", StringType(), False),
            StructField("crawl_run_id", StringType(), False),
            StructField("url", StringType(), False),
            StructField("canonical_url", StringType(), False),
            StructField("page_type", StringType(), False),
            StructField("fetch_status", StringType(), False),
            StructField("http_status", IntegerType(), True),
            StructField("fetched_at", TimestampType(), False),
            StructField("payload_format", StringType(), False),
            StructField("raw_payload", StringType(), False),
            StructField("raw_text", StringType(), False),
            StructField("checksum", StringType(), False),
            StructField("metadata", StringType(), False),
            StructField("created_at", TimestampType(), False),
        ]
    )


def news_articles_schema() -> Any:
    from pyspark.sql.types import BooleanType, StringType, StructField, StructType, TimestampType

    return StructType(
        [
            StructField("raw_id", StringType(), True),
            *articles_clean_schema().fields,
        ]
    )


def articles_clean_schema() -> Any:
    from pyspark.sql.types import BooleanType, StringType, StructField, StructType, TimestampType

    return StructType(
        [
            StructField("article_id", StringType(), False),
            StructField("source", StringType(), False),
            StructField("url", StringType(), False),
            StructField("canonical_url", StringType(), False),
            StructField("title", StringType(), False),
            StructField("summary_raw", StringType(), True),
            StructField("content", StringType(), False),
            StructField("category", StringType(), True),
            StructField("published_at", TimestampType(), True),
            StructField("crawled_at", TimestampType(), False),
            StructField("content_hash", StringType(), False),
            StructField("dedup_group_id", StringType(), False),
            StructField("is_duplicate", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("updated_at", TimestampType(), False),
        ]
    )


def ensure_database_objects(spark: Any, tables: TableNames) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {tables.schema_fqn}")
    ensure_raw_documents_table(spark, tables)
    ensure_news_articles_table(spark, tables)
    ensure_articles_clean_table(spark, tables)


def ensure_raw_documents_table(spark: Any, tables: TableNames) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.raw_documents_fqn} (
            raw_document_id STRING NOT NULL,
            source_name STRING NOT NULL,
            crawl_run_id STRING NOT NULL,
            url STRING NOT NULL,
            canonical_url STRING NOT NULL,
            page_type STRING NOT NULL,
            fetch_status STRING NOT NULL,
            http_status INT,
            fetched_at TIMESTAMP NOT NULL,
            payload_format STRING NOT NULL,
            raw_payload STRING NOT NULL,
            raw_text STRING NOT NULL,
            checksum STRING NOT NULL,
            metadata STRING NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
        USING DELTA
        """
    )


def ensure_news_articles_table(spark: Any, tables: TableNames) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.news_articles_fqn} (
            raw_id STRING,
            article_id STRING NOT NULL,
            source STRING NOT NULL,
            url STRING NOT NULL,
            canonical_url STRING NOT NULL,
            title STRING NOT NULL,
            summary_raw STRING,
            content STRING NOT NULL,
            category STRING,
            published_at TIMESTAMP,
            crawled_at TIMESTAMP NOT NULL,
            content_hash STRING NOT NULL,
            dedup_group_id STRING NOT NULL,
            is_duplicate BOOLEAN NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        USING DELTA
        """
    )


def ensure_articles_clean_table(spark: Any, tables: TableNames) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.articles_clean_fqn} (
            article_id STRING NOT NULL,
            source STRING NOT NULL,
            url STRING NOT NULL,
            canonical_url STRING NOT NULL,
            title STRING NOT NULL,
            summary_raw STRING,
            content STRING NOT NULL,
            category STRING,
            published_at TIMESTAMP,
            crawled_at TIMESTAMP NOT NULL,
            content_hash STRING NOT NULL,
            dedup_group_id STRING NOT NULL,
            is_duplicate BOOLEAN NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        USING DELTA
        """
    )
