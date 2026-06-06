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
    from pyspark.sql.types import StringType, StructField, StructType

    return StructType(
        [
            StructField("raw_id", StringType(), True),
            *articles_clean_schema().fields,
        ]
    )


def articles_clean_schema() -> Any:
    from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType, TimestampType

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
            StructField("source_category_name", StringType(), True),
            StructField("source_category_url", StringType(), True),
            StructField("published_at", TimestampType(), True),
            StructField("crawled_at", TimestampType(), False),
            StructField("content_hash", StringType(), False),
            StructField("dedup_group_id", StringType(), False),
            StructField("is_duplicate", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("updated_at", TimestampType(), False),
            StructField("primary_topic", StringType(), False),
            StructField("primary_topic_name", StringType(), False),
            StructField("topic_confidence", DoubleType(), False),
            StructField("secondary_topics_json", StringType(), False),
            StructField("entities_json", StringType(), False),
        ]
    )


def article_images_schema() -> Any:
    from pyspark.sql.types import BooleanType, IntegerType, StringType, StructField, StructType, TimestampType

    return StructType(
        [
            StructField("id", StringType(), False),
            StructField("article_id", StringType(), False),
            StructField("source", StringType(), False),
            StructField("canonical_url", StringType(), False),
            StructField("image_url", StringType(), False),
            StructField("caption", StringType(), True),
            StructField("alt_text", StringType(), True),
            StructField("credit", StringType(), True),
            StructField("position", IntegerType(), False),
            StructField("width", IntegerType(), True),
            StructField("height", IntegerType(), True),
            StructField("image_type", StringType(), False),
            StructField("is_representative", BooleanType(), False),
            StructField("created_at", TimestampType(), True),
        ]
    )


def ensure_database_objects(spark: Any, tables: TableNames) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {tables.schema_fqn}")
    ensure_raw_documents_table(spark, tables)
    ensure_news_articles_table(spark, tables)
    ensure_articles_clean_table(spark, tables)
    ensure_article_images_table(spark, tables)
    ensure_sports_tables(spark, tables)


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
            source_category_name STRING,
            source_category_url STRING,
            published_at TIMESTAMP,
            crawled_at TIMESTAMP NOT NULL,
            content_hash STRING NOT NULL,
            dedup_group_id STRING NOT NULL,
            is_duplicate BOOLEAN NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            primary_topic STRING NOT NULL,
            primary_topic_name STRING NOT NULL,
            topic_confidence DOUBLE NOT NULL,
            secondary_topics_json STRING NOT NULL,
            entities_json STRING NOT NULL
        )
        USING DELTA
        """
    )
    _add_columns_if_missing(
        spark,
        tables.news_articles_fqn,
        {
            "primary_topic": "STRING",
            "primary_topic_name": "STRING",
            "topic_confidence": "DOUBLE",
            "secondary_topics_json": "STRING",
            "entities_json": "STRING",
            "source_category_name": "STRING",
            "source_category_url": "STRING",
        },
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
            source_category_name STRING,
            source_category_url STRING,
            published_at TIMESTAMP,
            crawled_at TIMESTAMP NOT NULL,
            content_hash STRING NOT NULL,
            dedup_group_id STRING NOT NULL,
            is_duplicate BOOLEAN NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            primary_topic STRING NOT NULL,
            primary_topic_name STRING NOT NULL,
            topic_confidence DOUBLE NOT NULL,
            secondary_topics_json STRING NOT NULL,
            entities_json STRING NOT NULL
        )
        USING DELTA
        """
    )
    _add_columns_if_missing(
        spark,
        tables.articles_clean_fqn,
        {
            "primary_topic": "STRING",
            "primary_topic_name": "STRING",
            "topic_confidence": "DOUBLE",
            "secondary_topics_json": "STRING",
            "entities_json": "STRING",
            "source_category_name": "STRING",
            "source_category_url": "STRING",
        },
    )


def ensure_article_images_table(spark: Any, tables: TableNames) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.article_images_fqn} (
            id STRING NOT NULL,
            article_id STRING NOT NULL,
            source STRING NOT NULL,
            canonical_url STRING NOT NULL,
            image_url STRING NOT NULL,
            caption STRING,
            alt_text STRING,
            credit STRING,
            position INT NOT NULL,
            width INT,
            height INT,
            image_type STRING NOT NULL,
            is_representative BOOLEAN NOT NULL,
            created_at TIMESTAMP
        )
        USING DELTA
        """
    )


def ensure_sports_tables(spark: Any, tables: TableNames) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.sports_leagues_fqn} (
            league_id STRING NOT NULL,
            source STRING NOT NULL,
            source_url STRING,
            name STRING NOT NULL,
            country STRING,
            season STRING,
            created_at TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.sports_matches_fqn} (
            match_id STRING NOT NULL,
            source STRING NOT NULL,
            source_url STRING,
            league_id STRING,
            league_name STRING,
            season STRING,
            round STRING,
            home_team STRING,
            away_team STRING,
            kickoff_at TIMESTAMP,
            status STRING,
            home_score INT,
            away_score INT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.sports_standings_fqn} (
            standing_id STRING NOT NULL,
            source STRING NOT NULL,
            source_url STRING,
            league_id STRING,
            league_name STRING,
            season STRING,
            team STRING NOT NULL,
            rank INT,
            played INT,
            points INT,
            wins INT,
            draws INT,
            losses INT,
            goals_for INT,
            goals_against INT,
            updated_at TIMESTAMP
        )
        USING DELTA
        """
    )


def _add_columns_if_missing(spark: Any, table_name: str, columns: dict[str, str]) -> None:
    try:
        existing_columns = set(spark.table(table_name).columns)
    except Exception:
        return
    missing_columns = [(name, data_type) for name, data_type in columns.items() if name not in existing_columns]
    if not missing_columns:
        return
    columns_sql = ", ".join(f"{name} {data_type}" for name, data_type in missing_columns)
    spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({columns_sql})")
