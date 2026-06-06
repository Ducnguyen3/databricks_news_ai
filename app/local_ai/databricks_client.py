from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from app.config import load_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatabricksSqlConfig:
    server_hostname: str
    http_path: str
    token: str
    articles_table: str
    article_images_table: str

    @classmethod
    def from_env(cls) -> "DatabricksSqlConfig":
        load_dotenv()
        settings = load_settings()
        return cls(
            server_hostname=_required_env("DATABRICKS_SERVER_HOSTNAME"),
            http_path=_required_env("DATABRICKS_HTTP_PATH"),
            token=_required_env("DATABRICKS_TOKEN"),
            articles_table=os.getenv(
                "DATABRICKS_ARTICLES_TABLE",
                settings.local_ai.databricks_articles_table,
            ),
            article_images_table=os.getenv(
                "DATABRICKS_ARTICLE_IMAGES_TABLE",
                _default_article_images_table(settings.local_ai.databricks_articles_table),
            ),
        )


class DatabricksArticleClient:
    def __init__(self, config: DatabricksSqlConfig | None = None) -> None:
        self._config = config or DatabricksSqlConfig.from_env()

    def fetch_articles(
        self,
        limit: int | None = None,
        since: str | None = None,
        order_by_source: bool = False,
    ) -> list[dict[str, Any]]:
        query = self._build_query(limit, since=since, order_by_source=order_by_source)
        logger.info(
            "Loading articles from Databricks table=%s limit=%s since=%s",
            self._config.articles_table,
            limit or "all",
            since or "all",
        )

        from databricks import sql

        with sql.connect(
            server_hostname=self._config.server_hostname,
            http_path=self._config.http_path,
            access_token=self._config.token,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description]

        articles = [dict(zip(columns, row)) for row in rows]
        logger.info("Loaded %s articles from Databricks SQL", len(articles))
        return articles

    def fetch_articles_from_gold(
        self,
        limit: int | None = None,
        source: str | None = None,
        since: str | None = None,
        order_by_source: bool = False,
    ) -> list[dict[str, Any]]:
        articles = self.fetch_articles(limit=limit, since=since, order_by_source=order_by_source)
        if source:
            articles = [article for article in articles if str(article.get("source") or "") == source]
        return articles

    def fetch_article_images(self, article_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if article_ids is not None and not article_ids:
            return []
        where_clause = ""
        if article_ids:
            quoted_ids = ", ".join(_sql_string(article_id) for article_id in article_ids)
            where_clause = f"WHERE article_id IN ({quoted_ids})"
        query = f"""
        SELECT
            article_id,
            source,
            canonical_url,
            image_url,
            caption,
            alt_text,
            credit,
            position,
            width,
            height,
            is_representative
        FROM {self._config.article_images_table}
        {where_clause}
        ORDER BY article_id, is_representative DESC, position ASC
        """.strip()

        from databricks import sql

        try:
            with sql.connect(
                server_hostname=self._config.server_hostname,
                http_path=self._config.http_path,
                access_token=self._config.token,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    columns = [column[0] for column in cursor.description]
        except Exception:
            logger.warning("Could not load article images from %s", self._config.article_images_table, exc_info=True)
            return []
        images = [dict(zip(columns, row)) for row in rows]
        logger.info("Loaded %s article images from Databricks SQL", len(images))
        return images

    def _build_query(self, limit: int | None, since: str | None = None, order_by_source: bool = False) -> str:
        limit_clause = ""
        if limit is not None:
            safe_limit = int(limit)
            if safe_limit <= 0:
                raise ValueError("limit must be positive")
            limit_clause = f"\nLIMIT {safe_limit}"
        since_clause = ""
        if since:
            since_clause = f"\n          AND COALESCE(published_at, crawled_at) >= to_timestamp({_sql_string(since)})"

        order_clause = (
            "ORDER BY source ASC, COALESCE(published_at, crawled_at) DESC, updated_at DESC"
            if order_by_source
            else "ORDER BY COALESCE(published_at, crawled_at) DESC, updated_at DESC"
        )
        return f"""
        SELECT
            article_id,
            source,
            url,
            canonical_url,
            title,
            summary_raw,
            content,
            category,
            source_category_name,
            source_category_url,
            published_at,
            crawled_at,
            content_hash,
            primary_topic,
            primary_topic_name,
            topic_confidence,
            secondary_topics_json,
            entities_json
        FROM {self._config.articles_table}
        WHERE (is_duplicate = false OR is_duplicate IS NULL)
          AND content IS NOT NULL
          AND LENGTH(TRIM(content)) > 0
          {since_clause}
        {order_clause}
        {limit_clause}
        """.strip()


DatabricksArticlesClient = DatabricksArticleClient


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    raise RuntimeError(f"Missing required environment variable: {name}")


def _default_article_images_table(articles_table: str) -> str:
    parts = articles_table.split(".")
    if len(parts) >= 3:
        return ".".join([*parts[:-1], "news_article_images"])
    return "main.news_ai.news_article_images"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
