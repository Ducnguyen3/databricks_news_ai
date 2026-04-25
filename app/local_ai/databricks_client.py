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
        )


class DatabricksArticleClient:
    def __init__(self, config: DatabricksSqlConfig | None = None) -> None:
        self._config = config or DatabricksSqlConfig.from_env()

    def fetch_articles(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = self._build_query(limit)
        logger.info(
            "Loading articles from Databricks table=%s limit=%s",
            self._config.articles_table,
            limit or "all",
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

    def _build_query(self, limit: int | None) -> str:
        limit_clause = ""
        if limit is not None:
            safe_limit = int(limit)
            if safe_limit <= 0:
                raise ValueError("limit must be positive")
            limit_clause = f"\nLIMIT {safe_limit}"

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
            published_at,
            crawled_at,
            content_hash
        FROM {self._config.articles_table}
        WHERE (is_duplicate = false OR is_duplicate IS NULL)
          AND content IS NOT NULL
          AND LENGTH(TRIM(content)) > 0
        ORDER BY COALESCE(published_at, crawled_at) DESC, updated_at DESC
        {limit_clause}
        """.strip()


DatabricksArticlesClient = DatabricksArticleClient


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    raise RuntimeError(f"Missing required environment variable: {name}")
