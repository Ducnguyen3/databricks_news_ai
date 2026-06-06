from __future__ import annotations

import unittest

from app.local_ai.databricks_client import DatabricksArticleClient, DatabricksSqlConfig


class DatabricksClientQueryTest(unittest.TestCase):
    def test_incremental_query_can_order_by_source_then_recency(self) -> None:
        client = DatabricksArticleClient(
            DatabricksSqlConfig(
                server_hostname="host",
                http_path="path",
                token="token",
                articles_table="main.news_ai.articles_clean",
                article_images_table="main.news_ai.news_article_images",
            )
        )

        query = client._build_query(limit=10, order_by_source=True)

        self.assertIn("ORDER BY source ASC, COALESCE(published_at, crawled_at) DESC, updated_at DESC", query)


if __name__ == "__main__":
    unittest.main()
