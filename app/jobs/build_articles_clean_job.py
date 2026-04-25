from __future__ import annotations

import logging

from app.config import load_settings
from app.databricks.delta_tables import ensure_database_objects
from app.databricks.repositories import ArticlesCleanRepository
from app.databricks.session import get_spark
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    settings = load_settings()
    spark = get_spark()
    ensure_database_objects(spark, settings.tables)

    ArticlesCleanRepository(spark, settings.tables).rebuild_from_news_articles()
    logger.info("build_articles_clean_job finished")


if __name__ == "__main__":
    main()

