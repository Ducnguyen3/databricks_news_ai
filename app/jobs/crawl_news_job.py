from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Sequence

from app.config import load_settings
from app.databricks.delta_tables import ensure_database_objects
from app.databricks.session import get_spark
from app.ingestion.services.crawl_service import CrawlResult, CrawlService
from app.ingestion.services.source_management_service import SourceManagementService
from app.repositories.raw_document_repository import RawDocumentRepository
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> CrawlResult:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    settings = load_settings()

    source_names = args.source_names or _read_databricks_widget("source_names", "")
    dry_run = _parse_bool(args.dry_run if args.dry_run is not None else _read_databricks_widget("dry_run", "false"))
    max_articles_per_source = args.max_articles_per_source
    if max_articles_per_source is None:
        widget_value = _read_databricks_widget("max_articles_per_source", "")
        max_articles_per_source = int(widget_value) if widget_value else settings.crawl.max_articles_per_source
    max_pages_per_category = args.max_pages_per_category
    if max_pages_per_category is None:
        widget_value = _read_databricks_widget("max_pages_per_category", "")
        max_pages_per_category = int(widget_value) if widget_value else settings.crawl.max_pages_per_category
    stop_after_empty_pages = _int_arg_or_widget(
        args.stop_after_empty_pages,
        "stop_after_empty_pages",
        settings.crawl.stop_after_empty_pages,
    )
    stop_after_duplicate_pages = _int_arg_or_widget(
        args.stop_after_duplicate_pages,
        "stop_after_duplicate_pages",
        settings.crawl.stop_after_duplicate_pages,
    )
    request_delay_seconds = _float_arg_or_widget(
        args.request_delay_seconds,
        "request_delay_seconds",
        settings.crawl.request_delay_seconds,
    )
    discover_categories_value = args.discover_categories
    if discover_categories_value is None:
        discover_categories_value = _read_databricks_widget("discover_categories", "")
    discover_categories = (
        settings.crawl.discover_categories
        if not discover_categories_value
        else _parse_bool(discover_categories_value)
    )
    crawl_mode = args.crawl_mode or _read_databricks_widget("crawl_mode", "") or settings.crawl.crawl_mode
    crawl_settings = replace(
        settings.crawl,
        crawl_mode=crawl_mode,
        discover_categories=discover_categories,
        max_pages_per_category=max_pages_per_category,
        stop_after_empty_pages=stop_after_empty_pages,
        stop_after_duplicate_pages=stop_after_duplicate_pages,
        request_delay_seconds=request_delay_seconds,
    )

    repository = None
    if not dry_run:
        spark = get_spark()
        ensure_database_objects(spark, settings.tables)
        repository = RawDocumentRepository(spark, settings.tables)

    source_service = SourceManagementService(crawl_settings)
    crawl_service = CrawlService(
        settings=crawl_settings,
        source_management_service=source_service,
        raw_document_repository=repository,
    )
    result = crawl_service.run_crawl(
        source_names=source_names or None,
        dry_run=dry_run,
        max_articles_per_source=max_articles_per_source,
        crawl_run_id=args.crawl_run_id,
    )
    _log_summary(result)
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl news sources into raw Delta documents.")
    parser.add_argument("--source_names", default=None, help="Comma-separated source names.")
    parser.add_argument("--dry_run", default=None, help="true/false. Dry run does not write Delta.")
    parser.add_argument("--max_articles_per_source", type=int, default=None, help="Legacy; ignored in category_pagination mode.")
    parser.add_argument("--max_pages_per_category", type=int, default=None, help="0 means no fixed page limit.")
    parser.add_argument("--stop_after_empty_pages", type=int, default=None)
    parser.add_argument("--stop_after_duplicate_pages", type=int, default=None)
    parser.add_argument("--request_delay_seconds", type=float, default=None)
    parser.add_argument("--discover_categories", default=None)
    parser.add_argument("--crawl_mode", default=None)
    parser.add_argument("--crawl_run_id", default=None)
    parser.add_argument("--log_level", default="INFO")
    args, _unknown = parser.parse_known_args(argv)
    return args


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _read_databricks_widget(name: str, default: str) -> str:
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            return default
        dbutils = DBUtils(spark)
        try:
            dbutils.widgets.text(name, default)
        except Exception:
            pass
        value = dbutils.widgets.get(name)
        return value if value is not None else default
    except Exception:
        return default


def _int_arg_or_widget(value: int | None, widget_name: str, default: int) -> int:
    if value is not None:
        return value
    widget_value = _read_databricks_widget(widget_name, "")
    return int(widget_value) if widget_value else default


def _float_arg_or_widget(value: float | None, widget_name: str, default: float) -> float:
    if value is not None:
        return value
    widget_value = _read_databricks_widget(widget_name, "")
    return float(widget_value) if widget_value else default


def _log_summary(result: CrawlResult) -> None:
    logger.info(
        "Crawl finished:\n"
        "- crawl_run_id: %s\n"
        "- sources: %s\n"
        "- discovered urls: %s\n"
        "- fetched articles: %s\n"
        "- saved documents: %s\n"
        "- failed sources: %s",
        result.crawl_run_id,
        result.total_sources,
        result.total_discovered_urls,
        result.total_fetched_articles,
        result.total_saved_documents,
        ", ".join(result.failed_sources) if result.failed_sources else "none",
    )


if __name__ == "__main__":
    main()
