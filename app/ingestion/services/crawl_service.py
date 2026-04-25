from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from app.config import CrawlSettings
from app.domain.models import RawDocument
from app.ingestion.crawlers.base import BaseCrawler
from app.ingestion.crawlers.base_crawler import Category
from app.ingestion.crawlers.registry import CrawlerRegistry
from app.ingestion.services.source_management_service import SourceManagementService
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CrawlResult:
    crawl_run_id: str
    total_sources: int
    total_discovered_urls: int
    total_fetched_articles: int
    total_saved_documents: int
    failed_sources: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None


class CrawlService:
    def __init__(
        self,
        settings: CrawlSettings,
        source_management_service: SourceManagementService | None = None,
        crawler_registry: CrawlerRegistry | None = None,
        raw_document_repository: object | None = None,
    ) -> None:
        self._settings = settings
        self._source_management_service = source_management_service or SourceManagementService(settings)
        self._crawler_registry = crawler_registry or CrawlerRegistry()
        self._raw_document_repository = raw_document_repository

    def run_crawl(
        self,
        source_names: Iterable[str] | str | None = None,
        dry_run: bool = False,
        max_articles_per_source: int | None = None,
        crawl_run_id: str | None = None,
    ) -> CrawlResult:
        started_at = utc_now()
        run_id = crawl_run_id or str(uuid.uuid4())
        requested_sources = _normalize_source_names(source_names)
        source_configs = self._source_management_service.get_enabled_sources(requested_sources)
        if max_articles_per_source is not None:
            logger.info("[CRAWL] max_articles_per_source is ignored in category_pagination mode")

        total_discovered_urls = 0
        total_fetched_articles = 0
        total_saved_documents = 0
        failed_sources: list[str] = []

        for source_config in source_configs:
            source_name = source_config.source_name
            logger.info("[CRAWL] Start source=%s", source_name)
            try:
                crawler = self._crawler_registry.get_crawler(source_name, source_config)
                self._load_existing_keys(crawler, source_name)
                categories = crawler.discover_categories(source_config.homepage_url)
                logger.info("[CRAWL] Source categories source=%s count=%s", source_name, len(categories))

                source_result = self._crawl_source_categories(
                    crawler=crawler,
                    categories=categories,
                    crawl_run_id=run_id,
                    dry_run=dry_run,
                )
                total_discovered_urls += source_result.total_discovered_urls
                total_fetched_articles += source_result.total_fetched_articles
                total_saved_documents += source_result.total_saved_documents
            except Exception as exc:
                failed_sources.append(source_name)
                logger.exception("[CRAWL] Failed source=%s error=%s", source_name, exc)

        return CrawlResult(
            crawl_run_id=run_id,
            total_sources=len(source_configs),
            total_discovered_urls=total_discovered_urls,
            total_fetched_articles=total_fetched_articles,
            total_saved_documents=total_saved_documents,
            failed_sources=failed_sources,
            started_at=started_at,
            finished_at=utc_now(),
        )

    def crawl_all(self, crawlers: list[BaseCrawler]) -> list[RawDocument]:
        all_documents: list[RawDocument] = []
        crawl_run_id = str(uuid.uuid4())
        for crawler in crawlers:
            try:
                logger.info("[CRAWL] Start source=%s", crawler.source_name)
                urls = crawler.discover_article_urls(
                    max_articles=None
                )
                documents = crawler.fetch_article_pages(urls, crawl_run_id=crawl_run_id)
                all_documents.extend(documents)
            except Exception:
                logger.exception("[CRAWL] Failed source=%s", crawler.source_name)
        return all_documents

    def _save_documents(self, documents: list[RawDocument], source_name: str) -> int:
        if not documents:
            return 0
        if self._raw_document_repository is None:
            raise RuntimeError("raw_document_repository is required when dry_run=False")
        upsert = getattr(self._raw_document_repository, "upsert", None)
        if upsert is None:
            raise RuntimeError("raw_document_repository must expose an upsert(documents) method")
        result = upsert(documents)
        if isinstance(result, int):
            return result
        logger.info("[CRAWL] Repository saved documents source=%s count=%s", source_name, len(documents))
        return len(documents)

    def _crawl_source_categories(
        self,
        crawler: BaseCrawler,
        categories: list[Category],
        crawl_run_id: str,
        dry_run: bool,
    ) -> "_SourceCrawlTotals":
        totals = _SourceCrawlTotals()
        max_pages = max(1, self._settings.max_pages_per_category)
        stop_after_empty_pages = max(1, self._settings.stop_after_empty_pages)
        stop_after_duplicate_pages = max(1, self._settings.stop_after_duplicate_pages)

        for category in categories:
            logger.info("[CRAWL] Start category source=%s category=%s url=%s", crawler.source_name, category.name, category.url)
            empty_page_count = 0
            duplicate_page_count = 0

            for page in range(1, max_pages + 1):
                try:
                    category_page_url = crawler.build_category_page_url(category.url, page)
                    logger.info(
                        "[CRAWL] Page source=%s category=%s page=%s url=%s",
                        crawler.source_name,
                        category.name,
                        page,
                        category_page_url,
                    )
                    article_links = crawler.extract_article_links(category_page_url)
                    totals.total_discovered_urls += len(article_links)
                    logger.info(
                        "[CRAWL] Page links source=%s category=%s page=%s found=%s",
                        crawler.source_name,
                        category.name,
                        page,
                        len(article_links),
                    )

                    if not article_links:
                        empty_page_count += 1
                        logger.info(
                            "[CRAWL] Empty page source=%s category=%s page=%s empty_count=%s",
                            crawler.source_name,
                            category.name,
                            page,
                            empty_page_count,
                        )
                        if empty_page_count >= stop_after_empty_pages:
                            logger.info("[CRAWL] Stop category reason=empty_pages source=%s category=%s", crawler.source_name, category.name)
                            break
                        crawler.request_delay()
                        continue
                    empty_page_count = 0

                    new_links = crawler.filter_new_links(article_links)
                    logger.info(
                        "[CRAWL] New links source=%s category=%s page=%s new=%s duplicates=%s",
                        crawler.source_name,
                        category.name,
                        page,
                        len(new_links),
                        len(article_links) - len(new_links),
                    )
                    if not new_links:
                        duplicate_page_count += 1
                        if duplicate_page_count >= stop_after_duplicate_pages:
                            logger.info("[CRAWL] Stop category reason=duplicate_pages source=%s category=%s", crawler.source_name, category.name)
                            break
                        crawler.request_delay()
                        continue
                    duplicate_page_count = 0

                    if dry_run:
                        logger.info("[CRAWL] Dry run category=%s page=%s new_links=%s", category.name, page, len(new_links))
                        crawler.request_delay()
                        continue

                    page_documents = self._crawl_page_articles(
                        crawler=crawler,
                        category=category,
                        page=page,
                        article_links=new_links,
                        crawl_run_id=crawl_run_id,
                    )
                    totals.total_fetched_articles += len(page_documents)
                    saved_count = self._save_documents(page_documents, crawler.source_name)
                    totals.total_saved_documents += saved_count
                    logger.info(
                        "[CRAWL] Page saved source=%s category=%s page=%s success=%s saved=%s",
                        crawler.source_name,
                        category.name,
                        page,
                        len(page_documents),
                        saved_count,
                    )
                    crawler.request_delay()
                except Exception as exc:
                    logger.exception(
                        "[CRAWL] Failed category page source=%s category=%s page=%s error=%s",
                        crawler.source_name,
                        category.name,
                        page,
                        exc,
                    )
                    logger.info("[CRAWL] Stop category reason=request_error source=%s category=%s", crawler.source_name, category.name)
                    break
            else:
                logger.info("[CRAWL] Stop category reason=max_pages source=%s category=%s max_pages=%s", crawler.source_name, category.name, max_pages)

        return totals

    def _crawl_page_articles(
        self,
        crawler: BaseCrawler,
        category: Category,
        page: int,
        article_links: list[str],
        crawl_run_id: str,
    ) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for article_url in article_links:
            crawler.request_delay()
            article = crawler.crawl_article_detail(article_url, category=category, category_page=page)
            if article is None:
                continue
            content_hash = article.content_hash or crawler.build_checksum(article)
            if crawler.should_skip_content_hash(content_hash):
                logger.info("[CRAWL] Skip duplicate content source=%s url=%s", crawler.source_name, article.url)
                continue
            document = crawler.build_raw_document(article=article, crawl_run_id=crawl_run_id)
            crawler.mark_existing_link(document.url, document.canonical_url)
            documents.append(document)
        return documents

    def _load_existing_keys(self, crawler: BaseCrawler, source_name: str) -> None:
        if self._raw_document_repository is None:
            crawler.set_existing_keys()
            return
        get_existing_article_keys = getattr(self._raw_document_repository, "get_existing_article_keys", None)
        if get_existing_article_keys is None:
            crawler.set_existing_keys()
            return
        existing_keys = get_existing_article_keys(source_name)
        crawler.set_existing_keys(
            link_keys=getattr(existing_keys, "link_keys", set()),
            content_hashes=getattr(existing_keys, "content_hashes", set()),
        )


def _normalize_source_names(source_names: Iterable[str] | str | None) -> list[str] | None:
    if source_names is None:
        return None
    if isinstance(source_names, str):
        return [item.strip() for item in source_names.split(",") if item.strip()]
    return [item.strip() for item in source_names if item.strip()]


@dataclass(slots=True)
class _SourceCrawlTotals:
    total_discovered_urls: int = 0
    total_fetched_articles: int = 0
    total_saved_documents: int = 0
