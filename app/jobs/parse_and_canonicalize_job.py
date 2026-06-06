from __future__ import annotations

import logging

from app.config import load_settings
from app.databricks.delta_tables import ensure_database_objects
from app.databricks.repositories import ArticleImagesRepository, NewsArticlesRepository, RawDocumentsRepository
from app.databricks.session import get_spark
from app.domain.models import Article, ArticleImage
from app.ingestion.parsers.html_parser import parse_raw_document
from app.processing.deduplicator import Deduplicator
from app.processing.image_extractor import extract_article_images
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    settings = load_settings()
    spark = get_spark()
    ensure_database_objects(spark, settings.tables)

    raw_documents = RawDocumentsRepository(spark, settings.tables).read_all()
    logger.info("Loaded %s raw documents", len(raw_documents))

    parsed_articles: list[Article] = []
    images_by_article_id: dict[str, list[ArticleImage]] = {}
    for raw_document in raw_documents:
        article = parse_raw_document(raw_document)
        if article is not None:
            parsed_articles.append(article)
            images_by_article_id.setdefault(article.article_id, extract_article_images(raw_document, article))

    parsed_articles.sort(
        key=lambda article: (article.published_at or article.crawled_at, article.source, article.url),
        reverse=True,
    )
    unique_articles_by_id: dict[str, Article] = {}
    for article in parsed_articles:
        unique_articles_by_id.setdefault(article.article_id, article)

    deduplicated_articles = Deduplicator().mark_duplicates(list(unique_articles_by_id.values()))
    NewsArticlesRepository(spark, settings.tables).upsert(deduplicated_articles)

    image_ids: set[str] = set()
    article_images: list[ArticleImage] = []
    for article in deduplicated_articles:
        for image in images_by_article_id.get(article.article_id, []):
            if image.id in image_ids:
                continue
            image_ids.add(image.id)
            article_images.append(image)
    ArticleImagesRepository(spark, settings.tables).upsert(article_images)
    logger.info(
        "parse_and_canonicalize_job finished with %s articles and %s article images",
        len(deduplicated_articles),
        len(article_images),
    )


if __name__ == "__main__":
    main()
