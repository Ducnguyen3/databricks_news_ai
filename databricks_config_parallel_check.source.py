from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TableNames:
    catalog: str = "main"
    schema: str = "news_ai"
    raw_documents: str = "news_raw_documents"
    news_articles: str = "news_articles"
    articles_clean: str = "articles_clean"
    article_images: str = "news_article_images"
    sports_matches: str = "sports_matches"
    sports_standings: str = "sports_standings"
    sports_leagues: str = "sports_leagues"

    @property
    def schema_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def raw_documents_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.raw_documents}"

    @property
    def news_articles_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.news_articles}"

    @property
    def articles_clean_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.articles_clean}"

    @property
    def article_images_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.article_images}"

    @property
    def sports_matches_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.sports_matches}"

    @property
    def sports_standings_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.sports_standings}"

    @property
    def sports_leagues_fqn(self) -> str:
        return f"{self.schema_fqn}.{self.sports_leagues}"


@dataclass(frozen=True)
class CrawlSettings:
    sources: tuple[str, ...]
    max_articles_per_source: int
    crawl_mode: str
    discover_categories: bool
    max_pages_per_category: int
    stop_after_empty_pages: int
    stop_after_duplicate_pages: int
    request_delay_seconds: float
    max_concurrent_requests: int
    request_timeout_seconds: int
    retry_count: int
    user_agent: str


@dataclass(frozen=True)
class LocalAiSettings:
    databricks_articles_table: str
    embedding_model_name: str
    chroma_persist_dir: str
    chroma_collection_name: str
    ollama_base_url: str | None
    ollama_model: str | None
    rag_retrieve_top_n: int
    rag_retrieval_mode: str
    rag_broad_retrieve_top_n: int
    rag_top_k: int
    rag_min_score: float
    rag_max_chunks_per_article: int
    chunk_size: int
    chunk_overlap: int
    summary_max_chunks: int
    prompt_max_context_chars: int
    prompt_debug_max_chars: int


@dataclass(frozen=True)
class Settings:
    tables: TableNames
    crawl: CrawlSettings
    local_ai: LocalAiSettings


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_settings() -> Settings:
    tables = TableNames(
        catalog=os.getenv("DATABRICKS_CATALOG", "main"),
        schema=os.getenv("DATABRICKS_SCHEMA", "news_ai"),
    )
    crawl = CrawlSettings(
        sources=_split_csv(os.getenv("NEWS_SOURCES", "vnexpress,cafef,genk,diendandoanhnghiep")),
        max_articles_per_source=_int_env("MAX_ARTICLES_PER_SOURCE", 20),
        crawl_mode=os.getenv("CRAWL_MODE", "category_pagination"),
        discover_categories=_bool_env("DISCOVER_CATEGORIES", True),
        max_pages_per_category=_int_env("MAX_PAGES_PER_CATEGORY", 5),
        stop_after_empty_pages=_int_env("STOP_AFTER_EMPTY_PAGES", 3),
        stop_after_duplicate_pages=_int_env("STOP_AFTER_DUPLICATE_PAGES", 3),
        request_delay_seconds=_float_env("REQUEST_DELAY_SECONDS", 1.0),
        max_concurrent_requests=_int_env("MAX_CONCURRENT_REQUESTS", 4),
        request_timeout_seconds=_int_env("REQUEST_TIMEOUT_SECONDS", 15),
        retry_count=_int_env("REQUEST_RETRY_COUNT", 2),
        user_agent=os.getenv(
            "NEWS_CRAWLER_USER_AGENT",
            "databricks-news-ai-demo/1.0 (+https://databricks.com)",
        ),
    )
    local_ai = LocalAiSettings(
        databricks_articles_table=os.getenv(
            "DATABRICKS_ARTICLES_TABLE",
            "main.news_ai.articles_clean",
        ),
        embedding_model_name=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "BAAI/bge-m3",
        ),
        chroma_persist_dir=os.getenv(
            "CHROMA_PERSIST_DIR",
            "data/chroma",
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "news_articles",
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        ollama_model=os.getenv("OLLAMA_MODEL"),
        rag_retrieve_top_n=_int_env("RAG_RETRIEVE_TOP_N", 12),
        rag_retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower(),
        rag_broad_retrieve_top_n=_int_env("RAG_BROAD_RETRIEVE_TOP_N", 30),
        rag_top_k=_int_env("RAG_TOP_K", 4),
        rag_min_score=_float_env("RAG_MIN_SCORE", 0.35),
        rag_max_chunks_per_article=_int_env("RAG_MAX_CHUNKS_PER_ARTICLE", 1),
        chunk_size=_int_env("CHUNK_SIZE", 700),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 120),
        summary_max_chunks=_int_env("SUMMARY_MAX_CHUNKS", 10),
        prompt_max_context_chars=_int_env("PROMPT_MAX_CONTEXT_CHARS", 12000),
        prompt_debug_max_chars=_int_env("PROMPT_DEBUG_MAX_CHARS", 4000),
    )
    return Settings(tables=tables, crawl=crawl, local_ai=local_ai)
