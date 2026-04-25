from __future__ import annotations

from dataclasses import dataclass

from app.config import LocalAiSettings
from app.local_ai.chunker import ArticleChunker
from app.local_ai.databricks_client import DatabricksArticleClient
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.rag_service import RAGService
from app.local_ai.vector_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class IndexingResult:
    articles_loaded: int
    chunks_created: int
    chunks_upserted: int
    index_size: int


def create_embedding_model(settings: LocalAiSettings) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(settings.embedding_model_name)


def create_vector_store(settings: LocalAiSettings) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


def create_rag_service(
    settings: LocalAiSettings,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> RAGService:
    return RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ollama_client=OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        ),
        settings=settings,
    )


def index_articles(
    settings: LocalAiSettings,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    limit: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset_index: bool = False,
) -> IndexingResult:
    if reset_index:
        vector_store.reset_collection()

    articles = DatabricksArticleClient().fetch_articles(limit=limit)
    if not articles:
        raise RuntimeError("No articles returned from Databricks table.")

    chunker = ArticleChunker(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
    chunks = chunker.chunk_articles(articles)
    if not chunks:
        raise RuntimeError("No chunks were created from loaded articles.")

    embeddings = embedding_model.embed_texts([chunk.text for chunk in chunks])
    vector_store.upsert_chunks(chunks, embeddings)
    return IndexingResult(
        articles_loaded=len(articles),
        chunks_created=len(chunks),
        chunks_upserted=len(chunks),
        index_size=vector_store.count(),
    )
