from __future__ import annotations

import logging
import os
from time import perf_counter

import numpy as np
from dotenv import load_dotenv

from app.config import load_settings

logger = logging.getLogger(__name__)


class LocalEmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        settings = load_settings()
        self._model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", settings.local_ai.embedding_model_name)
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = perf_counter()
        embeddings = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = perf_counter() - started_at
        logger.info("Embedded %s texts in %.2fs model=%s", len(texts), elapsed, self._model_name)
        return np.asarray(embeddings, dtype=np.float32).tolist()

    def embed_query(self, query: str) -> list[float]:
        started_at = perf_counter()
        embedding = self._get_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        elapsed = perf_counter() - started_at
        logger.info("Embedded query in %.2fs model=%s", elapsed, self._model_name)
        return np.asarray(embedding, dtype=np.float32).tolist()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model
