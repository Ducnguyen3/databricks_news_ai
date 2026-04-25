from __future__ import annotations

import re
import unicodedata
from typing import Any


class SimpleReranker:
    def rerank(self, question: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_terms = normalized_terms(question)
        reranked: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            text = str(candidate.get("text") or "")
            title = str(metadata.get("title") or "")
            category = str(metadata.get("category") or "")

            vector_score = safe_float(candidate.get("score"))
            keyword_score = overlap_score(query_terms, normalized_terms(text))
            metadata_score = overlap_score(query_terms, normalized_terms(f"{title} {category}"))
            final_score = (vector_score * 0.7) + (keyword_score * 0.25) + (metadata_score * 0.05)

            reranked_candidate = dict(candidate)
            reranked_candidate["vector_score"] = vector_score
            reranked_candidate["keyword_score"] = keyword_score
            reranked_candidate["metadata_score"] = metadata_score
            reranked_candidate["final_score"] = final_score
            reranked.append(reranked_candidate)

        reranked.sort(
            key=lambda item: (
                safe_float(item.get("final_score")),
                safe_float(item.get("vector_score")),
            ),
            reverse=True,
        )
        return reranked[: max(1, int(top_k))]


def normalized_terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)
        if len(term) >= 3
    }


def overlap_score(query_terms: set[str], text_terms: set[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms.intersection(text_terms)) / max(1, len(query_terms))


def normalize_text(text: str) -> str:
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("\u0111", "d")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
