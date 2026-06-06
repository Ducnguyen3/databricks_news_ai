from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any


class BM25ChunkIndex:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = [normalize_chunk(chunk) for chunk in chunks]
        self._k1 = k1
        self._b = b
        self._documents: list[list[str]] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        for chunk in self._chunks:
            tokens = self.tokenize(build_search_text(chunk))
            self._documents.append(tokens)
            frequencies = Counter(tokens)
            self._term_freqs.append(frequencies)
            self._doc_freqs.update(frequencies.keys())
        self._avg_doc_len = (
            sum(len(document) for document in self._documents) / len(self._documents)
            if self._documents
            else 0.0
        )

    def count(self) -> int:
        return len(self._chunks)

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)

    def search(self, query: str, top_k: int = 30, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self._chunks:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        query_terms = set(query_tokens)
        scored: list[dict[str, Any]] = []
        for index, chunk in enumerate(self._chunks):
            if not matches_filters(chunk.get("metadata", {}), filters or {}):
                continue
            score = self._score_document(index, query_terms)
            if score <= 0:
                continue
            scored.append(
                {
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "document": str(chunk.get("document") or chunk.get("text") or ""),
                    "text": str(chunk.get("document") or chunk.get("text") or ""),
                    "metadata": dict(chunk.get("metadata") or {}),
                    "bm25_score": score,
                    "rank": 0,
                    "retrieval_source": "bm25",
                }
            )
        scored.sort(key=lambda item: float(item["bm25_score"]), reverse=True)
        for rank, item in enumerate(scored[: max(1, int(top_k))], start=1):
            item["rank"] = rank
        return scored[: max(1, int(top_k))]

    def _score_document(self, index: int, query_terms: set[str]) -> float:
        frequencies = self._term_freqs[index]
        doc_len = max(1, len(self._documents[index]))
        score = 0.0
        total_docs = max(1, len(self._documents))
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = self._doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + self._k1 * (1 - self._b + self._b * doc_len / max(self._avg_doc_len, 1.0))
            score += idf * (term_frequency * (self._k1 + 1)) / denominator
        return score


def normalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk.get("chunk_id") or chunk.get("id") or ""),
        "document": str(chunk.get("document") or chunk.get("text") or ""),
        "text": str(chunk.get("document") or chunk.get("text") or ""),
        "metadata": dict(chunk.get("metadata") or {}),
    }


def build_search_text(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata") or {})
    entities_text = entities_to_text(metadata.get("entities_json"))
    return "\n".join(
        [
            str(metadata.get("title") or ""),
            str(metadata.get("source") or ""),
            str(metadata.get("primary_topic_name") or metadata.get("primary_topic") or ""),
            str(metadata.get("entity_names") or ""),
            entities_text,
            str(chunk.get("document") or chunk.get("text") or ""),
        ]
    )


def entities_to_text(value: Any) -> str:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        items = parsed if isinstance(parsed, list) else []
    else:
        items = []
    output: list[str] = []
    for item in items:
        if isinstance(item, dict):
            output.append(str(item.get("normalized_name") or item.get("name") or ""))
        else:
            output.append(str(item))
    return " ".join(output)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9À-ỹĐđ-]+", text):
        plain = strip_accents(raw).lower()
        if len(plain) >= 2:
            tokens.append(plain)
        if raw.isupper() and 2 <= len(raw) <= 8:
            tokens.append(raw)
    return tokens


def strip_accents(value: str) -> str:
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d").replace("Đ", "D")


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    source = filters.get("source")
    if source and str(metadata.get("source") or "").lower() != str(source).lower():
        return False
    topic = filters.get("topic") or filters.get("primary_topic")
    if topic and str(metadata.get("primary_topic") or metadata.get("topic_category") or "").lower() != str(topic).lower():
        return False
    if filters.get("has_images") and not truthy(metadata.get("has_images")):
        return False
    return True


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
