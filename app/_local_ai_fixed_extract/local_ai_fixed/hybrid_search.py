from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.local_ai.bm25_index import BM25ChunkIndex, strip_accents, tokenize, truthy


class HybridSearchEngine:
    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any,
        bm25_index: BM25ChunkIndex | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._bm25_index = bm25_index

    @classmethod
    def from_vector_store(cls, vector_store: Any, embedding_model: Any) -> "HybridSearchEngine":
        get_all_chunks = getattr(vector_store, "get_all_chunks", None)
        chunks = get_all_chunks() if get_all_chunks is not None else []
        return cls(vector_store=vector_store, embedding_model=embedding_model, bm25_index=BM25ChunkIndex(chunks))

    def search(self, query: str, query_plan: dict[str, Any], top_n: int = 50) -> list[dict[str, Any]]:
        query_embedding = self._embedding_model.embed_query(query)
        vector_results = self._vector_store.search(query_embedding, top_k=top_n)
        bm25_results: list[dict[str, Any]] = []
        if self._bm25_index is not None and self._bm25_index.count():
            bm25_results = self._bm25_index.search(query, top_k=top_n, filters=filters_from_query_plan(query_plan))
        return merge_candidates(vector_results, bm25_results, query=query, query_plan=query_plan)


def filters_from_query_plan(query_plan: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if query_plan.get("source"):
        filters["source"] = query_plan.get("source")
    if query_plan.get("primary_topic"):
        filters["topic"] = query_plan.get("primary_topic")
    if query_plan.get("need_images") or query_plan.get("needs_images"):
        filters["has_images"] = True
    return filters


def merge_candidates(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    query: str,
    query_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = query_plan or {}
    vector_scores = [safe_float(item.get("score") or item.get("vector_score")) for item in vector_results]
    bm25_scores = [safe_float(item.get("bm25_score")) for item in bm25_results]
    vector_norm = normalizer(vector_scores)
    bm25_norm = normalizer(bm25_scores)
    merged: dict[str, dict[str, Any]] = {}
    for item in vector_results:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        score = safe_float(item.get("score") or item.get("vector_score"))
        merged[chunk_id] = {
            "chunk_id": chunk_id,
            "document": str(item.get("document") or item.get("text") or ""),
            "text": str(item.get("text") or item.get("document") or ""),
            "metadata": dict(item.get("metadata") or {}),
            "vector_score": score,
            "bm25_score": 0.0,
            "retrieval_sources": ["vector"],
        }
    for item in bm25_results:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        existing = merged.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "document": str(item.get("document") or item.get("text") or ""),
                "text": str(item.get("text") or item.get("document") or ""),
                "metadata": dict(item.get("metadata") or {}),
                "vector_score": 0.0,
                "bm25_score": 0.0,
                "retrieval_sources": [],
            },
        )
        existing["bm25_score"] = safe_float(item.get("bm25_score"))
        if "bm25" not in existing["retrieval_sources"]:
            existing["retrieval_sources"].append("bm25")
    vector_weight, bm25_weight = weights_for_query(query, plan)
    for item in merged.values():
        metadata = dict(item.get("metadata") or {})
        vector_score = safe_float(item.get("vector_score"))
        bm25_score = safe_float(item.get("bm25_score"))
        item["hybrid_score"] = (
            vector_weight * vector_norm(vector_score)
            + bm25_weight * bm25_norm(bm25_score)
            + exact_match_bonus(query, item)
            + entity_bonus(query, plan, metadata)
            + source_bonus(plan, metadata)
            + image_bonus(plan, metadata)
            + recency_bonus(plan, metadata)
        )
        item["score"] = max(vector_score, safe_float(item["hybrid_score"]))
    results = list(merged.values())
    results.sort(key=lambda candidate: safe_float(candidate.get("hybrid_score")), reverse=True)
    return results


def weights_for_query(query: str, query_plan: dict[str, Any]) -> tuple[float, float]:
    if query_plan.get("requires_lexical") or has_ticker(query):
        return 0.45, 0.55
    return 0.65, 0.35


def exact_match_bonus(query: str, item: dict[str, Any]) -> float:
    query_tokens = set(tokenize(query))
    haystack = strip_accents(
        f"{item.get('text') or item.get('document') or ''} "
        f"{(item.get('metadata') or {}).get('title') or ''}"
    ).lower()
    if not query_tokens:
        return 0.0
    matched = sum(1 for token in query_tokens if token.lower() in haystack or token in haystack)
    return min(0.15, matched / max(1, len(query_tokens)) * 0.15)


def entity_bonus(query: str, query_plan: dict[str, Any], metadata: dict[str, Any]) -> float:
    terms = [*query_plan.get("lexical_terms", []), *query_plan.get("entities", []), *query_plan.get("stock_symbols", [])]
    if not terms:
        return 0.0
    haystack = strip_accents(f"{metadata.get('title') or ''} {metadata.get('entity_names') or ''} {metadata.get('entities_json') or ''}").lower()
    matched = sum(1 for term in terms if strip_accents(str(term)).lower() in haystack)
    return min(0.15, matched / max(1, len(terms)) * 0.15)


def source_bonus(query_plan: dict[str, Any], metadata: dict[str, Any]) -> float:
    preferred = {str(item).lower() for item in query_plan.get("preferred_sources", [])}
    source = str(metadata.get("source") or "").lower()
    return 0.08 if source and source in preferred else 0.0


def image_bonus(query_plan: dict[str, Any], metadata: dict[str, Any]) -> float:
    if query_plan.get("need_images") or query_plan.get("needs_images"):
        return 0.1 if truthy(metadata.get("has_images")) else 0.0
    return 0.0


def recency_bonus(query_plan: dict[str, Any], metadata: dict[str, Any]) -> float:
    if not (query_plan.get("needs_recent") or query_plan.get("time_range") in {"today", "24h", "7d"}):
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(metadata.get("published_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).days)
    return max(0.0, 0.1 * (1.0 - min(age_days, 30) / 30))


def diversify_hybrid_results(
    results: list[dict[str, Any]],
    top_k: int,
    max_chunks_per_article: int = 2,
    max_articles_per_source: int = 3,
    preferred_sources: list[str] | None = None,
    requires_multi_source: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    article_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    preferred = {source.lower() for source in preferred_sources or []}
    source_limit = max_articles_per_source if requires_multi_source else max(top_k, max_articles_per_source)
    for result in results:
        metadata = dict(result.get("metadata") or {})
        article_id = str(metadata.get("article_id") or "")
        source = str(metadata.get("source") or "").lower()
        if article_id and article_counts.get(article_id, 0) >= max_chunks_per_article:
            continue
        if source and source not in preferred and source_counts.get(source, 0) >= source_limit:
            continue
        selected.append(result)
        article_counts[article_id] = article_counts.get(article_id, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def normalizer(values: list[float]):
    if not values:
        return lambda value: 0.0
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return lambda value: 1.0 if value > 0 else 0.0
    return lambda value: (safe_float(value) - min_value) / (max_value - min_value)


def has_ticker(query: str) -> bool:
    return bool(re.search(r"\b[A-Z]{2,8}\b", query))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
