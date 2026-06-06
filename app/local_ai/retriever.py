from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.local_ai.chunking.article_chunker import group_chunks_by_article
from app.local_ai.hybrid_search import HybridSearchEngine, diversify_hybrid_results
from app.local_ai.query_router import domain_from_topic
from app.local_ai.reranker import NewsReranker, normalize_text, safe_float
from app.local_ai.topic_guard import filter_context_for_topic

logger = logging.getLogger(__name__)
_VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "hpg": ("hpg", "hoa phat", "tap doan hoa phat"),
    "hoa phat": ("hpg", "hoa phat", "tap doan hoa phat"),
    "fpt": ("fpt", "tap doan fpt", "fpt corporation"),
    "nvl": ("nvl", "novaland"),
    "novaland": ("nvl", "novaland"),
    "vcb": ("vcb", "vietcombank", "ngan hang ngoai thuong"),
    "vietcombank": ("vcb", "vietcombank", "ngan hang ngoai thuong"),
    "vic": ("vic", "vingroup"),
    "vingroup": ("vic", "vingroup"),
    "vhm": ("vhm", "vinhomes"),
    "vinhomes": ("vhm", "vinhomes"),
    "msn": ("msn", "masan"),
    "masan": ("msn", "masan"),
    "mwg": ("mwg", "the gioi di dong", "mobile world"),
    "the gioi di dong": ("mwg", "the gioi di dong", "mobile world"),
}


class MetadataFilteringRetriever:
    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any,
        reranker: NewsReranker | None = None,
        retrieval_mode: str = "hybrid",
    ) -> None:
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._reranker = reranker or NewsReranker()
        self._retrieval_mode = retrieval_mode if retrieval_mode in {"vector", "hybrid"} else "hybrid"
        self._hybrid_engine: HybridSearchEngine | None = None

    def retrieve(self, query: str, query_plan: dict[str, Any], top_n: int = 50, top_k: int = 8) -> list[dict[str, Any]]:
        candidate_k = _candidate_k(query, query_plan, top_n)
        if self._retrieval_mode == "hybrid":
            candidates = self._hybrid_candidates(query, query_plan, top_n=candidate_k)
        else:
            candidates = self._vector_candidates(query, top_n=candidate_k)
        min_results = max(3, int(top_k or 0))
        topic_filter_applied = bool(query_plan.get("primary_topic")) and _is_high_confidence_topic_query(query_plan)
        scoped_candidates, topic_fallback = _scope_candidates_by_topic(candidates, query_plan, min_results=min_results)
        filtered = filter_results_by_metadata(scoped_candidates, query_plan)
        metadata_filter_applied = len(filtered) != len(scoped_candidates)
        metadata_fallback_used = False
        metadata_fallback_reason = ""
        metadata_fallback_strategy = ""
        metadata_filtered_before_guard = list(filtered)
        if not filtered and scoped_candidates and not _has_explicit_source_filter(query_plan):
            filtered, metadata_fallback_strategy = _metadata_filter_fallback(scoped_candidates, candidates, query_plan)
            if filtered:
                metadata_fallback_used = True
                metadata_fallback_reason = "metadata_filter_removed_all_candidates"
                metadata_filtered_before_guard = list(filtered)
        guard_filtered = filter_context_for_topic(query, str(query_plan.get("primary_topic") or ""), filtered)
        filtered = guard_filtered.kept
        query_plan["_topic_guard_filter"] = {
            "retrieved_count": guard_filtered.retrieved_count,
            "kept_after_topic_filter": guard_filtered.kept_after_topic_filter,
            "dropped_wrong_topic": guard_filtered.dropped_wrong_topic,
            "dropped_deny_keyword": guard_filtered.dropped_deny_keyword,
            "final_count": len(filtered),
            "allowed": guard_filtered.result.allowed,
            "reason": guard_filtered.result.reason,
            "violations": guard_filtered.result.violations,
            "dropped_chunks": _debug_candidate_entries(guard_filtered.dropped[:10]),
        }
        if not filtered and metadata_fallback_used:
            filtered = metadata_filtered_before_guard
            query_plan["_topic_guard_filter"]["fallback_after_guard"] = True
            query_plan["_topic_guard_filter"]["reason"] = "topic_guard_removed_metadata_fallback_candidates"
        if not filtered and not _has_explicit_source_filter(query_plan) and not _is_high_confidence_topic_query(query_plan):
            filtered = scoped_candidates
            topic_fallback = topic_fallback or not _is_high_confidence_topic_query(query_plan)
        fallback_used = bool(topic_fallback or metadata_fallback_used)
        filtered = _attach_retrieval_flags(filtered, topic_fallback=topic_fallback)
        if query_plan.get("debug_scores"):
            query_plan["_retrieval_debug"] = {
                "broad_retrieve_top_n": int(top_n or 0),
                "candidate_k": candidate_k,
                "raw_candidate_count": len(candidates),
                "metadata_filter_applied": metadata_filter_applied,
                "topic_filter_applied": topic_filter_applied,
                "selected_topic": str(query_plan.get("primary_topic") or ""),
                "candidate_count_before_topic_filter": len(candidates),
                "candidate_count_after_topic_filter": len(scoped_candidates),
                "candidate_count_after_metadata_filter": len(filtered),
                "topic_guard": query_plan.get("_topic_guard_filter") or {},
                "fallback_used": fallback_used,
                "fallback_reason": metadata_fallback_reason or ("topic_filter_too_narrow" if topic_fallback else ""),
                "fallback_strategy": metadata_fallback_strategy or ("topic_filter_fallback" if topic_fallback else ""),
                "min_results": min_results,
                "source_diversity_count": _source_diversity_count(filtered),
                "top_candidates_before_filter": _debug_candidate_entries(candidates[:10]),
                "top_candidates_after_topic_filter": _debug_candidate_entries(scoped_candidates[:10]),
                "top_candidates_after_metadata_filter": _debug_candidate_entries(filtered[:10]),
            }
        logger.debug(
            "Retriever query=%s topic=%s candidates_before=%s scoped=%s after_metadata=%s fallback_used=%s",
            query,
            query_plan.get("primary_topic"),
            len(candidates),
            len(scoped_candidates),
            len(filtered),
            topic_fallback,
        )
        reranked = self._reranker.rerank(query, filtered, query_context=query_plan)
        if query_plan.get("debug_scores"):
            query_plan["_rerank_debug"] = _debug_candidate_entries(reranked[:10], include_scores=True)
        return diversify_hybrid_results(
            reranked,
            top_k=top_k,
            max_chunks_per_article=2,
            max_articles_per_source=2 if query_plan.get("requires_multi_source") else max(2, top_k),
            preferred_sources=query_plan.get("preferred_sources") or [],
            requires_multi_source=bool(query_plan.get("requires_multi_source")),
        )

    def _vector_candidates(self, query: str, top_n: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_model.embed_query(query)
        return self._vector_store.search(query_embedding, top_k=top_n)

    def _hybrid_candidates(self, query: str, query_plan: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
        try:
            if self._hybrid_engine is None:
                self._hybrid_engine = HybridSearchEngine.from_vector_store(self._vector_store, self._embedding_model)
            return self._hybrid_engine.search(query, query_plan=query_plan, top_n=top_n)
        except Exception:
            return self._vector_candidates(query, top_n=top_n)


def filter_results_by_metadata(results: list[dict[str, Any]], query_plan: dict[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for result in results:
        metadata = _metadata(result)
        if not _matches_topic(
            metadata,
            query_plan.get("primary_topic"),
            strict_primary=_is_high_confidence_topic_query(query_plan),
        ):
            continue
        if not _matches_entities(
            metadata,
            query_plan.get("entities") or [],
            query_plan.get("stock_symbols") or [],
            result=result,
            lexical_terms=query_plan.get("lexical_terms") or [],
        ):
            continue
        if not _matches_time(
            metadata,
            str(query_plan.get("time_range") or "all"),
            query_plan.get("time_range_days"),
            query_plan.get("date_filter"),
        ):
            continue
        if not _matches_source(metadata, query_plan.get("source"), query_plan.get("preferred_sources") or []):
            continue
        if _requires_image_metadata(query_plan) and not _truthy(metadata.get("has_images")):
            continue
        filtered.append(result)
    return filtered


def _metadata_filter_fallback(
    scoped_candidates: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    query_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    relaxed_entity = [
        result
        for result in scoped_candidates
        if _matches_source(_metadata(result), query_plan.get("source"), query_plan.get("preferred_sources") or [])
        and _matches_entities(
            _metadata(result),
            query_plan.get("entities") or [],
            query_plan.get("stock_symbols") or [],
            result=result,
            lexical_terms=query_plan.get("lexical_terms") or [],
            relaxed=True,
        )
        and (not _requires_image_metadata(query_plan) or _truthy(_metadata(result).get("has_images")))
    ]
    if relaxed_entity:
        return relaxed_entity, "relaxed_entity_filter"

    relaxed_time = [
        result
        for result in scoped_candidates
        if _matches_topic(_metadata(result), query_plan.get("primary_topic"), strict_primary=False)
        and _matches_entities(
            _metadata(result),
            query_plan.get("entities") or [],
            query_plan.get("stock_symbols") or [],
            result=result,
            lexical_terms=query_plan.get("lexical_terms") or [],
            relaxed=True,
        )
        and _matches_source(_metadata(result), query_plan.get("source"), query_plan.get("preferred_sources") or [])
        and (not _requires_image_metadata(query_plan) or _truthy(_metadata(result).get("has_images")))
    ]
    if relaxed_time:
        return relaxed_time, "relaxed_time_filter"

    entity_over_topic = [
        result
        for result in raw_candidates
        if _matches_entities(
            _metadata(result),
            query_plan.get("entities") or [],
            query_plan.get("stock_symbols") or [],
            result=result,
            lexical_terms=query_plan.get("lexical_terms") or [],
            relaxed=True,
        )
        and _matches_source(_metadata(result), query_plan.get("source"), query_plan.get("preferred_sources") or [])
    ]
    if entity_over_topic:
        return entity_over_topic, "entity_over_topic_fallback"

    return scoped_candidates[: max(3, int(query_plan.get("latest_article_count") or 8))], "pre_filter_topk_fallback"


def _scope_candidates_by_topic(
    candidates: list[dict[str, Any]],
    query_plan: dict[str, Any],
    min_results: int,
) -> tuple[list[dict[str, Any]], bool]:
    topic = query_plan.get("primary_topic")
    if not topic or not _is_high_confidence_topic_query(query_plan):
        return candidates, False
    topic_candidates = [
        candidate
        for candidate in candidates
        if _matches_topic(_metadata(candidate), topic, strict_primary=True)
    ]
    if len(topic_candidates) >= max(1, int(min_results or 1)):
        return topic_candidates, False
    return candidates, True


def _is_high_confidence_topic_query(query_plan: dict[str, Any]) -> bool:
    if bool(query_plan.get("explicit_topic")):
        return True
    try:
        confidence = float(query_plan.get("topic_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence >= 0.75


def _attach_retrieval_flags(results: list[dict[str, Any]], topic_fallback: bool) -> list[dict[str, Any]]:
    if not topic_fallback:
        return results
    output: list[dict[str, Any]] = []
    for result in results:
        updated = dict(result)
        updated["topic_filter_fallback"] = True
        output.append(updated)
    return output


def _source_diversity_count(results: list[dict[str, Any]]) -> int:
    return len({str(_metadata(result).get("source") or "") for result in results if str(_metadata(result).get("source") or "").strip()})


def _debug_candidate_entries(results: list[dict[str, Any]], include_scores: bool = False) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in results:
        metadata = _metadata(result)
        entry: dict[str, Any] = {
            "article_id": str(metadata.get("article_id") or result.get("article_id") or ""),
            "chunk_id": str(result.get("chunk_id") or ""),
            "title": str(metadata.get("title") or result.get("title") or ""),
            "source": str(metadata.get("source") or result.get("source") or ""),
            "topic": str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or ""),
            "published_at": str(metadata.get("published_at") or ""),
            "score": safe_float(result.get("final_score") or result.get("hybrid_score") or result.get("score") or result.get("vector_score")),
            "entity_names": _debug_entity_names(metadata),
            "matched_terms": [str(item) for item in result.get("matched_keywords", []) if str(item).strip()],
            "reason": str(result.get("topic_guard_drop_reason") or result.get("why_selected") or _why_selected(result)),
        }
        if include_scores:
            entry.update(
                {
                    "vector_score": safe_float(result.get("vector_score") or result.get("score")),
                    "keyword_score": safe_float(result.get("keyword_score")),
                    "topic_score": safe_float(result.get("combined_topic_score") or result.get("topic_match")),
                    "topic_penalty": safe_float(result.get("topic_penalty")),
                    "mismatch_penalty": safe_float(result.get("topic_penalty")),
                    "recency_score": safe_float(result.get("recency_score")),
                    "entity_score": safe_float(result.get("combined_entity_score") or result.get("entity_match")),
                    "final_score": safe_float(result.get("final_score") or result.get("score")),
                    "weights_used": result.get("weights_used") or {},
                    "matched_keywords": result.get("matched_keywords", []),
                    "why_selected": _why_selected(result),
                }
            )
            if isinstance(result.get("score_breakdown"), dict):
                entry["score_breakdown"] = result["score_breakdown"]
        entries.append(entry)
    return entries


def _debug_entity_names(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(item.strip() for item in str(metadata.get("entity_names") or "").split(",") if item.strip())
    for entity in _json_list(metadata.get("entities_json")):
        if isinstance(entity, dict):
            value = str(entity.get("normalized_name") or entity.get("name") or "").strip()
        else:
            value = str(entity or "").strip()
        if value:
            values.append(value)
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _why_selected(result: dict[str, Any]) -> str:
    if safe_float(result.get("topic_penalty")) > 0:
        return "penalized: topic mismatch"
    keywords = result.get("matched_keywords")
    if isinstance(keywords, list) and keywords:
        return "selected: topic/domain keyword match"
    if safe_float(result.get("combined_topic_score") or result.get("topic_match")) > 0:
        return "selected: metadata topic match"
    return "selected: retrieval/rerank score"


def _candidate_k(query: str, query_plan: dict[str, Any], top_n: int) -> int:
    base_top_n = max(1, int(top_n or 1))
    candidate_k = max(base_top_n * 10, 50)
    normalized_query = normalize_text(query)
    intent = str(query_plan.get("intent") or "")
    has_time_filter = bool(query_plan.get("time_range_days")) or str(query_plan.get("time_range") or "all") not in {"", "all"}
    has_time_word = any(term in normalized_query for term in ("hom nay", "moi nhat", "gan day", "hien nay", "24h qua"))
    if (
        intent in {"latest_news", "stock_market_overview"}
        or (intent == "topic_news" and has_time_filter)
        or str(query_plan.get("time_range") or "") == "date"
        or has_time_word
    ):
        candidate_k = max(base_top_n * 15, 80)
    return candidate_k


def diversify_results(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    article_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for result in results:
        metadata = _metadata(result)
        article_id = str(metadata.get("article_id") or "")
        source = str(metadata.get("source") or "")
        if article_id and article_counts[article_id] >= 2:
            continue
        if source and source_counts[source] >= max(2, top_k // 2):
            continue
        selected.append(result)
        article_counts[article_id] += 1
        source_counts[source] += 1
        if len(selected) >= top_k:
            break
    return selected


def build_structured_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    best_by_article_id: dict[str, dict[str, Any]] = {}
    best_score_by_article_id: dict[str, float] = {}
    snippets_by_article_id: dict[str, str] = {}
    ordered_article_ids: list[str] = []
    fallback_index = 0
    for result in results:
        metadata = _metadata(result)
        article_id = str(metadata.get("article_id") or "")
        if not article_id:
            fallback_index += 1
            article_id = str(metadata.get("url") or result.get("chunk_id") or f"chunk:{fallback_index}")
        score = max(
            safe_float(result.get("final_score")),
            safe_float(result.get("score")),
            safe_float(result.get("vector_score")),
            safe_float(result.get("hybrid_score")),
        )
        if article_id not in best_by_article_id:
            ordered_article_ids.append(article_id)
            best_by_article_id[article_id] = result
            best_score_by_article_id[article_id] = score
            snippets_by_article_id[article_id] = _source_snippet(result)
            continue
        if score > best_score_by_article_id[article_id]:
            best_by_article_id[article_id] = result
            best_score_by_article_id[article_id] = score
            snippets_by_article_id[article_id] = _source_snippet(result)

    for article_id in ordered_article_ids:
        result = best_by_article_id[article_id]
        metadata = _metadata(result)
        topic = str(metadata.get("primary_topic") or "")
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "id": len(sources) + 1,
                "article_id": str(metadata.get("article_id") or article_id),
                "title": str(metadata.get("title") or ""),
                "url": str(metadata.get("url") or metadata.get("canonical_url") or ""),
                "source": str(metadata.get("source") or ""),
                "published_at": str(metadata.get("published_at") or ""),
                "primary_topic": topic,
                "topic": topic,
                "domain": domain_from_topic(topic),
                "score": best_score_by_article_id[article_id],
                "snippet": snippets_by_article_id[article_id],
            }
        )
    return sources


def _legacy_build_structured_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        metadata = _metadata(result)
        article_id = str(metadata.get("article_id") or "")
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "article_id": article_id,
                "title": str(metadata.get("title") or ""),
                "url": str(metadata.get("url") or ""),
                "source": str(metadata.get("source") or ""),
                "published_at": str(metadata.get("published_at") or ""),
                "primary_topic": str(metadata.get("primary_topic") or ""),
            }
        )
    return sources


def collect_related_images(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        metadata = _metadata(result)
        image_items = _json_list(metadata.get("images_json"))
        for image in image_items:
            if not isinstance(image, dict):
                continue
            image_url = str(image.get("image_url") or "")
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            images.append(
                {
                    "article_id": str(metadata.get("article_id") or ""),
                    "image_url": image_url,
                    "caption": str(image.get("caption") or ""),
                    "credit": str(image.get("credit") or ""),
                    "source": str(metadata.get("source") or ""),
                    "article_title": str(metadata.get("title") or ""),
                }
            )
    return images


def _matches_topic(metadata: dict[str, Any], primary_topic: Any, strict_primary: bool = False) -> bool:
    if not primary_topic:
        return True
    topic = str(primary_topic)
    metadata_topic = str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or "")
    if metadata_topic == topic:
        return True
    # Check secondary topics before applying strict gate.
    secondary = _json_list(metadata.get("secondary_topics_json"))
    if topic in {str(item) for item in secondary}:
        return True
    if strict_primary:
        # Accept articles with strong topic signals in title/description even when
        # the primary topic was indexed too broadly.
        topic_content_signals: dict[str, tuple[str, ...]] = {
            "economy_finance_stock": (
                "co phieu", "chung khoan", "vn-index", "vnindex", "hnx", "upcom",
                "loi nhuan", "eps", "p/e", "ma co phieu", "khoi ngoai", "tu doanh",
                "von hoa", "thanh khoan", "giao dich", "niem yet",
            ),
            "real_estate": (
                "bat dong san", "can ho", "chung cu", "dat nen", "du an nha o",
                "gia nha", "thi truong bds",
            ),
            "tech_ai_internet": (
                "tri tue nhan tao", "cong nghe", "chip ban dan", "ai generative",
                "openai", "google deepmind",
            ),
        }
        signals = topic_content_signals.get(topic)
        if signals:
            haystack = normalize_text(
                " ".join(
                    filter(
                        None,
                        [
                            str(metadata.get("title") or ""),
                            str(metadata.get("description") or ""),
                        ],
                    )
                )
            )
            if any(signal in haystack for signal in signals):
                return True
        return False
    return False


def _matches_entities(
    metadata: dict[str, Any],
    entities: list[str],
    stock_symbols: list[str],
    result: dict[str, Any] | None = None,
    lexical_terms: list[str] | tuple[str, ...] | None = None,
    relaxed: bool = False,
) -> bool:
    requested = {normalize_text(str(item)) for item in [*entities, *stock_symbols] if str(item).strip()}
    if not requested:
        return True
    requested_aliases = _expanded_aliases([*requested, *[normalize_text(str(item)) for item in stock_symbols if str(item).strip()]])
    entity_names = {normalize_text(item.strip()) for item in str(metadata.get("entity_names") or "").split(",") if item.strip()}
    entity_names.update(normalize_text(item.strip()) for item in str(metadata.get("stock_symbols") or "").split(",") if item.strip())
    entities_json = _json_list(metadata.get("entities_json"))
    for entity in entities_json:
        if isinstance(entity, dict):
            entity_names.add(normalize_text(str(entity.get("normalized_name") or entity.get("name") or "")))
        elif str(entity).strip():
            entity_names.add(normalize_text(str(entity)))
    if requested.intersection(entity_names) or requested_aliases.intersection(entity_names):
        return True
    for requested_name in requested_aliases:
        for entity_name in entity_names:
            if _entity_alias_match(requested_name, entity_name):
                return True
    content_haystack = normalize_text(
        " ".join(
            filter(
                None,
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("summary") or ""),
                    str(metadata.get("summary_raw") or ""),
                    str(metadata.get("description") or ""),
                    str(metadata.get("chunk_text") or ""),
                    str(result.get("text") or result.get("document") or "") if isinstance(result, dict) else "",
                ],
            )
        )
    )
    for requested_name in requested_aliases:
        if len(requested_name) >= 2 and requested_name in content_haystack:
            return True
    if relaxed:
        lexical_aliases = _expanded_aliases([normalize_text(str(item)) for item in lexical_terms or [] if str(item).strip()])
        strong_lexical_matches = [term for term in lexical_aliases if len(term) >= 3 and term in content_haystack]
        if strong_lexical_matches:
            return True
    return False


def _expanded_aliases(values: list[str]) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        aliases.add(normalized)
        aliases.update(_ENTITY_ALIASES.get(normalized, ()))
        compact = _compact_entity_name(normalized)
        if compact:
            aliases.add(compact)
            aliases.update(_ENTITY_ALIASES.get(compact, ()))
    return aliases


def _entity_alias_match(requested_name: str, entity_name: str) -> bool:
    if not requested_name or not entity_name:
        return False
    requested_compact = _compact_entity_name(requested_name)
    entity_compact = _compact_entity_name(entity_name)
    if requested_compact and requested_compact == entity_compact:
        return True
    if requested_name == "ai":
        return entity_name in {"ai", "openai"} or entity_name.endswith(" ai")
    if len(requested_name) >= 3 and requested_name in entity_name:
        return True
    return len(entity_name) >= 3 and entity_name in requested_name


def _compact_entity_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


def _matches_time(
    metadata: dict[str, Any],
    time_range: str,
    time_range_days: Any = None,
    date_filter: Any = None,
) -> bool:
    if date_filter:
        return _matches_date_filter(str(metadata.get("published_at") or ""), date_filter)
    if time_range in {"", "all"}:
        try:
            days = int(time_range_days or 0)
        except (TypeError, ValueError):
            days = 0
        if days <= 0:
            return True
        published = _parse_datetime(str(metadata.get("published_at") or ""))
        if published is None:
            return False
        return published >= datetime.now(timezone.utc) - timedelta(days=days)
    published = _parse_datetime(str(metadata.get("published_at") or ""))
    if published is None:
        return False
    now = datetime.now(timezone.utc)
    if time_range == "today":
        return published.date() == now.date()
    if time_range == "24h":
        return published >= now - timedelta(hours=24)
    if time_range == "7d":
        return published >= now - timedelta(days=7)
    try:
        days = int(time_range_days or 0)
    except (TypeError, ValueError):
        days = 0
    if days > 0:
        return published >= now - timedelta(days=days)
    return True


def _matches_date_filter(published_at: str, date_filter: Any) -> bool:
    if not isinstance(date_filter, dict):
        return True
    exact_date = str(date_filter.get("exact_date") or date_filter.get("date") or "").strip()
    if exact_date:
        return _published_on_date(published_at, exact_date)
    start_date = str(date_filter.get("start_date") or "").strip()
    end_date = str(date_filter.get("end_date") or "").strip()
    if not start_date:
        return True
    published = _parse_datetime(published_at)
    if published is None:
        return False
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) if end_date else start + timedelta(days=1)
    except ValueError:
        return True
    return start <= published < end


def _published_on_date(published_at: str, target_date: str) -> bool:
    text = str(published_at or "").strip()
    if not text:
        return False
    if re_match := _date_prefix(text):
        if re_match == target_date:
            return True
    published = _parse_datetime(text)
    if published is None:
        return False
    return published.astimezone(_VIETNAM_TZ).date().isoformat() == target_date


def _date_prefix(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return ""


def _matches_source(metadata: dict[str, Any], source: Any, preferred_sources: list[Any] | tuple[Any, ...] = ()) -> bool:
    requested = {str(item) for item in preferred_sources if str(item).strip()}
    if source:
        requested.add(str(source))
    if not requested:
        return True
    return str(metadata.get("source") or "") in requested


def _has_explicit_source_filter(query_plan: dict[str, Any]) -> bool:
    if str(query_plan.get("source") or "").strip():
        return True
    preferred_sources = query_plan.get("preferred_sources") or []
    if not isinstance(preferred_sources, (list, tuple, set)):
        preferred_sources = [preferred_sources]
    return any(str(source).strip() for source in preferred_sources)


def _requires_image_metadata(query_plan: dict[str, Any]) -> bool:
    return bool(query_plan.get("need_images")) and str(query_plan.get("intent") or "") != "media_lookup"


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _source_snippet(result: dict[str, Any], max_chars: int = 240) -> str:
    text = " ".join(str(result.get("text") or result.get("document") or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_datetime(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = [
    "MetadataFilteringRetriever",
    "build_structured_sources",
    "collect_related_images",
    "diversify_results",
    "filter_results_by_metadata",
    "group_chunks_by_article",
]
