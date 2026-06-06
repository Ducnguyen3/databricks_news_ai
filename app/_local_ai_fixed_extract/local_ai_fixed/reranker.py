from __future__ import annotations

import logging
import re
import unicodedata
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


TOPIC_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "economy_finance_stock": (
        "tài chính",
        "chứng khoán",
        "cổ phiếu",
        "vn-index",
        "ngân hàng",
        "lãi suất",
        "thanh khoản",
    ),
    "real_estate": (
        "dự án",
        "quy hoạch",
        "pháp lý",
        "giá",
        "thanh khoản",
        "hạ tầng",
        "nhà đất",
    ),
    "world_geopolitics": (
        "quốc gia",
        "xung đột",
        "ngoại giao",
        "quân sự",
        "chiến tranh",
        "tuyên bố",
        "ukraine",
    ),
}

STOCK_STRONG_KEYWORDS = (
    "chứng khoán",
    "cổ phiếu",
    "vn-index",
    "VNIndex",
    "hnx",
    "upcom",
    "hose",
    "thanh khoản",
    "khớp lệnh",
    "mã cổ phiếu",
    "tăng trần",
    "giảm sàn",
    "sắc tím",
    "khối ngoại",
    "tự doanh",
    "vốn hóa",
    "bluechip",
    "tím trần",
)
STOCK_WEAK_KEYWORDS = ("doanh nghiệp", "công ty", "ngân hàng", "tài chính")


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


class NewsReranker:
    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        query_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        context = query_context or {}
        query_terms = normalized_terms(query)
        requested_entities = {normalize_text(str(item)) for item in [*context.get("entities", []), *context.get("stock_symbols", [])]}
        requested_topic = str(context.get("primary_topic") or "")
        preferred_sources = {str(item) for item in context.get("preferred_sources", [])}
        need_images = bool(context.get("need_images"))
        ranked: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}

        for result in results:
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            text = str(result.get("text") or metadata.get("chunk_text") or "")
            title = str(metadata.get("title") or "")
            source = str(metadata.get("source") or "")
            vector_score = safe_float(result.get("score") or result.get("vector_score"))
            bm25_score = safe_float(result.get("bm25_score"))
            hybrid_score = safe_float(result.get("hybrid_score"))
            keyword_score = overlap_score(query_terms, normalized_terms(f"{title} {text}"))
            topic_score = _topic_match_score(requested_topic, metadata)
            topic_hint_score = _topic_hint_score(requested_topic, f"{title} {text} {metadata.get('category') or ''}")
            entity_score = _entity_match_score(requested_entities, str(metadata.get("entity_names") or ""))
            entity_primary_score = _entity_primary_score(requested_entities, f"{title} {text}")
            recency_score = compute_recency_score(
                str(metadata.get("published_at") or ""),
                half_life_days=_recency_half_life_days(query, context),
            )
            source_score = _source_score(source, preferred_sources)
            image_score = 0.05 if need_images and _truthy(metadata.get("has_images")) else 0.0
            diversity_penalty = min(0.2, source_counts.get(source, 0) * 0.05)
            stock_overview_score = _stock_overview_score(context, f"{title} {text}")
            finance_keyword_score, matched_keywords = _finance_keyword_score(requested_topic, f"{title} {text}")
            if finance_keyword_score:
                keyword_score = max(keyword_score, finance_keyword_score)
            stock_single_name_penalty = _single_stock_article_penalty(context, f"{title} {text}")
            peripheral_entity_penalty = _peripheral_entity_penalty(query, requested_entities, f"{title} {text}")
            source_counts[source] = source_counts.get(source, 0) + 1

            requires_lexical = bool(context.get("requires_lexical"))
            relevance_score = _relevance_score(
                vector_score=vector_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score,
                keyword_score=keyword_score,
                requires_lexical=requires_lexical,
            )
            combined_entity_score = max(entity_score, entity_primary_score) - peripheral_entity_penalty
            combined_entity_score = max(0.0, min(1.0, combined_entity_score))
            combined_topic_score = max(topic_score, topic_hint_score)
            topic_penalty = _topic_mismatch_penalty(requested_topic, metadata, context)
            quality_score = _quality_score(stock_overview_score, stock_single_name_penalty)
            weights = _weights_for_intent(str(context.get("intent") or ""), str(context.get("sub_intent") or ""), str(context.get("answer_mode") or ""))
            final_score = (
                relevance_score * weights.get("relevance", 0.0)
                + recency_score * weights.get("recency", 0.0)
                + combined_entity_score * weights.get("entity", 0.0)
                + combined_topic_score * weights.get("topic", 0.0)
                + quality_score * weights.get("quality", 0.0)
                + source_score * 0.05
                + image_score
                - diversity_penalty
                - topic_penalty
            )
            updated = dict(result)
            updated.update(
                {
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score,
                    "relevance_score": relevance_score,
                    "keyword_score": keyword_score,
                    "topic_match": topic_score,
                    "topic_hint_score": topic_hint_score,
                    "combined_topic_score": combined_topic_score,
                    "topic_penalty": topic_penalty,
                    "entity_match": entity_score,
                    "entity_primary_match": entity_primary_score,
                    "combined_entity_score": combined_entity_score,
                    "recency_score": recency_score,
                    "source_score": source_score,
                    "stock_overview_score": stock_overview_score,
                    "stock_single_name_penalty": stock_single_name_penalty,
                    "peripheral_entity_penalty": peripheral_entity_penalty,
                    "quality_score": quality_score,
                    "matched_keywords": matched_keywords,
                    "weights_used": weights,
                    "final_score": final_score,
                }
            )
            if context.get("debug_scores"):
                updated["debug_score"] = {
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "topic_score": combined_topic_score,
                    "topic_penalty": topic_penalty,
                    "recency_score": recency_score,
                    "entity_score": combined_entity_score,
                    "weights_used": weights,
                    "final_score": final_score,
                    "topic": str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or ""),
                    "matched_keywords": matched_keywords,
                }
            ranked.append(updated)

        ranked.sort(key=lambda item: safe_float(item.get("final_score")), reverse=True)
        logger.debug(
            "Reranker query=%s topic=%s top_scores=%s",
            query,
            requested_topic,
            [
                {
                    "title": str((item.get("metadata") or {}).get("title") or "")[:80],
                    "topic": str((item.get("metadata") or {}).get("primary_topic") or ""),
                    "final_score": round(safe_float(item.get("final_score")), 4),
                    "topic_penalty": round(safe_float(item.get("topic_penalty")), 4),
                    "matched_keywords": item.get("matched_keywords", []),
                }
                for item in ranked[:5]
            ],
        )
        return ranked


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


def compute_recency_score(
    published_at: str | None,
    now: datetime | None = None,
    half_life_days: float = 7.0,
) -> float:
    if not published_at:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(parsed.tzinfo)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    reference = reference.astimezone(parsed.tzinfo)
    days_old = max(0.0, (reference - parsed).total_seconds() / 86400.0)
    half_life = max(0.1, float(half_life_days or 7.0))
    return max(0.0, min(1.0, math.exp(-days_old / half_life)))


def _weights_for_intent(intent: str, sub_intent: str = "", answer_mode: str = "") -> dict[str, float]:
    if intent == "stock_market_overview" or sub_intent == "stock_market_overview":
        return {"relevance": 0.32, "recency": 0.12, "entity": 0.04, "topic": 0.28, "quality": 0.24}
    if answer_mode == "citation":
        return {"relevance": 0.46, "recency": 0.06, "entity": 0.34, "topic": 0.14}
    if answer_mode == "followup":
        return {"relevance": 0.36, "recency": 0.05, "entity": 0.24, "topic": 0.35}
    if answer_mode == "synthesis":
        return {"relevance": 0.38, "recency": 0.06, "entity": 0.08, "topic": 0.48}
    if intent == "latest_news":
        # tăng trọng số độ mới, nhưng topic vẫn chi phối — bài sai topic không thể vượt lên chỉ nhờ mới hơn
        return {"relevance": 0.38, "recency": 0.16, "entity": 0.08, "topic": 0.38}
    if intent == "entity_news":
        return {"relevance": 0.45, "recency": 0.10, "entity": 0.35, "topic": 0.10}
    if intent == "article_summary" or intent.startswith("followup_"):
        return {"relevance": 0.75, "recency": 0.0, "entity": 0.15, "topic": 0.10}
    if intent == "topic_news":
        # topic là tín hiệu chính; độ mới chỉ là phụ
        return {"relevance": 0.40, "recency": 0.08, "entity": 0.08, "topic": 0.44}
    # mặc định / news_summary
    return {"relevance": 0.50, "recency": 0.10, "entity": 0.15, "topic": 0.25}


def _recency_half_life_days(query: str, context: dict[str, Any]) -> float:
    normalized_query = normalize_text(query)
    intent = str(context.get("intent") or "")
    if (
        intent == "latest_news"
        or str(context.get("time_range") or "") in {"today", "24h"}
        or normalize_text("hôm nay") in normalized_query
    ):
        return 1.5
    if (
        normalize_text("gần đây") in normalized_query
        or normalize_text("mới nhất") in normalized_query
        or str(context.get("time_range") or "") == "7d"
    ):
        return 5.0  # siết từ 7.0 — độ mới giảm nhanh hơn, để topic có nhiều trọng số hơn
    if intent == "stock_market_overview":
        return 3.0
    return 30.0


def _relevance_score(
    vector_score: float,
    bm25_score: float,
    hybrid_score: float,
    keyword_score: float,
    requires_lexical: bool,
) -> float:
    vector_weight = 0.30 if not requires_lexical else 0.25
    bm25_weight = 0.25 if not requires_lexical else 0.35
    hybrid_weight = 0.35
    keyword_weight = 0.10 if not requires_lexical else 0.05
    return (
        _bounded_score(hybrid_score) * hybrid_weight
        + _bounded_score(vector_score) * vector_weight
        + _bounded_score(bm25_score) * bm25_weight
        + max(0.0, min(1.0, keyword_score)) * keyword_weight
    )


def _bounded_score(value: float) -> float:
    if value <= 0:
        return 0.0
    if value <= 1:
        return value
    return value / (1.0 + value)


def _entity_match_score(requested_entities: set[str], entity_names: str) -> float:
    if not requested_entities or not entity_names:
        return 0.0
    available = {normalize_text(item.strip()) for item in entity_names.split(",") if item.strip()}
    if not available:
        return 0.0
    return len(requested_entities.intersection(available)) / max(1, len(requested_entities))


def _entity_primary_score(requested_entities: set[str], text: str) -> float:
    if not requested_entities:
        return 0.0
    normalized = normalize_text(text)
    matches = sum(1 for entity in requested_entities if entity and entity in normalized)
    return matches / max(1, len(requested_entities))


def _stock_overview_score(context: dict[str, Any], text: str) -> float:
    if context.get("intent") != "stock_market_overview" and context.get("sub_intent") != "stock_market_overview":
        return 0.0
    terms = (
        "vn-index",
        "VNIndex",
        "vn index",
        "hnx-index",
        "hnx",
        "upcom",
        "thị trường chứng khoán",
        "chứng khoán Việt Nam",
        "thanh khoản",
        "nhóm ngành",
        "khối ngoại",
        "nhà đầu tư",
        "cổ phiếu ngân hàng",
        "cổ phiếu chứng khoán",
        "sàn hose",
        "hose",
        "tăng điểm",
        "giảm điểm",
        "sắc xanh",
        "sắc đỏ",
        "thị trường",
        "chỉ số",
    )
    normalized = normalize_text(text)
    hits = sum(1 for term in terms if normalize_text(term) in normalized)
    return min(1.0, hits / 4)


def _single_stock_article_penalty(context: dict[str, Any], text: str) -> float:
    if context.get("intent") != "stock_market_overview" and context.get("sub_intent") != "stock_market_overview":
        return 0.0
    noisy_terms = (
        "giải trình",
        "tăng trần",
        "trần nhiều phiên",
        "trần 8 phiên",
        "chia cổ tức",
        "tăng vốn",
        "lên sàn",
        "doanh thu nhiều quý",
        "cổ phiếu tăng giá gấp rưỡi",
        "phát hành riêng lẻ",
    )
    normalized = normalize_text(text)
    return 0.25 if any(normalize_text(term) in normalized for term in noisy_terms) else 0.0


def _quality_score(stock_overview_score: float, stock_single_name_penalty: float) -> float:
    return max(0.0, min(1.0, stock_overview_score - stock_single_name_penalty))


def _peripheral_entity_penalty(query: str, requested_entities: set[str], text: str) -> float:
    if not requested_entities:
        return 0.0
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    if any(normalize_text(term) in normalized_query for term in ("chuyển nhầm", "chuyển tiền", "giao dịch", "đòi lại tiền")):
        return 0.0
    incidental_terms = ("chuyển nhầm", "số tài khoản", "chủ tài khoản", "không quen biết", "đòi lại tiền", "bất cẩn")
    if "techcombank" in requested_entities and any(normalize_text(term) in normalized_text for term in incidental_terms):
        return 0.30
    return 0.0


def _topic_match_score(requested_topic: str, metadata: dict[str, Any]) -> float:
    if not requested_topic:
        return 0.0
    if requested_topic == _metadata_topic(metadata):
        return 1.0
    secondary = str(metadata.get("secondary_topics_json") or "")
    return 0.5 if requested_topic and requested_topic in secondary else 0.0


def _metadata_topic(metadata: dict[str, Any]) -> str:
    return str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or "")


def _topic_mismatch_penalty(requested_topic: str, metadata: dict[str, Any], context: dict[str, Any]) -> float:
    if not requested_topic:
        return 0.0
    if _topic_match_score(requested_topic, metadata) > 0:
        return 0.0
    # Nếu topic được xác nhận rõ ràng, áp dụng phạt nặng
    if bool(context.get("explicit_topic")):
        return 0.90  # tăng từ 0.85 — bài sai topic không thể vượt bài đúng topic
    try:
        confidence = float(context.get("topic_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence >= 0.75:
        return 0.80  # tăng từ 0.75
    if confidence >= 0.35:
        return 0.45  # ngưỡng trung bình mới
    return 0.25  # độ tin cậy thấp — chỉ phạt nhẹ


def _topic_hint_score(requested_topic: str, text: str) -> float:
    hints = TOPIC_KEYWORD_HINTS.get(requested_topic)
    if not hints:
        return 0.0
    text_terms = normalized_terms(text)
    hint_terms = normalized_terms(" ".join(hints))
    return overlap_score(hint_terms, text_terms)


def _finance_keyword_score(requested_topic: str, text: str) -> tuple[float, list[str]]:
    if requested_topic != "economy_finance_stock":
        return 0.0, []
    normalized = normalize_text(text)
    strong_matches = [keyword for keyword in STOCK_STRONG_KEYWORDS if normalize_text(keyword) in normalized]
    weak_matches = [keyword for keyword in STOCK_WEAK_KEYWORDS if normalize_text(keyword) in normalized]
    if not strong_matches:
        return 0.0, []
    score = min(1.0, (len(strong_matches) * 0.24) + min(0.16, len(weak_matches) * 0.04))
    return score, [*strong_matches, *weak_matches]


def _source_score(source: str, preferred_sources: set[str]) -> float:
    weights = {
        "vnexpress": 1.0,
        "cafef": 0.95,
        "genk": 0.9,
        "diendandoanhnghiep": 0.85,
    }
    if source in preferred_sources:
        return 1.0
    return weights.get(source, 0.7)


def _recency_score(value: str) -> float:
    return compute_recency_score(value, half_life_days=30.0)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
