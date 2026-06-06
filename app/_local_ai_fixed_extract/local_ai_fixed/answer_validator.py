from __future__ import annotations

import unicodedata
from typing import Any


SAFE_NO_ANSWER = (
    "Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có đủ thông tin đáng tin cậy "
    "để trả lời câu hỏi này."
)

SAFE_STOCK_FALLBACK = (
    "Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có đủ thông tin đáng tin cậy "
    "để tổng hợp tin chứng khoán tuần này. Bạn có thể thử hỏi hẹp hơn như: "
    "'VN-Index tuần này diễn biến thế nào?' hoặc 'khối ngoại mua bán ra sao?'"
)

_TECHNICAL_FORBIDDEN = (
    "tomcat",
    "sessiontrackingmode",
    "catalina.properties",
    "servlet",
    "java server",
    "apache tomcat",
    "web application",
    "deploy war",
)

_INSTRUCTION_LEAK_PHRASES = (
    "neu du lieu hien co",
    "neu khong du thong tin",
    "cau truc tra loi",
    "expected",
    "actual",
    "debug",
    "instruction",
    "system_instructions",
    "core_rag_rules",
    "retrieved_context",
)

_STOCK_QUERY_TERMS = (
    "chung khoan",
    "co phieu",
    "vn-index",
    "vnindex",
    "hose",
    "hnx",
    "upcom",
    "thanh khoan",
    "khoi ngoai",
    "tu doanh",
    "ma chung khoan",
)

_STOCK_ANSWER_TERMS = (
    "vn-index",
    "vnindex",
    "co phieu",
    "thi truong",
    "thanh khoan",
    "khoi ngoai",
    "hose",
    "hnx",
    "upcom",
    "ma chung khoan",
    "ma co phieu",
)


def validate_answer_against_topic(
    answer: str,
    query: str,
    topic: str,
    sources: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    normalized_answer = _normalize_text(answer)
    normalized_query = _normalize_text(query)
    normalized_topic = _normalize_topic(topic)
    violations: list[str] = []

    if normalized_topic == "economy_finance_stock":
        technical = [term for term in _TECHNICAL_FORBIDDEN if term in normalized_answer]
        if technical:
            violations.append(f"forbidden_technical_terms:{','.join(technical[:3])}")

    leaked = [term for term in _INSTRUCTION_LEAK_PHRASES if term in normalized_answer]
    if leaked:
        violations.append(f"instruction_leak:{','.join(leaked[:3])}")

    if normalized_topic == "economy_finance_stock" and _has_any(normalized_query, _STOCK_QUERY_TERMS):
        if not _has_any(normalized_answer, _STOCK_ANSWER_TERMS):
            violations.append("stock_answer_missing_stock_terms")

    if sources and normalized_topic and normalized_topic != "general_news":
        source_topics = {_normalize_topic(str(source.get("topic") or source.get("primary_topic") or "")) for source in sources}
        if source_topics and normalized_topic not in source_topics:
            violations.append("sources_do_not_match_answer_topic")

    return not violations, violations


def safe_fallback_for_topic(query: str, topic: str) -> str:
    normalized_query = _normalize_text(query)
    normalized_topic = _normalize_topic(topic)
    if normalized_topic == "economy_finance_stock" and _has_any(normalized_query, _STOCK_QUERY_TERMS):
        return SAFE_STOCK_FALLBACK
    return SAFE_NO_ANSWER


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _normalize_topic(topic: str | None) -> str:
    topic_text = str(topic or "").strip()
    if topic_text == "technology_ai_internet":
        return "tech_ai_internet"
    return topic_text


def _normalize_text(text: str) -> str:
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", str(text).casefold())
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")


__all__ = [
    "SAFE_NO_ANSWER",
    "SAFE_STOCK_FALLBACK",
    "safe_fallback_for_topic",
    "validate_answer_against_topic",
]
