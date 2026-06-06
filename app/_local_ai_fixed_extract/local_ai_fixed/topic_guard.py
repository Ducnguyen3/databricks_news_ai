from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TopicGuardResult:
    allowed: bool
    reason: str
    score: float
    violations: list[str]


@dataclass(frozen=True, slots=True)
class TopicFilteredContext:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    result: TopicGuardResult
    retrieved_count: int
    kept_after_topic_filter: int
    dropped_wrong_topic: int
    dropped_deny_keyword: int


_TOPIC_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "economy_finance_stock": {
        "allow": (
            "chung khoan",
            "co phieu",
            "vn-index",
            "vnindex",
            "hnx",
            "upcom",
            "hose",
            "thanh khoan",
            "khoi ngoai",
            "tu doanh",
            "bluechip",
            "ngan hang",
            "bat dong san",
            "thep",
            "dau khi",
            "ma co phieu",
            "ma chung khoan",
            "hpg",
            "fpt",
            "vcb",
            "vnm",
            "ssi",
            "vnd",
            "mbs",
            "vndirect",
            "loi nhuan",
            "eps",
            "p/e",
            "thi truong",
            "giao dich",
            "diem so",
            "tang giam",
            "ban rong",
            "mua rong",
        ),
        "deny": (
            "tomcat",
            "sessiontrackingmode",
            "catalina.properties",
            "servlet",
            "java server",
            "apache tomcat",
            "web application session",
            "deploy war",
            "spring boot",
            "kubernetes",
            "docker",
            "dependency",
            "package",
            "api key",
        ),
    },
    "tech_ai_internet": {
        "allow": (
            "ai",
            "tri tue nhan tao",
            "cong nghe",
            "chip",
            "internet",
            "du lieu",
            "bao mat",
            "an ninh mang",
            "phan mem",
            "startup cong nghe",
            "robot",
            "cloud",
            "openai",
            "google",
            "nvidia",
            "microsoft",
        ),
        "deny": ("vn-index", "vnindex", "hose", "khoi ngoai", "can ho", "du an dat nen"),
    },
    "real_estate": {
        "allow": (
            "bat dong san",
            "nha dat",
            "can ho",
            "chung cu",
            "dat nen",
            "quy hoach",
            "du an",
            "phap ly",
            "gia nha",
            "nha o xa hoi",
            "khu do thi",
            "chu dau tu",
            "so do",
            "mat bang",
            "lai vay mua nha",
        ),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties", "vn-index", "vnindex"),
    },
    "world_geopolitics": {
        "allow": (
            "quoc te",
            "the gioi",
            "my",
            "trung quoc",
            "nga",
            "ukraine",
            "israel",
            "gaza",
            "eu",
            "nato",
            "chien tranh",
            "xung dot",
            "ngoai giao",
            "dia chinh tri",
            "bau cu",
            "thue quan",
            "thuong mai quoc te",
        ),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties", "can ho ha noi", "dat nen ha noi"),
    },
    "politics_society": {
        "allow": (
            "chinh phu",
            "quoc hoi",
            "xa hoi",
            "phap luat",
            "giao thong",
            "chinh sach",
            "bo nganh",
            "dia phuong",
            "dan sinh",
            "giao duc xa hoi",
            "y te xa hoi",
            "cong an",
            "toa an",
        ),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties", "servlet", "java server"),
    },
    "business_startup": {
        "allow": (
            "doanh nghiep",
            "khoi nghiep",
            "startup",
            "ceo",
            "cong ty",
            "kinh doanh",
            "loi nhuan",
            "doanh thu",
            "goi von",
            "chien luoc",
            "san xuat",
            "xuat khau",
            "thuong hieu",
        ),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties", "servlet", "java server"),
    },
    "lifestyle_education_health_entertainment": {
        "allow": (
            "doi song",
            "giao duc",
            "suc khoe",
            "giai tri",
            "nguoi noi tieng",
            "hoc sinh",
            "sinh vien",
            "benh vien",
            "dinh duong",
            "phim",
            "am nhac",
            "du lich",
            "the thao",
        ),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties", "vn-index", "vnindex", "hose", "khoi ngoai"),
    },
    "general_news": {
        "allow": (),
        "deny": ("tomcat", "sessiontrackingmode", "catalina.properties"),
    },
}

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


def validate_context_for_topic(query: str, topic: str, chunks: list[dict[str, Any]]) -> TopicGuardResult:
    return filter_context_for_topic(query, topic, chunks).result


def filter_context_for_topic(query: str, topic: str, chunks: list[dict[str, Any]]) -> TopicFilteredContext:
    normalized_topic = _normalize_topic(topic)
    if not chunks:
        result = TopicGuardResult(False, "empty_context", 0.0, ["no_chunks"])
        return TopicFilteredContext([], [], result, 0, 0, 0, 0)

    query_norm = _normalize_text(query)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    dropped_wrong_topic = 0
    dropped_deny_keyword = 0
    topic_matches = 0
    allow_matches = 0
    violations: list[str] = []

    for index, chunk in enumerate(chunks):
        metadata = _metadata(chunk)
        text = _chunk_text(chunk)
        text_norm = _normalize_text(f"{metadata.get('title') or ''} {text}")
        chunk_topic = _metadata_topic(metadata)
        has_topic_match = not normalized_topic or normalized_topic == "general_news" or chunk_topic == normalized_topic
        has_allow = _has_any(text_norm, _profile_terms(normalized_topic, "allow"))
        deny_terms = _matched_terms(text_norm, _profile_terms(normalized_topic, "deny"))

        if has_topic_match:
            topic_matches += 1
        if has_allow:
            allow_matches += 1

        drop_reason = ""
        if deny_terms and not _has_any(query_norm, deny_terms):
            drop_reason = f"deny_keyword:{','.join(deny_terms[:3])}"
            dropped_deny_keyword += 1
        elif normalized_topic and normalized_topic != "general_news" and not has_topic_match:
            if not _allow_cross_topic(query_norm, normalized_topic, chunk_topic, text_norm):
                drop_reason = f"wrong_topic:{chunk_topic or 'unknown'}"
                dropped_wrong_topic += 1

        if drop_reason:
            updated = dict(chunk)
            updated["topic_guard_drop_reason"] = drop_reason
            dropped.append(updated)
        else:
            kept.append(chunk)

        if index < 5 and deny_terms:
            violations.append(f"top5_deny_keyword:{','.join(deny_terms[:3])}")

    retrieved_count = len(chunks)
    wrong_topic_ratio = dropped_wrong_topic / max(1, retrieved_count)
    deny_count = dropped_deny_keyword

    if len(kept) < min(2, retrieved_count):
        reason = "too_few_valid_chunks_after_guard"
        result = TopicGuardResult(False, reason, len(kept) / max(1, retrieved_count), [*violations, reason])
    elif wrong_topic_ratio > 0.4:
        reason = "wrong_topic_ratio_above_threshold"
        result = TopicGuardResult(False, reason, 1.0 - wrong_topic_ratio, [*violations, reason])
    elif normalized_topic != "general_news" and allow_matches == 0 and topic_matches == 0:
        reason = "no_topic_or_allow_keyword_evidence"
        result = TopicGuardResult(False, reason, 0.0, [*violations, reason])
    else:
        score = (len(kept) / max(1, retrieved_count)) * 0.6 + (topic_matches / max(1, retrieved_count)) * 0.3 + min(0.1, allow_matches * 0.03)
        result = TopicGuardResult(True, "ok", min(1.0, score), violations)

    return TopicFilteredContext(
        kept=kept,
        dropped=dropped,
        result=result,
        retrieved_count=retrieved_count,
        kept_after_topic_filter=len(kept),
        dropped_wrong_topic=dropped_wrong_topic,
        dropped_deny_keyword=dropped_deny_keyword,
    )


def topic_forbidden_terms(topic: str) -> tuple[str, ...]:
    return _profile_terms(_normalize_topic(topic), "deny")


def _allow_cross_topic(query_norm: str, topic: str, chunk_topic: str, text_norm: str) -> bool:
    if not chunk_topic:
        return False
    if topic == "economy_finance_stock" and chunk_topic == "business_startup":
        return _has_any(query_norm, _STOCK_QUERY_TERMS) and _has_any(text_norm, _profile_terms("economy_finance_stock", "allow"))
    return False


def _profile_terms(topic: str, key: str) -> tuple[str, ...]:
    profile = _TOPIC_PROFILES.get(topic) or _TOPIC_PROFILES["general_news"]
    return profile.get(key, ())


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _metadata_topic(metadata: dict[str, Any]) -> str:
    return _normalize_topic(str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or ""))


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("document") or chunk.get("content") or "")


def _matched_terms(text_norm: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _normalize_text(term) in text_norm]


def _has_any(text_norm: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(_normalize_text(term) in text_norm for term in terms if str(term).strip())


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
    "TopicFilteredContext",
    "TopicGuardResult",
    "filter_context_for_topic",
    "topic_forbidden_terms",
    "validate_context_for_topic",
]
