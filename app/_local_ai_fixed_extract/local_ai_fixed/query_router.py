from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.processing.entity_extractor import extract_entities
from app.processing.taxonomy import normalize_text, normalize_topic

logger = logging.getLogger(__name__)

_DOMAIN_BY_TOPIC = {
    "economy_finance_stock": "tai_chinh",
    "technology_ai_internet": "cong_nghe",
    "tech_ai_internet": "cong_nghe",
    "real_estate": "bat_dong_san",
    "lifestyle_education_health_entertainment": "doi_song",
    "politics_society": "chinh_tri_xa_hoi",
    "world_geopolitics": "the_gioi",
    "business_startup": "startup",
    "general_news": "all",
    None: "all",
}
_TICKER_WHITELIST = {
    "VNINDEX",
    "VN-INDEX",
    "HNX",
    "UPCOM",
    "HPG",
    "FPT",
    "VIC",
    "VHM",
    "VCB",
    "TCB",
    "BID",
    "CTG",
    "MWG",
    "VNM",
    "SSI",
    "VND",
    "MSN",
    "GAS",
    "VFS",
    "BSR",
}
_LATEST_KEYWORDS = ("tin moi nhat", "moi nhat", "hom nay", "24h qua", "gan day")
_IMAGE_KEYWORDS = ("co anh", "hinh anh", "anh ve", "xem anh", "cho toi xem anh", "photo", "image", "picture")
_SOURCE_KEYWORDS = ("nguon nao", "bao nao", "trich dan", "nguon")
_TIMELINE_KEYWORDS = ("dien bien", "timeline", "theo thoi gian")

# --- Answer mode detection ---
_SYNTHESIS_SIGNALS = (
    "co gi moi",
    "tin moi",
    "tin tuc",
    "tinh hinh",
    "xu huong",
    "tong hop",
    "tom tat",
    "phan tich",
    "anh huong",
    "tac dong",
    "dien bien",
    "co gi dang chu y",
    "nhu the nao",
    "ra sao",
    "co gi",
    "giai thich",
    "sao vay",
    "vi sao",
    "tai sao",
)
_CITATION_SIGNALS = (
    "trich dan",
    "trich doan",
    "bai bao noi gi",
    "noi chinh xac",
    "so lieu",
    "la bao nhieu",
    "ai noi",
    "khi nao",
    "ngay nao",
    "luc may gio",
    "o dau",
    "ten day du",
    "chinh xac la",
    "nguyen van",
    "copy",
    "dan cu the",
)
_FOLLOWUP_SIGNALS = (
    "vu nay",
    "su viec nay",
    "cai nay",
    "viec do",
    "chuyen do",
    "the nao",
    "sao nhi",
    "no la gi",
    "no anh huong",
    "giai thich them",
    "the co nghia la gi",
)
_STOCK_SYMBOLS = {
    # Bluechips & banks
    "HPG", "FPT", "VIC", "VHM", "VCB", "TCB", "BID", "CTG", "MWG", "VNM",
    "SSI", "VND", "MSN", "GAS", "ACB", "MBB", "VPB", "STB", "HDB", "LPB",
    "SHB", "EIB", "NAB", "BAB",
    # Oil & gas
    "GAS", "PVD", "PVS", "PLX", "BSR", "OIL", "PVC", "PGS", "PGD", "PSH",
    # Steel & materials
    "HPG", "NKG", "HSG", "TIS", "VGS", "POM",
    # Real estate
    "NVL", "PDR", "DIG", "KBC", "HHV", "VHM", "VRE", "CEO", "SCR", "IJC",
    "DXG", "HDG", "NLG", "KDH", "BCM", "SZC", "LDG",
    # Chemicals & fertilizers
    "DPM", "DCM", "BFC", "SFG", "PMC",
    # Tech & telecom
    "FPT", "CMG", "ELC", "VGI",
    # Consumer & retail
    "MWG", "FRT", "DGW", "PNJ", "ANV",
    # Aviation & logistics
    "HVN", "VJC", "GMD", "PAN",
    # Construction & infra
    "CTD", "HBC", "FCN", "VCG", "PC1", "CII",
    # Fin services & securities
    "VCI", "HCM", "ORS", "CTS", "EVF", "FTS", "APG",
    # Transport & port
    "HAH", "VSC", "VOS", "VFR",
    # VFS (Vinfast Financial Services) & others frequently searched
    "VFS", "VEA", "VGT", "TCM", "TNG", "MSH",
    # Indices
    "VNINDEX", "VN-INDEX", "HNX-INDEX", "UPCOM-INDEX",
}
_STRONG_STOCK_TOPIC_KEYWORDS = (
    "chung khoan",
    "co phieu",
    "vn-index",
    "vnindex",
    "hnx",
    "upcom",
    "hose",
    "thanh khoan",
    "khop lenh",
    "tang tran",
    "giam san",
    "sac tim",
    "ma co phieu",
    "khoi ngoai",
    "tu doanh",
    "von hoa",
    "bluechip",
)
_WEAK_STOCK_TOPIC_KEYWORDS = ("doanh nghiep", "cong ty", "ngan hang", "tai chinh")
_SPORTS_SCHEDULE_KEYWORDS = ("lich thi dau", "da luc may gio", "tran nao", "khi nao da", "lich bong da")
_SPORTS_RESULT_KEYWORDS = ("ket qua bong da", "ket qua tran", "ti so", "ty so", "vo dich", "ai vo dich")
_SPORTS_STANDING_KEYWORDS = ("bang xep hang", "bxh", "xep hang")
_SPORTS_CONTEXT_KEYWORDS = (
    "bong da",
    "the thao",
    "c1",
    "cup c1",
    "champions league",
    "ngoai hang anh",
    "world cup",
    "euro",
    "v-league",
    "vleague",
)


@dataclass(frozen=True, slots=True)
class TopicScore:
    topic: str | None
    scores: dict[str, int]
    matched_keywords: dict[str, list[str]]
    confidence: float
    confidence_label: str
    explicit_topic: bool


_TOPIC_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "tech_ai_internet": {
        "strong": (
            "ai",
            "tri tue nhan tao",
            "cong nghe",
            "internet",
            "an ninh mang",
            "du lieu",
            "chip",
            "ban dan",
            "phan mem",
            "ung dung",
            "smartphone",
            "dien thoai",
            "mang xa hoi",
            "zalo",
            "facebook",
            "tiktok",
            "google",
            "openai",
            "chatgpt",
            "nvidia",
            "robot",
            "blockchain",
            "chuyen doi so",
            "dien toan dam may",
            "cloud",
            "big tech",
            "mobile",
        ),
        "medium": (
            "nen tang so",
            "tai khoan so",
            "dinh danh dien tu",
            "vneid",
            "bao mat",
            "ma otp",
            "otp",
            "lua dao online",
            "qua mang",
            "website",
            "backlink",
            "seo",
        ),
        "weak": ("thiet bi thong minh", "gia dung thong minh", "xe dien", "do choi so"),
    },
    "economy_finance_stock": {
        "strong": (
            "kinh te",
            "tai chinh",
            "chung khoan",
            "co phieu",
            "thi truong chung khoan",
            "vn-index",
            "vnindex",
            "hnx",
            "upcom",
            "hose",
            "ngan hang",
            "tin dung",
            "lai suat",
            "ty gia",
            "trai phieu",
            "thanh khoan",
            "khop lenh",
            "ma co phieu",
            "ma chung khoan",
            "tang tran",
            "giam san",
            "sac tim",
            "khoi ngoai",
            "tu doanh",
            "bluechip",
            "von hoa",
            "niem yet",
            "san chung khoan",
            "fed",
            "ngan hang nha nuoc",
            "cpi",
            "gdp",
            "lam phat",
            "xuat khau",
            "nhap khau",
            "vang",
            "usd",
        ),
        "medium": (
            "dong tien",
            "loi nhuan",
            "doanh thu",
            "dau tu",
            "tai san",
            "thue",
            "ngan sach",
            "vay von",
            "huy dong von",
        ),
        "weak": ("thi truong",),
    },
    "politics_society": {
        "strong": (
            "chinh tri",
            "xa hoi",
            "thoi su",
            "quoc hoi",
            "chinh phu",
            "thu tuong",
            "chu tich nuoc",
            "bo cong an",
            "cong an",
            "canh sat",
            "dieu tra",
            "khoi to",
            "bat giu",
            "truy to",
            "toa an",
            "phap luat",
            "phap ly",
            "nghi dinh",
            "thong tu",
            "luat",
            "hanh chinh",
            "dan sinh",
            "nguoi dan",
            "tai nan",
            "giao thong",
            "trat tu",
            "an ninh trat tu",
            "vu an",
        ),
        "medium": ("chinh sach", "co quan chuc nang", "dia phuong"),
        "weak": (),
    },
    "world_geopolitics": {
        "strong": (
            "quoc te",
            "the gioi",
            "dia chinh tri",
            "chien tranh",
            "chien su",
            "xung dot",
            "ukraine",
            "nga",
            "my",
            "trung quoc",
            "chau au",
            "eu",
            "nato",
            "israel",
            "gaza",
            "palestine",
            "iran",
            "han quoc",
            "nhat ban",
            "asean",
            "lien hop quoc",
            "bau cu my",
            "tong thong my",
            "nha trang",
            "donald trump",
            "joe biden",
            "putin",
            "tap can binh",
            "quan su",
            "ngoai giao",
            "trung phat",
            "cang thang",
        ),
        "medium": ("thuong mai quoc te", "chuoi cung ung toan cau", "chinh sach toan cau"),
        "weak": (),
    },
    "business_startup": {
        "strong": (
            "doanh nghiep",
            "cong ty",
            "tap doan",
            "startup",
            "khoi nghiep",
            "ceo",
            "founder",
            "goi von",
            "von dau tu",
            "quy dau tu",
            "m&a",
            "sap nhap",
            "mua ban doanh nghiep",
            "chien luoc kinh doanh",
            "loi nhuan doanh nghiep",
            "ket qua kinh doanh",
            "bao cao tai chinh doanh nghiep",
            "co dong",
            "hoi dong quan tri",
            "thuong hieu",
            "san pham moi",
            "mo rong thi truong",
            "vcci",
        ),
        "medium": (
            "nha may",
            "san xuat",
            "chuoi cung ung",
            "ban le",
            "thuong mai dien tu",
            "marketing",
            "khach hang",
            "doanh nhan",
        ),
        "weak": (),
    },
    "real_estate": {
        "strong": (
            "bat dong san",
            "nha dat",
            "dat nen",
            "chung cu",
            "can ho",
            "nha o xa hoi",
            "du an bat dong san",
            "khu do thi",
            "khu cong nghiep",
            "quy hoach",
            "phap ly du an",
            "gia nha",
            "gia dat",
            "moi gioi",
            "chu dau tu",
            "condotel",
            "shophouse",
            "van phong cho thue",
            "mat bang ban le",
        ),
        "medium": ("xay dung", "ha tang", "giai phong mat bang", "dau gia dat", "so do", "quyen su dung dat"),
        "weak": ("du an",),
    },
    "lifestyle_education_health_entertainment": {
        "strong": (
            "doi song",
            "giao duc",
            "truong hoc",
            "hoc sinh",
            "sinh vien",
            "dai hoc",
            "tuyen sinh",
            "diem thi",
            "suc khoe",
            "benh vien",
            "bac si",
            "benh",
            "y te",
            "dinh duong",
            "giai tri",
            "showbiz",
            "nghe si",
            "ca si",
            "dien vien",
            "phim",
            "am nhac",
            "du lich",
            "am thuc",
            "the thao",
            "bong da",
            "gia dinh",
            "tieu dung",
        ),
        "medium": ("van hoa", "le hoi", "lam dep", "thoi trang", "nha hang", "khach san"),
        "weak": ("cup c1", "champions league"),
    },
}

_GENERIC_QUERY_TERMS = (
    "tin tuc",
    "tong hop",
    "moi nhat",
    "hom nay",
    "tuan nay",
    "thang nay",
    "nam 2026",
    "tinh hinh",
    "dien bien",
    "van de",
    "cap nhat",
    "co gi moi",
)


def domain_from_topic(primary_topic: str | None) -> str:
    return _DOMAIN_BY_TOPIC.get(primary_topic, "all")


def route_query(query: str) -> dict[str, Any]:
    normalized = normalize_text(query)
    ai_question_pronoun = _has_ai_question_pronoun(query, normalized)
    entities_result = extract_entities(title=query)
    entities = [
        str(entity.get("normalized_name") or entity.get("name") or "")
        for entity in entities_result["entities"]
        if entity.get("normalized_name") or entity.get("name")
    ]
    if ai_question_pronoun:
        entities = [entity for entity in entities if normalize_text(entity) != "ai"]
    stock_symbols = _extract_whitelisted_tickers(query, entities_result["entities"])
    topic_result = normalize_topic(source="", source_category=None, title=query)
    topic_score = _score_topics(normalized, stock_symbols, entities)
    primary_topic = topic_score.topic
    if primary_topic == "tech_ai_internet" and ai_question_pronoun:
        primary_topic = None
        topic_score = TopicScore(
            topic=None,
            scores=topic_score.scores,
            matched_keywords=topic_score.matched_keywords,
            confidence=0.0,
            confidence_label="low",
            explicit_topic=False,
        )
    if not primary_topic:
        fallback_topic = str(topic_result["primary_topic"] or "")
        primary_topic = None if fallback_topic == "general_news" else fallback_topic
    if stock_symbols:
        primary_topic = "economy_finance_stock"

    need_images = _has_image_intent(normalized)
    need_sources = any(keyword in normalized for keyword in _SOURCE_KEYWORDS) or True
    date_filter = _date_filter(normalized)
    time_range = "date" if date_filter else _time_range(normalized)
    sports_intent = _sports_structured_intent(normalized)
    if sports_intent:
        primary_topic = "lifestyle_education_health_entertainment"
    stock_overview = _is_stock_market_overview(normalized)
    explicit_stock_topic = _has_explicit_stock_topic(normalized)
    if stock_overview or explicit_stock_topic:
        primary_topic = "economy_finance_stock"
    intent = sports_intent or ("stock_market_overview" if stock_overview else _intent(normalized, primary_topic, entities, stock_symbols, need_images, time_range))
    data_source = "structured_sports" if sports_intent else "article_rag"
    ticker = stock_symbols[0] if stock_symbols else ""
    domain = "tai_chinh" if stock_symbols else domain_from_topic(primary_topic)
    answer_mode = _detect_answer_mode(normalized, intent)
    route = {
        "intent": intent,
        "answer_mode": answer_mode,
        "normalized_query": normalized,
        "primary_topic": primary_topic,
        "domain": domain,
        "ticker": ticker,
        "entities": entities,
        "stock_symbols": stock_symbols,
        "requires_lexical": _requires_lexical(query, normalized, entities, stock_symbols),
        "lexical_terms": _lexical_terms(query, entities, stock_symbols),
        "exact_entities": [*entities, *stock_symbols],
        "needs_recent": time_range in {"today", "24h", "7d", "date"},
        "needs_images": need_images,
        "preferred_sources": [_source_from_query(normalized)] if _source_from_query(normalized) else [],
        "time_range": time_range,
        "date_filter": date_filter,
        "source": _source_from_query(normalized),
        "need_images": need_images,
        "need_sources": need_sources,
        "need_timeline": any(keyword in normalized for keyword in _TIMELINE_KEYWORDS),
        "data_source": data_source,
        "sub_intent": "stock_market_overview" if stock_overview else "",
        "topic_confidence": 1.0 if explicit_stock_topic or stock_overview or stock_symbols else topic_score.confidence,
        "topic_confidence_label": "high" if explicit_stock_topic or stock_overview or stock_symbols else topic_score.confidence_label,
        "explicit_topic": bool(explicit_stock_topic or stock_overview or stock_symbols or topic_score.explicit_topic),
        "matched_keywords": topic_score.matched_keywords.get(primary_topic or "", []),
        "topic_scores": topic_score.scores,
        "topic_matched_keywords": topic_score.matched_keywords,
    }
    logger.debug(
        "Query route original=%s normalized=%s intent=%s selected_topic=%s confidence=%s scores=%s matched=%s",
        query,
        normalized,
        intent,
        primary_topic,
        route["topic_confidence_label"],
        topic_score.scores,
        route["matched_keywords"],
    )
    return route


def _intent(
    normalized: str,
    primary_topic: str | None,
    entities: list[str],
    stock_symbols: list[str],
    need_images: bool,
    time_range: str,
) -> str:
    if need_images:
        return "media_lookup"
    if stock_symbols or entities:
        if stock_symbols:
            return "entity_news"
        if time_range in {"today", "24h", "7d"} and any(keyword in normalized for keyword in _LATEST_KEYWORDS):
            return "latest_news"
        if primary_topic:
            return "topic_news"
        return "entity_news"
    if time_range in {"today", "24h", "7d"} and any(keyword in normalized for keyword in _LATEST_KEYWORDS):
        return "latest_news"
    if primary_topic:
        return "topic_news"
    return "news_summary"


def _time_range(normalized: str) -> str:
    if "hom nay" in normalized:
        return "today"
    if "24h" in normalized or "24 gio" in normalized:
        return "24h"
    if "7 ngay" in normalized or "tuan qua" in normalized or "tuan nay" in normalized or "gan day" in normalized or "moi nhat" in normalized:
        return "7d"
    return "all"


def _date_filter(normalized: str) -> dict[str, str] | None:
    match = re.search(
        r"(?:ngay\s+)?(?P<day>[0-3]?\d)\s*(?:/|-|thang\s+)(?P<month>1[0-2]|0?\d)"
        r"(?:\s*(?:/|-|nam\s+)?(?P<year>19\d{2}|20\d{2}))?",
        normalized,
    )
    if not match:
        return None
    try:
        day = int(match.group("day"))
        month = int(match.group("month"))
        year = int(match.group("year") or date.today().year)
        selected = date(year, month, day)
    except ValueError:
        return None
    next_day = selected + timedelta(days=1)
    return {
        "type": "exact_date",
        "exact_date": selected.isoformat(),
        "start_date": selected.isoformat(),
        "end_date": next_day.isoformat(),
    }


def _sports_structured_intent(normalized: str) -> str | None:
    if any(keyword in normalized for keyword in _SPORTS_SCHEDULE_KEYWORDS):
        return "sports_schedule"
    has_sports_context = any(keyword in normalized for keyword in _SPORTS_CONTEXT_KEYWORDS)
    if any(keyword in normalized for keyword in _SPORTS_RESULT_KEYWORDS) and has_sports_context:
        return "sports_result"
    if any(keyword in normalized for keyword in _SPORTS_STANDING_KEYWORDS):
        return "sports_standing"
    return None


def _has_image_intent(normalized: str) -> bool:
    if "anh huong" in normalized:
        return False
    return any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", normalized) for keyword in _IMAGE_KEYWORDS)


def _is_stock_market_overview(normalized: str) -> bool:
    overview_patterns = (
        "gia co phieu hom nay",
        "chung khoan hom nay",
        "thi truong chung khoan hom nay",
        "vn-index hom nay",
        "vnindex hom nay",
    )
    if any(pattern in normalized for pattern in overview_patterns):
        return True
    return bool(
        ("co phieu" in normalized or "chung khoan" in normalized or "vn-index" in normalized or "vnindex" in normalized)
        and ("hom nay" in normalized or "thi truong" in normalized)
    )


def _has_explicit_stock_topic(normalized: str) -> bool:
    return any(keyword in normalized for keyword in _STRONG_STOCK_TOPIC_KEYWORDS)


def _detect_answer_mode(normalized: str, intent: str) -> str:
    """Detect whether the query wants synthesis/analysis, exact citation, or follow-up."""
    if any(signal in normalized for signal in _CITATION_SIGNALS):
        return "citation"
    if any(signal in normalized for signal in _FOLLOWUP_SIGNALS):
        return "followup"
    if any(signal in normalized for signal in _SYNTHESIS_SIGNALS):
        return "synthesis"
    # Intent-based fallback
    if intent in {"latest_news", "topic_news", "news_summary", "stock_market_overview"}:
        return "synthesis"
    if intent in {"entity_news", "article_summary", "article_qa"}:
        return "citation"
    return "synthesis"


def _score_topics(normalized: str, stock_symbols: list[str], entities: list[str]) -> TopicScore:
    scores = {topic: 0 for topic in _TOPIC_KEYWORDS}
    matched: dict[str, list[str]] = {topic: [] for topic in _TOPIC_KEYWORDS}
    strong_hits: dict[str, int] = {topic: 0 for topic in _TOPIC_KEYWORDS}

    for topic, groups in _TOPIC_KEYWORDS.items():
        for group_name, weight in (("strong", 3), ("medium", 2), ("weak", 1)):
            for keyword in groups.get(group_name, ()):
                if _keyword_matches(normalized, keyword):
                    bonus = 1 if " " in keyword.strip() else 0
                    scores[topic] += weight + bonus
                    matched[topic].append(keyword)
                    if group_name == "strong":
                        strong_hits[topic] += 1

    if stock_symbols:
        scores["economy_finance_stock"] += 2
        matched["economy_finance_stock"].extend(stock_symbols)
        strong_hits["economy_finance_stock"] += 1

    normalized_entities = {normalize_text(str(entity)) for entity in entities if str(entity).strip()}
    if normalized_entities.intersection({"vcci"}):
        scores["business_startup"] += 2
        matched["business_startup"].extend(sorted(normalized_entities.intersection({"vcci"})))

    _apply_topic_conflict_rules(scores, matched, normalized)

    topic = max(scores, key=lambda item: (scores[item], _topic_tiebreaker(item)))
    top_score = scores[topic]
    if top_score < 2:
        return TopicScore(
            topic=None,
            scores=scores,
            matched_keywords=_dedupe_matched(matched),
            confidence=0.0,
            confidence_label="low",
            explicit_topic=False,
        )

    explicit = strong_hits.get(topic, 0) > 0 or top_score >= 4
    if strong_hits.get(topic, 0) > 0 or top_score >= 5:
        confidence = 1.0
        confidence_label = "high"
    elif top_score >= 3:
        confidence = 0.7
        confidence_label = "medium"
    else:
        confidence = 0.35
        confidence_label = "low"

    return TopicScore(
        topic=topic,
        scores=scores,
        matched_keywords=_dedupe_matched(matched),
        confidence=confidence,
        confidence_label=confidence_label,
        explicit_topic=explicit,
    )


def _apply_topic_conflict_rules(scores: dict[str, int], matched: dict[str, list[str]], normalized: str) -> None:
    finance_strong = any(_keyword_matches(normalized, keyword) for keyword in _TOPIC_KEYWORDS["economy_finance_stock"]["strong"])
    stock_market = any(_keyword_matches(normalized, keyword) for keyword in ("chung khoan", "co phieu", "vn-index", "vnindex", "ma co phieu", "ma chung khoan"))
    real_estate_strong = any(_keyword_matches(normalized, keyword) for keyword in _TOPIC_KEYWORDS["real_estate"]["strong"])
    tech_strong = any(_keyword_matches(normalized, keyword) for keyword in _TOPIC_KEYWORDS["tech_ai_internet"]["strong"])
    world_finance_terms = any(_keyword_matches(normalized, keyword) for keyword in ("fed", "lai suat my", "usd", "thi truong toan cau"))

    if finance_strong:
        scores["politics_society"] = max(0, scores["politics_society"] - 3)
    if stock_market:
        scores["business_startup"] = max(0, scores["business_startup"] - 3)
        scores["economy_finance_stock"] += 2
        matched["economy_finance_stock"].append("stock_conflict_boost")
    if real_estate_strong and not any(_keyword_matches(normalized, keyword) for keyword in ("tin dung", "ngan hang", "lai suat", "vay von")):
        scores["politics_society"] = max(0, scores["politics_society"] - 3)
    if real_estate_strong and any(_keyword_matches(normalized, keyword) for keyword in ("tin dung", "ngan hang", "lai suat", "vay mua nha", "cho vay")):
        scores["economy_finance_stock"] += 3
        matched["economy_finance_stock"].append("finance_real_estate_conflict_boost")
    if tech_strong and any(_keyword_matches(normalized, keyword) for keyword in ("zalo", "otp", "vneid", "an ninh mang", "lua dao online", "qua mang")):
        scores["politics_society"] = max(0, scores["politics_society"] - 2)
    if world_finance_terms:
        scores["economy_finance_stock"] += 2
        matched["economy_finance_stock"].append("world_finance_conflict_boost")


def _keyword_matches(normalized: str, keyword: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword.strip() or normalized_keyword in _GENERIC_QUERY_TERMS:
        return False
    if re.search(r"\W", normalized_keyword):
        return normalized_keyword in normalized
    return re.search(rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)", normalized) is not None


def _topic_tiebreaker(topic: str) -> int:
    priority = {
        "economy_finance_stock": 7,
        "real_estate": 6,
        "tech_ai_internet": 5,
        "world_geopolitics": 4,
        "business_startup": 3,
        "lifestyle_education_health_entertainment": 2,
        "politics_society": 1,
    }
    return priority.get(topic, 0)


def _dedupe_matched(matched: dict[str, list[str]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for topic, keywords in matched.items():
        seen: set[str] = set()
        output[topic] = []
        for keyword in keywords:
            if keyword in seen:
                continue
            seen.add(keyword)
            output[topic].append(keyword)
    return output


def _topic_from_rules(normalized: str) -> str | None:
    rules = (
        (
            "tech_ai_internet",
            (
                "tin ai",
                "tri tue nhan tao",
                "cong nghe",
                "internet",
                "mobile",
                "blockchain",
                "thu thuat cong nghe",
                "do choi so",
                "xe dien",
                "gia dung thong minh",
                "thiet bi thong minh",
            ),
        ),
        ("economy_finance_stock", (*_STRONG_STOCK_TOPIC_KEYWORDS, *_WEAK_STOCK_TOPIC_KEYWORDS)),
        ("real_estate", ("bat dong san", "nha dat", "chung cu")),
        ("world_geopolitics", ("tinh hinh the gioi", "the gioi", "quoc te", "ukraine", "trung dong", "nga", "my")),
        ("business_startup", ("doanh nghiep", "startup", "khoi nghiep", "vcci")),
        ("politics_society", ("thoi su", "chinh tri", "xa hoi", "phap luat")),
        (
            "lifestyle_education_health_entertainment",
            ("giao duc", "suc khoe", "giai tri", "du lich", "the thao", "bong da", "cup c1", "champions league"),
        ),
    )
    for topic, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return topic
    return None


def _has_ai_question_pronoun(query: str, normalized: str) -> bool:
    if not re.search(r"(?<!\w)ai(?!\w)", normalized):
        return False
    if re.search(r"\bAI\b", query):
        return False
    technology_markers = (
        "tin ai",
        "tri tue nhan tao",
        "openai",
        "chatgpt",
        "genai",
        "generative ai",
        "mo hinh ai",
        "cong nghe ai",
        "ung dung ai",
    )
    return not any(marker in normalized for marker in technology_markers)


def _source_from_query(normalized: str) -> str | None:
    for source in ("vnexpress", "cafef", "genk", "diendandoanhnghiep"):
        if source in normalized:
            return source
    return None


def _requires_lexical(query: str, normalized: str, entities: list[str], stock_symbols: list[str]) -> bool:
    if entities or stock_symbols:
        return True
    if re.search(r"\b[A-Z]{2,8}\b", query):
        return True
    if _source_from_query(normalized):
        return True
    if len(normalized.split()) <= 5:
        return True
    return bool(re.search(r"\d", query))


def _lexical_terms(query: str, entities: list[str], stock_symbols: list[str]) -> list[str]:
    terms: list[str] = []
    terms.extend(str(item) for item in stock_symbols if str(item).strip())
    terms.extend(str(item) for item in entities if str(item).strip())
    terms.extend(match.group(0) for match in re.finditer(r"\b[A-Z]{2,8}\b", query))
    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        output.append(term)
    return output


def _extract_whitelisted_tickers(query: str, entities: list[dict[str, Any]]) -> list[str]:
    candidates = [
        str(entity.get("normalized_name") or "").upper()
        for entity in entities
        if entity.get("type") == "stock_symbol"
    ]
    candidates.extend(match.group(0).upper() for match in re.finditer(r"\b[A-Z]{2,8}\b", query))
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = "VNINDEX" if candidate == "VN-INDEX" else candidate
        if normalized not in _TICKER_WHITELIST or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
