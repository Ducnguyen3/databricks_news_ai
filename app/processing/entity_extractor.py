from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

_STOCK_SYMBOLS = ("VNINDEX", "VN-INDEX", "HNX", "UPCOM", "HPG", "FPT", "VIC", "VHM", "VCB", "TCB", "BID", "CTG", "MWG", "VNM", "SSI", "VND", "MSN", "GAS")
_COMPANY_BY_SYMBOL = {
    "HPG": "Hòa Phát",
    "FPT": "FPT",
    "VIC": "Vingroup",
    "VHM": "Vinhomes",
    "VCB": "Vietcombank",
    "TCB": "Techcombank",
    "BID": "BIDV",
    "CTG": "VietinBank",
    "MWG": "Thế Giới Di Động",
    "VNM": "Vinamilk",
    "SSI": "SSI",
    "MSN": "Masan",
    "GAS": "PV Gas",
}

_ENTITY_ALIASES: list[tuple[str, str, str, tuple[str, ...], float]] = [
    ("country", "Hoa Kỳ", "Hoa Kỳ", ("Mỹ", "Hoa Kỳ", "USA", "United States"), 0.90),
    ("country", "Trung Quốc", "Trung Quốc", ("Trung Quốc",), 0.90),
    ("country", "Nga", "Nga", ("Nga",), 0.90),
    ("country", "Ukraine", "Ukraine", ("Ukraine",), 0.90),
    ("country", "Israel", "Israel", ("Israel",), 0.90),
    ("country", "Iran", "Iran", ("Iran",), 0.90),
    ("country", "Ấn Độ", "Ấn Độ", ("Ấn Độ",), 0.90),
    ("country", "Nhật Bản", "Nhật Bản", ("Nhật Bản",), 0.90),
    ("country", "Hàn Quốc", "Hàn Quốc", ("Hàn Quốc",), 0.90),
    ("country", "Đông Nam Á", "Đông Nam Á", ("Đông Nam Á",), 0.90),
    ("country", "Trung Đông", "Trung Đông", ("Trung Đông",), 0.90),
    ("country", "Châu Âu", "Châu Âu", ("Châu Âu",), 0.90),
    ("country", "EU", "EU", ("EU",), 0.90),
    ("organization", "Chính phủ", "Chính phủ", ("Chính phủ",), 0.90),
    ("organization", "Quốc hội", "Quốc hội", ("Quốc hội",), 0.90),
    ("organization", "Ngân hàng Nhà nước", "Ngân hàng Nhà nước", ("Ngân hàng Nhà nước",), 0.90),
    ("company", "Techcombank", "Techcombank", ("Techcombank", "TCB", "NgÃ¢n hÃ ng Techcombank", "Ngan hang Techcombank"), 0.92),
    ("organization", "Fed", "Fed", ("Fed",), 0.90),
    ("organization", "ECB", "ECB", ("ECB",), 0.90),
    ("organization", "NATO", "NATO", ("NATO",), 0.90),
    ("organization", "Liên Hợp Quốc", "Liên Hợp Quốc", ("Liên Hợp Quốc", "Liên Hiệp Quốc"), 0.90),
    ("organization", "WHO", "WHO", ("WHO",), 0.90),
    ("organization", "IMF", "IMF", ("IMF",), 0.90),
    ("organization", "World Bank", "World Bank", ("World Bank", "Ngân hàng Thế giới"), 0.90),
    ("location", "Hà Nội", "Hà Nội", ("Hà Nội",), 0.90),
    ("location", "TP.HCM", "TP.HCM", ("TP.HCM", "TP HCM", "TPHCM", "TP Hồ Chí Minh", "Thành phố Hồ Chí Minh"), 0.90),
    ("location", "Đà Nẵng", "Đà Nẵng", ("Đà Nẵng",), 0.90),
    ("location", "Bình Dương", "Bình Dương", ("Bình Dương",), 0.90),
    ("location", "Đồng Nai", "Đồng Nai", ("Đồng Nai",), 0.90),
    ("location", "Hải Phòng", "Hải Phòng", ("Hải Phòng",), 0.90),
    ("location", "Long An", "Long An", ("Long An",), 0.90),
    ("location", "Bà Rịa - Vũng Tàu", "Bà Rịa - Vũng Tàu", ("Bà Rịa - Vũng Tàu", "Bà Rịa Vũng Tàu"), 0.90),
]
_TOPIC_KEYWORDS = ("AI", "trí tuệ nhân tạo", "chứng khoán", "bất động sản", "lãi suất", "startup")


def extract_entities(
    title: str | None = None,
    summary: str | None = None,
    content: str | None = None,
    source_category: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    original = " ".join(part for part in (title or "", summary or "", content or "", source_category or "") if part)
    if not original.strip():
        return {"entities": []}

    found: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol in _STOCK_SYMBOLS:
        count = _count_mentions(original, symbol)
        if count <= 0:
            continue
        normalized_symbol = "VNINDEX" if symbol in {"VNINDEX", "VN-INDEX"} else symbol
        _merge_entity(found, normalized_symbol, normalized_symbol, "stock_symbol", count, 0.95)
        company = _COMPANY_BY_SYMBOL.get(normalized_symbol)
        if company:
            _merge_entity(found, company, company, "company", 1, 0.80)

    for entity_type, name, normalized_name, aliases, confidence in _ENTITY_ALIASES:
        count = sum(_count_mentions(original, alias) for alias in aliases)
        if count > 0:
            _merge_entity(found, name, normalized_name, entity_type, count, confidence)

    for keyword in _TOPIC_KEYWORDS:
        count = _count_mentions(original, keyword)
        if count > 0:
            _merge_entity(found, keyword, keyword, "topic_keyword", count, 0.80)

    entities = sorted(found.values(), key=lambda item: (str(item["type"]), str(item["normalized_name"])))
    return {"entities": entities}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return " ".join(without_marks.replace("đ", "d").split())


def _merge_entity(
    found: dict[tuple[str, str], dict[str, Any]],
    name: str,
    normalized_name: str,
    entity_type: str,
    mention_count: int,
    confidence: float,
) -> None:
    key = (entity_type, normalized_name)
    if key not in found:
        found[key] = {
            "name": name,
            "normalized_name": normalized_name,
            "type": entity_type,
            "mention_count": mention_count,
            "confidence": confidence,
        }
        return
    found[key]["mention_count"] = int(found[key]["mention_count"]) + mention_count
    found[key]["confidence"] = max(float(found[key]["confidence"]), confidence)


def _count_mentions(text: str, phrase: str) -> int:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return 0
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", flags=re.IGNORECASE)
    return len(pattern.findall(normalized_text))
