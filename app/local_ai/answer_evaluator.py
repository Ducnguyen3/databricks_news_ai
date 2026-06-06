from __future__ import annotations

from typing import Any


_INSUFFICIENT_DATA_PHRASES = (
    "không tìm thấy",
    "khong tim thay",
    "không có đủ dữ liệu",
    "khong co du du lieu",
    "chưa có dữ liệu",
    "chua co du lieu",
    "không có thông tin",
    "khong co thong tin",
    "không đủ thông tin",
    "khong du thong tin",
)


def evaluate_rag_answer(response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "").strip()
    sources = response.get("sources")
    images = response.get("images")
    source_count = len(sources) if isinstance(sources, list) else 0
    image_count = len(images) if isinstance(images, list) else 0
    has_answer = bool(answer)
    has_sources = source_count > 0

    if not has_answer:
        status = "FAIL"
        reason = "empty_answer"
    elif not has_sources:
        status = "FAIL"
        reason = "missing_sources"
    elif _has_insufficient_data_phrase(answer):
        status = "WARN"
        reason = "answer_indicates_insufficient_data"
    elif source_count >= 2:
        status = "OK"
        reason = "answer_has_multiple_sources"
    else:
        status = "WARN"
        reason = "single_source"

    return {
        "status": status,
        "has_answer": has_answer,
        "has_sources": has_sources,
        "source_count": source_count,
        "image_count": image_count,
        "reason": reason,
    }


def _has_insufficient_data_phrase(answer: str) -> bool:
    normalized = answer.casefold()
    return any(phrase in normalized for phrase in _INSUFFICIENT_DATA_PHRASES)
