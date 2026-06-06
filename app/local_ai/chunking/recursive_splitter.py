from __future__ import annotations

import re

from app.local_ai.chunking.models import ArticleBlock
from app.local_ai.chunking.semantic_blocker import split_paragraphs

_ABBREVIATIONS = (
    "TP.HCM",
    "PGS.TS",
    "TS.",
    "ThS.",
    "GS.",
    "Mr.",
    "Mrs.",
    "Dr.",
    "Ltd.",
    "JSC.",
)
_DOT_TOKEN = "<DOT>"
_BAD_START_PREFIXES = (
    "và ",
    "nhưng ",
    "tuy nhiên",
    "trong khi đó",
    "theo đó",
    "điều này",
    "điều này cho thấy",
    "do đó",
    "vì vậy",
    "ngoài ra",
    "mặt khác",
    "của ",
    "theo của",
    "cho thấy",
    "xử lý ",
    "này ",
    "đó ",
    "va ",
    "nhung ",
    "tuy nhien",
    "trong khi do",
    "theo do",
    "dieu nay",
    "do do",
    "vi vay",
    "ngoai ra",
    "mat khac",
    "cua ",
    "xu ly ",
    "nay ",
    "do ",
)


class RecursiveSplitter:
    def __init__(
        self,
        target_chunk_chars: int = 1000,
        max_chunk_chars: int = 1600,
        min_chunk_chars: int = 300,
        overlap_chars: int = 250,
        overlap_sentences: int = 1,
    ) -> None:
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars
        self.overlap_sentences = overlap_sentences

    def split_block(self, block: ArticleBlock) -> list[str]:
        text = normalize_whitespace(block.text)
        if not text:
            return []
        if len(text) <= self.target_chunk_chars:
            return [text]
        units = self._split_to_sentence_units(text)
        chunks: list[str] = []
        current: list[str] = []
        for unit in units:
            if len(unit) > self.max_chunk_chars:
                if current:
                    chunks.append(_join_sentences(current))
                    current = []
                chunks.extend(_hard_split(unit, self.max_chunk_chars, self.overlap_chars))
                continue

            candidate = _join_sentences([*current, unit])
            if current and len(candidate) > self.target_chunk_chars and len(_join_sentences(current)) >= self.min_chunk_chars:
                chunks.append(_join_sentences(current))
                overlap = current[-self.overlap_sentences :] if self.overlap_sentences > 0 else []
                current = [*overlap, unit]
            else:
                current.append(unit)
        if current:
            chunks.append(_join_sentences(current))
        chunks = _merge_small_chunks(chunks, self.min_chunk_chars, self.max_chunk_chars)
        return _repair_bad_chunk_starts(chunks, self.max_chunk_chars)

    def _split_to_sentence_units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in split_paragraphs(text):
            sentences = split_sentences_vi(paragraph)
            units.extend(sentences or [normalize_whitespace(paragraph)])
        return units


def normalize_whitespace(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    return "\n\n".join(paragraphs)


def split_sentences_vi(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    protected = _protect_abbreviations(normalized)
    parts = re.split(r"(?<=[.!?…;:])\s+|\n\s*\n+", protected)
    return [_restore_abbreviations(part.strip()) for part in parts if part.strip()]


def split_sentences(text: str) -> list[str]:
    return split_sentences_vi(text)


def is_bad_chunk_start(text: str) -> bool:
    normalized = normalize_whitespace(text).lstrip()
    if not normalized:
        return False
    if normalized[0] in {",", ".", ":", ";", "!", "?", "…"}:
        return True
    lowered = normalized.casefold()
    return any(lowered.startswith(prefix) for prefix in _BAD_START_PREFIXES)


def _join(first: str, second: str) -> str:
    if not first:
        return second.strip()
    if not second:
        return first.strip()
    return f"{first.strip()}\n\n{second.strip()}"


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
        if start > 0:
            next_space = text.find(" ", start)
            if 0 <= next_space < len(text):
                start = next_space + 1
    return [chunk for chunk in chunks if chunk]


def _merge_small_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1]) + len(chunk) + 2 <= max_chars + min_chars:
            merged[-1] = _join(merged[-1], chunk)
        else:
            merged.append(chunk)
    return merged


def _repair_bad_chunk_starts(chunks: list[str], max_chars: int) -> list[str]:
    repaired: list[str] = []
    for chunk in chunks:
        if repaired and is_bad_chunk_start(chunk):
            previous_sentence = _last_sentence(repaired[-1])
            if previous_sentence and not chunk.startswith(previous_sentence):
                candidate = _join_sentences([previous_sentence, chunk])
                if len(candidate) <= max_chars:
                    chunk = candidate
        repaired.append(chunk)
    return repaired


def _last_sentence(text: str) -> str:
    sentences = split_sentences_vi(text)
    return sentences[-1] if sentences else ""


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        tokenized = abbreviation.replace(".", _DOT_TOKEN)
        protected = re.sub(re.escape(abbreviation), tokenized, protected, flags=re.IGNORECASE)
    protected = re.sub(r"\b(VN|HNX|UPCOM)-Index\b", lambda match: match.group(0).replace(".", _DOT_TOKEN), protected)
    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace(_DOT_TOKEN, ".")
