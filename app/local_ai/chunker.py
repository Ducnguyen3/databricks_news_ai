from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from app.config import load_settings

logger = logging.getLogger(__name__)

_MIN_CHUNK_LENGTH = 80
_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sports": (
        "the thao",
        "bong da",
        "bong ro",
        "tennis",
        "giai dau",
        "tran dau",
        "van dong vien",
        "world cup",
        "champions league",
        "premier league",
        "v league",
        "sea games",
        "olympic",
    ),
    "tech": (
        "cong nghe",
        "tri tue nhan tao",
        "chip",
        "ban dan",
        "phan mem",
        "ung dung",
        "startup cong nghe",
        "robot",
        "du lieu",
        "internet",
        "mobile",
        "khoa hoc cong nghe",
    ),
    "business": (
        "kinh te",
        "kinh doanh",
        "tai chinh",
        "doanh nghiep",
        "dau tu",
        "chung khoan",
        "ngan hang",
        "thi truong",
        "bat dong san",
        "xuat khau",
        "gia vang",
        "lam phat",
    ),
    "politics": (
        "chinh tri",
        "quoc hoi",
        "chinh phu",
        "bo ngoai giao",
        "ngoai giao",
        "bau cu",
        "nghi quyet",
        "tong thong",
        "thu tuong",
    ),
}


@dataclass(frozen=True, slots=True)
class ArticleChunk:
    chunk_id: str
    article_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]


class ArticleChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        load_dotenv()
        settings = load_settings()
        self._chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", settings.local_ai.chunk_size))
        self._chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", settings.local_ai.chunk_overlap))
        if self._chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._chunk_overlap < 0 or self._chunk_overlap >= self._chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    def chunk_article(self, article: Mapping[str, Any]) -> list[ArticleChunk]:
        article_id = str(article.get("article_id") or "").strip()
        if not article_id:
            return []

        content = _build_article_text(article)
        if len(content) < _MIN_CHUNK_LENGTH:
            return []

        base_metadata = {
            "article_id": article_id,
            "title": str(article.get("title") or ""),
            "source": str(article.get("source") or ""),
            "url": str(article.get("url") or ""),
            "canonical_url": str(article.get("canonical_url") or ""),
            "category": str(article.get("category") or ""),
            "topic_category": infer_topic_category(article),
            "published_at": _to_optional_string(article.get("published_at")) or "",
            "content_hash": str(article.get("content_hash") or ""),
        }

        chunks: list[ArticleChunk] = []
        for index, chunk_text in enumerate(
            chunk_text_with_overlap(content, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        ):
            if len(chunk_text) < _MIN_CHUNK_LENGTH:
                continue
            metadata = dict(base_metadata)
            metadata["chunk_index"] = index
            chunks.append(
                ArticleChunk(
                    chunk_id=f"{article_id}:{index}",
                    article_id=article_id,
                    chunk_index=index,
                    text=chunk_text,
                    metadata=metadata,
                )
            )
        return chunks

    def chunk_articles(self, articles: list[dict[str, Any]] | Iterable[Mapping[str, Any]]) -> list[ArticleChunk]:
        rows = _article_rows(articles)
        chunks: list[ArticleChunk] = []
        for article in rows:
            chunks.extend(self.chunk_article(article))
        logger.info("Created %s chunks from %s articles", len(chunks), len(rows))
        return chunks


def chunk_articles(
    articles: Any,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ArticleChunk]:
    return ArticleChunker(chunk_size=chunk_size, chunk_overlap=overlap).chunk_articles(articles)


def chunk_text_with_overlap(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = split_paragraphs(text)
    segments: list[str] = []
    for paragraph in paragraphs:
        paragraph_sentences = split_sentences(paragraph)
        if not paragraph_sentences:
            continue
        for sentence in paragraph_sentences:
            segments.extend(_split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in segments:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence) + (1 if current_sentences else 0)
        if current_sentences and current_length + sentence_length > chunk_size:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) >= _MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)
            overlap_sentences = _tail_overlap_sentences(current_sentences, overlap)
            current_sentences = overlap_sentences.copy()
            current_length = _joined_length(current_sentences)

        if current_sentences and current_length + sentence_length > chunk_size and current_length > 0:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) >= _MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)
            current_sentences = []
            current_length = 0

        current_sentences.append(sentence)
        current_length = _joined_length(current_sentences)

    final_chunk = " ".join(current_sentences).strip()
    if len(final_chunk) >= _MIN_CHUNK_LENGTH:
        chunks.append(final_chunk)

    return _merge_short_chunks(chunks, min_length=_MIN_CHUNK_LENGTH)


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [" ".join(paragraph.split()) for paragraph in re.split(r"\n\s*\n+", normalized)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if paragraphs:
        return paragraphs
    fallback = " ".join(text.split())
    return [fallback] if fallback else []


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+|(?<=\))\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    if len(sentence) <= chunk_size:
        return [sentence]

    clauses = [part.strip() for part in re.split(r"(?<=[,])\s+", sentence) if part.strip()]
    if len(clauses) <= 1:
        return _split_hard(sentence, chunk_size)

    pieces: list[str] = []
    current = ""
    for clause in clauses:
        candidate = clause if not current else f"{current} {clause}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(clause) > chunk_size:
            pieces.extend(_split_hard(clause, chunk_size))
            current = ""
        else:
            current = clause
    if current:
        pieces.append(current)
    return pieces


def _split_hard(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join([*current_words, word]).strip()
        if current_words and len(candidate) > chunk_size:
            pieces.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words.append(word)
    if current_words:
        pieces.append(" ".join(current_words))
    return pieces


def _tail_overlap_sentences(sentences: list[str], overlap: int) -> list[str]:
    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        sentence_length = len(sentence) + (1 if selected else 0)
        if not selected and len(sentence) > overlap:
            break
        if selected and total + sentence_length > overlap:
            break
        selected.insert(0, sentence)
        total += sentence_length
        if total >= overlap:
            break
    return selected


def _merge_short_chunks(chunks: list[str], min_length: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_length:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
            continue
        merged.append(chunk)
    return merged


def _joined_length(parts: list[str]) -> int:
    return len(" ".join(parts).strip())


def _article_rows(articles: Any) -> list[Mapping[str, Any]]:
    if hasattr(articles, "to_dict"):
        return articles.to_dict("records")
    if isinstance(articles, Iterable):
        return list(articles)
    raise TypeError("articles must be a pandas DataFrame or an iterable of mappings")


def _to_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in {"NaT", "nan", "None"}:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return text


def _build_article_text(article: Mapping[str, Any]) -> str:
    parts = [
        str(article.get("title") or "").strip(),
        str(article.get("summary_raw") or "").strip(),
        str(article.get("content") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def infer_topic_category(article: Mapping[str, Any]) -> str:
    fields = [
        str(article.get("category") or "").strip(),
        str(article.get("title") or "").strip(),
        str(article.get("summary_raw") or "").strip(),
    ]
    haystack = normalize_topic_text(" ".join(part for part in fields if part))
    if not haystack:
        return ""

    best_category = ""
    best_score = 0
    for category, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def normalize_topic_text(text: str) -> str:
    normalized = text.casefold()
    replacements = {
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())
