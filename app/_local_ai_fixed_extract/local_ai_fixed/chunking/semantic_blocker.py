from __future__ import annotations

import re

from app.local_ai.chunking.models import ArticleBlock, ParentArticle


class SemanticBlocker:
    def __init__(self, target_block_chars: int = 2000, min_block_chars: int = 500, max_block_chars: int = 3000) -> None:
        self.target_block_chars = target_block_chars
        self.min_block_chars = min_block_chars
        self.max_block_chars = max_block_chars

    def build_blocks(self, article: ParentArticle) -> list[ArticleBlock]:
        paragraphs = split_paragraphs(article.content)
        lead_parts = [article.title.strip()]
        if article.summary:
            lead_parts.append(article.summary.strip())
        if paragraphs:
            lead_parts.append(paragraphs[0])
        blocks: list[ArticleBlock] = [
            _block(article, 0, "lead", "\n\n".join(part for part in lead_parts if part))
        ]

        current: list[str] = []
        current_type = "body"
        for paragraph in paragraphs[1:]:
            paragraph_type = classify_block_type(paragraph)
            candidate = "\n\n".join([*current, paragraph]).strip()
            if current and (len(candidate) > self.target_block_chars or paragraph_type != current_type):
                blocks.append(_block(article, len(blocks), current_type, "\n\n".join(current)))
                current = [paragraph]
                current_type = paragraph_type
                continue
            current.append(paragraph)
            current_type = paragraph_type if not current_type else current_type
            if len("\n\n".join(current)) >= self.max_block_chars:
                blocks.append(_block(article, len(blocks), current_type, "\n\n".join(current)))
                current = []
                current_type = "body"
        if current:
            text = "\n\n".join(current)
            if blocks and len(text) < self.min_block_chars and len(blocks[-1].text) + len(text) <= self.max_block_chars:
                previous = blocks[-1]
                previous.text = f"{previous.text}\n\n{text}".strip()
            else:
                blocks.append(_block(article, len(blocks), current_type, text))
        return [block for block in blocks if block.text.strip()]


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n+", normalized)
    if len(parts) <= 1:
        parts = normalized.split("\n")
    return [" ".join(part.split()) for part in parts if part.strip()]


def classify_block_type(text: str) -> str:
    normalized = text.casefold()
    if text.strip().startswith(("\"", "“", "'")) or text.count("\"") + text.count("“") + text.count("”") >= 2:
        return "quote"
    if any(keyword in normalized for keyword in ("trước đó", "bối cảnh", "theo hồ sơ", "lịch sử")):
        return "background"
    if any(keyword in normalized for keyword in ("theo chuyên gia", "nhận định", "phân tích", "dự báo", "cho rằng")):
        return "analysis"
    return "body"


def _block(article: ParentArticle, index: int, block_type: str, text: str) -> ArticleBlock:
    block_id = f"{article.article_id}::b{index}"
    return ArticleBlock(
        block_id=block_id,
        article_id=article.article_id,
        block_index=index,
        block_type=block_type,
        text=text.strip(),
        metadata={"block_type": block_type},
    )
