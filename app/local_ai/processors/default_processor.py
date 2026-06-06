from __future__ import annotations

from typing import Any

from app.local_ai.processors.base_processor import BaseNewsProcessor

_NO_CONTEXT_ANSWER = "Toi khong tim thay thong tin phu hop trong du lieu hien co."


class DefaultNewsProcessor(BaseNewsProcessor):
    topic_id = "default"
    domain_instructions = (
        "Tom tat va tra loi dua tren retrieved context. "
        "Neu context khong du du lieu thi noi ro. "
        "Khong bia thong tin ngoai retrieved context. "
        "Luon neu nguon neu co."
    )

    def build_context(
        self,
        query: str,
        route: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "query": query,
            "intent": route.get("intent"),
            "topic": route.get("primary_topic"),
            "entities": list(route.get("entities") or []),
            "stock_symbols": list(route.get("stock_symbols") or []),
            "time_range": route.get("time_range", "all"),
            "retrieved_chunks": list(retrieved_chunks or []),
            "sources": list(sources or []),
            "images": list(images or []),
            "domain_instructions": self.domain_instructions,
        }

    def build_prompt(self, context: dict[str, Any]) -> str:
        chunks_text = _format_chunks(context.get("retrieved_chunks") or [])
        sources_text = _format_sources(context.get("sources") or [])
        images_text = _format_images(context.get("images") or [])
        return (
            "Ban la chatbot doc bao dua tren du lieu da truy hoi.\n\n"
            "QUY TAC BAT BUOC:\n"
            "1. Chi su dung thong tin trong RETRIEVED_CONTEXT.\n"
            "2. Khong bia thong tin ngoai retrieved context.\n"
            f"3. Neu khong du du lieu, tra loi ro rang: \"{_NO_CONTEXT_ANSWER}\"\n"
            "4. Tra loi bang tieng Viet.\n"
            "5. Moi y quan trong phai co citation theo dang: (Theo source, published_at).\n"
            "6. Neu IMAGES_METADATA co anh phu hop, co the chen bang Markdown: ![caption](image_url).\n"
            "7. Neu co nguon, liet ke nguon da dung.\n\n"
            f"DOMAIN_INSTRUCTIONS:\n{context.get('domain_instructions') or self.domain_instructions}\n\n"
            f"QUERY: {context.get('query') or ''}\n"
            f"INTENT: {context.get('intent') or ''}\n"
            f"TOPIC: {context.get('topic') or ''}\n"
            f"ENTITIES: {', '.join(str(item) for item in context.get('entities') or [])}\n"
            f"STOCK_SYMBOLS: {', '.join(str(item) for item in context.get('stock_symbols') or [])}\n"
            f"TIME_RANGE: {context.get('time_range') or 'all'}\n\n"
            f"RETRIEVED_CONTEXT:\n{chunks_text}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            f"IMAGES_METADATA:\n{images_text}\n\n"
            "Hay tra loi theo format:\n\n"
            "Tra loi:\n"
            "...\n\n"
            "Nguon:\n"
            "1. title - source - url"
        )


def _format_chunks(chunks: list[dict[str, Any]], max_chars: int = 9000) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        blocks.append(
            "\n".join(
                [
                    f"[CHUNK {index}]",
                    f"Article ID: {metadata.get('article_id') or ''}",
                    f"Title: {metadata.get('title') or 'Untitled'}",
                    f"Source: {metadata.get('source') or 'unknown'}",
                    f"Published at: {metadata.get('published_at') or 'unknown'}",
                    f"URL: {metadata.get('url') or ''}",
                    "Content:",
                    str(chunk.get("text") or ""),
                ]
            )
        )
    text = "\n\n".join(blocks).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n[CONTEXT_TRUNCATED]"


def _format_sources(sources: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        lines.append(
            f"{index}. {source.get('title') or 'Untitled'} - "
            f"{source.get('source') or 'unknown'} - {source.get('url') or ''}"
        )
    return "\n".join(lines)


def _format_images(images: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            continue
        lines.append(
            f"{index}. article_id={image.get('article_id') or ''}; "
            f"url={image.get('image_url') or ''}; caption={image.get('caption') or ''}; "
            f"credit={image.get('credit') or ''}"
        )
    return "\n".join(lines)
