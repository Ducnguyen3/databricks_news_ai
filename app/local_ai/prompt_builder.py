from __future__ import annotations

from typing import Any

from app.config import LocalAiSettings, load_settings

_NO_CONTEXT_ANSWER = "Tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."


class PromptBuilder:
    def __init__(self, settings: LocalAiSettings | None = None) -> None:
        self._settings = settings or load_settings().local_ai
        self._max_context_chars = max(1000, int(self._settings.prompt_max_context_chars))

    def build_qa_prompt(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        context = self._build_chunk_context(context_chunks)
        return (
            "Bạn là chatbot đọc báo dựa trên dữ liệu đã truy hồi.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ sử dụng thông tin trong CONTEXT.\n"
            "2. Không bịa thông tin ngoài CONTEXT.\n"
            "3. Nếu CONTEXT không liên quan hoặc không đủ dữ liệu, trả lời đúng câu sau:\n"
            f'"{_NO_CONTEXT_ANSWER}"\n'
            "4. Trả lời bằng tiếng Việt.\n"
            "5. Trả lời ngắn gọn, đúng trọng tâm.\n"
            "6. Luôn liệt kê nguồn đã dùng.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Trả lời:\n"
            "...\n\n"
            "Nguồn:\n"
            "1. title - source - url"
        )

    def build_broad_topic_prompt(self, question: str, articles: list[dict[str, Any]]) -> str:
        context = self._build_articles_context(articles)
        return (
            "Bạn là trợ lý tổng hợp tin tức.\n\n"
            "NHIỆM VỤ:\n"
            "Dựa trên danh sách bài báo trong CONTEXT, hãy tổng hợp các tin/chủ đề phù hợp với câu hỏi.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng thông tin trong CONTEXT.\n"
            "2. Không tự thêm tin ngoài dữ liệu.\n"
            "3. Ưu tiên các bài mới hơn nếu có thông tin ngày đăng.\n"
            "4. Nếu dữ liệu không đủ, nói rõ chưa đủ dữ liệu.\n"
            "5. Trả lời bằng tiếng Việt.\n"
            "6. Mỗi chủ đề/tin phải có nguồn.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tổng hợp:\n"
            "1. ...\n"
            "   - Tóm tắt:\n"
            "   - Nguồn:"
        )

    def build_article_summary_prompt(self, article: dict[str, Any], article_chunks: list[dict[str, Any]]) -> str:
        context = self._build_single_article_context(article, article_chunks)
        return (
            "Bạn là trợ lý tóm tắt tin tức.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ tóm tắt dựa trên ARTICLE_CONTEXT.\n"
            "2. Không thêm thông tin ngoài bài báo.\n"
            "3. Trả lời bằng tiếng Việt.\n"
            "4. Tóm tắt rõ ràng, dễ hiểu.\n"
            "5. Luôn ghi nguồn bài viết.\n\n"
            f"ARTICLE_CONTEXT:\n{context}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tóm tắt:\n"
            "...\n\n"
            "Ý chính:\n"
            "- ...\n"
            "- ...\n\n"
            "Nhân vật / tổ chức liên quan:\n"
            "- ...\n\n"
            "Nguồn:\n"
            "- title - source - url"
        )

    def build_category_summary_prompt(self, question: str, articles: list[dict[str, Any]]) -> str:
        context = self._build_articles_context(articles)
        return (
            "Bạn là trợ lý tổng hợp tin tức theo chuyên mục.\n\n"
            "NHIỆM VỤ:\n"
            "Dựa trên CONTEXT, hãy liệt kê các bài báo phù hợp nhất với chuyên mục/câu hỏi của người dùng.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng bài báo trong CONTEXT.\n"
            "2. Không thêm bài ngoài dữ liệu.\n"
            "3. Ưu tiên bài mới hơn.\n"
            "4. Mỗi bài tóm tắt 1-2 câu.\n"
            "5. Mỗi bài phải có nguồn gồm title, source, url.\n"
            "6. Trả lời bằng tiếng Việt.\n"
            "7. Không đưa bài sai chuyên mục nếu CONTEXT không hỗ trợ.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tin mới theo chủ đề:\n"
            "1. ...\n"
            "   - Tóm tắt:\n"
            "   - Nguồn:"
        )

    def build_no_context_answer(self, question: str) -> str:
        return (
            "Bạn là chatbot đọc báo.\n"
            "Nếu không có dữ liệu phù hợp cho câu hỏi, hãy trả lời đúng một câu sau và không thêm gì khác:\n"
            f'"{_NO_CONTEXT_ANSWER}"\n\n'
            f"CÂU HỎI:\n{question}\n\n"
            "Trả lời:"
        )

    def _build_chunk_context(self, context_chunks: list[dict[str, Any]]) -> str:
        blocks = [
            self._format_chunk_block(index, chunk)
            for index, chunk in enumerate(context_chunks, start=1)
            if isinstance(chunk, dict)
        ]
        return self._truncate_context("\n\n".join(blocks))

    def _build_single_article_context(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> str:
        metadata = self._article_metadata(article, article_chunks)
        content = self._merge_article_content(article, article_chunks)
        block = "\n".join(
            [
                "[ARTICLE]",
                f"Title: {metadata.get('title') or 'Untitled'}",
                f"Source: {metadata.get('source') or 'unknown'}",
                f"Category: {metadata.get('category') or 'unknown'}",
                f"Published at: {metadata.get('published_at') or 'unknown'}",
                f"URL: {metadata.get('url') or ''}",
                "Content:",
                content,
            ]
        )
        return self._truncate_context(block)

    def _build_articles_context(self, articles: list[dict[str, Any]]) -> str:
        blocks = [
            self._format_article_block(index, article)
            for index, article in enumerate(articles, start=1)
            if isinstance(article, dict)
        ]
        return self._truncate_context("\n\n".join(blocks))

    def _format_chunk_block(self, index: int, chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return "\n".join(
            [
                f"[CHUNK {index}]",
                f"Title: {metadata.get('title') or 'Untitled'}",
                f"Source: {metadata.get('source') or 'unknown'}",
                f"Category: {metadata.get('category') or 'unknown'}",
                f"Published at: {metadata.get('published_at') or 'unknown'}",
                f"URL: {metadata.get('url') or ''}",
                "Content:",
                str(chunk.get("text") or ""),
            ]
        )

    def _format_article_block(self, index: int, article: dict[str, Any]) -> str:
        content = self._article_summary_or_content(article)
        return "\n".join(
            [
                f"[ARTICLE {index}]",
                f"Title: {article.get('title') or 'Untitled'}",
                f"Source: {article.get('source') or 'unknown'}",
                f"Category: {article.get('category') or 'unknown'}",
                f"Published at: {article.get('published_at') or 'unknown'}",
                f"URL: {article.get('url') or ''}",
                "Summary/Content:",
                content,
            ]
        )

    def _article_summary_or_content(self, article: dict[str, Any]) -> str:
        summary = str(article.get("summary") or "").strip()
        if summary:
            return summary
        chunks = article.get("chunks", [])
        if isinstance(chunks, list):
            return self._merge_chunk_texts(chunks)
        return ""

    def _article_metadata(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = dict(article)
        if metadata.get("title"):
            return metadata
        for chunk in article_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_metadata = chunk.get("metadata", {})
            if isinstance(chunk_metadata, dict):
                return dict(chunk_metadata)
        return metadata

    def _merge_article_content(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> str:
        chunks = article.get("chunks")
        if isinstance(chunks, list) and chunks:
            return self._merge_chunk_texts(chunks)
        return self._merge_chunk_texts(article_chunks)

    def _merge_chunk_texts(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = " ".join(str(chunk.get("text") or "").split())
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(text)
        return " ".join(parts)

    def _truncate_context(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= self._max_context_chars:
            return normalized
        truncated = normalized[: self._max_context_chars].rstrip()
        return f"{truncated}\n\n[CONTEXT ĐÃ ĐƯỢC RÚT GỌN ĐỂ PHÙ HỢP GIỚI HẠN PROMPT]"
