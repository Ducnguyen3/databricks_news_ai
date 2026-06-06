from __future__ import annotations

from typing import Any

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class LifestyleProcessor(DefaultNewsProcessor):
    topic_id = "lifestyle_education_health_entertainment"
    domain_instructions = (
        "Tap trung vao tom tat de hieu va thong tin thiet thuc. "
        "Voi giao duc/giai tri, uu tien dien giai ro rang theo retrieved context."
    )

    def build_context(
        self,
        query: str,
        route: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context = super().build_context(query, route, retrieved_chunks, sources, images)
        if _looks_health_related(query, retrieved_chunks):
            context["domain_instructions"] = (
                f"{self.domain_instructions} "
                "Neu la noi dung suc khoe: khong chan doan, khong thay the tu van y te, "
                "chi tom tat theo nguon da retrieve va khuyen doc gia tham khao chuyen gia y te khi can."
            )
        return context


def _looks_health_related(query: str, chunks: list[dict[str, Any]]) -> bool:
    haystack = query.lower()
    for chunk in chunks:
        if isinstance(chunk, dict):
            haystack += " " + str(chunk.get("text") or "").lower()
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                haystack += " " + str(metadata.get("category") or "").lower()
    keywords = ("suc khoe", "y te", "benh", "thuoc", "bac si", "dieu tri", "health", "medical")
    return any(keyword in haystack for keyword in keywords)
