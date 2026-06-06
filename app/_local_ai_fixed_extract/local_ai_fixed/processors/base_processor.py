from __future__ import annotations

from typing import Any


class BaseNewsProcessor:
    topic_id: str = "default"

    def build_context(
        self,
        query: str,
        route: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_prompt(self, context: dict[str, Any]) -> str:
        raise NotImplementedError
