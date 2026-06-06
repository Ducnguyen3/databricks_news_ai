from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SportsRetriever:
    def __init__(self, repository: Any | None = None) -> None:
        self._repository = repository

    def retrieve(self, query_plan: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        if query_plan.get("data_source") != "structured_sports":
            return []
        if self._repository is None:
            return []
        try:
            if query_plan.get("intent") == "sports_schedule":
                return list(self._repository.get_schedule(query_plan=query_plan, limit=limit))
            if query_plan.get("intent") == "sports_result":
                return list(self._repository.get_results(query_plan=query_plan, limit=limit))
            if query_plan.get("intent") == "sports_standing":
                return list(self._repository.get_standings(query_plan=query_plan, limit=limit))
        except Exception:
            logger.warning("Sports structured retrieval failed; falling back to empty results")
        return []
