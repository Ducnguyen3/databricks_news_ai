from __future__ import annotations

import unittest

from app.local_ai.sports_retriever import SportsRetriever


class _FailingRepository:
    def get_schedule(self, query_plan: dict[str, object], limit: int) -> list[dict[str, object]]:
        raise RuntimeError("not connected")


class _Repository:
    def get_schedule(self, query_plan: dict[str, object], limit: int) -> list[dict[str, object]]:
        return [{"home_team": "A", "away_team": "B"}]

    def get_results(self, query_plan: dict[str, object], limit: int) -> list[dict[str, object]]:
        return [{"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0}]

    def get_standings(self, query_plan: dict[str, object], limit: int) -> list[dict[str, object]]:
        return [{"team": "A", "rank": 1}]


class SportsRetrieverTest(unittest.TestCase):
    def test_fallback_without_repository(self) -> None:
        result = SportsRetriever().retrieve({"data_source": "structured_sports", "intent": "sports_schedule"})

        self.assertEqual([], result)

    def test_fallback_when_repository_raises(self) -> None:
        with self.assertLogs("app.local_ai.sports_retriever", level="WARNING"):
            result = SportsRetriever(_FailingRepository()).retrieve(
                {"data_source": "structured_sports", "intent": "sports_schedule"}
            )

        self.assertEqual([], result)

    def test_routes_to_repository_method(self) -> None:
        retriever = SportsRetriever(_Repository())

        self.assertEqual([{"home_team": "A", "away_team": "B"}], retriever.retrieve({"data_source": "structured_sports", "intent": "sports_schedule"}))
        self.assertEqual([{"team": "A", "rank": 1}], retriever.retrieve({"data_source": "structured_sports", "intent": "sports_standing"}))


if __name__ == "__main__":
    unittest.main()
