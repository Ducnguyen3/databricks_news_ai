from __future__ import annotations

import unittest

from app.local_ai.answer_evaluator import evaluate_rag_answer


class AnswerEvaluatorTest(unittest.TestCase):
    def test_empty_answer_fails(self) -> None:
        evaluation = evaluate_rag_answer({"answer": "", "sources": [{"id": "s1"}]})

        self.assertEqual("FAIL", evaluation["status"])
        self.assertFalse(evaluation["has_answer"])
        self.assertEqual("empty_answer", evaluation["reason"])

    def test_answer_without_sources_fails(self) -> None:
        evaluation = evaluate_rag_answer({"answer": "Co thong tin", "sources": []})

        self.assertEqual("FAIL", evaluation["status"])
        self.assertTrue(evaluation["has_answer"])
        self.assertFalse(evaluation["has_sources"])
        self.assertEqual("missing_sources", evaluation["reason"])

    def test_answer_with_two_sources_is_ok(self) -> None:
        evaluation = evaluate_rag_answer(
            {
                "answer": "Co thong tin tu hai nguon.",
                "sources": [{"id": "s1"}, {"id": "s2"}],
                "images": [{"id": "i1"}],
            }
        )

        self.assertEqual("OK", evaluation["status"])
        self.assertEqual(2, evaluation["source_count"])
        self.assertEqual(1, evaluation["image_count"])
        self.assertEqual("answer_has_multiple_sources", evaluation["reason"])

    def test_insufficient_data_answer_warns(self) -> None:
        evaluation = evaluate_rag_answer(
            {
                "answer": "He thong chua co du lieu de tra loi cau hoi nay.",
                "sources": [{"id": "s1"}, {"id": "s2"}],
            }
        )

        self.assertEqual("WARN", evaluation["status"])
        self.assertEqual("answer_indicates_insufficient_data", evaluation["reason"])

    def test_single_source_warns(self) -> None:
        evaluation = evaluate_rag_answer({"answer": "Co thong tin tu mot nguon.", "sources": [{"id": "s1"}]})

        self.assertEqual("WARN", evaluation["status"])
        self.assertEqual(1, evaluation["source_count"])
        self.assertEqual("single_source", evaluation["reason"])


if __name__ == "__main__":
    unittest.main()
