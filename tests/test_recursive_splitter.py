from __future__ import annotations

import unittest

from app.local_ai.chunking.models import ArticleBlock
from app.local_ai.chunking.recursive_splitter import RecursiveSplitter, is_bad_chunk_start, split_sentences_vi


class RecursiveSplitterTest(unittest.TestCase):
    def test_split_sentences_vi_handles_vietnamese_boundaries(self) -> None:
        text = (
            "Doanh nghiệp tăng tốc xây dựng SOC. "
            "Thị trường bảo mật xoay trục sang AI. "
            "Tuy nhiên, chi phí triển khai vẫn là rào cản."
        )

        sentences = split_sentences_vi(text)

        self.assertEqual(3, len(sentences))
        self.assertEqual("Doanh nghiệp tăng tốc xây dựng SOC.", sentences[0])
        self.assertEqual("Thị trường bảo mật xoay trục sang AI.", sentences[1])
        self.assertEqual("Tuy nhiên, chi phí triển khai vẫn là rào cản.", sentences[2])

    def test_split_sentences_vi_does_not_split_tp_hcm(self) -> None:
        text = "TP.HCM đang đẩy mạnh chuyển đổi số. Doanh nghiệp tăng đầu tư AI."

        sentences = split_sentences_vi(text)

        self.assertEqual(
            ["TP.HCM đang đẩy mạnh chuyển đổi số.", "Doanh nghiệp tăng đầu tư AI."],
            sentences,
        )

    def test_chunk_start_gets_previous_sentence_overlap_for_bad_start(self) -> None:
        block = ArticleBlock(
            block_id="a1::b0",
            article_id="a1",
            block_index=0,
            block_type="body",
            text=(
                "Các doanh nghiệp đang tăng đầu tư bảo mật để giảm rủi ro vận hành. "
                "Điều này cho thấy AI ngày càng quan trọng trong trung tâm điều hành an ninh mạng. "
                "Trong khi đó, chi phí triển khai vẫn là thách thức với nhiều công ty vừa và nhỏ."
            ),
        )
        splitter = RecursiveSplitter(target_chunk_chars=95, max_chunk_chars=240, min_chunk_chars=40, overlap_sentences=1)

        chunks = splitter.split_block(block)

        self.assertGreater(len(chunks), 1)
        self.assertFalse(is_bad_chunk_start(chunks[1]))
        self.assertTrue(chunks[1].startswith("Các doanh nghiệp đang tăng đầu tư bảo mật"))
        self.assertIn("Điều này cho thấy AI", chunks[1])


if __name__ == "__main__":
    unittest.main()
