from __future__ import annotations

import unittest

from app.processing.entity_extractor import extract_entities


def _by_normalized(result):
    return {entity["normalized_name"]: entity for entity in result["entities"]}


class EntityExtractorTest(unittest.TestCase):
    def test_extracts_stock_symbols(self) -> None:
        result = extract_entities(title="HPG và FPT tăng mạnh, VN-Index vượt mốc mới")
        entities = _by_normalized(result)

        self.assertEqual("stock_symbol", entities["HPG"]["type"])
        self.assertEqual("stock_symbol", entities["FPT"]["type"])
        self.assertEqual("stock_symbol", entities["VNINDEX"]["type"])

    def test_extracts_countries(self) -> None:
        result = extract_entities(title="Mỹ và Trung Quốc tiếp tục căng thẳng")
        entities = _by_normalized(result)

        self.assertEqual("country", entities["Hoa Kỳ"]["type"])
        self.assertEqual("country", entities["Trung Quốc"]["type"])

    def test_extracts_organizations(self) -> None:
        result = extract_entities(title="Ngân hàng Nhà nước và Fed phát tín hiệu mới về lãi suất")
        entities = _by_normalized(result)

        self.assertEqual("organization", entities["Ngân hàng Nhà nước"]["type"])
        self.assertEqual("organization", entities["Fed"]["type"])

    def test_extracts_locations(self) -> None:
        result = extract_entities(title="Giá chung cư tại Hà Nội và TP.HCM tiếp tục tăng")
        entities = _by_normalized(result)

        self.assertEqual("location", entities["Hà Nội"]["type"])
        self.assertEqual("location", entities["TP.HCM"]["type"])

    def test_merges_aliases_without_duplicate(self) -> None:
        result = extract_entities(title="Mỹ và Hoa Kỳ cùng xuất hiện trong bản tin")
        entities = [entity for entity in result["entities"] if entity["normalized_name"] == "Hoa Kỳ"]

        self.assertEqual(1, len(entities))
        self.assertEqual(2, entities[0]["mention_count"])

    def test_mention_count_for_repeated_symbol(self) -> None:
        result = extract_entities(title="HPG HPG HPG")
        entities = _by_normalized(result)

        self.assertEqual(3, entities["HPG"]["mention_count"])

    def test_empty_input_returns_empty_entities(self) -> None:
        self.assertEqual({"entities": []}, extract_entities(title=None, summary=None, content=None))


if __name__ == "__main__":
    unittest.main()
