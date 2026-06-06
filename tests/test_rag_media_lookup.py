from __future__ import annotations

import json
import unittest

from app.local_ai.query_router import route_query
from app.local_ai.retriever import collect_related_images, filter_results_by_metadata


def _chunk(chunk_id: str, article_id: str, has_images: bool):
    return {
        "chunk_id": chunk_id,
        "text": "Ukraine hôm nay",
        "score": 0.9,
        "metadata": {
            "article_id": article_id,
            "title": "Ảnh về Ukraine",
            "source": "vnexpress",
            "url": "https://example.com/ukraine",
            "primary_topic": "world_geopolitics",
            "entity_names": "Ukraine",
            "published_at": "",
            "has_images": has_images,
            "images_json": json.dumps(
                [{"image_url": "https://example.com/u.jpg", "caption": "Ukraine", "credit": "Reuters"}]
                if has_images
                else []
            ),
        },
    }


class RagMediaLookupTest(unittest.TestCase):
    def test_media_query_routes_to_need_images(self) -> None:
        plan = route_query("ảnh về Ukraine hôm nay")

        self.assertEqual("media_lookup", plan["intent"])
        self.assertTrue(plan["need_images"])

    def test_retriever_keeps_related_articles_without_images_for_media_lookup(self) -> None:
        plan = route_query("ảnh về Ukraine hôm nay")
        plan["time_range"] = "all"
        results = [_chunk("c1", "a1", True), _chunk("c2", "a2", False)]

        filtered = filter_results_by_metadata(results, plan)

        self.assertEqual(["a1", "a2"], [item["metadata"]["article_id"] for item in filtered])

    def test_middle_east_media_query_routes_to_world_with_default_image_limit(self) -> None:
        plan = route_query("cho toi cac anh lien quan den tinh hinh chien su Trung Dong")

        self.assertEqual("media_lookup", plan["intent"])
        self.assertEqual("world_geopolitics", plan["primary_topic"])
        self.assertEqual(4, plan["image_limit"])
        self.assertIn("Trung Đông", plan["entities"])

    def test_media_query_parses_requested_image_limit(self) -> None:
        plan = route_query("cho toi 6 anh ve Ukraine")

        self.assertEqual("media_lookup", plan["intent"])
        self.assertEqual(6, plan["image_limit"])

    def test_collect_related_images_dedupes_multiple_chunks_same_article(self) -> None:
        results = [_chunk("c1", "a1", True), _chunk("c2", "a1", True)]

        images = collect_related_images(results)

        self.assertEqual(1, len(images))
        self.assertEqual("https://example.com/u.jpg", images[0]["image_url"])
        self.assertEqual("a1", images[0]["article_id"])


if __name__ == "__main__":
    unittest.main()
