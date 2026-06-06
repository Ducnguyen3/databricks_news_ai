from __future__ import annotations

import json
import unittest

from app.local_ai.media_retriever import MediaRetriever


class _FailingRepository:
    def fetch_article_images(self, article_ids):
        raise RuntimeError("not available")


class _Repository:
    def fetch_article_images(self, article_ids):
        return [
            {
                "article_id": article_ids[0],
                "image_url": "https://example.com/repository.jpg",
                "caption": "Repository image",
            }
        ]


class MediaRetrieverTest(unittest.TestCase):
    def test_representative_image_is_prioritized(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {"article_id": "a1", "image_url": "https://example.com/normal.jpg", "caption": "Normal"},
                    {"article_id": "a1", "image_url": "https://example.com/rep.jpg", "is_representative": True},
                ]
            }
        )

        images = retriever.get_images_for_articles(["a1"], limit_per_article=2)

        self.assertEqual("https://example.com/rep.jpg", images[0]["image_url"])
        self.assertEqual("original", images[0]["type"])

    def test_captioned_image_is_prioritized_without_representative(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {"article_id": "a1", "image_url": "https://example.com/no-caption.jpg"},
                    {"article_id": "a1", "image_url": "https://example.com/caption.jpg", "caption": "Caption"},
                ]
            }
        )

        images = retriever.get_images_for_articles(["a1"], limit_per_article=1)

        self.assertEqual("https://example.com/caption.jpg", images[0]["image_url"])

    def test_filters_junk_images(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {"article_id": "a1", "image_url": "https://example.com/logo.png"},
                    {"article_id": "a1", "image_url": "https://example.com/banner.jpg"},
                    {"article_id": "a1", "image_url": "https://example.com/avatar.jpg"},
                    {"article_id": "a1", "image_url": "https://example.com/ad.jpg"},
                    {"article_id": "a1", "image_url": "https://example.com/icon.png"},
                    {"article_id": "a1", "image_url": "https://example.com/news.jpg"},
                ]
            }
        )

        images = retriever.get_images_for_articles(["a1"], limit_per_article=10)

        self.assertEqual(["https://example.com/news.jpg"], [image["image_url"] for image in images])

    def test_deduplicates_image_urls(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {"article_id": "a1", "image_url": "https://example.com/news.jpg"},
                    {"article_id": "a1", "image_url": "https://example.com/news.jpg"},
                ]
            }
        )

        images = retriever.get_images_for_articles(["a1"], limit_per_article=10)

        self.assertEqual(1, len(images))

    def test_limits_total_images_and_deduplicates_for_media_lookup(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {"article_id": "a1", "image_url": f"https://example.com/news-{index}.jpg", "caption": "Ukraine"}
                    for index in range(6)
                ],
                "a2": [
                    {"article_id": "a2", "image_url": "https://example.com/news-1.jpg", "caption": "Duplicate"},
                    {"article_id": "a2", "image_url": "https://example.com/news-6.jpg", "caption": "Ukraine"},
                    {"article_id": "a2", "image_url": "https://example.com/news-7.jpg", "caption": "Ukraine"},
                    {"article_id": "a2", "image_url": "https://example.com/news-8.jpg", "caption": "Ukraine"},
                    {"article_id": "a2", "image_url": "https://example.com/news-9.jpg", "caption": "Ukraine"},
                ],
            }
        )

        images = retriever.get_images_for_articles(["a1", "a2"], limit_per_article=10, max_images=4, query_terms=["Ukraine"])

        self.assertEqual(4, len(images))
        self.assertEqual(4, len({image["image_url"] for image in images}))

    def test_fallback_without_repository_or_failing_repository_does_not_crash(self) -> None:
        self.assertEqual([], MediaRetriever().get_images_for_articles(["a1"]))
        with self.assertLogs("app.local_ai.media_retriever", level="WARNING"):
            self.assertEqual([], MediaRetriever(image_repository=_FailingRepository()).get_images_for_articles(["a1"]))

    def test_repository_fallback_is_used_when_metadata_has_no_images(self) -> None:
        images = MediaRetriever(image_repository=_Repository()).get_images_for_articles(["a1"])

        self.assertEqual("https://example.com/repository.jpg", images[0]["image_url"])
        self.assertEqual("a1", images[0]["article_id"])

    def test_can_build_from_chunk_metadata(self) -> None:
        results = [
            {
                "metadata": {
                    "article_id": "a1",
                    "title": "Article",
                    "url": "https://example.com/a",
                    "source": "vnexpress",
                    "images_json": json.dumps([{"article_id": "a1", "image_url": "https://example.com/a.jpg"}]),
                }
            }
        ]

        images = MediaRetriever.from_retrieval_results(results).get_images_for_articles(["a1"])

        self.assertEqual("Article", images[0]["article_title"])
        self.assertEqual("https://example.com/a", images[0]["article_url"])

    def test_accepts_url_alias_from_metadata_images(self) -> None:
        retriever = MediaRetriever(
            metadata_images_by_article_id={
                "a1": [
                    {
                        "article_id": "a1",
                        "url": "https://example.com/alias.jpg",
                        "alt": "Alias image",
                    }
                ]
            }
        )

        images = retriever.get_images_for_articles(["a1"])

        self.assertEqual("https://example.com/alias.jpg", images[0]["image_url"])
        self.assertEqual("https://example.com/alias.jpg", images[0]["url"])
        self.assertEqual("Alias image", images[0]["caption"])


if __name__ == "__main__":
    unittest.main()
