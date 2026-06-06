from __future__ import annotations

import json
import unittest

from app.local_ai.prompt_builder import build_topic_rag_prompt
from app.local_ai.retrieved_article import (
    build_related_articles_from_retrieved_articles,
    build_retrieved_articles,
    build_sources_from_retrieved_articles,
)
from app.local_ai.topic_profiles import get_topic_profile


def _chunk(article_id: str, text: str, source: str = "vnexpress", score: float = 0.9) -> dict:
    return {
        "chunk_id": f"{article_id}:0",
        "text": text,
        "score": score,
        "metadata": {
            "article_id": article_id,
            "title": f"Title {article_id}",
            "source": source,
            "url": f"https://example.com/{article_id}",
            "published_at": "2026-06-03T00:00:00+00:00",
            "primary_topic": "tech_ai_internet",
            "entities_json": json.dumps([{"name": "OpenAI"}]),
            "images_json": json.dumps([{"article_id": article_id, "image_url": f"https://example.com/{article_id}.jpg"}]),
        },
    }


class RetrievedArticleTest(unittest.TestCase):
    def test_chunks_same_article_id_are_grouped(self) -> None:
        articles = build_retrieved_articles(
            [
                _chunk("a1", "Chunk one", score=0.8),
                _chunk("a1", "Chunk two", score=0.7),
                _chunk("a2", "Other article", score=0.6),
            ]
        )

        self.assertEqual(["a1", "a2"], [article["article_id"] for article in articles])
        self.assertEqual(2, len(articles[0]["matched_chunks"]))
        self.assertIn("Chunk one", articles[0]["selected_context"])
        self.assertIn("Chunk two", articles[0]["selected_context"])

    def test_parent_article_overrides_metadata_when_available(self) -> None:
        articles = build_retrieved_articles(
            [_chunk("a1", "Chunk text")],
            parent_articles=[
                {
                    "article_id": "a1",
                    "title": "Parent title",
                    "content": "Full parent content",
                    "source": "genk",
                    "url": "https://example.com/parent",
                    "published_at": "2026-06-04T00:00:00+00:00",
                    "primary_topic": "tech_ai_internet",
                }
            ],
        )

        self.assertEqual("Parent title", articles[0]["title"])
        self.assertEqual("Full parent content", articles[0]["content"])
        self.assertEqual("genk", articles[0]["source_name"])

    def test_no_parent_article_fallback_does_not_crash(self) -> None:
        articles = build_retrieved_articles([_chunk("a1", "Chunk text")], parent_articles=[])

        self.assertEqual("a1", articles[0]["article_id"])
        self.assertEqual("Title a1", articles[0]["title"])

    def test_image_url_alias_is_kept_from_chunk_metadata(self) -> None:
        chunk = _chunk("a1", "Chunk text")
        chunk["metadata"]["images_json"] = json.dumps([{"article_id": "a1", "url": "https://example.com/alias.jpg"}])

        articles = build_retrieved_articles([chunk], parent_articles=[])

        self.assertEqual("https://example.com/alias.jpg", articles[0]["images"][0]["image_url"])
        self.assertEqual("https://example.com/alias.jpg", articles[0]["images"][0]["url"])

    def test_sources_and_related_articles_are_deduped(self) -> None:
        articles = build_retrieved_articles([_chunk("a1", "One"), _chunk("a1", "Two"), _chunk("a2", "Three")])
        sources = build_sources_from_retrieved_articles(articles)
        related = build_related_articles_from_retrieved_articles(articles)

        self.assertEqual(["a1", "a2"], [source["article_id"] for source in sources])
        for key in ("source_name", "title", "url", "published_at"):
            self.assertIn(key, sources[0])
        self.assertEqual(["a1", "a2"], [article["article_id"] for article in related])

    def test_multi_source_prompt_contains_source_name_and_published_at(self) -> None:
        articles = build_retrieved_articles([_chunk("a1", "AI source one", "vnexpress"), _chunk("a2", "AI source two", "genk")])
        prompt = build_topic_rag_prompt(
            question="tin AI moi nhat",
            context_blocks=articles,
            topic_profile=get_topic_profile("tech_ai_internet"),
            query_plan={"intent": "topic_news"},
        )

        self.assertIn("Source name: vnexpress", prompt)
        self.assertIn("Source name: genk", prompt)
        self.assertIn("Published at: 2026-06-03T00:00:00+00:00", prompt)
        self.assertIn("[ARTICLE 1]", prompt)


if __name__ == "__main__":
    unittest.main()
