from __future__ import annotations

import json
import unittest
from datetime import datetime

from app.domain.models import Article
from app.ingestion.parsers.html_parser import parse_raw_document
from app.processing.image_extractor import extract_article_images


def _article() -> Article:
    now = datetime(2026, 1, 1)
    return Article(
        article_id="article-1",
        source="vnexpress",
        url="https://example.com/news/story.html",
        canonical_url="https://example.com/news/story.html",
        title="Story",
        summary_raw=None,
        content="This is an article body with enough content.",
        category="Tech",
        source_category_name="Tech",
        source_category_url="https://example.com/tech",
        published_at=None,
        crawled_at=now,
        content_hash="hash-1",
        dedup_group_id="article-1",
        is_duplicate=False,
        created_at=now,
        updated_at=now,
        raw_id="raw-1",
    )


def _raw_document(html: str) -> dict[str, object]:
    return {
        "raw_document_id": "raw-1",
        "source_name": "vnexpress",
        "url": "https://example.com/news/story.html",
        "canonical_url": "https://example.com/news/story.html",
        "raw_payload": json.dumps({"html": html, "title": "Story", "content": "Parsed content"}),
        "fetched_at": datetime(2026, 1, 1),
    }


class ArticleImageExtractionTest(unittest.TestCase):
    def test_extracts_multiple_article_images_with_caption(self) -> None:
        html = """
        <article>
          <figure>
            <img src="/images/first.jpg" alt="First alt" width="640" height="360">
            <figcaption>First caption</figcaption>
          </figure>
          <figure>
            <img data-src="https://cdn.example.com/second.jpg" alt="Second alt">
            <figcaption>Second caption</figcaption>
          </figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(2, len(images))
        self.assertEqual("https://example.com/images/first.jpg", images[0].image_url)
        self.assertEqual("First caption", images[0].caption)
        self.assertEqual("First alt", images[0].alt_text)
        self.assertEqual(640, images[0].width)
        self.assertEqual(360, images[0].height)

    def test_filters_logo_banner_avatar_and_tracking_images(self) -> None:
        html = """
        <article>
          <img src="/assets/logo.png">
          <img class="top-banner" src="/banner.jpg">
          <img alt="author avatar" src="/avatar.jpg">
          <img width="1" height="1" src="/tracking-pixel.gif">
          <figure>
            <img src="/images/real.jpg">
            <figcaption>Real image</figcaption>
          </figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(1, len(images))
        self.assertEqual("https://example.com/images/real.jpg", images[0].image_url)

    def test_first_captioned_body_image_is_representative(self) -> None:
        html = """
        <article>
          <img src="/images/no-caption.jpg" width="800" height="500">
          <figure>
            <img src="/images/captioned.jpg" width="300" height="180">
            <figcaption>Captioned image</figcaption>
          </figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(2, len(images))
        self.assertFalse(images[0].is_representative)
        self.assertTrue(images[1].is_representative)

    def test_article_without_images_still_parses_and_extracts_no_images(self) -> None:
        html = """
        <article>
          <h1>Story</h1>
          <p>This article has no images but still has valid text content for parsing.</p>
        </article>
        """
        raw_document = _raw_document(html)

        article = parse_raw_document(raw_document)
        images = extract_article_images(raw_document, article or _article())

        self.assertIsNotNone(article)
        self.assertEqual([], images)

    def test_relative_image_url_is_converted_to_absolute_url(self) -> None:
        html = """
        <article>
          <figure>
            <img src="../media/photo.jpg">
            <figcaption>Relative photo</figcaption>
          </figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual("https://example.com/media/photo.jpg", images[0].image_url)

    def test_duplicate_image_url_in_same_article_is_not_saved_twice(self) -> None:
        html = """
        <article>
          <figure><img src="/images/reused.jpg"><figcaption>First</figcaption></figure>
          <figure><img data-src="https://example.com/images/reused.jpg"><figcaption>Second</figcaption></figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(1, len(images))
        self.assertEqual("First", images[0].caption)

    def test_prefers_lazy_loaded_image_over_src_placeholder(self) -> None:
        html = """
        <article>
          <figure>
            <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
                 data-src="https://cdn.example.com/news/photo.jpg"
                 alt="Lazy photo">
            <figcaption>Lazy caption</figcaption>
          </figure>
        </article>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(1, len(images))
        self.assertEqual("https://cdn.example.com/news/photo.jpg", images[0].image_url)
        self.assertEqual("Lazy caption", images[0].caption)

    def test_uses_payload_image_when_html_has_no_valid_image(self) -> None:
        raw_document = _raw_document("<article><img src='/loading.gif'></article>")
        payload = json.loads(str(raw_document["raw_payload"]))
        payload["image"] = "https://cdn.example.com/lead.jpg"
        raw_document["raw_payload"] = json.dumps(payload)

        images = extract_article_images(raw_document, _article())

        self.assertEqual(1, len(images))
        self.assertEqual("https://cdn.example.com/lead.jpg", images[0].image_url)

    def test_uses_open_graph_image_when_body_images_are_not_valid(self) -> None:
        html = """
        <html>
          <head><meta property="og:image" content="https://cdn.example.com/og.jpg"></head>
          <body><article><img src="/loading.gif"></article></body>
        </html>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(1, len(images))
        self.assertEqual("https://cdn.example.com/og.jpg", images[0].image_url)

    def test_uses_json_ld_image_when_body_images_are_not_valid(self) -> None:
        html = """
        <html>
          <head>
            <script type="application/ld+json">
              {"@type": "NewsArticle", "image": {"url": "https://cdn.example.com/jsonld.jpg"}}
            </script>
          </head>
          <body><article><img src="/loading.gif"></article></body>
        </html>
        """

        images = extract_article_images(_raw_document(html), _article())

        self.assertEqual(1, len(images))
        self.assertEqual("https://cdn.example.com/jsonld.jpg", images[0].image_url)


if __name__ == "__main__":
    unittest.main()
