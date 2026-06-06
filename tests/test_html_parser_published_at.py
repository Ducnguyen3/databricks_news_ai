from __future__ import annotations

import json
import unittest
from datetime import datetime

from app.ingestion.parsers.html_parser import parse_datetime, parse_raw_document


def _raw_document(source: str, html: str, payload_values: dict[str, object] | None = None) -> dict[str, object]:
    payload = {
        "title": "Story",
        "summary": "Summary",
        "content": "This is a long enough article content body for parser tests.",
        "html": html,
    }
    payload.update(payload_values or {})
    return {
        "raw_document_id": f"raw-{source}",
        "source_name": source,
        "url": "https://example.com/story.html",
        "canonical_url": "https://example.com/story.html",
        "raw_payload": json.dumps(payload),
        "fetched_at": datetime(2026, 6, 4, 1, 0, 0),
    }


class HtmlParserPublishedAtTest(unittest.TestCase):
    def test_parse_vietnamese_datetime_with_gmt_offset(self) -> None:
        parsed = parse_datetime("Thứ tư, 3/6/2026, 14:30 (GMT+7)")

        self.assertEqual(datetime(2026, 6, 3, 7, 30, 0), parsed)

    def test_parse_vietnamese_datetime_without_weekday(self) -> None:
        parsed = parse_datetime("03/06/2026 14:30")

        self.assertEqual(datetime(2026, 6, 3, 14, 30, 0), parsed)

    def test_parse_vietnamese_datetime_with_pipe_separator_and_gmt(self) -> None:
        parsed = parse_datetime("04/06/2026 | 10:30 GMT+7")

        self.assertEqual(datetime(2026, 6, 4, 3, 30, 0), parsed)

    def test_parse_datetime_with_dash_separator_and_ampm(self) -> None:
        parsed = parse_datetime("04-06-2026 - 9:15 PM")

        self.assertEqual(datetime(2026, 6, 4, 21, 15, 0), parsed)

    def test_parse_us_style_datetime_with_ampm(self) -> None:
        parsed = parse_datetime("5/27/2026 2:51:01 PM")

        self.assertEqual(datetime(2026, 5, 27, 14, 51, 1), parsed)

    def test_parse_raw_document_uses_published_at_from_html_meta(self) -> None:
        html = """
        <html>
          <head><meta property="article:published_time" content="2026-06-03T14:30:00+07:00"></head>
          <body><article><p>Body</p></article></body>
        </html>
        """

        article = parse_raw_document(_raw_document("vnexpress", html, {"published_at": ""}))

        self.assertIsNotNone(article)
        self.assertEqual(datetime(2026, 6, 3, 7, 30, 0), article.published_at)

    def test_parse_raw_document_uses_vnexpress_date_selector(self) -> None:
        html = """
        <html>
          <body>
            <span class="date">Thứ năm, 4/6/2026, 09:05 (GMT+7)</span>
            <article><p>Body</p></article>
          </body>
        </html>
        """

        article = parse_raw_document(_raw_document("vnexpress", html, {"published_at": ""}))

        self.assertIsNotNone(article)
        self.assertEqual(datetime(2026, 6, 4, 2, 5, 0), article.published_at)

    def test_parse_raw_document_finds_date_text_without_known_selector(self) -> None:
        html = """
        <html>
          <body>
            <div class="unknown-time-box">Cập nhật: 04/06/2026 | 10:30 GMT+7</div>
            <article><p>Body</p></article>
          </body>
        </html>
        """

        article = parse_raw_document(_raw_document("diendandoanhnghiep", html, {"published_at": ""}))

        self.assertIsNotNone(article)
        self.assertEqual(datetime(2026, 6, 4, 3, 30, 0), article.published_at)

    def test_parse_raw_document_uses_diendandoanhnghiep_longform_publish_time(self) -> None:
        html = """
        <html>
          <body>
            <span class="sc-longform-header-date block-sc-publish-time">28/05/2026 09:00</span>
            <article><p>Body</p></article>
          </body>
        </html>
        """

        article = parse_raw_document(_raw_document("diendandoanhnghiep", html, {"published_at": ""}))

        self.assertIsNotNone(article)
        self.assertEqual(datetime(2026, 5, 28, 9, 0, 0), article.published_at)

    def test_parse_raw_document_uses_diendandoanhnghiep_name_article_published_time(self) -> None:
        html = """
        <html>
          <head>
            <meta name="article:published_time" content="5/27/2026 2:51:01 PM" />
            <meta name="datetimenow" content="01/06/2026 17:53:32" />
          </head>
          <body><article><p>Body</p></article></body>
        </html>
        """

        article = parse_raw_document(_raw_document("diendandoanhnghiep", html, {"published_at": ""}))

        self.assertIsNotNone(article)
        self.assertEqual(datetime(2026, 5, 27, 14, 51, 1), article.published_at)


if __name__ == "__main__":
    unittest.main()
