from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    unescaped = html.unescape(value)
    return _WHITESPACE_RE.sub(" ", unescaped).strip()


def strip_html_tags(value: str | None) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    return clean_text(parser.text)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self._parts)

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data)

