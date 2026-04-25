from __future__ import annotations

import hashlib
from dataclasses import replace

from app.domain.models import Article
from app.processing.cleaner import clean_text


def hash_content(content: str) -> str:
    normalized = clean_text(content).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Deduplicator:
    def mark_duplicates(self, articles: list[Article]) -> list[Article]:
        seen_by_url: dict[str, str] = {}
        seen_by_hash: dict[str, str] = {}
        output: list[Article] = []

        for article in articles:
            content_hash = article.content_hash or hash_content(article.content)
            duplicate_group_id = self._existing_group_id(article, content_hash, seen_by_url, seen_by_hash)
            if duplicate_group_id is None:
                duplicate_group_id = article.article_id
                is_duplicate = False
            else:
                is_duplicate = True

            updated = replace(
                article,
                content_hash=content_hash,
                dedup_group_id=duplicate_group_id,
                is_duplicate=is_duplicate,
            )
            if updated.canonical_url:
                seen_by_url.setdefault(updated.canonical_url, updated.dedup_group_id)
            if updated.content_hash:
                seen_by_hash.setdefault(updated.content_hash, updated.dedup_group_id)
            output.append(updated)

        return output

    @staticmethod
    def _existing_group_id(
        article: Article,
        content_hash: str,
        seen_by_url: dict[str, str],
        seen_by_hash: dict[str, str],
    ) -> str | None:
        if article.canonical_url and article.canonical_url in seen_by_url:
            return seen_by_url[article.canonical_url]
        if content_hash and content_hash in seen_by_hash:
            return seen_by_hash[content_hash]
        return None

