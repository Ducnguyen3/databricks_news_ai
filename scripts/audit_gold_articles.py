from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.databricks_client import DatabricksSqlConfig


REQUIRED_FIELDS = [
    "article_id",
    "title",
    "cleaned_content",
    "source_name",
    "topic",
    "published_at",
]


def main(argv: Sequence[str] | None = None) -> None:
    configure_stdout()
    load_dotenv()
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_gold_articles(
        table=args.table,
        output_dir=output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Audit Databricks Gold articles_clean required fields.")
    parser.add_argument("--table", default=settings.local_ai.databricks_articles_table)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output-dir", default="data/gold_audit")
    return parser.parse_args(argv)


def audit_gold_articles(
    *,
    table: str,
    output_dir: Path,
    limit: int | None = None,
    batch_size: int = 1000,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    config = DatabricksSqlConfig.from_env()
    query = build_query(table=table, limit=limit)
    valid_path = output_dir / "valid_articles.jsonl"
    invalid_path = output_dir / "invalid_articles.jsonl"
    summary_path = output_dir / "audit_summary.json"

    counters = AuditCounters()
    seen_article_ids: set[str] = set()

    from databricks import sql

    with sql.connect(
        server_hostname=config.server_hostname,
        http_path=config.http_path,
        access_token=config.token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            with valid_path.open("w", encoding="utf-8", newline="\n") as valid_file:
                with invalid_path.open("w", encoding="utf-8", newline="\n") as invalid_file:
                    while True:
                        rows = cursor.fetchmany(max(1, int(batch_size or 1000)))
                        if not rows:
                            break
                        for row in rows:
                            article = normalize_article(dict(zip(columns, row)))
                            reasons = validate_article(article, seen_article_ids=seen_article_ids)
                            counters.update(article, reasons)
                            if reasons:
                                invalid_file.write(json.dumps({"invalid_reasons": reasons, **article}, ensure_ascii=False, default=json_default) + "\n")
                            else:
                                valid_file.write(json.dumps(article, ensure_ascii=False, default=json_default) + "\n")

    summary = counters.summary(
        table=table,
        started_at=started_at,
        output_dir=output_dir,
        valid_path=valid_path,
        invalid_path=invalid_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    return summary


def build_query(*, table: str, limit: int | None = None) -> str:
    safe_table = validate_table_name(table)
    limit_clause = ""
    if limit is not None:
        safe_limit = int(limit)
        if safe_limit <= 0:
            raise ValueError("limit must be positive")
        limit_clause = f"\nLIMIT {safe_limit}"
    return f"""
    SELECT
        article_id,
        title,
        content AS cleaned_content,
        source AS source_name,
        primary_topic AS topic,
        published_at
    FROM {safe_table}
    ORDER BY COALESCE(published_at, crawled_at) DESC, updated_at DESC
    {limit_clause}
    """.strip()


def validate_table_name(table: str) -> str:
    value = str(table or "").strip()
    if not re.fullmatch(r"[A-Za-z_][\w]*(\.[A-Za-z_][\w]*){0,2}", value):
        raise ValueError(f"Invalid Databricks table name: {table!r}")
    return value


def normalize_article(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in REQUIRED_FIELDS}


def validate_article(article: dict[str, Any], *, seen_article_ids: set[str] | None = None) -> list[str]:
    reasons: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in article:
            reasons.append(f"missing_field:{field}")
            continue
        if is_empty(article.get(field)):
            reasons.append(f"empty_field:{field}")

    article_id = str(article.get("article_id") or "").strip()
    if seen_article_ids is not None and article_id:
        if article_id in seen_article_ids:
            reasons.append("duplicate_article_id")
        seen_article_ids.add(article_id)
    return reasons


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


class AuditCounters:
    def __init__(self) -> None:
        self.total = 0
        self.valid = 0
        self.invalid = 0
        self.reason_counts: Counter[str] = Counter()
        self.empty_field_counts: Counter[str] = Counter()
        self.missing_field_counts: Counter[str] = Counter()
        self.topic_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.min_content_chars: int | None = None
        self.max_content_chars = 0
        self.total_content_chars = 0

    def update(self, article: dict[str, Any], reasons: list[str]) -> None:
        self.total += 1
        if reasons:
            self.invalid += 1
        else:
            self.valid += 1
        self.reason_counts.update(reasons)
        for reason in reasons:
            if reason.startswith("empty_field:"):
                self.empty_field_counts.update([reason.split(":", 1)[1]])
            if reason.startswith("missing_field:"):
                self.missing_field_counts.update([reason.split(":", 1)[1]])
        topic = str(article.get("topic") or "").strip()
        source = str(article.get("source_name") or "").strip()
        if topic:
            self.topic_counts.update([topic])
        if source:
            self.source_counts.update([source])
        content_chars = len(str(article.get("cleaned_content") or ""))
        self.total_content_chars += content_chars
        self.max_content_chars = max(self.max_content_chars, content_chars)
        self.min_content_chars = content_chars if self.min_content_chars is None else min(self.min_content_chars, content_chars)

    def summary(
        self,
        *,
        table: str,
        started_at: datetime,
        output_dir: Path,
        valid_path: Path,
        invalid_path: Path,
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        return {
            "table": table,
            "required_fields": REQUIRED_FIELDS,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "output_dir": str(output_dir),
            "valid_articles_path": str(valid_path),
            "invalid_articles_path": str(invalid_path),
            "summary_path": str(output_dir / "audit_summary.json"),
            "total_articles": self.total,
            "valid_articles": self.valid,
            "invalid_articles": self.invalid,
            "valid_ratio": round(self.valid / self.total, 6) if self.total else 0.0,
            "reason_counts": dict(self.reason_counts),
            "missing_field_counts": dict(self.missing_field_counts),
            "empty_field_counts": dict(self.empty_field_counts),
            "topic_counts": dict(self.topic_counts),
            "source_counts": dict(self.source_counts),
            "content_chars": {
                "min": self.min_content_chars or 0,
                "max": self.max_content_chars,
                "avg": round(self.total_content_chars / self.total, 2) if self.total else 0.0,
            },
        }


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
