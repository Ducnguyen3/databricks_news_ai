from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.pipeline import create_embedding_model, create_rag_service
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging

DEFAULT_QUERY_FILE = Path("tests/fixtures/rag_quality_queries.json")
REQUIRED_RESPONSE_FIELDS = ("answer", "intent", "topic", "query_plan", "sources", "images", "related_articles")
FALLBACK_MARKERS = (
    "khong tim thay",
    "chua tim thay",
    "chua co du lieu",
    "chua co anh phu hop",
    "no suitable data",
    "not enough data",
)


@dataclass(frozen=True, slots=True)
class QualityEvalResult:
    id: str
    query: str
    status: str
    expected: dict[str, Any]
    scores: dict[str, str]
    actual: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    answer_preview: str = ""
    sources_preview: list[dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)
    report = run_quality_eval(args)
    if args.save_report:
        save_report(report, args.save_report)
    if args.json:
        print_json_report(report)
    else:
        print_text_report(report)
    if args.strict and report["summary"]["warn"]:
        raise SystemExit(1)
    if report["summary"]["fail"]:
        raise SystemExit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval/answer quality against a query fixture.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERY_FILE), help="Path to quality query JSON fixture.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--save_report", default=None, help="Save JSON report to this path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any query is WARN or FAIL.")
    parser.add_argument("--top_k", type=int, default=settings.local_ai.rag_top_k)
    parser.add_argument("--chroma_path", default=settings.local_ai.chroma_persist_dir)
    parser.add_argument("--collection_name", default=settings.local_ai.chroma_collection_name)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_quality_queries(path: str | Path = DEFAULT_QUERY_FILE) -> list[dict[str, Any]]:
    query_path = Path(path)
    data = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("quality query fixture must be a JSON list")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"quality query at index {index} must be object")
        if not item.get("id") or not item.get("query"):
            raise ValueError(f"quality query at index {index} must include id and query")
    return data


def run_quality_eval(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_quality_queries(args.queries)
    settings = load_settings()
    embedding_model = create_embedding_model(settings.local_ai)
    vector_store = ChromaVectorStore(
        persist_directory=args.chroma_path,
        collection_name=args.collection_name,
    )
    service = create_rag_service(settings.local_ai, embedding_model, vector_store)
    started_at = perf_counter()
    results = [
        evaluate_query(service, expected, top_k=max(1, int(args.top_k)))
        for expected in queries
    ]
    return build_report(results, duration_seconds=round(perf_counter() - started_at, 4))


def evaluate_query(service: Any, expected: dict[str, Any], top_k: int) -> QualityEvalResult:
    started_at = perf_counter()
    query = str(expected["query"])
    try:
        response = service.answer_structured(query, top_k=top_k)
    except Exception as exc:
        return QualityEvalResult(
            id=str(expected["id"]),
            query=query,
            status="FAIL",
            expected=_expected_preview(expected),
            scores={"runtime": "FAIL"},
            actual={},
            errors=[str(exc)],
            duration_seconds=round(perf_counter() - started_at, 4),
        )
    result = score_response(expected, response)
    return QualityEvalResult(
        id=str(expected["id"]),
        query=query,
        status=result["status"],
        expected=_expected_preview(expected),
        scores=result["scores"],
        actual=result["actual"],
        warnings=result["warnings"],
        errors=result["errors"],
        answer_preview=_clip(str(response.get("answer") or ""), 500) if isinstance(response, dict) else "",
        sources_preview=_source_preview(response.get("sources") if isinstance(response, dict) else []),
        duration_seconds=round(perf_counter() - started_at, 4),
    )


def score_response(expected: dict[str, Any], response: Any) -> dict[str, Any]:
    scores: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(response, dict):
        return _score_result({"schema": "FAIL"}, {}, [], ["response must be dict"])

    _merge_score(scores, warnings, errors, "schema", score_schema(response))
    _merge_score(scores, warnings, errors, "intent", score_intent(expected, response))
    _merge_score(scores, warnings, errors, "topic", score_topic(expected, response))
    _merge_score(scores, warnings, errors, "sources", score_sources(expected, response))
    _merge_score(scores, warnings, errors, "multi_source", score_multi_source(expected, response))
    _merge_score(scores, warnings, errors, "entities", score_entities(expected, response))
    _merge_score(scores, warnings, errors, "images", score_images(expected, response))
    _merge_score(scores, warnings, errors, "related_articles", score_related_articles(response))
    _merge_score(scores, warnings, errors, "answer", score_answer(response))
    _merge_score(scores, warnings, errors, "recency", score_recency(expected, response))

    return _score_result(scores, build_actual(response), warnings, errors)


def score_schema(response: dict[str, Any]) -> tuple[str, str | None]:
    missing = [field for field in REQUIRED_RESPONSE_FIELDS if field not in response]
    if missing:
        return "FAIL", f"missing fields: {', '.join(missing)}"
    for field_name in ("sources", "images", "related_articles"):
        if not isinstance(response.get(field_name), list):
            return "FAIL", f"{field_name} must be list"
    if not isinstance(response.get("query_plan"), dict):
        return "FAIL", "query_plan must be dict"
    return "OK", None


def score_intent(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    expected_intent = expected.get("expected_intent")
    if not expected_intent:
        return "SKIP", None
    actual = str(response.get("intent") or "")
    if actual == str(expected_intent):
        return "OK", None
    return "WARN", f"wrong_intent: expected {expected_intent}, got {actual or 'empty'}"


def score_topic(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    expected_topics = [str(item) for item in expected.get("expected_topic_contains") or []]
    if not expected_topics:
        return "SKIP", None
    actual = str(response.get("topic") or "")
    if any(topic_matches(expected_topic, actual) for expected_topic in expected_topics):
        return "OK", None
    return "WARN", f"wrong_topic: expected one of {expected_topics}, got {actual or 'empty'}"


def topic_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalize(expected)
    actual_norm = _normalize(actual)
    if expected_norm in actual_norm:
        return True
    aliases = {
        "world": ("world", "quoc_te", "geopolitics", "the_gioi"),
        "quoc_te": ("world", "quoc_te", "geopolitics", "the_gioi"),
        "finance": ("finance", "stock", "kinh_te", "chung_khoan", "tai_chinh"),
        "stock": ("finance", "stock", "kinh_te", "chung_khoan", "tai_chinh"),
        "ai": ("ai", "tech", "technology", "cong_nghe", "internet"),
        "technology": ("ai", "tech", "technology", "cong_nghe", "internet"),
        "business": ("business", "doanh_nghiep", "startup", "khoi_nghiep"),
        "doanh_nghiep": ("business", "doanh_nghiep", "startup", "khoi_nghiep"),
    }
    expected_aliases = aliases.get(expected_norm, (expected_norm,))
    return any(alias in actual_norm for alias in expected_aliases)


def score_sources(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    sources = _list(response.get("sources"))
    min_sources = int(expected.get("min_sources") or 0)
    if min_sources and len(sources) < min_sources:
        return "WARN", f"empty_sources: expected at least {min_sources}, got {len(sources)}"
    expected_source = str(expected.get("expected_source") or "")
    if expected_source and not any(expected_source.lower() in str(source.get("source") or "").lower() for source in sources):
        return "WARN", f"expected_source_missing: {expected_source}"
    return "OK", None


def score_multi_source(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    if not expected.get("requires_multi_source"):
        return "SKIP", None
    unique_sources = {
        str(source.get("source") or "").strip().lower()
        for source in _list(response.get("sources"))
        if str(source.get("source") or "").strip()
    }
    if len(unique_sources) >= 2:
        return "OK", None
    return "WARN", "multi_source_weak: requires multiple sources but fewer than two were returned"


def score_entities(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    expected_entities = [str(item) for item in expected.get("expected_entities") or []]
    if not expected_entities:
        return "SKIP", None
    haystack_parts: list[str] = []
    query_plan = response.get("query_plan")
    if isinstance(query_plan, dict):
        haystack_parts.extend(str(item) for item in query_plan.get("entities") or [])
        haystack_parts.extend(str(item) for item in query_plan.get("stock_symbols") or [])
    for source in _list(response.get("sources")):
        haystack_parts.append(str(source.get("title") or ""))
    for article in _list(response.get("related_articles")):
        haystack_parts.append(str(article.get("title") or ""))
    haystack = _normalize(" ".join(haystack_parts))
    if any(_normalize(entity) in haystack for entity in expected_entities):
        return "OK", None
    return "WARN", f"entity_missing: expected one of {expected_entities}"


def score_images(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    if not expected.get("requires_images"):
        return "SKIP", None
    images = _list(response.get("images"))
    min_images = int(expected.get("min_images") or 1)
    if len(images) >= min_images:
        return "OK", None
    return "WARN", f"images_missing: expected at least {min_images}, got {len(images)}"


def score_related_articles(response: dict[str, Any]) -> tuple[str, str | None]:
    related = _list(response.get("related_articles"))
    sources = _list(response.get("sources"))
    if not related and sources:
        return "WARN", "related_articles_empty"
    seen: set[str] = set()
    for article in related:
        key = str(article.get("article_id") or article.get("title") or "").strip().lower()
        if key and key in seen:
            return "WARN", "related_articles_duplicate"
        seen.add(key)
    return "OK", None


def score_answer(response: dict[str, Any]) -> tuple[str, str | None]:
    answer = str(response.get("answer") or "").strip()
    sources = _list(response.get("sources"))
    if not answer:
        return "FAIL", "answer is empty"
    normalized = _normalize(answer)
    has_fallback = any(marker in normalized for marker in FALLBACK_MARKERS)
    if sources and has_fallback:
        return "WARN", "weak_answer: fallback answer returned despite sources"
    if sources and len(answer) < 40:
        return "WARN", "weak_answer: answer is too short for sourced response"
    if not sources and not has_fallback:
        return "WARN", "hallucination_risk: answer without sources does not look like fallback"
    return "OK", None


def score_recency(expected: dict[str, Any], response: dict[str, Any]) -> tuple[str, str | None]:
    if not expected.get("requires_recent"):
        return "SKIP", None
    dates = [_parse_datetime(str(source.get("published_at") or "")) for source in _list(response.get("sources"))]
    dates = [item for item in dates if item is not None]
    if not dates:
        return "WARN", "stale_sources: no published_at values to evaluate recency"
    newest = max(dates)
    now = datetime.now(timezone.utc)
    age_days = (now - newest).days
    if age_days <= 30:
        return "OK", None
    return "WARN", f"stale_sources: newest source is {age_days} days old"


def build_actual(response: dict[str, Any]) -> dict[str, Any]:
    sources = _list(response.get("sources"))
    images = _list(response.get("images"))
    related = _list(response.get("related_articles"))
    return {
        "intent": response.get("intent"),
        "topic": response.get("topic"),
        "source_count": len(sources),
        "unique_source_count": len({str(source.get("source") or "").lower() for source in sources if source.get("source")}),
        "image_count": len(images),
        "related_count": len(related),
        "retrieval_mode": response.get("query_plan", {}).get("retrieval_mode") if isinstance(response.get("query_plan"), dict) else None,
    }


def build_report(results: list[QualityEvalResult], duration_seconds: float) -> dict[str, Any]:
    summary_counter = Counter(result.status.lower() for result in results)
    issues: Counter[str] = Counter()
    for result in results:
        for message in [*result.warnings, *result.errors]:
            issues[_issue_key(message)] += 1
    return {
        "summary": {
            "total": len(results),
            "ok": summary_counter["ok"],
            "warn": summary_counter["warn"],
            "fail": summary_counter["fail"],
            "duration_seconds": duration_seconds,
        },
        "issues": dict(sorted(issues.items())),
        "results": [asdict(result) for result in results],
    }


def save_report(report: dict[str, Any], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_json_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def print_text_report(report: dict[str, Any]) -> None:
    print("========================================")
    print("RAG QUALITY EVALUATION")
    print("========================================")
    print("")
    for result in report["results"]:
        print(f"Query: {result['query']}")
        print(f"Expected intent: {result['expected'].get('expected_intent') or ''}")
        print(f"Actual intent: {result['actual'].get('intent') or ''}")
        print(f"Expected topic: {', '.join(result['expected'].get('expected_topic_contains') or [])}")
        print(f"Actual topic: {result['actual'].get('topic') or ''}")
        print("")
        print("Scores:")
        for score_name, status in result["scores"].items():
            print(f"- {score_name}: {status}")
        if result["sources_preview"]:
            print("")
            print("Sources:")
            for index, source in enumerate(result["sources_preview"], start=1):
                print(f"{index}. [{source.get('source') or ''}] {source.get('title') or ''}")
        if result["answer_preview"]:
            print("")
            print("Answer preview:")
            print(result["answer_preview"])
        print("")
        print(f"Status: {result['status']}")
        reasons = [*result["warnings"], *result["errors"]]
        if reasons:
            print("Reason:")
            for reason in reasons:
                print(f"- {reason}")
        print("")
        print("----------------------------------------")
        print("")
    summary = report["summary"]
    print("Summary:")
    print(f"Total queries: {summary['total']}")
    print(f"OK: {summary['ok']}")
    print(f"WARN: {summary['warn']}")
    print(f"FAIL: {summary['fail']}")
    print(f"Duration: {summary['duration_seconds']}s")
    print("")
    print("Common issues:")
    if report["issues"]:
        for issue, count in report["issues"].items():
            print(f"- {issue}: {count}")
    else:
        print("- none: 0")
    print("========================================")


def _score_result(
    scores: dict[str, str],
    actual: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    if errors or any(status == "FAIL" for status in scores.values()):
        status = "FAIL"
    elif warnings or any(status == "WARN" for status in scores.values()):
        status = "WARN"
    else:
        status = "OK"
    return {"status": status, "scores": scores, "actual": actual, "warnings": warnings, "errors": errors}


def _merge_score(
    scores: dict[str, str],
    warnings: list[str],
    errors: list[str],
    name: str,
    scored: tuple[str, str | None],
) -> None:
    status, message = scored
    scores[name] = status
    if not message:
        return
    if status == "FAIL":
        errors.append(message)
    elif status == "WARN":
        warnings.append(message)


def _list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_preview(sources: Any, limit: int = 5) -> list[dict[str, str]]:
    preview: list[dict[str, str]] = []
    for source in _list(sources)[:limit]:
        preview.append(
            {
                "title": str(source.get("title") or ""),
                "source": str(source.get("source") or ""),
                "url": str(source.get("url") or ""),
                "published_at": str(source.get("published_at") or ""),
            }
        )
    return preview


def _issue_key(message: str) -> str:
    prefix = message.split(":", maxsplit=1)[0].strip().lower()
    return prefix.replace(" ", "_") or "unknown"


def _expected_preview(expected: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "expected_intent",
        "expected_topic_contains",
        "expected_entities",
        "expected_source",
        "min_sources",
        "min_images",
        "requires_images",
        "requires_recent",
        "requires_multi_source",
    )
    return {key: expected[key] for key in keys if key in expected}


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


if __name__ == "__main__":
    main()
