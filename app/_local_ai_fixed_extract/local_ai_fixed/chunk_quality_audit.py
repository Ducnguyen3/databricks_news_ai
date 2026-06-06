from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging

CRITICAL_METADATA = ("article_id", "source", "title", "url", "published_at", "primary_topic", "content_hash")
OPTIONAL_METADATA = (
    "primary_topic_name",
    "entities_json",
    "images_json",
    "has_images",
    "image_count",
    "secondary_topics_json",
    "indexed_at",
    "embedding_model",
    "chunking_version",
    "index_version",
)
JSON_FIELDS = ("entities_json", "images_json", "secondary_topics_json")
WEAK_ENDINGS = ("va", "cua", "voi", "trong khi", "nhung", "and", "of", "with", "but")
WEAK_STARTS = ("nay", "do", "ong", "ba", "ho", "no", "they", "this", "that", "he", "she")


@dataclass(frozen=True, slots=True)
class ChunkIssue:
    chunk_id: str
    article_id: str
    issue: str
    detail: str


@dataclass(frozen=True, slots=True)
class ChunkSample:
    chunk_id: str
    article_id: str
    source: str
    title: str
    topic: str
    published_at: str
    length: int
    has_images: bool
    text_preview: str


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)
    report = run_audit(args)
    if args.save_report:
        save_report(report, args.save_report)
    if args.json:
        print_json_report(report)
    else:
        print_text_report(report)
    if report["status"] == "FAIL":
        raise SystemExit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Audit chunk quality in the local Chroma collection.")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample chunks to print.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--save_report", default=None, help="Save JSON report to this path.")
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--max_chars", type=int, default=2500)
    parser.add_argument("--source", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks read from Chroma.")
    parser.add_argument("--chroma_path", default=settings.local_ai.chroma_persist_dir)
    parser.add_argument("--collection_name", default=settings.local_ai.chroma_collection_name)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    try:
        vector_store = ChromaVectorStore(
            persist_directory=args.chroma_path,
            collection_name=args.collection_name,
        )
        chunks = vector_store.get_all_chunks(limit=args.limit)
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": str(exc),
            "chroma_path": args.chroma_path,
            "collection_name": args.collection_name,
            "overview": {"total_chunks": 0},
        }
    chunks = filter_chunks(chunks, source=args.source, topic=args.topic)
    return audit_chunks(
        chunks,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        sample_size=args.sample,
        source_filter=args.source,
        topic_filter=args.topic,
    )


def audit_chunks(
    chunks: list[dict[str, Any]],
    chroma_path: str,
    collection_name: str,
    min_chars: int = 100,
    max_chars: int = 2500,
    sample_size: int = 5,
    source_filter: str | None = None,
    topic_filter: str | None = None,
) -> dict[str, Any]:
    length = compute_length_stats(chunks, min_chars=min_chars, max_chars=max_chars)
    metadata = compute_metadata_completeness(chunks)
    json_validity = compute_json_validity(chunks)
    image_consistency = compute_image_consistency(chunks)
    grouping = compute_article_grouping(chunks)
    duplicates = compute_duplicate_stats(chunks)
    boundary = compute_boundary_warnings(chunks, min_chars=min_chars)
    overview = compute_overview(chunks, chroma_path, collection_name, source_filter, topic_filter)
    status, reasons = build_status(length, metadata, duplicates, overview)
    return {
        "status": status,
        "status_reasons": reasons,
        "overview": overview,
        "length": length,
        "metadata": metadata,
        "json_validity": json_validity,
        "image_consistency": image_consistency,
        "article_grouping": grouping,
        "duplicates": duplicates,
        "boundary_warnings": boundary,
        "samples": [asdict(sample) for sample in build_samples(chunks, sample_size)],
    }


def filter_chunks(chunks: list[dict[str, Any]], source: str | None = None, topic: str | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = _metadata(chunk)
        if source and str(metadata.get("source") or "").lower() != source.lower():
            continue
        if topic and str(metadata.get("primary_topic") or metadata.get("topic_category") or "").lower() != topic.lower():
            continue
        output.append(chunk)
    return output


def compute_overview(
    chunks: list[dict[str, Any]],
    chroma_path: str,
    collection_name: str,
    source_filter: str | None = None,
    topic_filter: str | None = None,
) -> dict[str, Any]:
    article_ids = set()
    sources = set()
    topics = set()
    for chunk in chunks:
        metadata = _metadata(chunk)
        if metadata.get("article_id"):
            article_ids.add(str(metadata["article_id"]))
        if metadata.get("source"):
            sources.add(str(metadata["source"]))
        topic = metadata.get("primary_topic") or metadata.get("topic_category")
        if topic:
            topics.add(str(topic))
    return {
        "chroma_path": chroma_path,
        "collection_name": collection_name,
        "source_filter": source_filter,
        "topic_filter": topic_filter,
        "total_chunks": len(chunks),
        "unique_articles": len(article_ids),
        "unique_sources": len(sources),
        "unique_topics": len(topics),
    }


def compute_length_stats(chunks: list[dict[str, Any]], min_chars: int, max_chars: int) -> dict[str, Any]:
    lengths = [len(str(chunk.get("text") or "")) for chunk in chunks]
    if not lengths:
        return _empty_length_stats()
    too_short = sum(1 for value in lengths if value < min_chars)
    too_long = sum(1 for value in lengths if value > max_chars)
    return {
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
        "median_chars": statistics.median(lengths),
        "p90_chars": percentile(lengths, 90),
        "too_short_count": too_short,
        "too_short_ratio": ratio(too_short, len(lengths)),
        "too_long_count": too_long,
        "too_long_ratio": ratio(too_long, len(lengths)),
        "configured_min_chars": min_chars,
        "configured_max_chars": max_chars,
    }


def compute_metadata_completeness(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [*CRITICAL_METADATA, *OPTIONAL_METADATA]
    missing: dict[str, dict[str, Any]] = {}
    for field_name in fields:
        count = sum(1 for chunk in chunks if _is_missing(_metadata(chunk).get(field_name)))
        missing[field_name] = {"missing_count": count, "missing_ratio": ratio(count, len(chunks))}
    return {
        "critical_fields": list(CRITICAL_METADATA),
        "optional_fields": list(OPTIONAL_METADATA),
        "missing": missing,
    }


def compute_json_validity(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    validity: dict[str, dict[str, Any]] = {}
    for field_name in JSON_FIELDS:
        invalid = 0
        present = 0
        for chunk in chunks:
            metadata = _metadata(chunk)
            if _is_missing(metadata.get(field_name)):
                continue
            present += 1
            if parse_json_list(metadata.get(field_name)) is None:
                invalid += 1
        validity[field_name] = {
            "present_count": present,
            "invalid_count": invalid,
            "invalid_ratio": ratio(invalid, present),
        }
    return validity


def compute_image_consistency(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[ChunkIssue] = []
    for chunk in chunks:
        metadata = _metadata(chunk)
        images = parse_json_list(metadata.get("images_json")) or []
        image_count = _int(metadata.get("image_count"))
        has_images = _bool(metadata.get("has_images"))
        if has_images and image_count <= 0:
            issues.append(_issue(chunk, "has_images_without_count", "has_images=true but image_count <= 0"))
        if image_count > 0 and not images:
            issues.append(_issue(chunk, "image_count_without_images_json", "image_count > 0 but images_json is empty"))
        if images and image_count > 0 and abs(len(images) - image_count) > 1:
            issues.append(_issue(chunk, "image_count_mismatch", f"images_json has {len(images)} items but image_count={image_count}"))
    return {
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues[:50]],
    }


def compute_article_grouping(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    titles: dict[str, str] = {}
    for chunk in chunks:
        metadata = _metadata(chunk)
        article_id = str(metadata.get("article_id") or "")
        if not article_id:
            continue
        counts[article_id] += 1
        titles.setdefault(article_id, str(metadata.get("title") or ""))
    values = list(counts.values())
    if not values:
        return {
            "min_chunks_per_article": 0,
            "max_chunks_per_article": 0,
            "avg_chunks_per_article": 0,
            "median_chunks_per_article": 0,
            "articles_with_one_chunk": 0,
            "articles_with_many_chunks": 0,
            "top_articles_by_chunk_count": [],
        }
    return {
        "min_chunks_per_article": min(values),
        "max_chunks_per_article": max(values),
        "avg_chunks_per_article": round(sum(values) / len(values), 2),
        "median_chunks_per_article": statistics.median(values),
        "articles_with_one_chunk": sum(1 for value in values if value == 1),
        "articles_with_many_chunks": sum(1 for value in values if value >= 10),
        "top_articles_by_chunk_count": [
            {"article_id": article_id, "title": titles.get(article_id, ""), "chunk_count": count}
            for article_id, count in counts.most_common(10)
        ],
    }


def compute_duplicate_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    exact: Counter[str] = Counter()
    same_article: Counter[tuple[str, str]] = Counter()
    for chunk in chunks:
        text = _normalized_text(str(chunk.get("text") or ""))
        if not text:
            continue
        exact[text] += 1
        article_id = str(_metadata(chunk).get("article_id") or "")
        if article_id:
            same_article[(article_id, text)] += 1
    exact_duplicate_count = sum(count - 1 for count in exact.values() if count > 1)
    same_article_duplicate_count = sum(count - 1 for count in same_article.values() if count > 1)
    return {
        "exact_duplicate_chunks": exact_duplicate_count,
        "same_article_duplicate_chunks": same_article_duplicate_count,
        "duplicate_ratio": ratio(exact_duplicate_count, len(chunks)),
    }


def compute_boundary_warnings(chunks: list[dict[str, Any]], min_chars: int) -> dict[str, Any]:
    cut_chunks = 0
    weak_pronoun_starts = 0
    no_sentence_end = 0
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        lowered = _simple_ascii(text).lower()
        if not text:
            continue
        if not text.endswith((".", "!", "?", ":", ";", '"', "'")):
            no_sentence_end += 1
        if any(lowered.endswith(f" {ending}") or lowered == ending for ending in WEAK_ENDINGS):
            cut_chunks += 1
        if len(text) < max(min_chars * 2, 220) and any(lowered.startswith(f"{start} ") for start in WEAK_STARTS):
            weak_pronoun_starts += 1
    return {
        "possibly_cut_chunks": cut_chunks,
        "starts_with_weak_pronoun": weak_pronoun_starts,
        "missing_sentence_end": no_sentence_end,
    }


def build_samples(chunks: list[dict[str, Any]], sample_size: int) -> list[ChunkSample]:
    samples: list[ChunkSample] = []
    for chunk in chunks[: max(0, sample_size)]:
        metadata = _metadata(chunk)
        text = str(chunk.get("text") or "")
        samples.append(
            ChunkSample(
                chunk_id=str(chunk.get("chunk_id") or ""),
                article_id=str(metadata.get("article_id") or ""),
                source=str(metadata.get("source") or ""),
                title=str(metadata.get("title") or ""),
                topic=str(metadata.get("primary_topic") or metadata.get("topic_category") or ""),
                published_at=str(metadata.get("published_at") or ""),
                length=len(text),
                has_images=_bool(metadata.get("has_images")),
                text_preview=_clip(" ".join(text.split()), 300),
            )
        )
    return samples


def build_status(
    length: dict[str, Any],
    metadata: dict[str, Any],
    duplicates: dict[str, Any],
    overview: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(overview.get("total_chunks") or 0) <= 0:
        return "FAIL", ["collection is empty"]
    total = int(overview.get("total_chunks") or 0)
    critical_missing = {
        field: values
        for field, values in metadata.get("missing", {}).items()
        if field in CRITICAL_METADATA and values.get("missing_ratio", 0) >= 0.2
    }
    if critical_missing:
        return "FAIL", [f"critical metadata missing too often: {', '.join(sorted(critical_missing))}"]
    if float(length.get("too_short_ratio") or 0) >= 0.1:
        reasons.append("too_short_ratio >= 10%")
    if float(length.get("too_long_ratio") or 0) >= 0.05:
        reasons.append("too_long_ratio >= 5%")
    if float(duplicates.get("duplicate_ratio") or 0) >= 0.05:
        reasons.append("duplicate_ratio >= 5%")
    optional_missing_count = sum(
        1
        for field, values in metadata.get("missing", {}).items()
        if field in OPTIONAL_METADATA and total > 0 and values.get("missing_ratio", 0) >= 0.5
    )
    if optional_missing_count:
        reasons.append(f"{optional_missing_count} optional metadata fields missing in >= 50% chunks")
    return ("WARN", reasons) if reasons else ("OK", [])


def save_report(report: dict[str, Any], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_json_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def print_text_report(report: dict[str, Any]) -> None:
    overview = report.get("overview", {})
    length = report.get("length", {})
    metadata = report.get("metadata", {})
    grouping = report.get("article_grouping", {})
    duplicates = report.get("duplicates", {})
    boundary = report.get("boundary_warnings", {})
    json_validity = report.get("json_validity", {})
    print("========================================")
    print("CHUNK QUALITY AUDIT")
    print("========================================")
    print(f"Chroma path: {overview.get('chroma_path') or report.get('chroma_path') or ''}")
    print(f"Collection: {overview.get('collection_name') or report.get('collection_name') or ''}")
    print("")
    print("Overview:")
    print(f"- Total chunks: {overview.get('total_chunks', 0)}")
    print(f"- Unique articles: {overview.get('unique_articles', 0)}")
    print(f"- Unique sources: {overview.get('unique_sources', 0)}")
    print(f"- Unique topics: {overview.get('unique_topics', 0)}")
    print("")
    print("Length:")
    print(f"- Avg chars: {length.get('avg_chars', 0)}")
    print(f"- Median chars: {length.get('median_chars', 0)}")
    print(f"- P90 chars: {length.get('p90_chars', 0)}")
    print(f"- Too short (<{length.get('configured_min_chars', 0)}): {length.get('too_short_count', 0)} ({_pct(length.get('too_short_ratio', 0))})")
    print(f"- Too long (>{length.get('configured_max_chars', 0)}): {length.get('too_long_count', 0)} ({_pct(length.get('too_long_ratio', 0))})")
    print("")
    print("Metadata:")
    missing = metadata.get("missing", {})
    for field_name in ("article_id", "source", "primary_topic", "entities_json", "images_json"):
        values = missing.get(field_name, {})
        print(f"- {field_name} missing: {values.get('missing_count', 0)}")
    for field_name in ("entities_json", "images_json", "secondary_topics_json"):
        values = json_validity.get(field_name, {})
        print(f"- {field_name} invalid: {values.get('invalid_count', 0)}")
    print("")
    print("Article grouping:")
    print(f"- Avg chunks/article: {grouping.get('avg_chunks_per_article', 0)}")
    print(f"- Max chunks/article: {grouping.get('max_chunks_per_article', 0)}")
    print(f"- Articles with one chunk: {grouping.get('articles_with_one_chunk', 0)}")
    print("")
    print("Duplicates:")
    print(f"- Exact duplicate chunks: {duplicates.get('exact_duplicate_chunks', 0)}")
    print(f"- Duplicate ratio: {_pct(duplicates.get('duplicate_ratio', 0))}")
    print("")
    print("Boundary warnings:")
    print(f"- Possibly cut chunks: {boundary.get('possibly_cut_chunks', 0)}")
    print(f"- Starts with weak pronoun: {boundary.get('starts_with_weak_pronoun', 0)}")
    if report.get("samples"):
        print("")
        print("Samples:")
        for index, sample in enumerate(report["samples"], start=1):
            print(f"{index}. [{sample.get('source') or ''}] {sample.get('title') or sample.get('article_id') or ''}")
            print(f"   topic={sample.get('topic') or ''} length={sample.get('length', 0)} has_images={sample.get('has_images')}")
            print(f"   {sample.get('text_preview') or ''}")
    print("")
    print(f"Status: {report.get('status', 'FAIL')}")
    for reason in report.get("status_reasons") or []:
        print(f"- {reason}")
    print("========================================")


def percentile(values: list[int], p: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[index]


def ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part / total, 4)


def parse_json_list(value: Any) -> list[Any] | None:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _empty_length_stats() -> dict[str, Any]:
    return {
        "min_chars": 0,
        "max_chars": 0,
        "avg_chars": 0,
        "median_chars": 0,
        "p90_chars": 0,
        "too_short_count": 0,
        "too_short_ratio": 0,
        "too_long_count": 0,
        "too_long_ratio": 0,
        "configured_min_chars": 0,
        "configured_max_chars": 0,
    }


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _issue(chunk: dict[str, Any], issue: str, detail: str) -> ChunkIssue:
    metadata = _metadata(chunk)
    return ChunkIssue(
        chunk_id=str(chunk.get("chunk_id") or ""),
        article_id=str(metadata.get("article_id") or ""),
        issue=issue,
        detail=detail,
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _simple_ascii(value: str) -> str:
    replacements = {"à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a", "â": "a", "ă": "a", "è": "e", "é": "e", "ê": "e", "ì": "i", "í": "i", "ò": "o", "ó": "o", "ô": "o", "ơ": "o", "ù": "u", "ú": "u", "ư": "u", "ỳ": "y", "ý": "y", "đ": "d"}
    output = value
    for src, dst in replacements.items():
        output = output.replace(src, dst).replace(src.upper(), dst.upper())
    return output


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


if __name__ == "__main__":
    main()
