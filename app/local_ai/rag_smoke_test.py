from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Sequence

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.pipeline import create_embedding_model, create_rag_service
from app.local_ai.vector_store import ChromaVectorStore
from app.utils.logging import configure_logging

REQUIRED_RESPONSE_FIELDS = ("answer", "intent", "topic", "query_plan", "sources", "images", "related_articles")


@dataclass(frozen=True, slots=True)
class QuerySmokeResult:
    query: str
    status: str
    duration_seconds: float
    response: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.log_level)
    results = run_smoke_tests(args)
    if args.json:
        print_json_report(results)
    else:
        print_text_report(
            results,
            max_sources=args.max_sources,
            max_images=args.max_images,
            max_related=args.max_related,
        )
    if args.strict and any(result.status != "OK" for result in results):
        raise SystemExit(1)
    if any(result.status == "FAIL" for result in results):
        raise SystemExit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Smoke-test structured RAG responses against local Chroma.")
    parser.add_argument("--query", action="append", default=None, help="Query to test. Can be passed multiple times.")
    parser.add_argument("--max_sources", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=5)
    parser.add_argument("--max_related", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    parser.add_argument("--retrieval_only", action="store_true", help="Accepted for compatibility; uses structured RAG path.")
    parser.add_argument("--top_k", type=int, default=settings.local_ai.rag_top_k)
    parser.add_argument("--chroma_path", default=settings.local_ai.chroma_persist_dir)
    parser.add_argument("--collection_name", default=settings.local_ai.chroma_collection_name)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def get_default_queries() -> list[str]:
    return [
        "tin AI moi nhat",
        "tinh hinh the gioi hom nay",
        "HPG co gi moi",
        "tin bat dong san gan day",
        "cho toi cac bai co anh ve Ukraine",
        "tin tu CafeF ve chung khoan",
        "tong hop tin doanh nghiep moi nhat tu nhieu nguon",
    ]


def run_smoke_tests(args: argparse.Namespace) -> list[QuerySmokeResult]:
    settings = load_settings()
    embedding_model = create_embedding_model(settings.local_ai)
    vector_store = ChromaVectorStore(
        persist_directory=args.chroma_path,
        collection_name=args.collection_name,
    )
    service = create_rag_service(settings.local_ai, embedding_model, vector_store)
    queries = list(args.query or get_default_queries())
    return [run_query(service, query, top_k=args.top_k, strict=bool(args.strict)) for query in queries]


def run_query(service: Any, query: str, top_k: int, strict: bool = False) -> QuerySmokeResult:
    started_at = perf_counter()
    try:
        response = service.answer_structured(query, top_k=top_k)
    except Exception as exc:
        return QuerySmokeResult(
            query=query,
            status="FAIL",
            duration_seconds=round(perf_counter() - started_at, 4),
            errors=[str(exc)],
        )
    validation = validate_response(response, strict=strict)
    return QuerySmokeResult(
        query=query,
        status=str(validation["status"]),
        duration_seconds=round(perf_counter() - started_at, 4),
        response=response,
        warnings=list(validation.get("warnings") or []),
        errors=list(validation.get("errors") or []),
    )


def validate_response(response: Any, strict: bool = False) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"status": "FAIL", "warnings": [], "errors": ["response must be dict"]}
    missing = [field for field in REQUIRED_RESPONSE_FIELDS if field not in response]
    if missing:
        return {"status": "FAIL", "warnings": [], "errors": [f"missing fields: {', '.join(missing)}"]}
    for field_name in ("sources", "images", "related_articles"):
        if not isinstance(response.get(field_name), list):
            return {"status": "FAIL", "warnings": [], "errors": [f"{field_name} must be list"]}
    warnings: list[str] = []
    if not response.get("sources"):
        warnings.append("sources is empty")
    if strict and warnings:
        return {"status": "FAIL", "warnings": warnings, "errors": ["strict mode rejects warnings"]}
    return {"status": "WARN" if warnings else "OK", "warnings": warnings, "errors": []}


def print_text_report(
    results: list[QuerySmokeResult],
    max_sources: int = 5,
    max_images: int = 5,
    max_related: int = 5,
) -> None:
    started_at = perf_counter()
    print("========================================")
    print("RAG STRUCTURED SMOKE TEST")
    print("========================================")
    print("")
    for result in results:
        response = result.response
        print(f"Query: {result.query}")
        print(f"Intent: {response.get('intent') or ''}")
        print(f"Topic: {response.get('topic') or ''}")
        print(f"Sources: {len(response.get('sources') or [])}")
        print(f"Images: {len(response.get('images') or [])}")
        print(f"Related articles: {len(response.get('related_articles') or [])}")
        print(f"Duration: {result.duration_seconds}s")
        print(f"Status: {result.status}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
        for error in result.errors:
            print(f"Error: {error}")
        _print_sources(response.get("sources") or [], max_sources=max_sources)
        _print_images(response.get("images") or [], max_images=max_images)
        _print_related(response.get("related_articles") or [], max_related=max_related)
        answer = str(response.get("answer") or "")
        if answer:
            print("")
            print("Answer preview:")
            print(_clip(answer, 500))
        print("")
        print("----------------------------------------")
        print("")
    passed = sum(1 for result in results if result.status == "OK")
    failed = sum(1 for result in results if result.status == "FAIL")
    warned = sum(1 for result in results if result.status == "WARN")
    print("Summary:")
    print(f"Total queries: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Warned: {warned}")
    print(f"Failed: {failed}")
    print(f"Duration: {round(perf_counter() - started_at, 4)}s")
    print(f"Status: {'FAIL' if failed else 'WARN' if warned else 'OK'}")
    print("========================================")


def print_json_report(results: list[QuerySmokeResult]) -> None:
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def _print_sources(sources: list[Any], max_sources: int) -> None:
    if not sources:
        return
    print("")
    print("Sources:")
    for index, source in enumerate(sources[: max(0, max_sources)], start=1):
        if not isinstance(source, dict):
            continue
        print(f"{index}. [{source.get('source') or ''}] {source.get('title') or ''}")
        print(f"   URL: {source.get('url') or ''}")
        print(f"   Published at: {source.get('published_at') or ''}")


def _print_images(images: list[Any], max_images: int) -> None:
    if not images:
        return
    print("")
    print("Images:")
    for index, image in enumerate(images[: max(0, max_images)], start=1):
        if not isinstance(image, dict):
            continue
        print(f"{index}. {image.get('image_url') or ''}")
        print(f"   Article: {image.get('article_title') or image.get('article_id') or ''}")
        print(f"   Caption: {image.get('caption') or ''}")


def _print_related(related_articles: list[Any], max_related: int) -> None:
    if not related_articles:
        return
    print("")
    print("Related articles:")
    for index, article in enumerate(related_articles[: max(0, max_related)], start=1):
        if not isinstance(article, dict):
            continue
        print(f"{index}. {article.get('title') or article.get('article_id') or ''}")


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


if __name__ == "__main__":
    main()
