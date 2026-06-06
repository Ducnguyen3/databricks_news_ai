from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import load_settings
from app.local_ai.answer_validator import validate_answer_against_topic
from app.local_ai.pipeline import create_embedding_model, create_rag_service, create_vector_store
from app.local_ai.query_router import route_query
from app.local_ai.retriever import MetadataFilteringRetriever
from app.local_ai.topic_guard import filter_context_for_topic


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Debug topic contamination guard for one RAG query.")
    parser.add_argument("--query", required=True, help="User query to inspect.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="Also call answer_structured and validate final answer.")
    args = parser.parse_args()

    settings = load_settings().local_ai
    embedding_model = create_embedding_model(settings)
    vector_store = create_vector_store(settings)
    route = route_query(args.query)
    route["debug_scores"] = True

    retriever = MetadataFilteringRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        retrieval_mode=settings.rag_retrieval_mode,
    )
    results = retriever.retrieve(
        query=str(route.get("standalone_query") or args.query),
        query_plan=route,
        top_n=max(settings.rag_broad_retrieve_top_n, args.top_k * 5),
        top_k=max(args.top_k * 3, args.top_k),
    )
    guard = filter_context_for_topic(args.query, str(route.get("primary_topic") or ""), results)

    print_section("ROUTE")
    print_json(
        {
            "intent": route.get("intent"),
            "answer_mode": route.get("answer_mode"),
            "topic": route.get("primary_topic"),
            "topic_confidence": route.get("topic_confidence"),
            "explicit_topic": route.get("explicit_topic"),
            "matched_keywords": route.get("matched_keywords"),
        }
    )

    print_section("RETRIEVAL DEBUG")
    print_json(route.get("_retrieval_debug") or {})

    print_section("RETRIEVED CHUNKS")
    print_json([chunk_summary(item) for item in results])

    print_section("DROPPED CHUNKS")
    print_json([chunk_summary(item) | {"drop_reason": item.get("topic_guard_drop_reason")} for item in guard.dropped])

    print_section("FINAL CHUNKS AFTER GUARD")
    print_json([chunk_summary(item) for item in guard.kept])

    print_section("GUARD RESULT")
    print_json(
        {
            "allowed": guard.result.allowed,
            "reason": guard.result.reason,
            "score": guard.result.score,
            "violations": guard.result.violations,
            "retrieved_count": guard.retrieved_count,
            "kept_after_topic_filter": guard.kept_after_topic_filter,
            "dropped_wrong_topic": guard.dropped_wrong_topic,
            "dropped_deny_keyword": guard.dropped_deny_keyword,
        }
    )

    if args.generate:
        service = create_rag_service(settings, embedding_model, vector_store)
        response = service.answer_structured(args.query, top_k=args.top_k, debug_retrieval=True)
        valid, violations = validate_answer_against_topic(
            answer=str(response.get("answer") or ""),
            query=args.query,
            topic=str(route.get("primary_topic") or ""),
            sources=list(response.get("sources") or []),
        )
        print_section("ANSWER VALIDATOR")
        print_json({"valid": valid, "violations": violations, "answer": response.get("answer")})


def chunk_summary(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "chunk_id": str(item.get("chunk_id") or ""),
        "article_id": str(metadata.get("article_id") or ""),
        "title": str(metadata.get("title") or ""),
        "source": str(metadata.get("source") or ""),
        "topic": str(metadata.get("primary_topic") or metadata.get("topic") or ""),
        "score": item.get("final_score") or item.get("score"),
        "text_preview": " ".join(str(item.get("text") or "").split())[:160],
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
