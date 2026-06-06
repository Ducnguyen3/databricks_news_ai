from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.pipeline import create_rag_service
from app.local_ai.query_router import route_query
from app.local_ai.vector_store import ChromaVectorStore


DEFAULT_QUERIES = (
    "tong hop tin tuc tai chinh",
    "tong hop tin tuc chung khoan",
    "co phieu nao dang tang",
    "co phieu nao tang tran",
    "BSR co con du dia tang truong khong",
    "tin AI moi nhat",
    "tinh hinh the gioi hom nay",
    "tin bat dong san moi nhat",
    "tin giao duc moi nhat",
)


def main() -> None:
    configure_stdout()
    load_dotenv()
    args = parse_args()
    os.environ["RAG_DEBUG"] = "true"
    settings = load_settings().local_ai
    queries = list(DEFAULT_QUERIES if args.all else args.queries)
    if not queries:
        raise SystemExit('Usage: python scripts/debug_rag_query.py "query"')
    service = None
    service_error = ""
    try:
        service = build_service(settings, no_llm=args.no_llm)
    except Exception as exc:
        service_error = f"{type(exc).__name__}: {exc}"

    for index, query in enumerate(queries, start=1):
        if index > 1:
            print("\n" + "=" * 100 + "\n")
        run_query(service, query, top_k=args.top_k, service_error=service_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a layered RAG diagnostic trace for one or more queries.")
    parser.add_argument("queries", nargs="*", help="Query text. Use --all to run the built-in diagnostic set.")
    parser.add_argument("--all", action="store_true", help="Run the known failing/regression query set.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama and use extractive fallback after retrieval.")
    return parser.parse_args()


def build_service(settings: Any, no_llm: bool) -> Any:
    embedding_model = LocalEmbeddingModel(settings.embedding_model_name)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    if no_llm:
        return create_rag_service_without_llm(settings, embedding_model, vector_store)
    return create_rag_service(settings, embedding_model, vector_store)


def create_rag_service_without_llm(settings: Any, embedding_model: Any, vector_store: Any) -> Any:
    from app.local_ai.rag_service import RAGService

    return RAGService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        ollama_client=None,
        settings=settings,
    )


def run_query(service: Any, query: str, top_k: int | None, service_error: str = "") -> None:
    print_header(f"QUERY: {query}")
    router = route_query(query)
    print_json("Router result", compact_router(router))
    if service is None:
        print_json(
            "Diagnostic stopped",
            {
                "layer": "vector_store_init",
                "error": service_error,
                "meaning": "Chroma could not be opened, so retriever/reranker/context/prompt/generation did not run.",
            },
        )
        return

    try:
        response = service.answer_structured(
            query,
            top_k=top_k,
            debug_retrieval=True,
            debug_prompt=True,
        )
    except Exception as exc:
        print_json(
            "Diagnostic stopped",
            {
                "layer": "answer_structured",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return
    trace = ((response.get("debug") or {}).get("rag_trace") if isinstance(response, dict) else {}) or {}
    print_json("Topic scores", trace.get("router", {}).get("topic_scores") or router.get("topic_scores") or {})
    print_json(
        "Routing summary",
        {
            "answer_mode": (trace.get("router") or {}).get("answer_mode") or router.get("answer_mode"),
            "explicit_topic": (trace.get("router") or {}).get("explicit_topic") if trace.get("router") else router.get("explicit_topic"),
            "topic_confidence": (trace.get("router") or {}).get("topic_confidence") or router.get("topic_confidence"),
            "topic_confidence_label": (trace.get("router") or {}).get("topic_confidence_label") or router.get("topic_confidence_label"),
            "standalone_query": (trace.get("router") or {}).get("standalone_query") or router.get("standalone_query"),
        },
    )
    retrieval = trace.get("retrieval") or {}
    print_json("Retrieval debug", retrieval)
    print_json("Top retrieved candidates before filter", retrieval.get("top_candidates_before_filter") or [])
    print_json("Top candidates after topic filter", retrieval.get("top_candidates_after_topic_filter") or [])
    print_json("Top reranked candidates", trace.get("rerank") or [])
    print_json("Final context titles", (trace.get("context_builder") or {}).get("titles_in_context") or [])
    print_json("Generation/fallback", trace.get("generation") or {})
    print_header("Final answer")
    print(response.get("answer") if isinstance(response, dict) else response)


def compact_router(router: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": router.get("intent"),
        "answer_mode": router.get("answer_mode"),
        "selected_topic": router.get("primary_topic"),
        "explicit_topic": router.get("explicit_topic"),
        "topic_confidence": router.get("topic_confidence"),
        "topic_confidence_label": router.get("topic_confidence_label"),
        "matched_keywords": router.get("matched_keywords"),
        "normalized_query": router.get("normalized_query"),
    }


def print_header(title: str) -> None:
    print(f"\n## {title}")


def print_json(title: str, value: Any) -> None:
    print_header(title)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
