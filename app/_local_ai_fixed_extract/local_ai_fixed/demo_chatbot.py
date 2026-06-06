from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from app.config import load_settings
from app.local_ai.pipeline import create_embedding_model, create_rag_service, create_vector_store, index_articles
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    args = _parse_args(settings)
    configure_logging(args.log_level)

    embedding_model = create_embedding_model(settings.local_ai)
    vector_store = create_vector_store(settings.local_ai)

    if args.reset_index:
        vector_store.reset_collection()

    if args.index:
        result = index_articles(
            settings=settings.local_ai,
            embedding_model=embedding_model,
            vector_store=vector_store,
            limit=args.limit,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Loaded articles: {result.articles_loaded}")
        print(f"Created chunks: {result.chunks_created}")
        print(f"Upserted chunks: {result.chunks_upserted}")

    if args.question or args.summarize_url or args.summarize_title:
        if vector_store.count() == 0:
            raise RuntimeError("Chroma index is empty. Run with --index first or point to an existing local index.")

        rag_service = create_rag_service(settings.local_ai, embedding_model, vector_store)

        if args.summarize_url:
            result = rag_service.summarize_url(
                args.summarize_url,
                debug_retrieval=args.debug_retrieval,
                debug_prompt=args.debug_prompt,
            )
        elif args.summarize_title:
            result = rag_service.summarize_title(
                args.summarize_title,
                debug_retrieval=args.debug_retrieval,
                debug_prompt=args.debug_prompt,
            )
        else:
            result = rag_service.answer(
                args.question,
                top_k=args.top_k,
                debug_retrieval=args.debug_retrieval,
                debug_prompt=args.debug_prompt,
            )

        _print_answer(
            result,
            debug_retrieval=args.debug_retrieval,
            debug_prompt=args.debug_prompt,
            debug_prompt_max_chars=settings.local_ai.prompt_debug_max_chars,
        )

    if not args.index and not args.question and not args.summarize_url and not args.summarize_title:
        raise RuntimeError("Provide at least one action: --index, --question, --summarize_url, or --summarize_title.")


def _print_answer(
    result: dict[str, object],
    debug_retrieval: bool,
    debug_prompt: bool,
    debug_prompt_max_chars: int,
) -> None:
    print("\nIntent:")
    print(str(result.get("intent") or ""))
    print("\nQuestion:")
    print(str(result.get("question") or ""))
    print("\nAnswer:")
    print(str(result.get("answer") or ""))

    sources = result.get("sources", [])
    if isinstance(sources, list) and sources:
        print("\nSources:")
        for source in sources:
            if not isinstance(source, dict):
                continue
            print(f"- {source.get('title') or ''} | {source.get('source') or ''} | {source.get('url') or ''}")

    if debug_prompt:
        prompt_text = str(result.get("prompt_debug") or "")
        print("\nPrompt Debug:")
        if not prompt_text:
            print("- No prompt captured.")
        else:
            print(_truncate_debug_prompt(prompt_text, debug_prompt_max_chars))

    if not debug_retrieval:
        return

    debug_items = result.get("retrieval_debug", [])
    if not isinstance(debug_items, list) or not debug_items:
        print("\nRetrieval Debug:")
        print("- No retrieval debug data.")
        return

    print("\nRetrieval Debug:")
    for item in debug_items:
        if not isinstance(item, dict):
            continue
        print(f"- chunk_id: {item.get('chunk_id') or ''}")
        print(f"  article_id: {item.get('article_id') or ''}")
        print(f"  title: {item.get('title') or ''}")
        print(f"  source: {item.get('source') or ''}")
        print(f"  category: {item.get('category') or ''}")
        print(f"  vector_score: {item.get('vector_score') or ''}")
        print(f"  final_score: {item.get('final_score') or ''}")
        print(f"  url: {item.get('url') or ''}")
        print(f"  preview text: {item.get('preview_text') or ''}")


def _parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RAG chatbot over Databricks articles_clean.")
    parser.add_argument("--index", action="store_true", help="Load articles from Databricks and upsert chunks into Chroma.")
    parser.add_argument("--reset_index", action="store_true", help="Delete the existing Chroma collection before indexing.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of articles loaded from Databricks.")
    parser.add_argument("--question", default=None, help="Question to ask the chatbot.")
    parser.add_argument("--top_k", type=int, default=settings.local_ai.rag_top_k, help="Number of chunks to retrieve.")
    parser.add_argument("--debug_retrieval", action="store_true", help="Print retrieval debug information.")
    parser.add_argument("--debug_prompt", action="store_true", help="Print the final prompt before sending it to Ollama.")
    parser.add_argument("--summarize_url", default=None, help="Summarize an article by exact URL.")
    parser.add_argument("--summarize_title", default=None, help="Summarize the best matching article by title or keyword.")
    parser.add_argument("--chunk_size", type=int, default=settings.local_ai.chunk_size, help="Chunk size in characters.")
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=settings.local_ai.chunk_overlap,
        help="Chunk overlap in characters.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _truncate_debug_prompt(prompt_text: str, max_chars: int) -> str:
    if len(prompt_text) <= max_chars:
        return prompt_text
    clipped = prompt_text[:max_chars].rstrip()
    return f"{clipped}\n\n...[prompt da duoc cat bot]..."


if __name__ == "__main__":
    main()
