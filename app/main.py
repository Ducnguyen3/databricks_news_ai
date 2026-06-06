from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import load_settings
from app.local_ai.pipeline import create_embedding_model, create_rag_service, create_vector_store
from app.local_ai.prompt_builder import NO_INFO_FALLBACK
from app.utils.logging import configure_logging

load_dotenv(override=True)
configure_logging()

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str | None = None
    question: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    filters: dict[str, Any] | None = None
    current_context: dict[str, Any] | None = None
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False


app = FastAPI(title="Databricks News AI RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def log_runtime_config() -> None:
    settings = load_settings().local_ai
    logger.info("Using Chroma path: %s", settings.chroma_persist_dir)
    logger.info("Using embedding model: %s", settings.embedding_model_name)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = load_settings().local_ai
    return {
        "status": "ok",
        "chroma_path": settings.chroma_persist_dir,
        "embedding_model": settings.embedding_model_name,
        "collection": settings.chroma_collection_name,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = (payload.message or payload.question or "").strip()
    if not question:
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
            "images": [],
            "debug": _runtime_debug(),
        }

    filters = payload.filters or {}
    top_k = payload.top_k or filters.get("top_k")
    service = _rag_service()
    response = service.answer_structured(
        question=question,
        top_k=int(top_k) if top_k else None,
        debug_retrieval=payload.debug,
        debug_prompt=payload.debug,
        current_context=payload.current_context,
        filters=filters,
    )
    response = _retry_relaxed_latest_query(
        service=service,
        question=question,
        response=response,
        top_k=int(top_k) if top_k else None,
        debug=payload.debug,
    )
    return _normalize_chat_response(response)


@lru_cache(maxsize=1)
def _rag_service():
    settings = load_settings().local_ai
    embedding_model = create_embedding_model(settings)
    vector_store = create_vector_store(settings)
    return create_rag_service(settings, embedding_model, vector_store)


def _runtime_debug() -> dict[str, Any]:
    settings = load_settings().local_ai
    return {
        "chroma_path": settings.chroma_persist_dir,
        "embedding_model": settings.embedding_model_name,
        "collection": settings.chroma_collection_name,
    }


def _normalize_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(response)
    query_plan = normalized.get("query_plan")
    if isinstance(query_plan, dict):
        normalized.setdefault("domain", query_plan.get("domain"))
        normalized.setdefault("ticker", query_plan.get("ticker"))
    normalized.setdefault("sources", [])
    normalized.setdefault("images", [])
    normalized.setdefault("related_articles", [])

    normalized["images"] = [_normalize_image(image) for image in normalized.get("images", []) if isinstance(image, dict)]

    debug = normalized.get("debug")
    if isinstance(debug, dict):
        merged_debug = dict(debug)
    else:
        merged_debug = {}
    merged_debug.update(_runtime_debug())
    normalized["debug"] = merged_debug
    return normalized


def _retry_relaxed_latest_query(
    service: Any,
    question: str,
    response: dict[str, Any],
    top_k: int | None,
    debug: bool,
) -> dict[str, Any]:
    if response.get("answer") != NO_INFO_FALLBACK or response.get("sources"):
        return response
    relaxed_question = _relaxed_latest_question(question)
    if relaxed_question == question:
        return response
    retry_response = service.answer_structured(
        question=relaxed_question,
        top_k=max(8, int(top_k or 0)),
        debug_retrieval=debug,
        debug_prompt=debug,
        current_context=None,
    )
    if retry_response.get("answer") == NO_INFO_FALLBACK or not retry_response.get("sources"):
        return response
    retry_response = dict(retry_response)
    debug_payload = retry_response.get("debug") if isinstance(retry_response.get("debug"), dict) else {}
    retry_response["debug"] = {
        **debug_payload,
        "relaxed_retry": True,
        "original_question": question,
        "relaxed_question": relaxed_question,
    }
    return retry_response


def _relaxed_latest_question(question: str) -> str:
    replacements = (
        ("mới nhất hôm nay", "đáng chú ý gần đây"),
        ("moi nhat hom nay", "dang chu y gan day"),
        ("hôm nay", ""),
        ("hom nay", ""),
        ("mới nhất", "gần đây"),
        ("moi nhat", "gan day"),
    )
    relaxed = question.strip()
    lowered = relaxed.casefold()
    changed = False
    for old, new in replacements:
        if old in lowered:
            relaxed = _replace_casefold(relaxed, old, new)
            lowered = relaxed.casefold()
            changed = True
    if not changed:
        return question
    relaxed = " ".join(relaxed.split())
    if len(relaxed.split()) <= 3:
        relaxed = f"{relaxed} có tin gì đáng chú ý"
    return relaxed


def _replace_casefold(text: str, old: str, new: str) -> str:
    index = text.casefold().find(old)
    if index < 0:
        return text
    return f"{text[:index]}{new}{text[index + len(old):]}"


def _normalize_image(image: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(image)
    url = (
        normalized.get("url")
        or normalized.get("image_url")
        or normalized.get("src")
        or normalized.get("thumbnail_url")
        or normalized.get("thumbnailUrl")
        or normalized.get("thumb_url")
        or normalized.get("thumb")
        or normalized.get("thumbnail")
    )
    if url:
        normalized["url"] = url
        normalized["image_url"] = url
    return normalized
