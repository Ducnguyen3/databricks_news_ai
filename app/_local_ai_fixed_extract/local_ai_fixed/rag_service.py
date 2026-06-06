from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import LocalAiSettings, load_settings
from app.local_ai.answer_validator import safe_fallback_for_topic, validate_answer_against_topic
from app.local_ai.chunker import infer_topic_category
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.media_retriever import MediaRetriever
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.parent_article_loader import ParentArticleLoader
from app.local_ai.prompt_builder import NO_INFO_FALLBACK, PromptBuilder, build_topic_rag_prompt
from app.local_ai.processors.default_processor import DefaultNewsProcessor
from app.local_ai.processors.registry import get_processor
from app.local_ai.query_router import domain_from_topic, route_query
from app.local_ai.reranker import SimpleReranker, normalize_text, normalized_terms, overlap_score, safe_float
from app.local_ai.retriever import MetadataFilteringRetriever, build_structured_sources
from app.local_ai.retrieved_article import (
    article_metadata_by_id,
    build_related_articles_from_retrieved_articles,
    build_retrieved_articles,
    build_sources_from_retrieved_articles,
    images_by_article_id,
)
from app.local_ai.topic_profiles import TopicProfile, get_topic_profile
from app.local_ai.topic_guard import validate_context_for_topic
from app.local_ai.vector_store import ChromaUnavailableError, ChromaVectorStore

logger = logging.getLogger(__name__)

_BROAD_KEYWORDS = (
    "nhung gi noi bat",
    "cac tin",
    "chu de",
    "top",
    "nhieu",
    "tong hop",
    "gan day",
    "xu huong",
    "dang chu y",
)
_ARTICLE_SUMMARY_KEYWORDS = (
    "tom tat bai",
    "tom tat bai bao",
    "tom tat tin",
    "tom tat noi dung",
    "bai nay noi gi",
    "bai bao nay noi gi",
    "bai co tieu de",
)
_BROAD_MIN_ARTICLES = 3
_BROAD_TARGET_ARTICLES = 8
_BROAD_MAX_ARTICLES = 10
_SUMMARY_NOT_FOUND = "Toi khong tim thay bai bao phu hop de tom tat trong du lieu hien co."
_NO_ANSWER = "Toi khong tim thay thong tin phu hop trong du lieu hien co."
_NO_BROAD_DATA = "Toi chua co du du lieu de tong hop cac chu de lien quan."
_NO_CATEGORY_DATA = "Toi chua co du lieu tin tuc cho chu de nay."
_CATEGORY_MAX_ARTICLES = 10
_CATEGORY_CHUNKS_PER_ARTICLE = 2


@dataclass(frozen=True, slots=True)
class ReferencedSourceResolution:
    source: dict[str, Any] | None = None
    needs_clarification: bool = False
    message: str = ""


class RAGService:
    def __init__(
        self,
        embedding_model: LocalEmbeddingModel,
        vector_store: ChromaVectorStore,
        ollama_client: OllamaClient | None = None,
        settings: LocalAiSettings | None = None,
        reranker: SimpleReranker | None = None,
    ) -> None:
        self._settings = settings or load_settings().local_ai
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._ollama_client = ollama_client
        self._reranker = reranker or SimpleReranker()
        self._prompt_builder = PromptBuilder(self._settings)
        self._last_generation_debug: dict[str, Any] = {}

    def answer_structured(
        self,
        question: str,
        top_k: int | None = None,
        debug_retrieval: bool = False,
        debug_prompt: bool = False,
        current_context: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        resolved_top_k = max(1, int(top_k or self._settings.rag_top_k))
        rag_debug = debug_retrieval or debug_prompt or _env_flag("RAG_DEBUG")
        followup_response = self._answer_follow_up(
            question=question,
            current_context=current_context,
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )
        if followup_response is not None:
            return followup_response

        detected_intent = detect_question_intent(question)
        query_plan = route_query(question)
        query_plan = _apply_frontend_filters(query_plan, filters or {}, latest_article_count=resolved_top_k)
        query_plan = _resolve_followup_route(question, query_plan, current_context)
        if query_plan.get("answer_mode") == "followup" and query_plan.get("needs_clarification"):
            return _clarification_response(
                question,
                str(
                    query_plan.get("clarification_message")
                    or "Mình chưa xác định được bạn đang hỏi tiếp về tin/bài nào. Bạn hãy nhắc lại chủ đề hoặc chọn một nguồn cụ thể."
                ),
                query_plan,
            )
        if rag_debug:
            query_plan["debug_scores"] = True
        if detected_intent == "article_summary":
            try:
                summary_response = self._summarize_from_question(
                    question,
                    debug_retrieval=debug_retrieval,
                    debug_prompt=debug_prompt,
                )
            except ChromaUnavailableError as exc:
                return _chroma_unavailable_response(query_plan, exc)
            return _structured_summary_response(summary_response, query_plan)

        topic_profile = get_topic_profile(query_plan.get("primary_topic"))
        retriever = MetadataFilteringRetriever(
            vector_store=self._vector_store,
            embedding_model=self._embedding_model,
            retrieval_mode=self._settings.rag_retrieval_mode,
        )
        try:
            retrieval_query = str(query_plan.get("standalone_query") or question)
            results = retriever.retrieve(
                query=retrieval_query,
                query_plan=query_plan,
                top_n=max(self._settings.rag_broad_retrieve_top_n, resolved_top_k * 5),
                top_k=max(resolved_top_k * 3, resolved_top_k),
            )
        except ChromaUnavailableError as exc:
            return _chroma_unavailable_response(query_plan, exc)
        if query_plan.get("prefer_latest"):
            results = _select_latest_article_results(results, resolved_top_k)
        retrieval_trace = dict(query_plan.get("_retrieval_debug") or {})
        rerank_trace = list(query_plan.get("_rerank_debug") or [])
        if not results:
            response = {
                "answer": NO_INFO_FALLBACK,
                "intent": query_plan["intent"],
                "topic": query_plan.get("primary_topic"),
                "query_plan": query_plan,
                "sources": [],
                "images": [],
                "related_articles": [],
            }
            if rag_debug:
                _attach_rag_trace(
                    response,
                    query_plan=query_plan,
                    question=question,
                    retrieval_trace=retrieval_trace,
                    rerank_trace=rerank_trace,
                    results=[],
                    sources=[],
                    prompt="",
                    generation_debug={"used_llm": False},
                    fallback_used=True,
                    no_answer_triggered=True,
                    no_answer_reason="no retrieval results",
                    settings=self._settings,
                )
            return response
        if not _has_sufficient_evidence(results):
            response = {
                "answer": NO_INFO_FALLBACK,
                "intent": query_plan["intent"],
                "topic": query_plan.get("primary_topic"),
                "query_plan": query_plan,
                "sources": [],
                "images": [],
                "related_articles": [],
            }
            if rag_debug:
                _attach_rag_trace(
                    response,
                    query_plan=query_plan,
                    question=question,
                    retrieval_trace=retrieval_trace,
                    rerank_trace=rerank_trace,
                    results=results,
                    sources=[],
                    prompt="",
                    generation_debug={"used_llm": False},
                    fallback_used=True,
                    no_answer_triggered=True,
                    no_answer_reason="retrieval scores below evidence threshold",
                    settings=self._settings,
                )
            return response
        topic_guard_result = validate_context_for_topic(question, str(query_plan.get("primary_topic") or ""), results)
        if not topic_guard_result.allowed:
            response = _guard_failed_response(question, query_plan, topic_guard_result.reason, topic_guard_result.violations)
            if rag_debug:
                _attach_rag_trace(
                    response,
                    query_plan=query_plan,
                    question=question,
                    retrieval_trace=retrieval_trace,
                    rerank_trace=rerank_trace,
                    results=results,
                    sources=[],
                    prompt="",
                    generation_debug={"used_llm": False, "answer_validator_reason": topic_guard_result.reason},
                    fallback_used=True,
                    no_answer_triggered=True,
                    no_answer_reason=topic_guard_result.reason,
                    settings=self._settings,
                )
            return response

        initial_sources = build_structured_sources(results)
        article_ids = [str(source.get("article_id") or "") for source in initial_sources]
        try:
            parent_articles = ParentArticleLoader(self._vector_store).load_parent_articles(
                article_ids=article_ids,
                chunks_per_article=self._settings.rag_max_chunks_per_article,
            )
        except ChromaUnavailableError as exc:
            return _chroma_unavailable_response(query_plan, exc)
        retrieved_articles = build_retrieved_articles(results, parent_articles)
        sources = build_sources_from_retrieved_articles(retrieved_articles) or initial_sources
        _copy_citation_ids_to_articles(retrieved_articles, sources)
        article_ids = [str(source.get("article_id") or "") for source in sources]
        image_limit = 2 if query_plan.get("need_images") or query_plan.get("intent") == "media_lookup" else 1
        images = MediaRetriever(
            metadata_images_by_article_id=images_by_article_id(retrieved_articles),
            article_metadata_by_id=article_metadata_by_id(retrieved_articles),
        ).get_images_for_articles(
            article_ids=article_ids,
            limit_per_article=image_limit,
        )
        prompt = self._build_structured_prompt(
            question=question,
            query_plan=query_plan,
            results=results,
            sources=sources,
            images=images,
            topic_profile=topic_profile,
            retrieved_articles=retrieved_articles,
        )
        llm_answer = self._generate_answer(prompt)
        answer_validator_reason = _invalid_generated_answer_reason(llm_answer, sources)
        if answer_validator_reason:
            self._last_generation_debug["answer_validator_reason"] = answer_validator_reason
            self._last_generation_debug["llm_answer_rejected"] = True
            llm_answer = None
        answer = llm_answer or extractive_answer(question, results)
        extractive_answer_used = llm_answer is None
        if (query_plan.get("need_images") or query_plan.get("intent") == "media_lookup") and not images:
            answer = f"{answer}\n\nHiện dữ liệu đã crawl chưa có ảnh phù hợp cho câu hỏi này."
        answer_valid, answer_violations = validate_answer_against_topic(
            answer=str(answer),
            query=question,
            topic=str(query_plan.get("primary_topic") or ""),
            sources=sources,
        )
        if not answer_valid:
            self._last_generation_debug["answer_validator_reason"] = ",".join(answer_violations)
            self._last_generation_debug["answer_replaced_by_safe_fallback"] = True
            answer = safe_fallback_for_topic(question, str(query_plan.get("primary_topic") or ""))
            extractive_answer_used = True
        citation_warnings = _citation_warnings(answer, sources)
        response: dict[str, object] = {
            "answer": answer,
            "intent": query_plan["intent"],
            "topic": query_plan.get("primary_topic"),
            "query_plan": {
                "entities": query_plan.get("entities", []),
                "stock_symbols": query_plan.get("stock_symbols", []),
                "primary_topic": query_plan.get("primary_topic"),
                "domain": query_plan.get("domain", "all"),
                "ticker": query_plan.get("ticker", ""),
                "time_range": query_plan.get("time_range", "all"),
                "need_images": bool(query_plan.get("need_images")),
                "need_sources": bool(query_plan.get("need_sources")),
                "answer_mode": query_plan.get("answer_mode", "synthesis"),
                "standalone_query": query_plan.get("standalone_query", ""),
                "explicit_topic": bool(query_plan.get("explicit_topic")),
                "topic_confidence": query_plan.get("topic_confidence"),
                "topic_confidence_label": query_plan.get("topic_confidence_label"),
                "data_source": query_plan.get("data_source", "article_rag"),
                "retrieval_mode": self._settings.rag_retrieval_mode,
                "requires_lexical": bool(query_plan.get("requires_lexical")),
                "lexical_terms": query_plan.get("lexical_terms", []),
                "preferred_sources": query_plan.get("preferred_sources", []),
                "latest_article_count": query_plan.get("latest_article_count"),
                "prefer_latest": bool(query_plan.get("prefer_latest")),
            },
            "sources": sources,
            "images": images,
            "related_articles": build_related_articles_from_retrieved_articles(retrieved_articles)
            or build_related_articles(parent_articles, sources),
        }
        if citation_warnings:
            response["debug"] = {"citation_warnings": citation_warnings}
        if debug_retrieval:
            response["retrieval_debug"] = build_debug_entries(results)
        if debug_prompt:
            response["prompt_debug"] = prompt
        if rag_debug:
            _attach_rag_trace(
                response,
                query_plan=query_plan,
                question=question,
                retrieval_trace=retrieval_trace,
                rerank_trace=rerank_trace,
                results=results,
                sources=sources,
                prompt=prompt,
                generation_debug={
                    **self._last_generation_debug,
                    "fallback_used": extractive_answer_used,
                    "extractive_answer_used": extractive_answer_used,
                    "no_answer_triggered": False,
                    "no_answer_reason": "",
                },
                fallback_used=extractive_answer_used,
                no_answer_triggered=False,
                no_answer_reason="",
                settings=self._settings,
                retrieved_articles=retrieved_articles,
            )
        return response

    def _answer_follow_up(
        self,
        question: str,
        current_context: dict[str, Any] | None,
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object] | None:
        if not current_context:
            return None
        followup_intent = detect_follow_up_intent(question)
        if followup_intent is None:
            return None

        query_plan = _followup_query_plan(followup_intent, current_context)

        if followup_intent == "followup_simplify":
            previous_answer = str(current_context.get("previous_answer") or "").strip()
            if not previous_answer:
                return _clarification_response(
                    question,
                    "Mình chưa có câu trả lời trước đó để rút gọn. Bạn hãy hỏi lại nội dung cần tóm tắt.",
                    query_plan,
                )
            prompt = (
                "Rut gon cau tra loi sau thanh 2-3 y ngan gon, chi giu thong tin da co.\n\n"
                f"CAU TRA LOI TRUOC:\n{previous_answer}\n\n"
                f"YEU CAU:\n{question}\n"
            )
            answer = self._generate_answer(prompt) or summarize_text_for_article(previous_answer, question, max_sentences=2)
            return {
                "answer": answer,
                "intent": followup_intent,
                "topic": _context_topic(current_context),
                "query_plan": query_plan,
                "sources": _context_sources(current_context),
                "images": [],
                "related_articles": build_related_articles([], _context_sources(current_context)),
                **({"prompt_debug": prompt} if debug_prompt else {}),
            }

        resolved = resolve_referenced_source(question, current_context)
        if resolved.needs_clarification:
            return _clarification_response(question, resolved.message, query_plan)
        if resolved.source is None:
            return None

        source = resolved.source
        if followup_intent == "followup_media_lookup":
            images = _context_images_for_source(current_context, source)
            answer = (
                f"Mình tìm thấy {len(images)} ảnh liên quan đến nguồn [{source.get('citation_id')}]."
                if images
                else f"Trong ngữ cảnh hiện tại chưa có ảnh cho nguồn [{source.get('citation_id')}]."
            )
            return {
                "answer": answer,
                "intent": followup_intent,
                "topic": str(source.get("topic") or source.get("primary_topic") or query_plan.get("primary_topic") or ""),
                "query_plan": query_plan,
                "sources": [_source_context_to_source(source)],
                "images": images,
                "related_articles": build_related_articles([], [_source_context_to_source(source)]),
            }

        try:
            summary = self._summarize_referenced_source(
                question=question,
                source=source,
                debug_retrieval=debug_retrieval,
                debug_prompt=debug_prompt,
            )
        except ChromaUnavailableError as exc:
            return _chroma_unavailable_response(query_plan, exc)
        return _structured_summary_response(summary, query_plan)

    def _summarize_referenced_source(
        self,
        question: str,
        source: dict[str, Any],
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object]:
        article_id = str(source.get("article_id") or "").strip()
        if article_id:
            chunks = self._vector_store.get_chunks_by_article_id(article_id, limit=self._settings.summary_max_chunks)
            if chunks:
                return self._summarize_article_chunks(
                    question,
                    chunks,
                    debug_retrieval=debug_retrieval,
                    debug_prompt=debug_prompt,
                )
        url = str(source.get("url") or "").strip()
        if url:
            return self._summarize_by_url(
                url,
                prompt_question=question,
                debug_retrieval=debug_retrieval,
                debug_prompt=debug_prompt,
            )
        title = str(source.get("title") or "").strip()
        return self._summarize_by_title(
            title or question,
            prompt_question=question,
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )

    def _build_structured_prompt(
        self,
        question: str,
        query_plan: dict[str, Any],
        results: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
        topic_profile: TopicProfile | None = None,
        retrieved_articles: list[dict[str, Any]] | None = None,
    ) -> str:
        profile = topic_profile or get_topic_profile(query_plan.get("primary_topic"))
        try:
            return build_topic_rag_prompt(
                question=question,
                context_blocks=retrieved_articles or results,
                topic_profile=profile,
                query_plan=query_plan,
                answer_mode=str(query_plan.get("answer_mode") or ""),
            )
        except Exception:
            logger.warning("Topic prompt builder failed; falling back to domain processor", exc_info=True)
        try:
            processor = get_processor(query_plan.get("primary_topic"))
            context = processor.build_context(
                query=question,
                route=query_plan,
                retrieved_chunks=results,
                sources=sources,
                images=images,
            )
            return processor.build_prompt(context)
        except Exception:
            logger.warning("Domain processor failed; falling back to default processor", exc_info=True)
        try:
            fallback_processor = DefaultNewsProcessor()
            context = fallback_processor.build_context(
                query=question,
                route=query_plan,
                retrieved_chunks=results,
                sources=sources,
                images=images,
            )
            return fallback_processor.build_prompt(context)
        except Exception:
            logger.warning("Default processor failed; falling back to legacy prompt", exc_info=True)
            return self._prompt_builder.build_qa_prompt(question, results)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        debug_retrieval: bool = False,
        debug_prompt: bool = False,
    ) -> dict[str, object]:
        resolved_top_k = max(1, int(top_k or self._settings.rag_top_k))
        intent = detect_question_intent(question)
        detected_category = detect_topic_category(question)
        if intent == "article_summary":
            result = self._summarize_from_question(question, debug_retrieval=debug_retrieval, debug_prompt=debug_prompt)
        elif intent == "category_summary" and detected_category:
            result = self._answer_category_summary(
                question,
                topic_category=detected_category,
                debug_retrieval=debug_retrieval,
                debug_prompt=debug_prompt,
            )
        elif intent == "broad_topic":
            result = self._answer_broad_topic(question, debug_retrieval=debug_retrieval, debug_prompt=debug_prompt)
        else:
            result = self._answer_specific_qa(
                question,
                top_k=resolved_top_k,
                debug_retrieval=debug_retrieval,
                debug_prompt=debug_prompt,
            )
        result["intent"] = intent
        if detected_category:
            result["detected_category"] = detected_category
        return result

    def summarize_url(
        self,
        url: str,
        debug_retrieval: bool = False,
        debug_prompt: bool = False,
    ) -> dict[str, object]:
        result = self._summarize_by_url(
            url,
            prompt_question=f"Tom tat bai bao nay: {url}",
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )
        result["intent"] = "article_summary"
        return result

    def summarize_title(
        self,
        title: str,
        debug_retrieval: bool = False,
        debug_prompt: bool = False,
    ) -> dict[str, object]:
        result = self._summarize_by_title(
            title,
            prompt_question=f"Tom tat bai co tieu de {title}",
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )
        result["intent"] = "article_summary"
        return result

    def _answer_specific_qa(
        self,
        question: str,
        top_k: int,
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object]:
        candidates = self._retrieve_candidates(question, top_n=self._settings.rag_retrieve_top_n)
        reranked = self._rerank(question, candidates)
        diverse = select_diverse_chunks(
            reranked,
            max_chunks_per_article=max(1, self._settings.rag_max_chunks_per_article),
            target_articles=max(top_k, _BROAD_TARGET_ARTICLES),
        )
        selected = diverse[:top_k]
        if not selected or is_weak_context(selected, self._settings.rag_min_score):
            prompt = self._prompt_builder.build_no_context_answer(question) if debug_prompt else None
            return self._result(question, _NO_ANSWER, [], selected if debug_retrieval else None, prompt=prompt)

        prompt = self._prompt_builder.build_qa_prompt(question, selected)
        answer = self._generate_answer(prompt) or extractive_answer(question, selected)
        return self._result(
            question,
            answer,
            build_sources(selected),
            selected if debug_retrieval else None,
            prompt=prompt if debug_prompt else None,
        )

    def _answer_broad_topic(self, question: str, debug_retrieval: bool, debug_prompt: bool) -> dict[str, object]:
        candidates = self._retrieve_candidates(question, top_n=self._settings.rag_broad_retrieve_top_n)
        reranked = self._rerank(question, candidates)
        diverse = select_diverse_chunks(
            reranked,
            max_chunks_per_article=max(2, self._settings.rag_max_chunks_per_article),
            target_articles=_BROAD_MAX_ARTICLES,
        )
        articles = aggregate_articles(
            diverse,
            question,
            max_articles=_BROAD_MAX_ARTICLES,
        )
        if len(articles) < _BROAD_MIN_ARTICLES:
            prompt = self._prompt_builder.build_no_context_answer(question) if debug_prompt else None
            return self._result(
                question,
                _NO_BROAD_DATA,
                build_article_sources(articles),
                diverse if debug_retrieval else None,
                prompt=prompt,
            )

        prompt = self._prompt_builder.build_broad_topic_prompt(question, articles)
        answer = self._generate_answer(prompt) or extractive_broad_answer(articles)
        return self._result(
            question,
            answer,
            build_article_sources(articles),
            diverse if debug_retrieval else None,
            prompt=prompt if debug_prompt else None,
        )

    def _answer_category_summary(
        self,
        question: str,
        topic_category: str,
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object]:
        category_chunks = self._vector_store.get_chunks_by_topic_category(topic_category, limit=500)
        if not category_chunks:
            prompt = self._prompt_builder.build_no_context_answer(question) if debug_prompt else None
            return self._result(question, _NO_CATEGORY_DATA, [], [] if debug_retrieval else None, prompt=prompt)

        articles = build_recent_articles_for_category(
            question,
            category_chunks,
            max_articles=_CATEGORY_MAX_ARTICLES,
            max_chunks_per_article=_CATEGORY_CHUNKS_PER_ARTICLE,
        )
        if not articles:
            prompt = self._prompt_builder.build_no_context_answer(question) if debug_prompt else None
            return self._result(
                question,
                _NO_CATEGORY_DATA,
                [],
                category_chunks if debug_retrieval else None,
                prompt=prompt,
            )

        prompt = self._prompt_builder.build_category_summary_prompt(question, articles)
        answer = self._generate_answer(prompt) or extractive_category_answer(articles)
        debug_items = flatten_article_chunks(articles) if debug_retrieval else None
        return self._result(
            question,
            answer,
            build_article_sources(articles),
            debug_items,
            prompt=prompt if debug_prompt else None,
        )

    def _summarize_from_question(self, question: str, debug_retrieval: bool, debug_prompt: bool) -> dict[str, object]:
        url = extract_url(question)
        if url:
            return self._summarize_by_url(
                url,
                prompt_question=question,
                debug_retrieval=debug_retrieval,
                debug_prompt=debug_prompt,
            )
        title_hint = extract_title_hint(question)
        return self._summarize_by_title(
            title_hint or question,
            prompt_question=question,
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )

    def _summarize_by_url(
        self,
        url: str,
        prompt_question: str,
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object]:
        chunks = self._vector_store.get_chunks_by_url(url, limit=self._settings.summary_max_chunks)
        if not chunks:
            prompt = self._prompt_builder.build_no_context_answer(prompt_question) if debug_prompt else None
            return self._result(prompt_question, _SUMMARY_NOT_FOUND, [], chunks if debug_retrieval else None, prompt=prompt)
        return self._summarize_article_chunks(
            prompt_question,
            chunks,
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
        )

    def _summarize_by_title(
        self,
        title: str,
        prompt_question: str,
        debug_retrieval: bool,
        debug_prompt: bool,
    ) -> dict[str, object]:
        candidates = self._retrieve_candidates(title, top_n=self._settings.rag_retrieve_top_n)
        reranked = self._rerank(title, candidates)
        best_article_id = resolve_best_article_id(reranked, self._settings.rag_min_score)
        if not best_article_id:
            prompt = self._prompt_builder.build_no_context_answer(prompt_question) if debug_prompt else None
            return self._result(prompt_question, _SUMMARY_NOT_FOUND, [], reranked if debug_retrieval else None, prompt=prompt)

        chunks = self._vector_store.get_chunks_by_article_id(best_article_id, limit=self._settings.summary_max_chunks)
        if not chunks:
            prompt = self._prompt_builder.build_no_context_answer(prompt_question) if debug_prompt else None
            return self._result(prompt_question, _SUMMARY_NOT_FOUND, [], reranked if debug_retrieval else None, prompt=prompt)
        return self._summarize_article_chunks(
            prompt_question,
            chunks,
            debug_retrieval=debug_retrieval,
            debug_prompt=debug_prompt,
            debug_candidates=reranked,
        )

    def _summarize_article_chunks(
        self,
        prompt_question: str,
        chunks: list[dict[str, object]],
        debug_retrieval: bool,
        debug_prompt: bool,
        debug_candidates: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        ordered_chunks = sorted(chunks, key=lambda item: chunk_index_of(item))
        if not ordered_chunks:
            prompt = self._prompt_builder.build_no_context_answer(prompt_question) if debug_prompt else None
            return self._result(prompt_question, _SUMMARY_NOT_FOUND, [], ordered_chunks if debug_retrieval else None, prompt=prompt)

        first_chunk = ordered_chunks[0]
        article = article_from_chunks(ordered_chunks)
        prompt = self._prompt_builder.build_article_summary_prompt(article, ordered_chunks)
        answer = self._generate_answer(prompt) or extractive_article_summary(ordered_chunks)
        sources = build_sources([first_chunk])
        debug_items = debug_candidates if debug_candidates is not None else ordered_chunks
        return self._result(
            prompt_question,
            answer,
            sources,
            debug_items if debug_retrieval else None,
            prompt=prompt if debug_prompt else None,
        )

    def _retrieve_candidates(self, query: str, top_n: int) -> list[dict[str, object]]:
        query_embedding = self._embedding_model.embed_query(query)
        raw_results = self._vector_store.search(query_embedding, top_k=top_n)
        return [item for item in raw_results if safe_float(item.get("score")) >= self._settings.rag_min_score]

    def _rerank(self, query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        if not candidates:
            return []
        return self._reranker.rerank(query, candidates, top_k=len(candidates))

    def _generate_answer(self, prompt: str) -> str | None:
        self._last_generation_debug = {
            "ollama_model": _ollama_model(self._ollama_client),
            "used_llm": False,
            "llm_error": "",
            "prompt_chars": len(prompt),
        }
        try:
            if self._ollama_client is None:
                self._last_generation_debug["llm_error"] = "ollama_client_not_configured"
                return None
            answer = self._ollama_client.generate(prompt)
            self._last_generation_debug["used_llm"] = True
            return answer
        except Exception as exc:
            self._last_generation_debug["llm_error"] = str(exc)
            logger.warning("Falling back to extractive answer because Ollama failed", exc_info=True)
            return None

    def _result(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, str]],
        debug_candidates: list[dict[str, object]] | None,
        prompt: str | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
        if debug_candidates is not None:
            result["retrieval_debug"] = build_debug_entries(debug_candidates)
        if prompt is not None:
            result["prompt_debug"] = prompt
        return result


RagService = RAGService


def detect_question_intent(question: str) -> str:
    normalized = normalize_text(question)
    if extract_url(question):
        return "article_summary"
    if any(keyword in normalized for keyword in _ARTICLE_SUMMARY_KEYWORDS):
        return "article_summary"
    if detect_topic_category(question):
        return "category_summary"
    if any(keyword in normalized for keyword in _BROAD_KEYWORDS):
        return "broad_topic"
    return "specific_qa"


def detect_follow_up_intent(question: str) -> str | None:
    normalized = normalize_text(question)
    if not normalized:
        return None
    has_reference = _has_context_reference(normalized) or _extract_citation_number(normalized) is not None
    if any(term in normalized for term in ("tom tat ngan hon", "noi ngan gon", "ngan gon hon", "rut gon", "viet ngan hon")):
        return "followup_simplify"
    if any(term in normalized for term in ("cho toi xem anh", "xem anh", "hinh anh", "anh cua")) and has_reference:
        return "followup_media_lookup"
    if any(term in normalized for term in ("tom tat", "noi gi", "noi dung gi", "bai nay", "tin nay", "nguon nay")) and has_reference:
        return "followup_article_summary"
    if _extract_citation_number(normalized) is not None:
        return "followup_citation_question"
    if any(term in normalized for term in ("giai thich them", "y tren", "doan tren")):
        return "followup_expand"
    return None


def resolve_referenced_source(question: str, current_context: dict[str, Any] | None) -> ReferencedSourceResolution:
    if not current_context:
        return ReferencedSourceResolution()
    sources = _context_sources(current_context)
    if not sources:
        return ReferencedSourceResolution()

    citation_id = _extract_citation_number(normalize_text(question))
    if citation_id is None:
        raw_citation = current_context.get("selected_citation_id")
        try:
            citation_id = int(raw_citation) if raw_citation is not None else None
        except (TypeError, ValueError):
            citation_id = None
    if citation_id is not None:
        for source in sources:
            try:
                if int(source.get("citation_id")) == citation_id:
                    return ReferencedSourceResolution(source=source)
            except (TypeError, ValueError):
                continue
        return ReferencedSourceResolution(
            needs_clarification=True,
            message=f"Mình không tìm thấy nguồn [{citation_id}] trong câu trả lời trước. Bạn hãy chọn lại nguồn có trong danh sách.",
        )

    selected_article_id = str(current_context.get("selected_article_id") or "").strip()
    selected_url = str(current_context.get("selected_url") or "").strip()
    if selected_article_id or selected_url:
        for source in sources:
            if selected_article_id and str(source.get("article_id") or "") == selected_article_id:
                return ReferencedSourceResolution(source=source)
            if selected_url and str(source.get("url") or "") == selected_url:
                return ReferencedSourceResolution(source=source)

    normalized = normalize_text(question)
    if _has_context_reference(normalized):
        if len(sources) == 1:
            return ReferencedSourceResolution(source=sources[0])
        choices = ", ".join(f"[{source.get('citation_id')}]" for source in sources if source.get("citation_id"))
        return ReferencedSourceResolution(
            needs_clarification=True,
            message=f"Bạn muốn mình xử lý bài nào? Hãy chọn nguồn {choices or '[1], [2] hoặc [3]'}."
        )
    return ReferencedSourceResolution()


def _apply_frontend_filters(
    query_plan: dict[str, Any],
    filters: dict[str, Any],
    latest_article_count: int,
) -> dict[str, Any]:
    if not filters:
        updated = dict(query_plan)
        updated["latest_article_count"] = max(1, int(latest_article_count))
        return updated

    updated = dict(query_plan)
    topic = _normalize_frontend_topic(str(filters.get("topic") or ""))
    if topic:
        updated["primary_topic"] = topic
        updated["domain"] = domain_from_topic(topic)

    sources = filters.get("sources")
    if isinstance(sources, list):
        preferred_sources = [str(source) for source in sources if str(source).strip()]
        if preferred_sources:
            updated["preferred_sources"] = preferred_sources
            updated["source"] = preferred_sources[0] if len(preferred_sources) == 1 else None

    ticker = str(filters.get("ticker") or "").strip().upper()
    if ticker:
        updated["ticker"] = ticker
        stock_symbols = list(updated.get("stock_symbols") or [])
        if ticker not in stock_symbols:
            stock_symbols.insert(0, ticker)
        updated["stock_symbols"] = stock_symbols
        updated["domain"] = "tai_chinh"
        updated["primary_topic"] = "economy_finance_stock"

    try:
        time_range_days = int(filters.get("time_range_days") or 0)
    except (TypeError, ValueError):
        time_range_days = 0
    if time_range_days > 0:
        updated["time_range_days"] = time_range_days
        updated["time_range"] = "all"
    else:
        updated["time_range_days"] = 0

    updated["latest_article_count"] = max(1, int(latest_article_count))
    updated["prefer_latest"] = True
    return updated


def _normalize_frontend_topic(topic: str) -> str:
    aliases = {
        "technology_ai_internet": "tech_ai_internet",
        "tech_ai_internet": "tech_ai_internet",
        "economy_finance_stock": "economy_finance_stock",
        "politics_society": "politics_society",
        "world_geopolitics": "world_geopolitics",
        "business_startup": "business_startup",
        "real_estate": "real_estate",
        "lifestyle_education_health_entertainment": "lifestyle_education_health_entertainment",
        "general_news": "general_news",
    }
    return aliases.get(topic.strip(), "")


def detect_topic_category(question: str) -> str | None:
    normalized = normalize_text(question)
    keyword_map: dict[str, tuple[str, ...]] = {
        "sports": ("the thao", "tin the thao", "bong da", "giai dau", "tran dau"),
        "tech": ("cong nghe", "tin cong nghe", "tri tue nhan tao", "chip", "ban dan"),
        "business": ("kinh te", "kinh doanh", "tai chinh", "chung khoan", "doanh nghiep"),
        "politics": ("chinh tri", "ngoai giao", "quoc hoi", "chinh phu"),
    }
    for category, keywords in keyword_map.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return None


def select_diverse_chunks(
    results: list[dict[str, object]],
    max_chunks_per_article: int,
    target_articles: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    per_article_count: dict[str, int] = defaultdict(int)
    article_ids: set[str] = set()

    for result in results:
        article_id = article_id_of(result)
        if not article_id:
            continue
        if per_article_count[article_id] >= max_chunks_per_article:
            continue
        if article_id not in article_ids and len(article_ids) >= target_articles:
            continue
        selected.append(result)
        per_article_count[article_id] += 1
        article_ids.add(article_id)

    return selected


def aggregate_articles(
    results: list[dict[str, object]],
    question: str,
    max_articles: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        article_id = article_id_of(result)
        if article_id:
            grouped[article_id].append(result)

    articles: list[dict[str, object]] = []
    for article_id, chunks in grouped.items():
        ordered = sorted(chunks, key=lambda item: safe_float(item.get("final_score")), reverse=True)
        best = ordered[0]
        metadata = best.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        merged_text = merge_chunk_texts(sorted(chunks, key=lambda item: chunk_index_of(item)))
        articles.append(
            {
                "article_id": article_id,
                "title": str(metadata.get("title") or ""),
                "source": str(metadata.get("source") or ""),
                "url": str(metadata.get("url") or ""),
                "category": str(metadata.get("category") or ""),
                "published_at": str(metadata.get("published_at") or ""),
                "score": safe_float(best.get("final_score")),
                "summary": summarize_text_for_article(merged_text, question, max_sentences=3),
            }
        )

    articles.sort(key=lambda item: safe_float(item.get("score")), reverse=True)
    return articles[:max_articles]


def build_recent_articles_for_category(
    question: str,
    chunks: list[dict[str, object]],
    max_articles: int,
    max_chunks_per_article: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk in chunks:
        article_id = article_id_of(chunk)
        if article_id:
            grouped[article_id].append(chunk)

    articles: list[dict[str, object]] = []
    for article_id, article_chunks in grouped.items():
        ordered_by_relevance = sort_chunks_for_article(question, article_chunks)
        selected_chunks = ordered_by_relevance[:max_chunks_per_article]
        best_chunk = ordered_by_relevance[0]
        metadata = best_chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        ordered_text_chunks = sorted(selected_chunks, key=lambda item: chunk_index_of(item))
        merged_text = merge_chunk_texts(ordered_text_chunks)
        articles.append(
            {
                "article_id": article_id,
                "title": str(metadata.get("title") or ""),
                "source": str(metadata.get("source") or ""),
                "url": str(metadata.get("url") or ""),
                "category": str(metadata.get("category") or ""),
                "topic_category": str(metadata.get("topic_category") or infer_topic_category(metadata)),
                "published_at": str(metadata.get("published_at") or ""),
                "score": safe_float(best_chunk.get("final_score") or best_chunk.get("score")),
                "summary": summarize_text_for_article(merged_text, question, max_sentences=3),
                "chunks": ordered_text_chunks,
            }
        )

    articles.sort(
        key=lambda item: (
            parse_published_at(str(item.get("published_at") or "")),
            safe_float(item.get("score")),
        ),
        reverse=True,
    )
    return articles[:max_articles]


def sort_chunks_for_article(question: str, chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    query_terms = normalized_terms(question)
    ranked: list[dict[str, object]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        text = str(chunk.get("text") or "")
        title = str(metadata.get("title") or "")
        category = str(metadata.get("category") or "")
        topic_category = str(metadata.get("topic_category") or "")
        score = safe_float(chunk.get("score"))
        keyword_score = overlap_score(query_terms, normalized_terms(text))
        metadata_score = overlap_score(query_terms, normalized_terms(f"{title} {category} {topic_category}"))
        ranked_chunk = dict(chunk)
        ranked_chunk["final_score"] = (score * 0.55) + (keyword_score * 0.3) + (metadata_score * 0.15)
        ranked.append(ranked_chunk)
    ranked.sort(
        key=lambda item: (
            safe_float(item.get("final_score")),
            -chunk_index_of(item),
        ),
        reverse=True,
    )
    return ranked


def resolve_best_article_id(results: list[dict[str, object]], min_score: float) -> str | None:
    seen: set[str] = set()
    for result in results:
        article_id = article_id_of(result)
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        if safe_float(result.get("final_score")) >= min_score:
            return article_id
    return None


def is_weak_context(results: list[dict[str, object]], min_score: float) -> bool:
    if not results:
        return True
    strongest_score = max(safe_float(item.get("final_score")) for item in results)
    return strongest_score < min_score


def build_sources(results: list[dict[str, object]]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = str(result.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        topic = str(metadata.get("primary_topic") or "")
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "article_id": str(metadata.get("article_id") or ""),
                "title": str(metadata.get("title") or ""),
                "source": str(metadata.get("source") or ""),
                "url": str(metadata.get("url") or ""),
                "category": str(metadata.get("category") or ""),
                "published_at": str(metadata.get("published_at") or ""),
                "primary_topic": topic,
                "topic": topic,
                "domain": domain_from_topic(topic),
                "score": max(safe_float(result.get("final_score")), safe_float(result.get("score"))),
                "snippet": _snippet(result.get("text") or result.get("document") or ""),
                "chunk_id": chunk_id,
            }
        )
    return sources


def build_article_sources(articles: list[dict[str, object]]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for article in articles:
        article_id = str(article.get("article_id") or "")
        if not article_id:
            continue
        topic = str(article.get("primary_topic") or article.get("topic") or "")
        sources.append(
            {
                "citation_id": len(sources) + 1,
                "article_id": article_id,
                "title": str(article.get("title") or ""),
                "source": str(article.get("source") or ""),
                "url": str(article.get("url") or ""),
                "category": str(article.get("category") or ""),
                "published_at": str(article.get("published_at") or ""),
                "primary_topic": topic,
                "topic": topic,
                "domain": domain_from_topic(topic),
                "score": safe_float(article.get("relevance_score") or article.get("score")),
                "snippet": _snippet(article.get("summary") or article.get("content") or ""),
                "chunk_id": article_id,
            }
        )
    return sources


def _copy_citation_ids_to_articles(articles: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    citation_by_article_id = {
        str(source.get("article_id") or ""): source.get("citation_id")
        for source in sources
        if str(source.get("article_id") or "")
    }
    for index, article in enumerate(articles, start=1):
        article_id = str(article.get("article_id") or "")
        article["citation_id"] = citation_by_article_id.get(article_id) or index


def _citation_warnings(answer: str, sources: list[dict[str, Any]]) -> list[str]:
    cited_ids = {int(match.group(1)) for match in re.finditer(r"\[(\d+)\]", str(answer or ""))}
    if not cited_ids:
        return []
    valid_ids: set[int] = set()
    for source in sources:
        try:
            valid_ids.add(int(source.get("citation_id")))
        except (TypeError, ValueError):
            continue
    invalid_ids = sorted(cited_ids - valid_ids)
    return [f"citation_id_not_in_sources: [{citation_id}]" for citation_id in invalid_ids]


def _snippet(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _structured_summary_response(response: dict[str, object], query_plan: dict[str, Any]) -> dict[str, object]:
    structured = dict(response)
    sources = structured.get("sources") if isinstance(structured.get("sources"), list) else []
    topic = ""
    if sources and isinstance(sources[0], dict):
        topic = str(sources[0].get("primary_topic") or sources[0].get("topic") or "")
    structured["intent"] = "article_summary"
    structured["topic"] = topic or query_plan.get("primary_topic")
    structured["query_plan"] = {
        **query_plan,
        "intent": "article_summary",
        "data_source": "article_rag",
    }
    structured["sources"] = sources
    structured.setdefault("images", [])
    structured.setdefault("related_articles", build_related_articles([], sources))
    return structured


def _chroma_unavailable_response(query_plan: dict[str, Any], exc: ChromaUnavailableError) -> dict[str, object]:
    return {
        "answer": "Hiện chưa thể truy vấn vì Chroma index local đang lỗi hoặc chưa sẵn sàng. Cần chạy healthcheck/rebuild index.",
        "intent": query_plan.get("intent", "unknown"),
        "topic": query_plan.get("primary_topic"),
        "query_plan": query_plan,
        "sources": [],
        "images": [],
        "related_articles": [],
        "debug": {
            "error": "CHROMA_UNAVAILABLE",
            "error_code": exc.error_code,
            "message": str(exc),
            "suggestion": "Run scripts/inspect_chroma_health.py or rebuild Chroma from Gold",
        },
    }


def _guard_failed_response(
    question: str,
    query_plan: dict[str, Any],
    reason: str,
    violations: list[str],
) -> dict[str, object]:
    topic = str(query_plan.get("primary_topic") or "")
    return {
        "answer": safe_fallback_for_topic(question, topic),
        "intent": query_plan.get("intent", "unknown"),
        "topic": query_plan.get("primary_topic"),
        "query_plan": query_plan,
        "sources": [],
        "images": [],
        "related_articles": [],
        "debug": {
            "guard_failed": True,
            "guard_reason": reason,
            "guard_violations": violations,
        },
    }


def article_from_chunks(chunks: list[dict[str, object]]) -> dict[str, object]:
    if not chunks:
        return {}
    first = chunks[0]
    metadata = first.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "article_id": str(metadata.get("article_id") or ""),
        "title": str(metadata.get("title") or ""),
        "source": str(metadata.get("source") or ""),
        "url": str(metadata.get("url") or ""),
        "category": str(metadata.get("category") or ""),
        "published_at": str(metadata.get("published_at") or ""),
        "chunks": chunks,
    }


def flatten_article_chunks(articles: list[dict[str, object]]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for article in articles:
        chunks = article.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        flattened.extend(chunk for chunk in chunks if isinstance(chunk, dict))
    return flattened


def build_debug_entries(results: list[dict[str, object]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        preview = " ".join(str(result.get("text") or "").split())[:180]
        entry = {
                "chunk_id": str(result.get("chunk_id") or ""),
                "article_id": str(metadata.get("article_id") or result.get("article_id") or ""),
                "title": str(metadata.get("title") or result.get("title") or ""),
                "source": str(metadata.get("source") or result.get("source") or ""),
                "category": str(metadata.get("category") or result.get("category") or ""),
                "vector_score": f"{safe_float(result.get('vector_score') or result.get('score')):.4f}",
                "keyword_score": f"{safe_float(result.get('keyword_score')):.4f}",
                "topic_score": f"{safe_float(result.get('combined_topic_score') or result.get('topic_match')):.4f}",
                "topic_penalty": f"{safe_float(result.get('topic_penalty')):.4f}",
                "recency_score": f"{safe_float(result.get('recency_score')):.4f}",
                "entity_score": f"{safe_float(result.get('combined_entity_score') or result.get('entity_match')):.4f}",
                "final_score": f"{safe_float(result.get('final_score') or result.get('score')):.4f}",
                "topic": str(metadata.get("primary_topic") or metadata.get("topic") or metadata.get("topic_category") or ""),
                "matched_keywords": ", ".join(str(item) for item in result.get("matched_keywords", []) if str(item).strip()),
                "url": str(metadata.get("url") or result.get("url") or ""),
                "preview_text": preview,
            }
        if isinstance(result.get("debug_score"), dict):
            entry["debug_score"] = result["debug_score"]
        entries.append(entry)
    return entries


def _attach_rag_trace(
    response: dict[str, object],
    *,
    query_plan: dict[str, Any],
    question: str,
    retrieval_trace: dict[str, Any],
    rerank_trace: list[dict[str, Any]],
    results: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    prompt: str,
    generation_debug: dict[str, Any],
    fallback_used: bool,
    no_answer_triggered: bool,
    no_answer_reason: str,
    settings: LocalAiSettings,
    retrieved_articles: list[dict[str, Any]] | None = None,
) -> None:
    trace = {
        "router": _router_debug(question, query_plan),
        "retrieval": retrieval_trace
        or {
            "broad_retrieve_top_n": settings.rag_broad_retrieve_top_n,
            "raw_candidate_count": len(results),
            "metadata_filter_applied": False,
            "topic_filter_applied": False,
            "selected_topic": str(query_plan.get("primary_topic") or ""),
            "candidate_count_before_topic_filter": len(results),
            "candidate_count_after_topic_filter": len(results),
            "fallback_used": False,
            "min_results": 0,
            "source_diversity_count": _source_diversity_count_from_results(results),
        },
        "rerank": rerank_trace or _rerank_debug_entries(results[:10]),
        "context_builder": _context_builder_debug(results, sources, prompt, settings, retrieved_articles),
        "prompt": _prompt_debug(prompt, settings),
        "generation": {
            "ollama_model": generation_debug.get("ollama_model") or "",
            "used_llm": bool(generation_debug.get("used_llm")),
            "llm_error": str(generation_debug.get("llm_error") or ""),
            "fallback_used": bool(generation_debug.get("fallback_used", fallback_used)),
            "no_answer_triggered": bool(generation_debug.get("no_answer_triggered", no_answer_triggered)),
            "no_answer_reason": str(generation_debug.get("no_answer_reason") or no_answer_reason),
            "answer_validator_reason": str(generation_debug.get("answer_validator_reason") or ""),
            "extractive_answer_used": bool(generation_debug.get("extractive_answer_used", fallback_used)),
        },
    }
    query_debug = response.get("query_plan")
    if isinstance(query_debug, dict):
        query_debug["debug_trace"] = trace
        query_debug["router_debug"] = trace["router"]
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    debug["rag_trace"] = trace
    response["debug"] = debug


def _router_debug(question: str, query_plan: dict[str, Any]) -> dict[str, Any]:
    selected_topic = str(query_plan.get("primary_topic") or "")
    topic_scores = query_plan.get("topic_scores") if isinstance(query_plan.get("topic_scores"), dict) else {}
    matched = query_plan.get("topic_matched_keywords") if isinstance(query_plan.get("topic_matched_keywords"), dict) else {}
    return {
        "original_query": question,
        "normalized_query": str(query_plan.get("normalized_query") or normalize_text(question)),
        "intent": str(query_plan.get("intent") or ""),
        "answer_mode": str(query_plan.get("answer_mode") or ""),
        "standalone_query": str(query_plan.get("standalone_query") or ""),
        "selected_topic": selected_topic,
        "explicit_topic": bool(query_plan.get("explicit_topic")),
        "topic_confidence": query_plan.get("topic_confidence"),
        "topic_confidence_label": str(query_plan.get("topic_confidence_label") or ""),
        "topic_scores": {topic: int(topic_scores.get(topic, 0) or 0) for topic in _REQUIRED_TOPICS},
        "matched_keywords": {topic: list(matched.get(topic, []) or []) for topic in _REQUIRED_TOPICS},
        "reason": _router_reason(query_plan),
    }


_REQUIRED_TOPICS = (
    "tech_ai_internet",
    "economy_finance_stock",
    "politics_society",
    "world_geopolitics",
    "business_startup",
    "real_estate",
    "lifestyle_education_health_entertainment",
)


def _router_reason(query_plan: dict[str, Any]) -> str:
    topic = str(query_plan.get("primary_topic") or "")
    if not topic:
        return "no domain topic selected"
    if bool(query_plan.get("explicit_topic")):
        return f"explicit topic/domain keyword selected {topic}"
    matched = query_plan.get("matched_keywords")
    if isinstance(matched, list) and matched:
        return f"highest topic score for {topic}; matched keywords: {', '.join(str(item) for item in matched)}"
    return f"highest topic score or taxonomy fallback selected {topic}"


def _context_builder_debug(
    results: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    prompt: str,
    settings: LocalAiSettings,
    retrieved_articles: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    articles = retrieved_articles or []
    titles = [str(source.get("title") or "") for source in sources if str(source.get("title") or "").strip()]
    topics = sorted({str(source.get("topic") or source.get("primary_topic") or "") for source in sources if str(source.get("topic") or source.get("primary_topic") or "").strip()})
    return {
        "final_chunk_count": len(results),
        "final_article_count": len(articles) if articles else len(sources),
        "context_chars": _context_chars_from_prompt(prompt),
        "max_context_chars": settings.prompt_max_context_chars,
        "titles_in_context": titles,
        "topics_in_context": topics,
        "whether_context_empty": not bool(results or sources),
    }


def _prompt_debug(prompt: str, settings: LocalAiSettings) -> dict[str, Any]:
    limit = max(0, int(settings.prompt_debug_max_chars or 0))
    preview = prompt[:limit] if limit else ""
    return {
        "prompt_chars": len(prompt),
        "context_chars": _context_chars_from_prompt(prompt),
        "prompt_preview": preview,
        "no_answer_rules": [
            "Return NO_INFO_FALLBACK only when context is empty, unrelated, or insufficient.",
            "If context has direct evidence/title keyword overlap, answer from context with citations.",
        ],
    }


def _context_chars_from_prompt(prompt: str) -> int:
    if not prompt:
        return 0
    markers = ("RETRIEVED_CONTEXT:", "CONTEXT:", "ARTICLE_CONTEXT:")
    for marker in markers:
        start = prompt.find(marker)
        if start < 0:
            continue
        tail = prompt[start + len(marker) :]
        end_positions = [pos for token in ("\n\nUSER_QUESTION:", "\n\nCAU HOI:", "\n\nCÂU HỎI:") if (pos := tail.find(token)) >= 0]
        end = min(end_positions) if end_positions else len(tail)
        return len(tail[:end].strip())
    return 0


def _rerank_debug_entries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": str(_result_metadata(result).get("title") or ""),
            "source": str(_result_metadata(result).get("source") or ""),
            "topic": str(_result_metadata(result).get("primary_topic") or _result_metadata(result).get("topic") or ""),
            "published_at": str(_result_metadata(result).get("published_at") or ""),
            "article_id": str(_result_metadata(result).get("article_id") or ""),
            "vector_score": safe_float(result.get("vector_score") or result.get("score")),
            "keyword_score": safe_float(result.get("keyword_score")),
            "topic_score": safe_float(result.get("combined_topic_score") or result.get("topic_match")),
            "topic_penalty": safe_float(result.get("topic_penalty")),
            "recency_score": safe_float(result.get("recency_score")),
            "entity_score": safe_float(result.get("combined_entity_score") or result.get("entity_match")),
            "final_score": safe_float(result.get("final_score") or result.get("score")),
            "matched_keywords": result.get("matched_keywords", []),
            "why_selected": "penalized: topic mismatch" if safe_float(result.get("topic_penalty")) > 0 else "selected",
        }
        for result in results
    ]


def _source_diversity_count_from_results(results: list[dict[str, Any]]) -> int:
    return len({str(_result_metadata(result).get("source") or "") for result in results if str(_result_metadata(result).get("source") or "").strip()})


def _ollama_model(client: Any) -> str:
    if client is None:
        return ""
    runtime_config = getattr(client, "runtime_config", None)
    if callable(runtime_config):
        try:
            config = runtime_config()
            if isinstance(config, dict):
                return str(config.get("model") or "")
        except Exception:
            return ""
    return str(getattr(client, "_model", "") or "")


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip().lower() in {"1", "true", "yes", "y", "on"})


def extractive_answer(question: str, results: list[dict[str, object]]) -> str:
    query_terms = normalized_terms(question)
    scored_sentences: list[tuple[float, int, str]] = []
    for result_index, result in enumerate(results, start=1):
        for sentence in split_sentences(str(result.get("text") or "")):
            score = overlap_score(query_terms, normalized_terms(sentence))
            scored_sentences.append((score, result_index, sentence))

    scored_sentences.sort(key=lambda item: (item[0], -item[1], len(item[2])), reverse=True)
    selected = [item for item in scored_sentences if item[0] > 0][:3]
    if not selected:
        selected = scored_sentences[:3]
    if not selected:
        return _NO_ANSWER

    snippets = [f"{sentence} [{result_index}]" for _, result_index, sentence in selected]
    return " ".join(snippets)


def extractive_broad_answer(articles: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for index, article in enumerate(articles, start=1):
        lines.append(
            f"{index}. {article.get('title') or 'Untitled'}\n"
            f"   - Tom tat: {article.get('summary') or ''}\n"
            f"   - Nguon: {article.get('source') or 'unknown'} | {article.get('url') or ''}"
        )
    return "\n".join(lines) if lines else _NO_BROAD_DATA


def extractive_category_answer(articles: list[dict[str, object]]) -> str:
    if not articles:
        return _NO_CATEGORY_DATA

    lines: list[str] = []
    for index, article in enumerate(articles, start=1):
        lines.append(
            f"{index}. {article.get('title') or 'Untitled'}\n"
            f"   - Tom tat: {article.get('summary') or ''}\n"
            f"   - Nguon: {article.get('source') or 'unknown'} ({article.get('url') or ''})"
        )
    return "\n".join(lines)


def extractive_article_summary(chunks: list[dict[str, object]]) -> str:
    article_text = merge_chunk_texts(chunks)
    summary = summarize_text_for_article(article_text, "tom tat bai bao", max_sentences=4)
    source = build_sources([chunks[0]])[0] if chunks else {}
    bullets = split_sentences(summary)[:3]
    bullet_text = "\n".join(f"- {item}" for item in bullets)
    return (
        f"Tom tat:\n{summary}\n\n"
        f"Y chinh:\n{bullet_text}\n\n"
        "Nguon:\n"
        f"- {source.get('title') or ''}\n"
        f"- {source.get('source') or ''}\n"
        f"- {source.get('url') or ''}"
    )


def summarize_text_for_article(text: str, question: str, max_sentences: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    query_terms = normalized_terms(question)
    ranked = sorted(
        (
            (
                overlap_score(query_terms, normalized_terms(sentence)) + max(0.0, 0.12 - (index * 0.02)),
                index,
                sentence,
            )
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = sorted(ranked[:max_sentences], key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in selected)


def merge_chunk_texts(chunks: list[dict[str, object]]) -> str:
    texts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = " ".join(str(chunk.get("text") or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return " ".join(texts)


def build_related_articles(parent_articles: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(parent.get("article_id") or ""): parent for parent in parent_articles}
    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        article_id = str(source.get("article_id") or "")
        if not article_id or article_id in seen:
            continue
        seen.add(article_id)
        parent = by_id.get(article_id, {})
        related.append(
            {
                "article_id": article_id,
                "title": str(parent.get("title") or source.get("title") or ""),
                "url": str(parent.get("url") or source.get("url") or ""),
                "source": str(parent.get("source") or source.get("source") or ""),
                "published_at": str(parent.get("published_at") or source.get("published_at") or ""),
                "primary_topic": str(parent.get("primary_topic") or source.get("primary_topic") or ""),
            }
        )
    return related


def _has_sufficient_evidence(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    best = max(
        max(safe_float(result.get("final_score")), safe_float(result.get("score")), safe_float(result.get("vector_score")))
        for result in results
    )
    best_vector = max(max(safe_float(result.get("score")), safe_float(result.get("vector_score"))) for result in results)
    return best_vector >= 0.05 and best >= 0.15


def _invalid_generated_answer_reason(answer: str | None, sources: list[dict[str, Any]]) -> str:
    if not answer or not str(answer).strip():
        return "empty_llm_answer"
    if not sources:
        return ""
    normalized = normalize_text(str(answer))
    evasive_markers = (
        "cau hoi khong duoc chi ro",
        "cau hoi chua duoc chi ro",
        "khong cung cap cu the",
        "khong duoc cung cap cu the",
        "vui long cung cap them",
        "xin vui long cung cap them",
        "doan van ban cung cap",
        "doan van duoc cung cap",
        "noi dung da duoc chia se",
        "khach hang yeu cau",
        "khong ro lieu co lien quan",
        "chua ro ban dang hoi",
    )
    if any(marker in normalized for marker in evasive_markers):
        return "llm_evasive_despite_retrieved_sources"
    source_titles = [normalize_text(str(source.get("title") or "")) for source in sources]
    if source_titles and not any(_title_token_overlap(normalized, title) for title in source_titles):
        if len(normalized) < 180 and any(term in normalized for term in ("khong tim thay", "khong co thong tin", "chua co thong tin")):
            return "llm_no_answer_despite_retrieved_sources"
    return ""


def _title_token_overlap(normalized_answer: str, normalized_title: str) -> bool:
    title_terms = {term for term in normalized_title.split() if len(term) >= 3}
    if not title_terms:
        return False
    answer_terms = {term for term in normalized_answer.split() if len(term) >= 3}
    return len(title_terms.intersection(answer_terms)) >= min(2, len(title_terms))


def _select_latest_article_results(results: list[dict[str, Any]], article_count: int) -> list[dict[str, Any]]:
    if not results:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback_index = 0
    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        article_id = str(metadata.get("article_id") or "")
        if not article_id:
            fallback_index += 1
            article_id = str(metadata.get("url") or result.get("chunk_id") or f"fallback:{fallback_index}")
        grouped[article_id].append(result)

    ordered_article_ids = sorted(
        grouped,
        key=lambda article_id: (
            max(parse_published_at(str(_result_metadata(item).get("published_at") or "")) for item in grouped[article_id]),
            max(safe_float(item.get("final_score") or item.get("score")) for item in grouped[article_id]),
        ),
        reverse=True,
    )[: max(1, int(article_count))]

    selected: list[dict[str, Any]] = []
    for article_id in ordered_article_ids:
        chunks = sorted(
            grouped[article_id],
            key=lambda item: (
                -chunk_index_of(item),
                safe_float(item.get("final_score") or item.get("score")),
            ),
            reverse=True,
        )
        selected.extend(chunks[:2])
    return selected


def _result_metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _extract_citation_number(normalized_question: str) -> int | None:
    patterns = (
        r"\[(\d{1,2})\]",
        r"(?:nguon|source)\s*(?:so)?\s*(\d{1,2})",
        r"(?:bai|tin)\s*(?:so)?\s*(\d{1,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_question)
        if match:
            return int(match.group(1))
    return None


def _has_context_reference(normalized_question: str) -> bool:
    references = (
        "bai nay",
        "bai do",
        "bai bao nay",
        "bai bao do",
        "tin nay",
        "tin do",
        "nguon nay",
        "nguon do",
        "y tren",
        "doan tren",
    )
    return any(reference in normalized_question for reference in references)


def _context_sources(current_context: dict[str, Any]) -> list[dict[str, Any]]:
    sources = current_context.get("previous_sources")
    if not isinstance(sources, list):
        return []
    return [dict(source) for source in sources if isinstance(source, dict)]


def _context_topic(current_context: dict[str, Any]) -> str:
    explicit_topic = str(current_context.get("previous_topic") or current_context.get("topic") or "").strip()
    if explicit_topic:
        return explicit_topic
    query_plan = current_context.get("previous_query_plan")
    if isinstance(query_plan, dict):
        topic = str(query_plan.get("primary_topic") or query_plan.get("topic") or "").strip()
        if topic:
            return topic
    sources = _context_sources(current_context)
    if sources:
        return str(sources[0].get("topic") or sources[0].get("primary_topic") or "")
    return ""


def _resolve_followup_route(
    question: str,
    query_plan: dict[str, Any],
    current_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if query_plan.get("answer_mode") != "followup":
        return query_plan
    updated = dict(query_plan)
    if not current_context:
        updated.update(
            {
                "needs_clarification": True,
                "clarification_message": "Mình chưa có ngữ cảnh trước đó để hiểu 'vụ này' là vụ nào. Bạn hãy nhắc lại chủ đề hoặc chọn một nguồn cụ thể.",
                "standalone_query": "",
            }
        )
        return updated

    previous_plan = current_context.get("previous_query_plan")
    previous_plan = previous_plan if isinstance(previous_plan, dict) else {}
    topic = _context_topic(current_context)
    sources = _context_sources(current_context)
    entities = _context_list(previous_plan, "entities")
    stock_symbols = _context_list(previous_plan, "stock_symbols")
    if not entities and sources:
        entities = _entities_from_sources(sources)
    if not topic and not entities and not stock_symbols:
        updated.update(
            {
                "needs_clarification": True,
                "clarification_message": "Mình chưa xác định được bạn đang hỏi tiếp về chủ đề nào. Bạn hãy nhắc lại chủ đề hoặc nguồn cần hỏi.",
                "standalone_query": "",
            }
        )
        return updated

    updated["primary_topic"] = topic or updated.get("primary_topic")
    updated["domain"] = domain_from_topic(str(updated.get("primary_topic") or "") or None)
    updated["entities"] = entities
    updated["stock_symbols"] = stock_symbols
    updated["ticker"] = stock_symbols[0] if stock_symbols else str(previous_plan.get("ticker") or "")
    updated["exact_entities"] = [*entities, *stock_symbols]
    updated["lexical_terms"] = [*entities, *stock_symbols]
    updated["requires_lexical"] = bool(entities or stock_symbols)
    updated["time_range"] = str(previous_plan.get("time_range") or updated.get("time_range") or "all")
    updated["needs_recent"] = bool(previous_plan.get("needs_recent") or updated.get("needs_recent"))
    updated["preferred_sources"] = list(previous_plan.get("preferred_sources") or updated.get("preferred_sources") or [])
    updated["topic_confidence"] = max(float(updated.get("topic_confidence") or 0.0), float(previous_plan.get("topic_confidence") or 0.0), 0.8 if topic else 0.0)
    updated["topic_confidence_label"] = "high" if topic else str(updated.get("topic_confidence_label") or "")
    updated["explicit_topic"] = bool(topic or updated.get("explicit_topic"))
    updated["data_source"] = "article_rag"
    updated["standalone_query"] = _standalone_followup_query(question, updated, previous_plan)
    updated["followup_inherited_from_context"] = True
    return updated


def _context_list(plan: dict[str, Any], key: str) -> list[str]:
    values = plan.get(key)
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _entities_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for source in sources:
        for key in ("title", "source"):
            value = str(source.get(key) or "").strip()
            if value:
                values.append(value)
    return values[:3]


def _standalone_followup_query(question: str, query_plan: dict[str, Any], previous_plan: dict[str, Any]) -> str:
    topic = str(query_plan.get("primary_topic") or "")
    entities = [str(item) for item in query_plan.get("entities", []) if str(item).strip()]
    stock_symbols = [str(item) for item in query_plan.get("stock_symbols", []) if str(item).strip()]
    previous_query = str(previous_plan.get("normalized_query") or previous_plan.get("standalone_query") or "").strip()
    if topic == "tech_ai_internet":
        base = "phan tich anh huong cua cac tin AI gan day"
    elif topic == "economy_finance_stock":
        target = ", ".join(stock_symbols or entities) or "thi truong tai chinh chung khoan"
        base = f"phan tich anh huong den {target}"
    elif topic == "real_estate":
        base = "phan tich anh huong cua cac tin bat dong san gan day"
    elif topic == "world_geopolitics":
        base = "phan tich tac dong cua dien bien quoc te gan day"
    elif topic == "business_startup":
        base = "phan tich tac dong cua tin doanh nghiep startup gan day"
    elif topic == "politics_society":
        base = "phan tich tac dong xa hoi cua su kien thoi su gan day"
    elif topic == "lifestyle_education_health_entertainment":
        base = "phan tich tac dong cua tin doi song giao duc suc khoe giai tri gan day"
    else:
        base = previous_query or normalize_text(question)
    if previous_query and previous_query not in base:
        return f"{base}; ngu canh truoc: {previous_query}"
    return base


def _followup_query_plan(followup_intent: str, current_context: dict[str, Any]) -> dict[str, Any]:
    topic = _context_topic(current_context)
    previous_plan = current_context.get("previous_query_plan")
    previous_plan = previous_plan if isinstance(previous_plan, dict) else {}
    entities = _context_list(previous_plan, "entities")
    stock_symbols = _context_list(previous_plan, "stock_symbols")
    return {
        "intent": followup_intent,
        "answer_mode": "followup",
        "primary_topic": topic or None,
        "domain": domain_from_topic(topic or None),
        "ticker": stock_symbols[0] if stock_symbols else str(previous_plan.get("ticker") or ""),
        "entities": entities,
        "stock_symbols": stock_symbols,
        "requires_lexical": bool(entities or stock_symbols),
        "lexical_terms": [*entities, *stock_symbols],
        "exact_entities": [*entities, *stock_symbols],
        "needs_recent": False,
        "needs_images": followup_intent == "followup_media_lookup",
        "preferred_sources": [],
        "time_range": str(previous_plan.get("time_range") or "all"),
        "source": None,
        "need_images": followup_intent == "followup_media_lookup",
        "need_sources": True,
        "need_timeline": False,
        "data_source": "current_context",
        "standalone_query": _standalone_followup_query("", {"primary_topic": topic, "entities": entities, "stock_symbols": stock_symbols}, previous_plan),
    }


def _source_context_to_source(source: dict[str, Any]) -> dict[str, Any]:
    topic = str(source.get("topic") or source.get("primary_topic") or "")
    return {
        "citation_id": source.get("citation_id"),
        "article_id": str(source.get("article_id") or ""),
        "title": str(source.get("title") or ""),
        "source": str(source.get("source") or ""),
        "url": str(source.get("url") or ""),
        "published_at": str(source.get("published_at") or ""),
        "primary_topic": topic,
        "topic": topic,
        "domain": domain_from_topic(topic),
        "score": safe_float(source.get("score")),
        "snippet": str(source.get("snippet") or ""),
    }


def _context_images_for_source(current_context: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    images = current_context.get("previous_images")
    if not isinstance(images, list):
        return []
    article_id = str(source.get("article_id") or "")
    title = str(source.get("title") or "")
    output: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_article_id = str(image.get("article_id") or "")
        image_title = str(image.get("article_title") or "")
        if article_id and image_article_id and article_id != image_article_id:
            continue
        if not article_id and title and image_title and title != image_title:
            continue
        normalized = dict(image)
        if normalized.get("image_url") and not normalized.get("url"):
            normalized["url"] = normalized["image_url"]
        if normalized.get("url") and not normalized.get("image_url"):
            normalized["image_url"] = normalized["url"]
        output.append(normalized)
    return output


def _clarification_response(question: str, message: str, query_plan: dict[str, Any]) -> dict[str, object]:
    return {
        "question": question,
        "answer": message,
        "intent": query_plan.get("intent", "followup_clarification"),
        "topic": query_plan.get("primary_topic"),
        "query_plan": query_plan,
        "sources": [],
        "images": [],
        "related_articles": [],
    }


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", normalized) if part.strip()]


def extract_url(text: str) -> str | None:
    match = re.search(r"https?://\S+", text)
    return match.group(0).rstrip(").,]") if match else None


def extract_title_hint(question: str) -> str | None:
    normalized = question.strip()
    if ":" in normalized:
        return normalized.split(":", maxsplit=1)[1].strip() or None
    quoted = re.findall(r'"([^"]+)"', normalized)
    if quoted:
        return quoted[0].strip()
    suffix_markers = (
        " tóm tắt",
        " tom tat",
        " hãy tóm tắt",
        " hay tom tat",
    )
    lowered = normalized.casefold()
    for marker in suffix_markers:
        index = lowered.find(marker)
        if index > 0:
            return normalized[:index].strip(" -:;,.") or None
    prefix_patterns = (
        r"^\s*(?:hãy\s+)?tóm\s+tắt\s+(?:bài\s+báo|bài|tin|nội\s+dung)?\s*(?:này|sau)?\s*:?\s*(.+)$",
        r"^\s*(?:hay\s+)?tom\s+tat\s+(?:bai\s+bao|bai|tin|noi\s+dung)?\s*(?:nay|sau)?\s*:?\s*(.+)$",
        r"^\s*bài\s+(?:báo\s+)?có\s+tiêu\s+đề\s+(.+)$",
        r"^\s*bai\s+(?:bao\s+)?co\s+tieu\s+de\s+(.+)$",
    )
    for pattern in prefix_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" -:;,.") or None
    return None


def article_id_of(result: dict[str, object]) -> str:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("article_id") or "")


def chunk_index_of(result: dict[str, object]) -> int:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    try:
        return int(metadata.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def parse_published_at(value: str) -> datetime:
    text = value.strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min
