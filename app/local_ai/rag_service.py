from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from app.config import LocalAiSettings, load_settings
from app.local_ai.chunker import infer_topic_category
from app.local_ai.embeddings import LocalEmbeddingModel
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.prompt_builder import PromptBuilder
from app.local_ai.reranker import SimpleReranker, normalize_text, normalized_terms, overlap_score, safe_float
from app.local_ai.vector_store import ChromaVectorStore

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
        try:
            return self._ollama_client.generate(prompt) if self._ollama_client is not None else None
        except Exception:
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
        sources.append(
            {
                "title": str(metadata.get("title") or ""),
                "source": str(metadata.get("source") or ""),
                "url": str(metadata.get("url") or ""),
                "category": str(metadata.get("category") or ""),
                "published_at": str(metadata.get("published_at") or ""),
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
        sources.append(
            {
                "title": str(article.get("title") or ""),
                "source": str(article.get("source") or ""),
                "url": str(article.get("url") or ""),
                "category": str(article.get("category") or ""),
                "published_at": str(article.get("published_at") or ""),
                "chunk_id": article_id,
            }
        )
    return sources


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


def build_debug_entries(results: list[dict[str, object]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        preview = " ".join(str(result.get("text") or "").split())[:180]
        entries.append(
            {
                "chunk_id": str(result.get("chunk_id") or ""),
                "article_id": str(metadata.get("article_id") or result.get("article_id") or ""),
                "title": str(metadata.get("title") or result.get("title") or ""),
                "source": str(metadata.get("source") or result.get("source") or ""),
                "category": str(metadata.get("category") or result.get("category") or ""),
                "vector_score": f"{safe_float(result.get('vector_score') or result.get('score')):.4f}",
                "final_score": f"{safe_float(result.get('final_score') or result.get('score')):.4f}",
                "url": str(metadata.get("url") or result.get("url") or ""),
                "preview_text": preview,
            }
        )
    return entries


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
