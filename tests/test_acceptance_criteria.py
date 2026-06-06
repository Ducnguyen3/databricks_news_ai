from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from typing import Any

from app.local_ai.chunking.metadata_builder import ChunkMetadataBuilder
from app.local_ai.chunking.models import ArticleBlock, ParentArticle
from app.local_ai.media_retriever import MediaRetriever
from app.local_ai.processors import DefaultNewsProcessor, get_processor
from app.local_ai.query_router import route_query
from app.local_ai.rag_service import RAGService


class _EmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _VectorStore:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self.get_chunks_calls: list[str] = []

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        return self._results[:top_k]

    def get_chunks_by_article_id(self, article_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        self.get_chunks_calls.append(article_id)
        chunks = [item for item in self._results if item.get("metadata", {}).get("article_id") == article_id]
        return chunks[:limit] if limit else chunks


class _OllamaClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str) -> str:
        return self.answer


class _ImageRepository:
    def fetch_article_images(self, article_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "article_id": article_ids[0],
                "image_url": f"https://example.com/{article_ids[0]}-repository.jpg",
                "caption": "Anh that tu bang article images",
                "credit": "Example Photo",
                "source": "vnexpress",
                "article_title": "Tin AI moi nhat",
                "article_url": f"https://example.com/{article_ids[0]}",
                "is_representative": True,
            }
        ]


def _chunk(
    article_id: str = "article-ai",
    topic: str = "tech_ai_internet",
    entity_names: str = "OpenAI",
    has_images: bool = False,
    published_at: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": f"{article_id}:0",
        "text": "OpenAI cong bo tin cong nghe moi trong tuan nay.",
        "score": 0.95,
        "metadata": {
            "chunk_id": f"{article_id}:0",
            "article_id": article_id,
            "block_id": f"{article_id}:block-0",
            "title": "Tin AI moi nhat",
            "source": "vnexpress",
            "url": f"https://example.com/{article_id}",
            "published_at": published_at or datetime.now(timezone.utc).isoformat(),
            "primary_topic": topic,
            "entity_names": entity_names,
            "entity_types": "organization",
            "has_images": has_images,
            "image_count": 1 if has_images else 0,
            "images_json": json.dumps(
                [
                    {
                        "article_id": article_id,
                        "image_url": f"https://example.com/{article_id}.jpg",
                        "caption": "Anh dai dien",
                        "credit": "Example",
                        "is_representative": True,
                    }
                ]
                if has_images
                else []
            ),
        },
    }


class AcceptanceCriteriaTest(unittest.TestCase):
    def test_answer_structured_always_returns_required_schema(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk()]),
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3)

        for key in ("answer", "intent", "topic", "query_plan", "sources", "images", "related_articles"):
            self.assertIn(key, response)
        self.assertIsInstance(response["answer"], str)
        self.assertIsInstance(response["query_plan"], dict)
        self.assertIsInstance(response["sources"], list)
        self.assertIsInstance(response["images"], list)
        self.assertIsInstance(response["related_articles"], list)

    def test_answer_structured_debug_prompt_uses_topic_profile(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk()]),
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3, debug_prompt=True)

        for key in ("answer", "intent", "topic", "query_plan", "sources", "images", "related_articles"):
            self.assertIn(key, response)
        self.assertIn("TOPIC_PROFILE", response["prompt_debug"])
        self.assertIn("Cong nghe - AI - Internet", response["prompt_debug"])

    def test_answer_structured_no_results_does_not_crash_or_hallucinate(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([]),
            ollama_client=None,
        )

        response = service.answer_structured("du lieu khong co trong index", top_k=3)

        self.assertEqual([], response["sources"])
        self.assertEqual([], response["images"])
        self.assertEqual(
            "Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có thông tin đủ để trả lời câu hỏi này.",
            response["answer"],
        )

    def test_answer_structured_low_evidence_returns_index_fallback(self) -> None:
        low_score_chunk = _chunk()
        low_score_chunk["score"] = 0.0
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([low_score_chunk]),
            ollama_client=None,
        )

        response = service.answer_structured("du lieu khong co trong index", top_k=3)

        self.assertEqual([], response["sources"])
        self.assertEqual(
            "Dựa trên các bài báo hệ thống đã thu thập hiện tại, tôi chưa có thông tin đủ để trả lời câu hỏi này.",
            response["answer"],
        )

    def test_answer_structured_returns_parent_child_related_articles(self) -> None:
        vector_store = _VectorStore([_chunk(article_id="article-parent")])
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3)

        self.assertIn("article-parent", vector_store.get_chunks_calls)
        self.assertEqual(1, len(response["related_articles"]))
        self.assertEqual("article-parent", response["related_articles"][0]["article_id"])
        self.assertEqual("Tin AI moi nhat", response["related_articles"][0]["title"])

    def test_frontend_top_k_filter_returns_latest_n_articles(self) -> None:
        chunks = [
            _chunk(article_id="old", published_at="2026-06-01T00:00:00+00:00"),
            _chunk(article_id="newest", published_at="2026-06-05T00:00:00+00:00"),
            _chunk(article_id="middle", published_at="2026-06-03T00:00:00+00:00"),
        ]
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore(chunks),
            ollama_client=None,
        )

        response = service.answer_structured(
            "tin AI moi nhat",
            top_k=2,
            filters={"top_k": 2, "time_range_days": 30, "topic": "technology_ai_internet"},
        )

        self.assertEqual(["newest", "middle"], [source["article_id"] for source in response["sources"]])
        self.assertEqual(2, len(response["sources"]))
        self.assertEqual(2, response["query_plan"]["latest_article_count"])

    def test_answer_structured_article_summary_uses_title_context(self) -> None:
        vector_store = _VectorStore([_chunk(article_id="article-summary")])
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=vector_store,
            ollama_client=None,
        )

        response = service.answer_structured("Tin AI moi nhat tom tat bai bao nay", top_k=3)

        self.assertEqual("article_summary", response["intent"])
        self.assertEqual("article_summary", response["query_plan"]["intent"])
        self.assertIn("article-summary", vector_store.get_chunks_calls)
        self.assertEqual("article-summary", response["sources"][0]["article_id"])
        self.assertIn("related_articles", response)

    def test_legacy_answer_api_still_returns_result_with_answer_string(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk()]),
            ollama_client=None,
        )

        response = service.answer("tin AI moi nhat", top_k=3)

        self.assertIsInstance(response, dict)
        self.assertIsInstance(response.get("answer"), str)

    def test_required_query_routes(self) -> None:
        ai_plan = route_query("tin AI moi nhat")
        self.assertIn(ai_plan["intent"], {"latest_news", "topic_news"})
        self.assertEqual("tech_ai_internet", ai_plan["primary_topic"])
        self.assertNotEqual("all", ai_plan["time_range"])

        world_plan = route_query("tinh hinh the gioi hom nay")
        self.assertEqual("world_geopolitics", world_plan["primary_topic"])
        self.assertIn(world_plan["time_range"], {"today", "24h"})

        stock_plan = route_query("HPG co gi moi")
        self.assertEqual("entity_news", stock_plan["intent"])
        self.assertIn("HPG", stock_plan["stock_symbols"])
        self.assertIn("HPG", stock_plan["entities"])

        real_estate_plan = route_query("bat dong san Ha Noi")
        self.assertEqual("real_estate", real_estate_plan["primary_topic"])

        image_plan = route_query("anh ve Ukraine hom nay")
        self.assertEqual("media_lookup", image_plan["intent"])
        self.assertTrue(image_plan["need_images"])
        self.assertIn("Ukraine", image_plan["entities"])

    def test_source_and_image_objects_have_required_fields(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore(
                [
                    _chunk(
                        article_id="article-ukraine",
                        topic="world_geopolitics",
                        entity_names="Ukraine",
                        has_images=True,
                    )
                ]
            ),
            ollama_client=None,
        )

        response = service.answer_structured("anh ve Ukraine hom nay", top_k=3)

        source = response["sources"][0]
        for key in ("citation_id", "id", "article_id", "title", "url", "source", "published_at", "primary_topic", "score", "snippet", "topic", "domain"):
            self.assertIn(key, source)
        self.assertEqual(1, source["citation_id"])
        self.assertEqual(1, source["id"])

        image = response["images"][0]
        for key in ("article_id", "citation_id", "image_url", "caption", "credit", "source", "article_title", "article_url", "is_representative", "type"):
            self.assertIn(key, image)
        self.assertEqual(1, image["citation_id"])
        self.assertEqual("original", image["type"])
        self.assertEqual([], response["generated_image_prompts"])

    def test_answer_structured_returns_generation_prompt_when_source_has_no_image(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk(article_id="article-no-image", has_images=False)]),
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3)

        self.assertEqual([], response["images"])
        prompts = response["generated_image_prompts"]
        self.assertEqual(1, len(prompts))
        self.assertEqual("article-no-image", prompts[0]["article_id"])
        self.assertEqual("Tin AI moi nhat", prompts[0]["article_title"])
        self.assertIn("neutral editorial illustration", prompts[0]["prompt"])
        self.assertIn("not a real photo", prompts[0]["prompt"])

    def test_answer_structured_uses_article_image_repository_before_generation_prompt(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk(article_id="article-repository-image", has_images=False)]),
            ollama_client=None,
            image_repository=_ImageRepository(),
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3)

        self.assertEqual([], response["generated_image_prompts"])
        self.assertEqual(1, len(response["images"]))
        self.assertEqual("https://example.com/article-repository-image-repository.jpg", response["images"][0]["image_url"])
        self.assertEqual("original", response["images"][0]["type"])

    def test_answer_structured_debug_rag_trace_has_audit_schema(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk(article_id="article-trace")]),
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3, debug_retrieval=True)

        trace = response["debug"]["rag_trace"]
        for key in ("query_plan", "router", "retrieval", "rerank", "context_builder", "prompt", "generation"):
            self.assertIn(key, trace)
        retrieval = trace["retrieval"]
        for key in (
            "raw_candidate_count",
            "candidate_count_before_topic_filter",
            "candidate_count_after_topic_filter",
            "candidate_count_after_metadata_filter",
            "fallback_used",
            "fallback_reason",
            "fallback_strategy",
            "top_candidates_before_filter",
            "top_candidates_after_metadata_filter",
            "topic_guard",
        ):
            self.assertIn(key, retrieval)
        self.assertIn("top_results", trace["rerank"])
        self.assertIn("score_breakdown_available", trace["rerank"])
        self.assertIn("context_char_count", trace["prompt"])
        self.assertIn("no_answer_reason", trace["generation"])
        self.assertIn("debug_trace", response["query_plan"])

    def test_multiple_chunks_same_article_create_one_citation_source(self) -> None:
        first = _chunk(article_id="article-same")
        second = _chunk(article_id="article-same")
        second["chunk_id"] = "article-same:1"
        second["text"] = "OpenAI co them thong tin trong cung mot bai."
        second["score"] = 0.75
        second["metadata"]["chunk_id"] = "article-same:1"
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([first, second]),
            ollama_client=None,
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3, debug_prompt=True)

        self.assertEqual(1, len(response["sources"]))
        self.assertEqual(1, response["sources"][0]["citation_id"])
        self.assertIn("[BÀI BÁO [1]]", response["prompt_debug"])
        self.assertEqual(1, response["prompt_debug"].count("Citation ID: [1]"))

    def test_answer_structured_query_plan_keeps_compatibility_domain_and_ticker(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk(article_id="article-hpg", topic="economy_finance_stock", entity_names="HPG")]),
            ollama_client=None,
        )

        response = service.answer_structured("HPG co gi moi", top_k=3)

        self.assertEqual("tai_chinh", response["query_plan"]["domain"])
        self.assertEqual("HPG", response["query_plan"]["ticker"])
        self.assertEqual("economy_finance_stock", response["topic"])

    def test_answer_structured_filters_invalid_images_from_response(self) -> None:
        chunk = _chunk(article_id="article-image", topic="world_geopolitics", entity_names="Ukraine", has_images=True)
        chunk["metadata"]["images_json"] = json.dumps(
            [
                {"article_id": "article-image", "image_url": "https://example.com/logo.svg", "caption": "Logo"},
                {"article_id": "article-image", "image_url": "https://example.com/news.jpg", "caption": "News"},
            ]
        )
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([chunk]),
            ollama_client=None,
        )

        response = service.answer_structured("anh ve Ukraine hom nay", top_k=3)

        self.assertEqual(["https://example.com/news.jpg"], [image["image_url"] for image in response["images"]])

    def test_invalid_citation_ids_are_reported_in_debug(self) -> None:
        service = RAGService(
            embedding_model=_EmbeddingModel(),
            vector_store=_VectorStore([_chunk()]),
            ollama_client=_OllamaClient("Noi dung tra loi co citation sai [99]."),
        )

        response = service.answer_structured("tin AI moi nhat", top_k=3)

        self.assertIn("debug", response)
        self.assertEqual(["citation_id_not_in_sources: [99]"], response["debug"]["citation_warnings"])

    def test_image_fallback_does_not_error(self) -> None:
        self.assertEqual([], MediaRetriever().get_images_for_articles(["missing-article"]))

    def test_unknown_topic_falls_back_to_default_processor(self) -> None:
        self.assertIsInstance(get_processor("unknown_topic"), DefaultNewsProcessor)
        self.assertIsInstance(get_processor(None), DefaultNewsProcessor)

    def test_chunk_metadata_is_chroma_compatible_and_serializes_json(self) -> None:
        article = ParentArticle(
            article_id="a1",
            source="cafef",
            url="https://example.com/a1",
            title="Title",
            summary="Summary",
            content="Content",
            published_at="2026-01-01T00:00:00+00:00",
            source_category="Cong nghe",
            primary_topic="tech_ai_internet",
            primary_topic_name="Cong nghe - AI - Internet",
            secondary_topics=["business_startup"],
            entities=[{"normalized_name": "OpenAI", "type": "organization"}],
            images=[{"image_url": "https://example.com/a1.jpg", "caption": "Image"}],
        )
        block = ArticleBlock("a1::b0", "a1", 0, "lead", "Content")

        metadata = ChunkMetadataBuilder().build_metadata(article, block, "a1::b0::c0", 0, "lead")

        required_keys = (
            "article_id",
            "chunk_id",
            "block_id",
            "title",
            "source",
            "url",
            "published_at",
            "primary_topic",
            "entities_json",
            "entity_names",
            "entity_types",
            "has_images",
            "images_json",
        )
        for key in required_keys:
            self.assertIn(key, metadata)
        self.assertIsInstance(metadata["entities_json"], str)
        self.assertIsInstance(metadata["images_json"], str)
        self.assertEqual([{"normalized_name": "OpenAI", "type": "organization"}], json.loads(metadata["entities_json"]))
        self.assertEqual([{"image_url": "https://example.com/a1.jpg", "caption": "Image"}], json.loads(metadata["images_json"]))
        for value in metadata.values():
            self.assertIsInstance(value, (str, int, float, bool))


if __name__ == "__main__":
    unittest.main()
