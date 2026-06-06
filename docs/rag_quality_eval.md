# RAG Quality Evaluation

Quality eval checks retrieval and answer quality against a small expectation fixture. It does not crawl, rebuild Gold, rebuild Chroma, or call Databricks jobs.

## Smoke Test vs Quality Eval

Smoke test checks that `answer_structured()` runs and returns the required schema.

Quality eval checks whether retrieval, routing, sources, images, related articles, recency, and answer quality match query expectations.

## Commands

Default fixture:

```powershell
python -m app.local_ai.rag_quality_eval
```

Custom fixture:

```powershell
python -m app.local_ai.rag_quality_eval --queries tests/fixtures/rag_quality_queries.json
```

JSON output:

```powershell
python -m app.local_ai.rag_quality_eval --json
```

Save report:

```powershell
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_report.json
```

Strict mode:

```powershell
python -m app.local_ai.rag_quality_eval --strict
```

## Status Meaning

- `OK`: required quality checks pass.
- `WARN`: the pipeline works, but quality needs review.
- `FAIL`: schema/runtime error, empty answer, or high hallucination risk.

## Debug Guide

- `wrong_intent`: check `app/local_ai/query_router.py`.
- `wrong_topic`: check `app/local_ai/query_router.py`, `app/processing/taxonomy.py`, and Gold `primary_topic`.
- `empty_sources`: check Chroma index size, filters, and evidence threshold.
- `multi_source_weak`: check candidate pool, reranker, and source diversity.
- `images_missing`: check `news_article_images`, chunk `images_json`, `MediaRetriever`, and `need_images`.
- `weak_answer`: check context building, domain processors, prompt, and Ollama availability.
- `stale_sources`: check crawl freshness, Gold refresh, and recency scoring.

## Demo Checklist

- `python -m app.local_ai.index_sync --rebuild_mode incremental` finished with `Status: OK`.
- `python -m app.local_ai.rag_smoke_test` has no `FAIL`.
- `python -m app.local_ai.rag_quality_eval` has no `FAIL`.
- WARN items are understood before demo.
- Image query returns images if image metadata exists in Chroma.
