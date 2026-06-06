# RAG Structured Smoke Test

The smoke test checks `answer_structured()` against the current local Chroma index. It does not crawl, rebuild Gold, or rebuild Chroma.

Run this before `rag_quality_eval`. Smoke test checks that the structured RAG path runs and returns the right schema. Quality eval checks retrieval and answer quality against expectations.

## When To Run

Run it:

- after full Chroma rebuild
- after incremental Chroma sync
- after retriever, reranker, router, or parent-child retrieval changes
- before frontend demos

## Commands

Default query set:

```powershell
python -m app.local_ai.rag_smoke_test
```

Single query:

```powershell
python -m app.local_ai.rag_smoke_test --query "HPG co gi moi"
```

Multiple queries:

```powershell
python -m app.local_ai.rag_smoke_test --query "HPG co gi moi" --query "tin AI moi nhat"
```

Limit output:

```powershell
python -m app.local_ai.rag_smoke_test --max_sources 5 --max_images 5 --max_related 5
```

JSON output:

```powershell
python -m app.local_ai.rag_smoke_test --json
```

Strict mode:

```powershell
python -m app.local_ai.rag_smoke_test --strict
```

## Status Meaning

- `OK`: schema is valid and no warnings were found.
- `WARN`: schema is valid, but sources are empty or another non-fatal warning exists.
- `FAIL`: command crashed, response is not a dict, required fields are missing, or list fields have wrong types.

## Debug Guide

### Sources Empty

Check:

- Chroma index size
- query topic filter
- source/topic metadata
- retriever candidate pool

### Topic Wrong

Check:

- `app/local_ai/query_router.py`
- `app/processing/taxonomy.py`
- article `primary_topic` in Gold

### Intent Wrong

Check query keyword rules in `query_router.py`.

### Images Empty

Check:

- `news_article_images`
- `images_json` in chunk metadata
- `need_images` in query plan
- image enrichment before Chroma sync

### Related Articles Empty

Check:

- `ParentArticleLoader`
- `get_chunks_by_article_id`
- chunk metadata `article_id`

### Too Many Fallback Answers

Check:

- Chroma sync ran after Gold refresh
- vector index path
- evidence threshold in `rag_service.py`
- embedding model compatibility

## Demo Checklist

- `python -m app.local_ai.index_sync --rebuild_mode incremental` completed with `Status: OK`.
- `Index size` is greater than zero.
- `python -m app.local_ai.rag_smoke_test` returns no `FAIL`.
- `python -m app.local_ai.rag_quality_eval` returns no `FAIL`.
- Key queries have sources.
- Image query returns images when image data exists.
- `related_articles` appears for queries with evidence.
