# Acceptance Validation

This project currently validates the local/backend RAG path. Streamlit has been removed from the repo; frontend validation should be done against the structured backend response when a separate frontend is added.

## Automated Validation

Run:

```powershell
python -m unittest discover -s tests
```

Expected:

```text
OK
```

The acceptance coverage includes:

- query routing for AI, world, stock symbol, real estate, and image/media queries
- structured RAG response schema
- legacy `answer()` path still returning a result object with an `answer` string
- source and image object fields
- media fallback without hallucinated image URLs
- domain processor fallback for unknown topics
- Chroma-compatible chunk metadata with JSON serialization for list/dict fields

## Manual Backend Checklist

1. Deploy/run Databricks jobs in order:

```powershell
databricks bundle run crawl_news_job
databricks bundle run parse_and_canonicalize_job
databricks bundle run build_articles_clean_job
```

2. Rebuild the local Chroma index after metadata/chunking changes:

```powershell
python -m app.local_ai.demo_chatbot --reset_index --index --limit 100
```

3. Test representative queries through the backend/CLI:

```powershell
python -m app.local_ai.demo_chatbot --question "tin AI moi nhat"
python -m app.local_ai.demo_chatbot --question "tinh hinh the gioi hom nay"
python -m app.local_ai.demo_chatbot --question "HPG co gi moi"
python -m app.local_ai.demo_chatbot --question "bat dong san Ha Noi"
python -m app.local_ai.demo_chatbot --question "anh ve Ukraine hom nay"
```

Expected:

- response has an answer
- structured responses have `sources`, `images`, and `related_articles`
- image queries do not crash when `images` is empty
- no image URL is invented if crawled metadata has no image
- missing or weak context is reported instead of hallucinated

## Databricks Notes

- No new schema change is required for this acceptance pass.
- If `articles_clean` or chunk metadata does not yet contain image fields, media retrieval can safely return `[]`.
- If `articles_clean` has stale rows, rerun parse/build jobs before rebuilding Chroma.
