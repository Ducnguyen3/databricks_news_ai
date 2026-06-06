# End-to-End Databricks + Local RAG Runbook

This runbook describes the full data flow from crawling news websites to querying the local RAG index.

## Pipeline Overview

```text
News websites
-> Bronze: main.news_ai.news_raw_documents
-> Silver: main.news_ai.news_articles
-> Gold: main.news_ai.articles_clean
-> Local ChromaDB: data/chroma / news_articles
-> RAG API or CLI query
```

Databricks is responsible for crawl, parse, clean, deduplicate, and Gold table build. The local/backend side is responsible for chunking, embedding, Chroma indexing, and RAG. Chroma is a derived index and can always be rebuilt from Gold.

## Environment

Required for local Chroma sync from Databricks SQL:

```text
DATABRICKS_SERVER_HOSTNAME
DATABRICKS_HTTP_PATH
DATABRICKS_TOKEN
```

Local RAG defaults:

```text
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=news_articles
LOCAL_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=700
CHUNK_OVERLAP=120
```

## Step 1: Crawl Raw Documents

Databricks job entrypoint:

```powershell
python -m app.jobs.crawl_news_job
```

Databricks bundle command:

```powershell
databricks bundle run crawl_news_job
```

Output table:

```text
main.news_ai.news_raw_documents
```

## Step 2: Parse, Canonicalize, Deduplicate

```powershell
python -m app.jobs.parse_and_canonicalize_job
```

Databricks bundle command:

```powershell
databricks bundle run parse_and_canonicalize_job
```

Output table:

```text
main.news_ai.news_articles
main.news_ai.news_article_images
```

## Step 3: Build Gold

```powershell
python -m app.jobs.build_articles_clean_job
```

Databricks bundle command:

```powershell
databricks bundle run build_articles_clean_job
```

Output table:

```text
main.news_ai.articles_clean
```

## Step 4: Validate Gold

Run these SQL checks in Databricks:

```sql
SELECT source, COUNT(*) AS total
FROM main.news_ai.articles_clean
GROUP BY source
ORDER BY total DESC;
```

```sql
SELECT primary_topic_name, COUNT(*) AS total
FROM main.news_ai.articles_clean
GROUP BY primary_topic_name
ORDER BY total DESC;
```

```sql
SELECT COUNT(*) AS total_articles
FROM main.news_ai.articles_clean;
```

```sql
SELECT COUNT(*) AS articles_with_entities
FROM main.news_ai.articles_clean
WHERE entities_json IS NOT NULL AND entities_json <> '[]';
```

```sql
SELECT COUNT(*) AS article_images
FROM main.news_ai.news_article_images;
```

## Step 5: Sync Local Chroma

First run, empty Chroma, changed embedding model, or changed chunking logic:

```powershell
python -m app.local_ai.index_sync --rebuild_mode full
```

Normal run after new crawl/Gold refresh:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental
```

Fast validation run:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental --limit 100
```

Useful filters:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental --source cafef
python -m app.local_ai.index_sync --rebuild_mode incremental --topic economy_finance_stock
```

## Step 6: Check Chroma Sync Stats

The CLI output should include:

```text
CHROMA INDEX SYNC
Mode
Gold table
Chroma path
Collection
Embedding model
Chunking version
Index version
Articles loaded
Articles skipped
Articles indexed
Articles reindexed
Chunks generated
Chunks upserted
Articles with images
Chunks with images
Index size
Duration
Status: OK
```

`Index size` should be greater than zero after a successful full or incremental sync with new content.

## Step 7: Audit Chunk Quality

Run chunk audit against the current local Chroma collection:

```powershell
python -m app.local_ai.chunk_quality_audit
```

Save a JSON audit report:

```powershell
python -m app.local_ai.chunk_quality_audit --save_report data/chunk_quality_report.json
```

Use this before smoke test and quality eval when chunking or metadata changed.

## Step 8: Retrieval Mode

Hybrid retrieval is the default:

```powershell
$env:RAG_RETRIEVAL_MODE="hybrid"
```

Vector-only mode is available for comparison:

```powershell
$env:RAG_RETRIEVAL_MODE="vector"
```

Compare quality reports:

```powershell
$env:RAG_RETRIEVAL_MODE="vector"
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_vector.json

$env:RAG_RETRIEVAL_MODE="hybrid"
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_hybrid.json
```

## Step 9: Smoke Test RAG Queries

Run the structured smoke-test CLI:

```powershell
python -m app.local_ai.rag_smoke_test
```

Run one query:

```powershell
python -m app.local_ai.rag_smoke_test --query "HPG co gi moi"
```

Print JSON:

```powershell
python -m app.local_ai.rag_smoke_test --json
```

Run quality evaluation after the smoke test:

```powershell
python -m app.local_ai.rag_quality_eval
```

Save a JSON quality report:

```powershell
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_report.json
```

Representative queries:

```text
tin AI moi nhat
tinh hinh the gioi hom nay
HPG co gi moi
tin bat dong san gan day
cho toi cac bai co anh ve Ukraine
tin tu CafeF ve chung khoan
tong hop tin doanh nghiep moi nhat tu nhieu nguon
```

For each query, verify:

- `intent` is reasonable.
- `topic` maps to one of the seven standard topics.
- `sources[]` includes `title`, `url`, `source`, and `published_at`.
- `images[]` is populated when the query needs images and image metadata exists.
- `related_articles[]` is populated from parent-child retrieval when evidence exists.
- If there is not enough evidence, the answer says the index has no suitable data instead of hallucinating.

Read the smoke-test output:

- `Intent` and `Topic` should match the query.
- `Sources` should be non-empty when evidence exists.
- `Images` should be non-empty for image queries when image data exists.
- `Related articles` should be populated by parent-child retrieval when evidence exists.
- `WARN` usually means schema is valid but sources are empty.
- `FAIL` means crash, missing schema field, or wrong field type.

## Standard Flow After Each Crawl

```powershell
python -m app.jobs.crawl_news_job
python -m app.jobs.parse_and_canonicalize_job
python -m app.jobs.build_articles_clean_job
python -m app.local_ai.index_sync --rebuild_mode incremental
python -m app.local_ai.chunk_quality_audit
python -m app.local_ai.rag_smoke_test
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_hybrid.json
python -m unittest discover -s tests
```

If those jobs are executed on Databricks, use the equivalent bundle/job commands for the first three steps, then run the local index sync command from the backend machine.

## When To Run Full Rebuild

Run full rebuild if any of these change:

- `LOCAL_EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- semantic chunking logic
- recursive chunking logic
- chunk metadata contract
- `embedding_text` format

Command:

```powershell
python -m app.local_ai.index_sync --rebuild_mode full
```

Do not use incremental in these cases because old vectors are no longer comparable with new vectors.

## Troubleshooting

### Cannot Connect To Databricks SQL

Likely causes:

- wrong `DATABRICKS_SERVER_HOSTNAME`
- wrong `DATABRICKS_HTTP_PATH`
- expired `DATABRICKS_TOKEN`
- wrong Databricks CLI profile

Check:

```powershell
databricks auth profiles
```

### Gold Has New Articles But RAG Does Not Find Them

Likely causes:

- incremental index sync was not run
- source/topic filter excluded the articles
- `content_hash` did not change
- backend points to the wrong Chroma path

Fix:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental
```

If still wrong:

```powershell
python -m app.local_ai.index_sync --rebuild_mode full
```

### Query Returns Only One Source

Likely causes:

- not enough articles from other sources in Gold
- top-k too small
- candidate pool before rerank too small
- reranker/source diversity needs tuning

Check:

- `app/local_ai/retriever.py`
- `RAG_RETRIEVE_TOP_N`
- `RAG_BROAD_RETRIEVE_TOP_N`
- source distribution in `articles_clean`

### Image Query Returns No Images

Likely causes:

- `news_article_images` has no rows
- image enrichment did not run before chunking
- `images_json` is missing from chunk metadata
- query router did not set `need_images=True`

Check:

```sql
SELECT COUNT(*) FROM main.news_ai.news_article_images;
```

Then rerun:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental
```

## Acceptance Checklist

Databricks:

- `news_raw_documents` has rows.
- `news_articles` has rows.
- `articles_clean` has rows.
- `primary_topic` is populated.
- `entities_json` is populated when entities exist.
- `news_article_images` has rows when crawler extracts images.

Local Chroma:

- collection `news_articles` exists.
- index size is greater than zero.
- chunks have `article_id`.
- chunks have `source`.
- chunks have `primary_topic`.
- chunks have `entities_json`.
- chunks have `images_json` when articles have images.

RAG:

- `answer_structured()` returns the required schema.
- `sources[]` is not empty when evidence exists.
- `images[]` appears when image evidence exists.
- fallback works when there is no evidence.
- `related_articles[]` is populated by parent-child retrieval.
