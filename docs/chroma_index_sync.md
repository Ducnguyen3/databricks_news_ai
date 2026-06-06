# Chroma Index Sync

For the full crawl -> Gold -> Chroma -> RAG flow, see [end_to_end_runbook.md](end_to_end_runbook.md).

Gold is the source of truth. In this project, Gold is the Databricks Delta table:

```text
main.news_ai.articles_clean
```

ChromaDB is only a derived local/backend vector index:

```text
data/chroma
```

Default collection:

```text
news_articles
```

## Full Rebuild

Use full rebuild when:

- Chroma index is empty or corrupted.
- The embedding model changed.
- Chunking logic changed.
- You want to rebuild all vectors from Gold.

```powershell
python -m app.local_ai.index_sync --rebuild_mode full
```

Full mode resets the Chroma collection, loads articles from Gold, enriches images, chunks, embeds, and upserts all chunks.

## Incremental Rebuild

Use incremental rebuild after normal crawl/parse/Gold refresh.

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental
```

Incremental mode reads indexed article hashes from Chroma, skips unchanged articles, indexes new articles, and delete/reindexes articles whose `content_hash` changed.

Useful filters:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental --limit 100
python -m app.local_ai.index_sync --rebuild_mode incremental --source cafef
python -m app.local_ai.index_sync --rebuild_mode incremental --topic economy_finance_stock
python -m app.local_ai.index_sync --rebuild_mode incremental --topic kinh_te_tai_chinh_chung_khoan
```

Dry run:

```powershell
python -m app.local_ai.index_sync --rebuild_mode incremental --dry_run
```

Dry run computes what would be indexed but does not reset, embed, delete, or upsert Chroma data.

## Standard Flow After Crawling

```text
crawl raw
parse/clean/dedup
build articles_clean
run incremental index sync
test RAG query
```

## Quick Checks

Run tests:

```powershell
python -m unittest discover -s tests
```

Try RAG queries after sync:

```text
tin AI moi nhat
HPG co gi moi
tinh hinh the gioi hom nay
cho toi bai co anh ve Ukraine
```

## Notes

- Chroma is not the main database.
- Do not manually edit Chroma as source data.
- If Chroma is deleted, rebuild it from Gold.
- Gold schema does not need to change for index sync.
