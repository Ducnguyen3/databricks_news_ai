# Chroma Surgery Repair Runbook

This repair is an internal Chroma SQLite/HNSW rollback for a local demo index. Databricks Gold remains the source of truth. Prefer a full Chroma rebuild when time allows.

Do not run `--apply` directly on `data/chroma`. The surgery script refuses that path by default.

## A. Backup

```powershell
Copy-Item data\chroma data\chroma_backup_before_surgery -Recurse
```

## B. Create Test Copy

```powershell
Copy-Item data\chroma data\chroma_surgery_test -Recurse
```

## C. Healthcheck Current Index

```powershell
python scripts\inspect_chroma_health.py --chroma-dir data/chroma
```

Expected current failure pattern:

```text
SQLite embeddings: 37410
HNSW id_to_label count: 36672
missing_in_hnsw: 738
```

## D. Dry Run On Test Copy

```powershell
python scripts\chroma_surgery_tail_rollback.py
```

The default target is:

```text
data/chroma_surgery_test
```

The dry run should report planned deletes for:

```text
embeddings id 36673..37410
embedding_metadata id 36673..37410
embedding_metadata_array id 36673..37410 if present
embeddings_queue seq_id > 36928
embedding_fulltext_search rowid 36673..37410
```

## E. Apply On Test Copy

```powershell
python scripts\chroma_surgery_tail_rollback.py --apply --vacuum
```

The script backs up `data/chroma_surgery_test/chroma.sqlite3` before modifying it.

## F. Verify Test Copy

```powershell
python scripts\inspect_chroma_health.py --chroma-dir data/chroma_surgery_test
python scripts\test_chroma_client.py --chroma-dir data/chroma_surgery_test
```

Acceptance target:

```text
SQLite embeddings: 36672
embeddings_queue has no seq_id > 36928
collection.count() returns 36672
collection.get(limit=3) returns documents and metadatas
```

## G. Replace Only If Test Copy Works

```powershell
Rename-Item data\chroma data\chroma_broken_before_surgery
Rename-Item data\chroma_surgery_test chroma
```

Then verify the live path:

```powershell
python scripts\test_chroma_client.py --chroma-dir data/chroma
python -m app.local_ai.rag_smoke_test
```

## H. If Surgery Still Fails

Do not keep editing SQLite manually. Rebuild Chroma from Gold:

```powershell
Rename-Item data\chroma data\chroma_failed_surgery
python -m app.local_ai.index_sync --rebuild_mode full
```

## Notes

- This is not an official Chroma repair API.
- The active HNSW folder `792fc08a-52f7-44bb-8c67-712399c5afdb` is referenced by SQLite and must not be deleted alone.
- The orphan folder `acca2986-5130-447b-af09-3bc4edab4724` can be removed after backup, but doing so does not fix the active HNSW mismatch.
- If Chroma cannot load the active HNSW segment after rollback, the only reliable path is a full rebuild from `main.news_ai.articles_clean`.
