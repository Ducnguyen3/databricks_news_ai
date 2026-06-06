# Chunk Quality Audit

Chunk audit checks the current local Chroma index. It does not crawl, rebuild Gold, rebuild Chroma, call Databricks jobs, or change RAG behavior.

## Purpose

Use it to inspect whether indexed chunks are healthy before tuning retrieval or answer generation:

- chunk length distribution
- metadata completeness
- JSON metadata validity
- image metadata consistency
- chunks per article
- duplicate chunks
- boundary warnings
- readable sample chunks

## Chunk Audit vs RAG Quality Eval

Chunk audit checks the index and chunk metadata.

RAG quality eval checks the output of `answer_structured()` for real query expectations.

## When To Run

Run chunk audit:

- after a full Chroma rebuild
- after changing chunking logic
- after changing `metadata_builder`
- after changing image/entity metadata
- before tuning retriever or reranker

## Commands

Default:

```powershell
python -m app.local_ai.chunk_quality_audit
```

More samples:

```powershell
python -m app.local_ai.chunk_quality_audit --sample 10
```

JSON output:

```powershell
python -m app.local_ai.chunk_quality_audit --json
```

Save report:

```powershell
python -m app.local_ai.chunk_quality_audit --save_report data/chunk_quality_report.json
```

Custom length thresholds:

```powershell
python -m app.local_ai.chunk_quality_audit --min_chars 100 --max_chars 2500
```

Filter by source or topic:

```powershell
python -m app.local_ai.chunk_quality_audit --source cafef
python -m app.local_ai.chunk_quality_audit --topic economy_finance_stock
```

## Status Meaning

- `OK`: index is non-empty, critical metadata is mostly present, and length/duplicate ratios are acceptable.
- `WARN`: index is readable, but chunk quality or optional metadata should be reviewed.
- `FAIL`: Chroma cannot be read, collection is empty, or critical metadata is missing too often.

## Fix Guide

### Chunk Too Short

Check semantic block merging and recursive splitter minimum size. Too many short chunks usually weakens retrieval context.

### Chunk Too Long

Check recursive splitter max size and separators. Too many long chunks can dilute embeddings.

### Missing Metadata

Check `app/local_ai/chunking/metadata_builder.py` and index sync metadata fields.

### Invalid images_json or entities_json

Ensure metadata values are serialized JSON strings, not raw list/dict objects.

### Duplicate Ratio High

Check deduplication in Gold and stable `chunk_id` generation before Chroma upsert.

### Boundary Warnings High

Review semantic chunking boundaries and recursive splitter separators. Boundary warnings are advisory, not automatic failures.
