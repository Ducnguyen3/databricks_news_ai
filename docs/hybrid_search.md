# Hybrid Search

Hybrid search combines vector retrieval from Chroma with an in-memory BM25 lexical index built from the same Chroma chunks.

Gold remains the source of truth. Chroma remains the derived vector index. BM25 is also derived and in-memory only.

## Why Hybrid Search

Vector search is strong for semantic queries:

- `tin AI moi nhat`
- `tinh hinh the gioi hom nay`
- `thi truong bat dong san ra sao`

BM25/lexical search is strong for exact terms:

- tickers such as `HPG`, `FPT`, `VNM`
- index names such as `VN-Index`
- organizations such as `OpenAI`, `Nvidia`
- countries such as `Ukraine`
- sources such as `CafeF`
- finance keywords such as `lai suat`, `ty gia`, `vang`, `chung khoan`

## Flow

```text
query router
-> vector candidates from Chroma
-> BM25 candidates from Chroma chunks
-> merge by chunk_id
-> metadata filter
-> rerank
-> diversify by article/source
-> parent-child retrieval
-> answer_structured()
```

## Retrieval Mode

Hybrid is the default:

```powershell
$env:RAG_RETRIEVAL_MODE="hybrid"
```

Vector-only mode:

```powershell
$env:RAG_RETRIEVAL_MODE="vector"
```

Use vector-only mode to compare before/after quality:

```powershell
$env:RAG_RETRIEVAL_MODE="vector"
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_vector.json

$env:RAG_RETRIEVAL_MODE="hybrid"
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_hybrid.json
```

## Debug

### BM25 Has No Result

Check chunk text, title, entities metadata, and token normalization.

### Vector Has Result But BM25 Does Not

The query may be semantic rather than lexical. This is acceptable.

### Source Is Wrong

Check `preferred_sources` in `query_plan` and source metadata in Chroma.

### Entity Query Is Weak

Check entity extraction, ticker detection, `entities_json`, `entity_names`, and BM25 search text.

### Multi-Source Is Weak

Check diversification limits and candidate pool size before rerank.

## Validation

```powershell
python -m app.local_ai.chunk_quality_audit
python -m app.local_ai.rag_smoke_test
python -m app.local_ai.rag_quality_eval --save_report data/rag_quality_hybrid.json
```
