# News AI / RAG Pipeline Audit Prompt

You are a senior RAG engineer.

Your task is not to answer the user question. Your task is to audit the entire retrieval pipeline and determine exactly where information was lost.

## Input

1. User Query
2. Final Answer
3. Sources
4. Retrieval Trace
5. Metadata Filter Trace
6. Rerank Trace
7. Prompt Context, if available

## Step 1 - User Intent Analysis

Determine:

- Primary intent
- Secondary intent
- Expected answer type

Output:

```text
Intent:
Expected Evidence:
Expected Source Types:
```

## Step 2 - Retrieval Audit

Inspect `retrieval_debug`.

Check:

- `raw_candidate_count`
- `retrieval_latency`
- `vector_hits`
- `bm25_hits`
- `hybrid_hits`

For every candidate, inspect:

- Relevance score
- Source
- Topic
- Publication date

Output:

```text
Retrieved Candidates:
Relevant Candidates:
Irrelevant Candidates:
Verdict: PASS / FAIL
```

## Step 3 - Metadata Filter Audit

Inspect `metadata_filter_debug`.

Check:

- `candidate_count_before_filter`
- `candidate_count_after_filter`
- Topic filter
- Entity filter
- Source filter
- Date filter

Questions:

- Did filtering remove relevant documents?
- If yes, list removed documents and explain why removal was incorrect.

Output:

```text
Filter Status: PASS / FAIL
Lost Relevant Documents:
```

## Step 4 - Topic Guard Audit

Inspect `topic_guard`.

Check:

- `detected_topic`
- `requested_topic`
- `rejection_reason`

Determine whether topic classification incorrectly routed the query.

Example:

```text
Query: HPG co gi moi
Detected: technology
Expected: stock/company
Verdict: FAIL
```

Output:

```text
Topic Classification:
Expected Topic:
Verdict:
```

## Step 5 - Reranking Audit

Inspect `rerank_debug`.

Compare:

- `top_candidates_before_rerank`
- `top_candidates_after_rerank`

Questions:

- Did reranking demote relevant articles?
- Did reranking keep irrelevant articles?

Output:

```text
Rerank Status: PASS / FAIL
Lost Relevant Results:
```

## Step 6 - Context Sufficiency

Using documents after rerank, determine:

- FULLY ANSWERABLE
- PARTIALLY ANSWERABLE
- NOT ANSWERABLE

Explain why.

## Step 7 - LLM Generation Audit

Inspect the prompt context.

Determine:

- Was sufficient evidence passed to the LLM?
- If yes, could the answer have been generated?

If the final answer says there is not enough information while context contains evidence, classify as:

- LLM Reasoning Failure
- Prompt Constraint Failure

Output:

```text
Generation Verdict: PASS / FAIL
```

## Step 8 - Failure Classification

Assign probabilities:

```text
A. Chroma Retrieval Failure:
B. BM25 Retrieval Failure:
C. Metadata Filter Failure:
D. Topic Guard Failure:
E. Reranking Failure:
F. Context Truncation:
G. Prompt Too Conservative:
H. LLM Reasoning Failure:
```

## Step 9 - Best Possible Answer

Using only retrieved evidence, generate the answer that should have been returned.

Never refuse unless no relevant documents exist.

## Step 10 - Root Cause

Summarize:

```text
ROOT CAUSE:

Component:
Reason:
Suggested Fix:
Priority: HIGH / MEDIUM / LOW
```
