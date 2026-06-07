# Cairn Dev Workflow Token/Cost/Speed Report

**Date:** 2026-06-06 20:01:29

**Mode:** WITH-LLM (embeddings enabled, embed=nomic-embed-text)

**Indexing time:** 465.40s

**Host resources:** CPU=12, RAM total=15.5GB, RAM avail=13.4GB, GPU=NVIDIA GeForce RTX 2060, VRAM=6.0GB

**Repo:** django (/mnt/c/Users/alfre/Projects/cairn-hardtest/corpus/django)

**Total indexed source tokens:** 1120.0k (850 files)

**Price per 1M input tokens:** \$3.0 (ESTIMATE)

## Methodology

Four baselines per task:
- **CAIRN**: tokens in `ContextAssembler.assemble(query)` — compressed context the agent receives
- **BASELINE_FILES**: tokens in the single ground-truth source file (what an agent reads in full)
- **BASELINE_RGDUMP**: tokens in up to 10 files found by `rg -l` with query content words
- **BASELINE_REPO**: tokens in ALL indexed source files (whole-repo dump)

Token counting uses `tiktoken` cl100k_base (proxy for Claude/Sonnet). Cost estimate: input tokens $ per 1M (configurable via `--price-per-mtok`). The **recall gate** checks whether the ground-truth symbol (e.g. `class Paginator`) appears in the CAIRN context; a miss is a quality failure.

## Per-Task Results

| # | Task | CAIRN tok | Base tok | Reduction% | RGDUMP tok (#files) | Recall | Latency | $ CAIRN | $ Base | $ Saved |
|---|------|-----------|----------|------------|----------------------|--------|---------|---------|--------|---------|
| 1 | how does QuerySet.filter build the query | 844 | 23.7k | 96.4% | 34.7k (10) | NO | 19.04s | $0.002532 | $0.071049 | $0.068517 |
| 2 | where and how is the CSRF token validated | 788 | 4.1k | 80.9% | 39.6k (10) | NO | 11.45s | $0.002364 | $0.012363 | $0.009999 |
| 3 | how does Paginator compute the number of pages | 861 | 3.3k | 74.2% | 51.4k (10) | YES | 18.18s | $0.002583 | $0.009996 | $0.007413 |
| 4 | how are URL patterns resolved to a view | 785 | 6.2k | 87.3% | 28.7k (10) | NO | 19.31s | $0.002355 | $0.018498 | $0.016143 |
| 5 | how does the model save() persist a row | 972 | 18.8k | 94.8% | 49.1k (10) | YES | 20.88s | $0.002916 | $0.056475 | $0.053559 |
| 6 | how is a QuerySet turned into SQL | 784 | 18.0k | 95.6% | 22.4k (10) | NO | 13.56s | $0.002352 | $0.053985 | $0.051633 |
| 7 | how does the template engine render a template | 763 | 9.1k | 91.6% | 52.1k (10) | YES | 12.58s | $0.002289 | $0.027186 | $0.024897 |
| 8 | how are forms validated | 826 | 3.3k | 74.7% | 61.9k (10) | NO | 21.43s | $0.002478 | $0.009789 | $0.007311 |
| 9 | how does the cache framework get/set values | 827 | 885 | 6.6% | 26.3k (10) | NO | 29.16s | $0.002481 | $0.002655 | $0.000174 |
| 10 | how does signing protect a value | 970 | 2.3k | 58.0% | 33.2k (10) | YES | 23.00s | $0.002910 | $0.006927 | $0.004017 |

## Aggregate

- **Reduction vs BASELINE_FILES:** mean=76.0%, median=84.1%
- **Recall:** 4/10 tasks passed the recall gate
- **Total CAIRN tokens:** 8.4k vs **total BASELINE_FILES tokens:** 89.6k
- **Estimated $ saved (vs BASELINE_FILES):** $0.243663 (CAIRN cost: $0.025260 vs base: $0.268923)
- **Median latency per CAIRN call:** 19.18s
- **CAIRN vs BASELINE_REPO ratio:** 0.8% (CAIRN uses 8.4k of 1120.0k repo tokens)
- **Freshness:** PASS (zzfreshmarker appeared in context after re-index)

## Findings & Limitations

- **Baseline fairness:** BASELINE_FILES is a single-file read — generous to the baseline, since a real agent would likely need 2-5 files to understand surrounding context. Cairn's reduction vs BASELINE_FILES is therefore a *conservative* lower bound on actual savings.

- **RGDUMP overestimates:** `rg -l` returns files containing query keywords but many are tangential (e.g. test files, docstrings mentioning the symbol). A real agent would not read all 10 files in full, making RGDUMP an upper bound on naive file-based retrieval cost.

- **WITH-LLM indexing:** This benchmark uses embeddings via ollama nomic-embed-text for semantic retrieval. Embedding-augmented retrieval typically improves recall but adds indexing cost.

- **Recall misses:** Tasks [1, 2, 4, 6, 8, 9] did not have the gt_symbol in CAIRN context. With embeddings the confidence guard may still reject if the semantic match is weak.

- **Ranking:** WITH-LLM mode uses lexical+structural retrieval + embedding-based retrieval + cross-encoder reranker. Semantic similarity improves query-concept-to-code mapping.

## Headline

Cairn reduces agent input context by **76.0% on average** (median 84.1%) vs reading the single source file in full, across 10 representative Django development tasks. Recall: **4/10**. Estimated cost saving: **$0.243663** per task set (at \$3.0/1M input tokens). Median assembly latency: **19.18s**.

