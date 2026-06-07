# Cairn Full Test Matrix Report

**Date:** 2026-06-06T23:54:26.732225+00:00
**Modes:** all
**Total planned entries:** 63

## Host Resources
- CPU cores: 12
- RAM total: 15.5 GB
- RAM available: 12.4 GB
- GPU: NVIDIA GeForce RTX 2060
- VRAM free: 2.9 GB
- gemma4:latest availability: present
- lancedb importable: True
- fastembed importable: True

## Results

| group | capability | variant | clean? | result | evidence | elapsed_s | notes |
|-------|-----------|---------|--------|--------|----------|-----------|-------|
| G1 | init (no LLM) | without LLM | yes | PASS | .cairn exists: True | 1.0 |  |
| G1 | init (with LLM) | with LLM | yes | PASS | .cairn exists: True, rc=0 | 2.8 |  |
| G1 | config | once | yes | PASS | rc=0, output_len=878 | 0.6 |  |
| G1 | profile (show+set) | once | yes | PASS | profile show rc=0, set rc=0 | 0.8 |  |
| G1 | status | once | yes | PASS | rc=0, output_len=194 | 1.2 |  |
| G1 | doctor (no LLM) | without LLM | yes | PASS | rc=0 | 1.7 |  |
| G1 | doctor (with LLM) | with LLM | yes | PASS | rc=0 | 47.3 |  |
| G1 | reindex quick (no LLM) | without LLM | yes | PASS | blocks=8, rc=0 | 1.3 |  |
| G1 | reindex full (no LLM) | without LLM | yes | PASS | rc=0 | 1.3 |  |
| G1 | search (no LLM) | without LLM | yes | PASS | rc=0, output_len=668 | 1.6 |  |
| G1 | search (with LLM) | with LLM | yes | PASS | rc=0, output_len=1008 | 2.5 |  |
| G1 | dry-run (no LLM) | without LLM | yes | PASS | rc=0 | 1.6 |  |
| G1 | suggest-local | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | token-stats | once | yes | PASS | rc=0 | 0.6 |  |
| G1 | token-history | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | cache stats | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | cache clear | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | memory status | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | memory clear | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | metrics | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | dashboard | once | yes | PASS | rc=0 | 0.7 |  |
| G1 | janitor start+stop | once | yes | PASS | start spawned, stop rc=0 | 2.5 |  |
| G1 | mcp smoke | once | yes | PASS | MCP process started, returncode=-15 | 3.2 | stderr_len=0 |
| G1 | run (start-all) | once | yes | PASS | rc=0 | 1.2 |  |
| G1 | dry-run --show-prompt | without LLM | yes | PASS | rc=0, has_context=True | 1.7 |  |
| G2 | search_code | without LLM | yes | PASS | result_len=585, has_process=True | 0.9 |  |
| G2 | assemble_context | without LLM | yes | PASS | result_len=801 | 0.2 |  |
| G2 | set_profile | without LLM | yes | PASS | response: Profile set to: iac. Retrieval: hybrid (structural, lexical). Embeddings: OFF. Config updated. | 0.2 |  |
| G2 | orchestrate (context-only) | without LLM | yes | PASS | result_len=801 | 0.2 |  |
| G2 | orchestrate (with instruction) | with LLM | yes | PASS | WorkClass=local_map_reduce, tokens=4921, chunks=2 | 42.6 | reason: split into 2 chunks; result_len=968 |
| G2 | cache_set/cache_get | without LLM | yes | PASS | set=cached, get_hit=yes, miss=CACHE_MISS | 0.2 |  |
| G2 | list_repos (single) | without LLM | yes | PASS | result_len=55 | 0.2 |  |
| G2 | list_repos (workspace) | without LLM | yes | PASS | result_len=143 | 0.1 |  |
| G2 | remember+recall (all kinds) | without LLM | yes | PASS | sections_found=5/5 | 0.2 | result_len=422 |
| G3 | store_backend chroma | chroma | yes | PASS | blocks=8, search_len=586 | 0.2 |  |
| G3 | store_backend lance | lance | yes | PASS | blocks=8, search_len=585 | 0.2 |  |
| G3 | index_location in_project | in_project | yes | PASS | blocks=8, index_in_project=True | 0.2 | path=/tmp/claude-1000/cairn_matrix_ad2gu0xu/g3_loc_project/python-repo/.cairn/chroma |
| G3 | index_location native | native | yes | PASS | blocks=8, native_index_exists=True | 0.2 | path=/home/alfredo/.cache/cairn/05f59ba0b17e/chroma |
| G3 | profile iac | explicit | yes | PASS | set=iac, detected=iac, embeddings=False, legs=['structural', 'lexical'] | 0.2 |  |
| G3 | profile code | explicit | yes | PASS | set=code, detected=python, embeddings=True, legs=['embeddings', 'lexical', 'structural'] | 0.2 |  |
| G3 | profile python | explicit | yes | PASS | set=python, detected=python, embeddings=True, legs=['embeddings', 'lexical', 'structural'] | 0.2 |  |
| G3 | profile shell | explicit | yes | PASS | set=shell, detected=python, embeddings=False, legs=['structural', 'lexical'] | 0.2 |  |
| G3 | profile auto-detect helm | auto | yes | PASS | detected=iac, expected=iac | 0.2 | ext_counts={'.yaml': 5, '.json': 2, '.sh': 1} |
| G3 | profile auto-detect python | auto | yes | PASS | detected=python, expected=python | 0.2 | ext_counts={'.py': 3, '.json': 1, '.yaml': 1} |
| G3 | embed mode placeholder | no-LLM | yes | PASS | blocks=8, search_len=582 | 0.2 |  |
| G3 | embed mode fastembed | no-LLM | yes | PASS | blocks=8, search_len=874 | 0.6 |  |
| G3 | embed mode ollama | with LLM | yes | PASS | blocks=8, search_len=878 | 0.6 |  |
| G3 | reranker cross_encoder | no-LLM | yes | PASS | result_len=802 | 0.2 |  |
| G3 | reranker none | no-LLM | yes | PASS | result_len=804 | 0.2 |  |
| G3 | reranker llm | with LLM | yes | PASS | result_len=1502 | 139.8 |  |
| G3 | compression none | none | yes | PASS | before=213, after=213, reduction=0.0% | 0.2 | level=none |
| G3 | compression minimal | minimal | yes | PASS | before=219, after=102, reduction=53.4% | 0.2 | level=minimal |
| G3 | compression aggressive | aggressive | yes | PASS | before=131, after=48, reduction=63.4% | 0.2 | level=aggressive |
| G3 | memory.scope repo | repo | yes | PASS | found=yes | 0.2 | result_len=148 |
| G3 | memory.scope workspace | workspace | yes | PASS | found=yes | 0.6 | result_len=160 |
| G3 | language python | fixture | yes | PASS | blocks=8 | 0.2 | lang=python |
| G3 | language helm-yaml | fixture | yes | PASS | blocks=10 | 0.2 | lang=helm-yaml |
| G3 | language terraform-hcl | fixture | yes | PASS | blocks=14 | 0.2 | lang=terraform-hcl |
| G3 | language k8s-yaml | fixture | yes | PASS | blocks=4 | 0.2 | lang=k8s-yaml |
| G3 | workspace fan-out search_all | multi-repo | yes | PASS | result_len=1567, has_deployment=True | 0.8 |  |
| G3 | failsafe empty repo | edge | yes | PASS | blocks=0 | 0.1 | empty repo, no crash |
| G3 | failsafe pathological k8s | edge | yes | PASS | blocks=505 | 0.7 | pathological k8s, no crash |
| G3 | failsafe malformed config | edge | yes | PASS | clean error: ParserError | 0.0 | malformed config handled gracefully |

## Summary

**Total entries:** 63  |  PASS: 63  |  FAIL: 0  |  SKIP: 0

### By Group

| Group | PASS | FAIL | SKIP |
|-------|------|------|------|
| G1 | 25 | 0 | 0 |
| G2 | 9 | 0 | 0 |
| G3 | 29 | 0 | 0 |

### Failures

No failures.

_Report completed at 2026-06-06T23:58:57.339551+00:00_
