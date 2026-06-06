# Cairn Full Test Matrix Report

**Date:** 2026-06-06T23:01:07.239729+00:00
**Modes:** all
**Total planned entries:** 63

## Host Resources
- CPU cores: 12
- RAM total: 15.5 GB
- RAM available: 12.7 GB
- GPU: NVIDIA GeForce RTX 2060
- VRAM free: 2.6 GB
- gemma4:latest availability: present
- lancedb importable: True
- fastembed importable: True

## Results

| group | capability | variant | clean? | result | evidence | elapsed_s | notes |
|-------|-----------|---------|--------|--------|----------|-----------|-------|
| G1 | init (no LLM) | without LLM | yes | PASS | .cairn exists: True | 1.2 |  |
| G1 | init (with LLM) | with LLM | yes | PASS | .cairn exists: True, rc=0 | 7.6 |  |
| G1 | config | once | yes | PASS | rc=0, output_len=878 | 0.5 |  |
| G1 | profile (show+set) | once | yes | PASS | profile show rc=0, set rc=0 | 0.7 |  |
| G1 | status | once | yes | PASS | rc=0, output_len=194 | 1.1 |  |
| G1 | doctor (no LLM) | without LLM | yes | PASS | rc=0 | 2.3 |  |
| G1 | doctor (with LLM) | with LLM | yes | PASS | rc=0 | 13.5 |  |
| G1 | reindex quick (no LLM) | without LLM | yes | PASS | blocks=8, rc=0 | 1.3 |  |
| G1 | reindex full (no LLM) | without LLM | yes | PASS | rc=0 | 1.2 |  |
| G1 | search (no LLM) | without LLM | yes | PASS | rc=0, output_len=1023 | 1.7 |  |
| G1 | search (with LLM) | with LLM | yes | PASS | rc=0, output_len=1093 | 2.3 |  |
| G1 | dry-run (no LLM) | without LLM | yes | PASS | rc=0 | 1.7 |  |
| G1 | suggest-local | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | token-stats | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | token-history | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | cache stats | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | cache clear | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | memory status | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | memory clear | once | yes | PASS | rc=0 | 0.5 |  |
| G1 | metrics | once | yes | PASS | rc=0 | 0.4 |  |
| G1 | dashboard | once | yes | PASS | rc=0 | 0.7 |  |
| G1 | janitor start+stop | once | yes | PASS | start spawned, stop rc=0 | 2.4 |  |
| G1 | mcp smoke | once | yes | PASS | MCP process started, returncode=-15 | 3.2 | stderr_len=0 |
| G1 | run (start-all) | once | yes | PASS | rc=0 | 1.3 |  |
| G1 | dry-run --show-prompt | without LLM | yes | PASS | rc=0, has_context=True | 2.0 |  |
| G2 | search_code | without LLM | yes | PASS | result_len=903, has_process=True | 0.9 |  |
| G2 | assemble_context | without LLM | yes | PASS | result_len=1658 | 0.3 |  |
| G2 | set_profile | without LLM | yes | PASS | response: Profile set to: iac. Retrieval: hybrid (structural, lexical). Embeddings: OFF. Config updated. | 0.2 |  |
| G2 | orchestrate (context-only) | without LLM | yes | PASS | result_len=1658 | 0.3 |  |
| G2 | orchestrate (with instruction) | with LLM | yes | PASS | WorkClass=local_map_reduce, tokens=4921, chunks=2 | 48.7 | reason: split into 2 chunks; result_len=918 |
| G2 | cache_set/cache_get | without LLM | yes | PASS | set=cached, get_hit=yes, miss=CACHE_MISS | 0.2 |  |
| G2 | list_repos (single) | without LLM | yes | PASS | result_len=55 | 0.2 |  |
| G2 | list_repos (workspace) | without LLM | yes | PASS | result_len=143 | 0.1 |  |
| G2 | remember+recall (all kinds) | without LLM | yes | PASS | sections_found=5/5 | 0.2 | result_len=422 |
| G3 | store_backend chroma | chroma | yes | PASS | blocks=8, search_len=906 | 0.3 |  |
| G3 | store_backend lance | lance | yes | PASS | blocks=8, search_len=903 | 0.3 |  |
| G3 | index_location in_project | in_project | yes | PASS | blocks=8, index_in_project=True | 0.2 | path=/tmp/claude-1000/cairn_matrix_nvo8n8ya/g3_loc_project/python-repo/.cairn/chroma |
| G3 | index_location native | native | yes | PASS | blocks=8, native_index_exists=True | 0.2 | path=/home/alfredo/.cache/cairn/4babb3d764b4/chroma |
| G3 | profile iac | explicit | yes | PASS | set=iac, detected=iac, embeddings=False, legs=['structural', 'lexical'] | 0.3 |  |
| G3 | profile code | explicit | yes | PASS | set=code, detected=python, embeddings=True, legs=['embeddings', 'lexical', 'structural'] | 0.2 |  |
| G3 | profile python | explicit | yes | PASS | set=python, detected=python, embeddings=True, legs=['embeddings', 'lexical', 'structural'] | 0.2 |  |
| G3 | profile shell | explicit | yes | PASS | set=shell, detected=python, embeddings=False, legs=['structural', 'lexical'] | 0.2 |  |
| G3 | profile auto-detect helm | auto | yes | PASS | detected=iac, expected=iac | 0.2 | ext_counts={'.yaml': 5, '.json': 2, '.sh': 1} |
| G3 | profile auto-detect python | auto | yes | PASS | detected=python, expected=python | 0.2 | ext_counts={'.py': 3, '.json': 1, '.yaml': 1} |
| G3 | embed mode placeholder | no-LLM | yes | PASS | blocks=8, search_len=894 | 0.3 |  |
| G3 | embed mode fastembed | no-LLM | yes | PASS | blocks=8, search_len=970 | 0.7 |  |
| G3 | embed mode ollama | with LLM | yes | PASS | blocks=8, search_len=976 | 0.8 |  |
| G3 | reranker cross_encoder | no-LLM | yes | PASS | result_len=1663 | 0.3 |  |
| G3 | reranker none | no-LLM | yes | PASS | result_len=1601 | 0.2 |  |
| G3 | reranker llm | with LLM | yes | PASS | result_len=1668 | 122.8 |  |
| G3 | compression none | none | yes | PASS | before=483, after=483, reduction=0.0% | 0.3 | level=none |
| G3 | compression minimal | minimal | yes | PASS | before=493, after=213, reduction=56.8% | 0.3 | level=minimal |
| G3 | compression aggressive | aggressive | yes | PASS | before=303, after=148, reduction=51.2% | 0.3 | level=aggressive |
| G3 | memory.scope repo | repo | yes | PASS | found=yes | 0.2 | result_len=148 |
| G3 | memory.scope workspace | workspace | yes | PASS | found=yes | 0.7 | result_len=160 |
| G3 | language python | fixture | yes | PASS | blocks=8 | 0.2 | lang=python |
| G3 | language helm-yaml | fixture | yes | PASS | blocks=10 | 0.2 | lang=helm-yaml |
| G3 | language terraform-hcl | fixture | yes | PASS | blocks=14 | 0.2 | lang=terraform-hcl |
| G3 | language k8s-yaml | fixture | yes | PASS | blocks=4 | 0.3 | lang=k8s-yaml |
| G3 | workspace fan-out search_all | multi-repo | yes | PASS | result_len=2312, has_deployment=True | 1.2 |  |
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

_Report completed at 2026-06-06T23:05:01.404015+00:00_
