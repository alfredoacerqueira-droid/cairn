# Cairn — Capability & Configuration Map

Authoritative, code-grounded reference for everything Cairn (v2, branch `v2/cli-mcp`) can do.
Cairn is a **local-first semantic context engine** for AI coding agents. It indexes a codebase,
retrieves surgically-relevant code for a query, and exposes it via **CLI + MCP** (the HTTP gateway
was removed in v2). Two runtime pieces: the **MCP server** (`cairn mcp`, on-demand) and the
background **Janitor** (file watcher → indexer). **The index + search work with no local LLM**; the
local LLM is optional (embeddings + memory summaries + the orchestrator).

---

## 1. CLI commands (entrypoint `cairn = cli.main:main`)

Global option: `--debug/--no-debug` (also accepted on search/reindex/status/dry-run/mcp).

| Command | Key options | Purpose | Uses local LLM? |
|---|---|---|---|
| `init` | `--no-index`, `-y/--yes`, `--force`, `--offline` | Detect layout, write `.cairn/config.yaml`, build index | Yes (embeddings) |
| `config` | — | Print current config | No |
| `profile [NAME]` | — | Show or set the repo profile | No |
| `status` | `--debug` | Index status + DB freshness + index location | No |
| `doctor` | — | Preflight: python/ollama/disk/git/chroma/ripgrep/flashrank + **System Resources** + **load-tests the configured model** | Yes (model load test) |
| `reindex` | `--mode quick\|full`, `--debug` | Re-index the project | Yes (embeddings) |
| `search <q>` | `-k/--top-k`, `--debug` | Semantic search (same path as MCP) | Yes (embeddings if enabled) |
| `dry-run <q>` | `-k`, `--show-prompt`, `--debug` | Show assembled+compressed context without sending | Yes (embeddings) |
| `suggest-local <q>` | — | System resources + model recommendation + query complexity | No |
| `memory update` | `--commits N` | Summarize git diffs → sectioned memory | Yes (summary; deterministic fallback) |
| `memory status` | — | Print memory.md | No |
| `memory clear` | — | Clear memory | No |
| `cache stats` / `cache clear` | — | Session cache introspection / clear | No |
| `token-stats` | `--days`, `--format text\|json` | Compression analytics | No |
| `token-history` | `--limit` | Recent compression entries | No |
| `janitor start` | `--debounce` | Background watcher → incremental index (+memory if enabled) | Conditional |
| `janitor stop` | — | Stop janitor (SIGTERM via pid file) | No |
| `metrics` | `-w/--watch` | Metrics/perf stats | No |
| `dashboard` | `-w/--watch` | Live observability dashboard | No |
| `mcp` | `--debug` | Run the stdio MCP server (Claude Code / OpenCode) | Yes (via tools) |
| `start-all` / `run` | `--no-janitor`, `--no-index`, `-y` (`--host/--port` deprecated) | Orchestrate: health → index → cache/memory → janitor | Yes |

---

## 2. MCP tools (`cairn mcp`, FastMCP stdio)

Binding (`_classify_binding`): **WORKSPACE** if the bound dir has ≥2 indexed child repos (checked
first); else **SINGLE** if it has `.cairn/`; else **UNBOUND** (tools return a bind-error). Set via
`CAIRN_PROJECT`. Every tool output is token-budget-capped (`_emit`: per-tool `tool_max_tokens`, then
the per-session 18%/36K cap).

| Tool | Args | Returns | LLM? |
|---|---|---|---|
| `search_code` | `query`, `top_k=5` | Ranked `[repo] file:func` hits + scores + code preview (workspace = merged across repos) | embeddings only |
| `assemble_context` | `query` | Compressed markdown context (functions + repo map + memory) | embeddings only |
| `set_profile` | `profile_name` (iac/dotnet/python/code/shell/auto) | Sets + persists profile (SINGLE only) | No |
| `orchestrate` | `query`, `instruction=""`, `payload=""` | Routes: CONTEXT_ONLY / LOCAL_ONE_SHOT / LOCAL_MAP_REDUCE / DEFER_TO_CLOUD | Yes (if instruction + enabled) |
| `cache_get` | `query` | Cached value (exact MD5 or semantic ≥ threshold) or `CACHE_MISS` | embedder for semantic |
| `cache_set` | `query`, `value`, `ttl_seconds=0` | `cached` | embedder for semantic |
| `list_repos` | — | Workspace layout: each repo + profile + block count | No |
| `remember` | `note`, `kind=change` (task/decision/convention/change/prompt) | Writes to the sectioned memory | No |
| `recall` | `max_entries=10` | Structured, budget-trimmed memory view (per `memory.scope`) | No |

**Multi-repo (workspace):** `discover_repos` finds child repos with `.cairn/`; `search_code`/
`assemble_context` **fan out across all repos** and return merged, `[repo]`-labeled results
(`route_multi` guarantees `per_repo_min` per repo so none is starved); hard per-repo isolation
(separate `project_id` + collection). `orchestrate` no-instruction → multi-repo context.

---

## 3. `config.yaml` schema (every field)

### profile (root): `auto|iac|dotnet|python|shell|code` (default `code`) · embeddings_enabled (default True)

### enabled
`file_watcher=True`, `vector_indexing=True`, `memory_summarizer=True`, `auto_start=False`

### resources
`max_cpu_percent=50`, `max_memory_mb=4096`, `vram_priority="gateway"` (`gateway|janitor`)

### indexing
`file_patterns` (default py/rs/go/c/h/cpp/hpp/cs/java/rb/sh/bash/tf/tfvars/toml; JS/TS excluded by
default), `exclude_patterns` (node_modules/.git/tests/vendor/minified/…), `source_roots=["."]`,
`batch_size=50`, `delay_ms=500`, `max_file_kb=0` (0=no limit), `parse_timeout_s=10.0`,
`embedding_model="nomic-embed-text"`, **`store_backend="chroma"`** (`chroma|lance`),
**`index_location="auto"`** (`auto|native|in_project`; auto = native `~/.cache/cairn/<id>` for `/mnt/*`
projects, else in-project `.cairn/`).

### stale_db
`auto_quick_reindex=True`, `quick_reindex_threshold=1000`, `full_reindex_threshold=10000`

### memory
`trigger="manual"` (`manual|post-commit|periodic`), `max_entries=50`,
`compaction_model="qwen2.5-coder:1.5b"`, `period_minutes=5`, **`scope="auto"`**
(`auto|both|workspace|repo`), section caps: `max_tasks=20`, `max_decisions=40`,
`max_conventions=40`, `max_changes=40`, `max_prompts=10`.

### routing
`mode="cloud_only"` (`cloud_only|conservative|aggressive`), `require_user_confirm=True`

### cache
`enabled=True`, `ttl_seconds=300` (session/embedding cache), **`semantic_ttl_seconds=1800`**
(prompt/response cache), `max_entries=100`

### budget
`session_window=200000`, `session_pct=0.18` (→ ~36K cap on Cairn's session-start injection),
`tool_max_tokens=8000` (per-MCP-tool cap), `tokenizer_model="claude"` (tiktoken cl100k_base proxy)

### retrieval
`mode="hybrid"` (`hybrid|embeddings|bm25|ast`), `weights=[0.4,0.3,0.3]` (bm25/ast/embeddings),
**`rrf_k=60`**, **`max_merged=24`**, **`per_repo_min=3`**, `rerank_enabled=True`,
`reranker_type="cross_encoder"` (`cross_encoder|llm|none`), `rerank_min_score=0.47`,
`min_confidence=0.82` (raw-cosine fallback gate), `offline=False`, `ca_bundle=None`

### compression
`enabled=True`, `level="minimal"` (`none|minimal|aggressive`; honors `COMPRESSION_LEVEL` env)

### local_llm
`enabled=False`, `backend="ollama"` (`ollama|openai_compatible`), `base_url=None`, `model=None`,
`embed_model=None`, `context_window=8192`, `max_local_tokens=6000`, `reduce_reserve_tokens=1024`,
`chunk_overlap_pct=0.12`, `one_shot_threshold=0.75`, **`embedder="ollama"`** (`ollama|fastembed|none`),
`fastembed_model="BAAI/bge-small-en-v1.5"`, `map_concurrency=1`, **`ollama_options={}`** (forwarded
to Ollama generate+embed: `num_ctx`, `num_gpu`, `low_vram`, `num_thread`, `num_batch`, …).

---

## 4. Profiles (`core/profiles.py`)

| Profile | file_patterns | embeddings | legs | embed model | rerank_min_score |
|---|---|---|---|---|---|
| iac | tf/tfvars/hcl/yaml/yml/sh/bash | OFF | structural+lexical | — | 0.15 |
| dotnet | cs | ON | embeddings+lexical+structural | qwen3-embedding:0.6b | 0.10 |
| python | py | ON | embeddings+lexical+structural | nomic-embed-text | 0.47 |
| shell | sh/bash | OFF | structural+lexical | — | 0.15 |
| code | js/ts/tsx/jsx/go/rs/java/cpp/c/h/hpp/rb/toml | ON | embeddings+lexical+structural | nomic-embed-text | 0.47 |

**detect_profile** (in order): infra markers (Chart.yaml/kustomization/k8s) → iac; `.tf/.hcl` → iac;
yaml-dominant (>50%) w/ no dominant language → iac; `.cs` present → dotnet; python-dominant (>50%) →
python; shell-dominant → shell; else → code.

---

## 5. Retrieval

**Legs:** ripgrep (live working-tree exact match; falls back to **BM25** if `rg` missing) ·
**structural** (exact block-id / symbol / resource matching) · **embeddings** (vector cosine; runs
only if embeddings enabled for the profile + an embedder is available) · ast_rank (PageRank call-graph
— only used in `mode="ast"`, not built in default hybrid). **Fusion:** Reciprocal Rank Fusion
(`rrf_k=60`, weights `[0.4,0.3,0.3]`). **Rerank:** `cross_encoder` (FlashRank `ms-marco-MiniLM-L-12-v2`,
CPU, ms, default) · `llm` (local Ollama scorer, ~19–39s/gen, opt-in) · `none`. **Confidence guard:**
drops results below `rerank_min_score` (or `min_confidence` raw cosine when rerank off) — but a
**symbol-name match bypasses the guard** so searching a class/function by name always returns it.

---

## 6. Embedding modes & store backends

**3 embedding modes** (`make_embedder`): `placeholder` (dim=1, embeddings OFF — lexical+structural
only) · `fastembed` (in-process ONNX `bge-small`, semantic search with **no Ollama**) · `ollama`
(`local_llm.enabled` → real embeddings via the configured model). **2 store backends:** `chroma`
(default; single SQLite-backed dir) · `lance` (optional `[local]` extra; native hybrid + versioning).
Both isolated per repo via `functions_<project_id>` collection + provenance stamping. The collection
**rebuilds automatically when the embedding dimension changes** (switching embed model / toggling LLM).

---

## 7. Parser language tiers (`pipeline/ast_parser.py`)

**Real tree-sitter AST** (functions/classes/methods): Python, C#, Bash, HCL/Terraform, Go, Rust,
Java, JavaScript, TypeScript/TSX, C++, Ruby. **YAML:** structured — one block per **top-level key**
(helm `values.yaml`) and one per **resource** (multi-doc k8s, `{kind}.{name}`). JSON/TOML: top-level
keys. Hang-safety: tree-sitter ML languages capped at 1500 KB input (fall back to regex); Python path
has a `parse_timeout_s` guard. EXTENSION_MAP covers .py/.js/.ts/.go/.rs/.java/.cs/.cpp/.rb/.sh/.yaml/
.tf/.hcl/.json/.toml/Dockerfile.

---

## 8. Memory (sectioned, bounded — `core/memory_doc.py`)

`.cairn/memory.md` is a structured doc with 5 capped sections: **Open Tasks** (20), **Decisions**
(40), **Conventions** (40), **Recent Changes** (40; oldest collapse into a `(history) N earlier
changes` line), **Recent User Prompts** (10). Writers: git-diff summaries → Recent Changes
(local-LLM if enabled, **deterministic fallback otherwise**); the `remember(note, kind)` MCP tool →
the section for that kind. `recall`/`load_memory` return a token-budgeted structured view. **Scope**
(`memory.scope`): `repo` (per-repo), `workspace` (one memory above the repos), `both`/`auto`
(workspace + per-repo). Old flat logs migrate losslessly.

---

## 9. Compression (`server/token_compressor.py`)

RTK-style, applied to assembled context. Levels: `none` (0%), `minimal` (~20–40%, default),
`aggressive` (~60–90%). 8-stage pipeline (ANSI strip → comments → docstrings[agg] → whitespace →
imports → truncate funcs[agg] → dedup[agg] → max_lines). Marked `[already-compressed]`.

---

## 10. Caches (3 layers)

**Session cache** (`core/cache.py`): in-memory, TTL 300s, LRU, keyed incl. git commit. **Persistent
cache** (`core/persistent_cache.py`): cross-process assembled-context cache under `.cairn/cache/`.
**Semantic cache** (`core/semantic_cache.py`): the MCP prompt/response cache — exact (MD5) + semantic
(cosine ≥ 0.92) lookup, TTL `semantic_ttl_seconds` (1800s), LRU; SET by local LLM, GET by cloud LLM
via `cache_set`/`cache_get`.

---

## 11. Local LLM (optional) + resources

`local_llm.enabled=False` by default → indexing + lexical/structural + rerank work with **no LLM**.
When enabled: embeddings (mode 1) + memory summaries + orchestrator + optional llm-rerank. Mode 2
(`embedder=fastembed`) gives semantic search with no Ollama. **`ollama_options`** forwards llama.cpp
knobs (num_ctx for bigger context, num_gpu to control GPU/RAM split, low_vram, …) into Ollama
generate+embed. **`core/resources.py`** detects RAM (psutil) + VRAM (nvidia-smi, graceful) and
`recommend_local_models` picks the best-fitting worker/embed model + a suggested `num_ctx`; surfaced
in `cairn suggest-local` and `cairn doctor` (which also load-tests the configured model).

---

## 12. Orchestrator (`server/orchestrator.py`) — small-LLM task splitting

`orchestrate` sizes the work in tokens and routes: **CONTEXT_ONLY** (no LLM) · **LOCAL_ONE_SHOT**
(fits one call) · **LOCAL_MAP_REDUCE** (split into token-bounded chunks with `chunk_overlap_pct`
overlap → map each through the local LLM → reduce, tree-reduce if needed) · **DEFER_TO_CLOUD** (too
big → budget-capped context only). Keeps every local call within the model's window. `SessionBudget`
caps total session output at 18%/36K.

---

## 13. Works WITHOUT a local LLM (degradation matrix)

| Capability | With local LLM | Without (default) |
|---|---|---|
| Indexing | real embeddings (mode 1) | placeholder vectors; or fastembed (mode 2) for semantic w/o Ollama |
| Search | embeddings + lexical + structural + rerank | lexical (ripgrep/BM25) + structural + rerank |
| Memory summaries | fluent LLM-written | deterministic (file/type heuristic) |
| `orchestrate` w/ instruction | one-shot or map-reduce on the LLM | context-only (no generation) |
| Reranking (FlashRank) | unaffected (no LLM) | unaffected |
| Compression / caches / parsing | unaffected | unaffected |
