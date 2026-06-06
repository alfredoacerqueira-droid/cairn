# Configuration Reference

Cairn's per-project configuration lives in `.cairn/config.yaml`. This document describes every field.

## Quick Start

After `cairn init`, you'll have a `config.yaml` that looks like this:

```yaml
profile: python
embeddings_enabled: true
enabled:
  file_watcher: true
  vector_indexing: true
  memory_summarizer: true
  auto_start: false
resources:
  max_cpu_percent: 50
  max_memory_mb: 4096
  vram_priority: gateway
indexing:
  file_patterns: [...]
  exclude_patterns: [...]
  source_roots: ['.']
  index_location: auto
  store_backend: chroma
  batch_size: 50
  delay_ms: 500
  embedding_model: nomic-embed-text
retrieval:
  mode: hybrid
  rerank_enabled: true
  rerank_min_score: 0.47
cache:
  enabled: true
  ttl_seconds: 300
memory:
  trigger: manual
memory_summarizer:
  max_entries: 50
budget:
  session_window: 200000
  session_pct: 0.18
  tool_max_tokens: 8000
local_llm:
  enabled: false
  backend: ollama
  embedder: ollama
```

## Configuration Sections

### `profile` (string)

Repository profile; determines retrieval strategy and which models are used.

**Valid values:**
- `iac` — Terraform, Helm, Kubernetes (embeddings OFF, structural + lexical)
- `python` — Django, FastAPI (embeddings ON, nomic-embed-text)
- `dotnet` — C#, MediatR, Roslyn (embeddings ON, qwen3-embedding:0.6b)
- `code` — Generic: JS/TS, Go, Rust, Java, C++, Ruby (embeddings ON)
- `shell` — Shell scripts, bash (embeddings OFF, lexical + structural)
- `auto` — Auto-detect at init (becomes concrete after init)

**Example:**
```yaml
profile: python
```

### `embeddings_enabled` (boolean)

Global switch: whether embeddings are used. When `false`, Cairn uses only lexical + structural retrieval (no semantic search). Saves VRAM; useful for IaC profiles or when Ollama is unavailable.

**Default:** `true`

**Example:**
```yaml
embeddings_enabled: false
```

### `enabled` (object)

Toggles for individual Cairn subsystems.

**Fields:**
- `file_watcher` (bool) — Watch for file changes and trigger reindexing. Default: `true`
- `vector_indexing` (bool) — Index files into vector DB. Default: `true`
- `memory_summarizer` (bool) — Summarize git diffs into memory. Default: `true`
- `auto_start` (bool) — Auto-start janitor on `cairn run`. Default: `false`

**Example:**
```yaml
enabled:
  file_watcher: true
  vector_indexing: true
  memory_summarizer: true
  auto_start: false
```

### `resources` (object)

Resource limits for the indexing/summarization pipeline.

**Fields:**
- `max_cpu_percent` (int) — Max CPU % for janitor jobs. Default: `50`
- `max_memory_mb` (int) — Max RAM (MB) for janitor jobs. Default: `4096`
- `vram_priority` (string) — VRAM priority group. Default: `gateway`

**Example:**
```yaml
resources:
  max_cpu_percent: 50
  max_memory_mb: 4096
  vram_priority: gateway
```

### `indexing` (object)

Vector database and AST parsing configuration.

**Fields:**

- **`file_patterns`** (list of strings) — Glob patterns to include in indexing. Default includes `.py`, `.rs`, `.go`, `.c`, `.h`, `.cpp`, `.hpp`, `.cs`, `.java`, `.rb`, `.sh`, `.bash`, `.tf`, `.tfvars`, `.toml`. Add `.js`/`.ts` explicitly if needed.

- **`exclude_patterns`** (list of strings) — Glob patterns to exclude. Default excludes `node_modules/`, `.git/`, `tests/`, `examples/`, `vendor/`, `static/`, and build artifacts.

- **`source_roots`** (list of strings) — Directories to index (relative to project root). Default: `['.']`

- **`index_location`** (string) — Where the vector DB lives. Default: `auto`
  - `auto` — On WSL with `/mnt/*` projects, use `~/.cache/cairn/<project-id>/`; otherwise `.cairn/`
  - `native` — Force `~/.cache/cairn/<project-id>/` (Linux native FS)
  - `in_project` — Force `.cairn/` (in the project directory)

- **`store_backend`** (string) — Vector DB backend. Default: `chroma`
  - `chroma` — Lighter, recommended default (built-in)
  - `lance` — Requires `pip install -e ".[local]"`; adds native hybrid search + versioning

- **`batch_size`** (int) — Files to batch per indexing job. Default: `50`

- **`delay_ms`** (int) — Debounce file-watch events (ms). Default: `500`

- **`max_file_kb`** (int) — Skip files larger than this (KB). `0` = no limit. Default: `0`

- **`parse_timeout_s`** (float) — Timeout per-file AST parse (seconds). `0` = no limit. Default: `10.0`

- **`embedding_model`** (string) — Ollama embedding model name (for semantic search). Default: `nomic-embed-text`. For code-specific embeddings, try `qwen3-embedding:0.6b` or other code-trained models (requires `cairn reindex --mode full` after changing).

**Example:**
```yaml
indexing:
  file_patterns:
    - '*.py'
    - '*.go'
  exclude_patterns:
    - '**/tests/**'
    - '**/node_modules/**'
  source_roots: ['.']
  index_location: auto
  store_backend: chroma
  batch_size: 50
  delay_ms: 500
  max_file_kb: 0
  parse_timeout_s: 10.0
  embedding_model: nomic-embed-text
```

### `stale_db` (object)

Automatic reindexing thresholds.

**Fields:**
- `auto_quick_reindex` (bool) — Auto-reindex if behind. Default: `true`
- `quick_reindex_threshold` (int) — Commits behind before quick reindex. Default: `1000`
- `full_reindex_threshold` (int) — Commits behind before full reindex warning. Default: `10000`

**Example:**
```yaml
stale_db:
  auto_quick_reindex: true
  quick_reindex_threshold: 1000
  full_reindex_threshold: 10000
```

### `memory` (object)

Git-diff memory configuration.

**Fields:**
- `trigger` (string) — When to summarize diffs. Default: `manual`
  - `manual` — Only on `cairn memory update` command
  - `post-commit` — After each commit (requires git hook)
  - `periodic` — Every N minutes (see `period_minutes`)

- `max_entries` (int) — Max number of memory entries to keep. Default: `50`

- `compaction_model` (string) — Ollama model for summarization. Default: `qwen2.5-coder:1.5b`

- `period_minutes` (int) — For periodic trigger, how often to summarize (minutes). Default: `5`

**Example:**
```yaml
memory:
  trigger: manual
  max_entries: 50
  compaction_model: qwen2.5-coder:1.5b
  period_minutes: 5
```

### `routing` (object)

Smart routing (local vs cloud LLM decisions).

**Fields:**
- `mode` (string) — Routing strategy. Default: `cloud_only`
  - `cloud_only` — Always forward to cloud (no local decision)
  - `conservative` — Use local judgement when confident
  - `aggressive` — Prefer local handling

- `require_user_confirm` (bool) — Require approval before routing locally. Default: `true`

**Example:**
```yaml
routing:
  mode: cloud_only
  require_user_confirm: true
```

### `cache` (object)

Semantic response cache (local, embedded).

**Fields:**
- `enabled` (bool) — Enable caching. Default: `true`
- `ttl_seconds` (int) — In-memory session/embedding cache entry lifetime. Default: `300` (5 minutes)
- `semantic_ttl_seconds` (int) — MCP prompt/response semantic cache entry lifetime (separate from session cache). Default: `1800` (30 minutes)
- `max_entries` (int) — Max cached responses. Default: `100`

**Example:**
```yaml
cache:
  enabled: true
  ttl_seconds: 300
  semantic_ttl_seconds: 1800
  max_entries: 100
```

### `budget` (object)

Token budget for Cairn's contributions to the session.

**Fields:**
- `session_window` (int) — Your LLM's total context window (tokens). Default: `200000` (Sonnet 3.5, non-1M)
- `session_pct` (float) — Max % of session_window Cairn can use. Default: `0.18` (caps contribution to ~36K)
- `tool_max_tokens` (int) — Cap per MCP tool output. Default: `8000`
- `tokenizer_model` (string) — Tokenizer model for estimation. Default: `claude` (cl100k_base)

**Example:**
```yaml
budget:
  session_window: 200000
  session_pct: 0.18
  tool_max_tokens: 8000
  tokenizer_model: claude
```

### `retrieval` (object)

Semantic search and reranking.

**Fields:**
- `mode` (string) — Retrieval strategy. Default: `hybrid`
  - `hybrid` — Fuses lexical (BM25) + semantic (embeddings) + structural (AST) via RRF, then reranks
  - `embeddings` — Semantic search only
  - `bm25` — Lexical search only (no embeddings)
  - `ast` — Structural search only

- `weights` (list of floats) — Fusion weights for `hybrid` mode. Default: `[0.4, 0.3, 0.3]` (bm25, ast, embeddings)

- `rerank_enabled` (bool) — Use cross-encoder reranking. Default: `true`

- `reranker_type` (string) — Which reranker to use. Default: `cross_encoder`
  - `cross_encoder` — FlashRank (CPU, ~milliseconds, recommended)
  - `llm` — Score with local Ollama (slow; ~19–39s per generation on 6GB GPU)
  - `none` — No reranking

- `rerank_min_score` (float) — Confidence threshold (0–1 scale). Default: `0.47` (measured on Django; separates relevant from off-topic queries)

- `min_confidence` (float) — Minimum embedding cosine (fallback when reranking off). Default: `0.82` (model-specific; measured on Django)

- `offline` (bool) — Disable reranker (skip FlashRank download). Default: `false`. Useful behind corporate proxies.

- `ca_bundle` (string or null) — Custom CA certificate bundle path. Default: `null`. Overrides `CAIRN_CA_BUNDLE`/`REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`.

**Example:**
```yaml
retrieval:
  mode: hybrid
  rerank_enabled: true
  reranker_type: cross_encoder
  rerank_min_score: 0.47
  min_confidence: 0.82
  offline: false
  ca_bundle: null
```

### `compression` (object)

Lossless token compression for assembled context.

**Fields:**
- `enabled` (bool) — Enable compression. Default: `true`
- `level` (string) — Compression intensity. Default: `minimal`
  - `none` — No compression (0%)
  - `minimal` — Light compression (20–40%)
  - `aggressive` — Heavy compression (60–90%)

**Example:**
```yaml
compression:
  enabled: true
  level: minimal
```

### `local_llm` (object)

Local LLM configuration (optional; disabled by default).

**Fields:**
- `enabled` (bool) — Enable local LLM (Ollama or OpenAI-compatible). Default: `false`

- `backend` (string) — Backend type. Default: `ollama`
  - `ollama` — Ollama
  - `openai_compatible` — LM Studio, llama.cpp, vLLM, etc.

- `base_url` (string or null) — Backend URL. Default: `null` (uses backend defaults: `http://127.0.0.1:11434` for Ollama)

- `model` (string or null) — Text generation model. Default: `null` (uses profile-specific: qwen2.5-coder:1.5b)

- `embed_model` (string or null) — Embedding model. Default: `null` (uses profile-specific: nomic-embed-text)

- `context_window` (int) — Local model's true context window (tokens). Default: `8192`

- `max_local_tokens` (int) — Usable input budget per local call (context minus output reserve). Default: `6000`

- `reduce_reserve_tokens` (int) — Tokens reserved for map/reduce answer. Default: `1024`

- `chunk_overlap_pct` (float) — Sliding-window overlap between map chunks (0–1 scale). Default: `0.12` (10–15% overlap)

- `one_shot_threshold` (float) — Work <= this fraction of `max_local_tokens` → single call (vs. map/reduce). Default: `0.75`

- `embedder` (string) — Embedder type. Default: `ollama`
  - `ollama` — Use Ollama embedding model
  - `fastembed` — In-process ONNX (requires `pip install -e ".[local]"`)
  - `none` — No embeddings (structural + lexical only)

- `fastembed_model` (string) — ONNX model name (for fastembed). Default: `BAAI/bge-small-en-v1.5`

- `map_concurrency` (int) — Parallel local LLM map calls. Default: `1`. Increase for fast local inference (Apple Silicon) on 6GB+ VRAM; risks OOM on smaller GPUs.

**Example (Disabled):**
```yaml
local_llm:
  enabled: false
  backend: ollama
  embedder: ollama
```

**Example (Enabled with Ollama):**
```yaml
local_llm:
  enabled: true
  backend: ollama
  base_url: http://127.0.0.1:11434
  model: qwen2.5-coder:1.5b
  embed_model: nomic-embed-text
  context_window: 8192
  max_local_tokens: 6000
  embedder: ollama
```

**Example (Enabled with fastembed, no Ollama):**
```yaml
local_llm:
  enabled: false  # fastembed only, no LLM generation
  embedder: fastembed
  fastembed_model: BAAI/bge-small-en-v1.5
```

## Environment Variables

Cairn respects these environment variables (override config.yaml):

| Variable | Example | Purpose |
|----------|---------|---------|
| `CAIRN_PROJECT` | `/path/to/project` | MCP server: bind to this project |
| `COMPRESSION_LEVEL` | `minimal` `aggressive` | Override `compression.level` |
| `CAIRN_CA_BUNDLE` | `/path/to/ca.pem` | Custom CA bundle for HTTPS |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Keep N models in Ollama memory (small GPU tip) |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep models in Ollama memory for this duration |

## Typical Workflows

### Small Django Project (1–5K functions)

```yaml
profile: python
embeddings_enabled: true
indexing:
  store_backend: chroma
retrieval:
  mode: hybrid
  rerank_min_score: 0.47
local_llm:
  enabled: false
```

### Large Python Repo (10K+ functions)

```yaml
profile: python
embeddings_enabled: true
indexing:
  store_backend: lance  # requires pip install -e ".[local]"
retrieval:
  mode: embeddings  # pure embeddings faster on large repos
  min_confidence: 0.82
local_llm:
  enabled: false
```

### Terraform / Helm Project

```yaml
profile: iac
embeddings_enabled: false  # structural + lexical only
retrieval:
  mode: bm25
local_llm:
  enabled: false
```

### C# / Roslyn Project

```yaml
profile: dotnet
embeddings_enabled: true
indexing:
  embedding_model: qwen3-embedding:0.6b  # code-tuned
local_llm:
  enabled: false
```

## Troubleshooting Config

### Searches return irrelevant results
- Increase `retrieval.rerank_min_score` (e.g., to 0.6)
- Switch `retrieval.mode` to `embeddings` (if using `hybrid`)
- Change `indexing.embedding_model` to a code-tuned model and `cairn reindex --mode full`

### Index is too large
- Add directories to `indexing.exclude_patterns`
- Reduce `indexing.file_patterns`
- Use `local_llm.embedder: fastembed` (smaller DB)

### Ollama out of memory
- Reduce `indexing.batch_size` (default 50)
- Set `OLLAMA_MAX_LOADED_MODELS=1` in your environment
- Use IaC profile (embeddings_enabled: false) to avoid loading embedding model

### Token budget exceeded
- Reduce `memory.max_entries` (fewer memory lines in context)
- Reduce `budget.tool_max_tokens` (cap per tool output)
- Use `compression.level: aggressive`

## See Also

- [README.md](../README.md) — Overview and quick start
- [RUNBOOK.md](../RUNBOOK.md) — Step-by-step setup guide
- [AGENTS.md](../AGENTS.md) — MCP tool reference for Claude Code / OpenCode
