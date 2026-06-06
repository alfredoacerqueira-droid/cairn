# Cairn

**Version 0.6.0** — A **local-first** semantic context engine for AI coding agents. It indexes your
codebase into a per-repo vector DB, retrieves the surgically-relevant functions for a query,
compresses them, and serves them via CLI or MCP tools — cutting the tokens you send to the cloud LLM while
keeping all your code on your machine. Works natively with **Claude Code and OpenCode via MCP tools**.

```
Claude Code / OpenCode ──MCP tool── Cairn ──► search + assemble + orchestrate (local)
                                       ↓
                  per-repo .cairn/  (vector DB + lexical + structural + cache + memory)
                       ↑ auto-synced on file edits & commits (never out of date)
```

## ⚡ Quick Start (install once, init per repo)

```bash
# 1. Install the CLI (once, globally — pipx keeps it isolated)
pipx install .            # or: pip install -e ".[dev]" from this repo

# 2. In the repo you want to work on:
cd /path/to/your-project
cairn init         # auto-detects profile → write config → build the index → ready
cairn doctor       # verify environment

# 3. Point your agent at it:
# Option A: MCP (recommended) — agents call it as a native tool
#   (See Use with Claude Code / OpenCode below)
# Option B: CLI — use cairn search, cairn assemble_context directly
cairn search "query" -k 5
```

Note: Ollama is optional (required only for embeddings-heavy profiles; can be disabled per config).

## 🔌 Use with Claude Code / OpenCode (MCP — recommended)

Cairn runs as a native **MCP server**, so either agent can call it as a tool. No proxy wiring.

**OpenCode** (`opencode.json`):
```json
{
  "mcp": {
    "cairn": {
      "type": "local",
      "command": "cairn",
      "args": ["mcp"],
      "env": { "CAIRN_PROJECT": "/absolute/path/to/your/project" }
    }
  }
}
```

**Claude Code** (`.mcp.json`):
```json
{
  "mcpServers": {
    "cairn": {
      "command": "cairn",
      "args": ["mcp"],
      "env": { "CAIRN_PROJECT": "/absolute/path/to/your/project" }
    }
  }
}
```

Exposes three tools: `search_code(query, top_k)`, `assemble_context(query)`, and
`set_profile(name)`. See [AGENTS.md](AGENTS.md) for detailed tool guidance.

## ✨ What's New in v0.6.0

- **Repository profiles** — auto-detects repo type (IaC/dotnet/python/generic) at init; each profile
  optimizes the retrieval strategy. IaC uses structural + lexical (embeddings OFF to save VRAM);
  dotnet/python use full hybrid (embeddings + lexical + structural).
- **Real tree-sitter parsing** — native grammar support for HCL, YAML, C#, bash (in addition to
  Python), with a regex fallback if a grammar fails to load.
- **Structural retrieval** — block-level scope analysis for better function isolation in large files.
- **FlashRank reranker** — cross-encoder confidence guard separates relevant queries (~0.92 mean score)
  from off-topic (~0.007 mean score) with 0.47 confidence threshold (measured on Django, 8331 functions).
- **Sectioned memory** — git-diff summaries now structured into sections (Open Tasks, Decisions, Conventions,
  Recent Changes, Recent User Prompts) with per-section entry caps; `remember(note, kind)` MCP tool routes
  to the right section; `recall()` returns trimmed, budget-aware view.
- **Compression on all paths** — MCP tools, CLI, and proxy all benefit from lossless context compression.
- **Configurable worker model** — choose the local Ollama model for optional reranking/summarization
  (default: qwen2.5-coder:1.5b).
- **Multi-repo workspace** — a session can bind to a folder with multiple repos; `search_code` and
  `assemble_context` fan out across all repos with `[repo]`-labeled results; hard per-repo isolation.
- **Persistent session cache** — cache keys include git commit hash, auto-evicting stale results on commits.
- **MCP server** — `cairn mcp`; works with Claude Code & OpenCode as a native tool (no proxy needed).

> **Measured token reductions:** 95–99% across Python, Go, Rust, and IaC repos at good recall.
> **Measured retrieval:** Terraform top-1 recall improved from ~17% (BM25) to ~60% (IaC profile strategy).
> See [RUNBOOK](RUNBOOK.md#benchmarking) for methodology and real caveats.

## 🚀 Daily Operations

```bash
# Smart start (checks health, reindexes if stale, starts janitor)
cairn start-all

# Or: just the janitor (background indexing)
cairn janitor start

# Or: just the CLI (manual search + assemble)
cairn search "query"
cairn assemble_context "what does handle_request do?"

# Agents invoke the MCP server directly (no manual start needed)
# cairn mcp runs on-demand when agents call it
```

## 📦 Installation

### System Requirements

- Python ≥ 3.10
- Git (to track commits for staleness detection)
- ~2GB free disk space (for ChromaDB + models)
- Ollama (any OS: macOS, Linux, Windows via WSL)

### Install via pipx (recommended)

```bash
# Install once globally
pipx install .

# Now from any repo:
cd /path/to/repo
cairn init
```

**⚠️ Avoid `pip install --break-system-packages`** on your system Python.
Use `pipx` or a venv. (Ollama is optional; only required for embedding profiles.)

### Install via venv (for development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cairn --help
```

### Optional: Install Local Embedding Backend

For embedding-heavy projects without Ollama, use fastembed (ONNX, in-process):

```bash
pip install -e ".[local]"  # adds lancedb + fastembed
# Then set local_llm.embedder: fastembed in config.yaml
```

## 🎯 How It Works

### Architecture

Two processes:

1. **MCP Server** (`cairn mcp`) — on-demand, high priority
   - Listens on stdio (not HTTP)
   - Claude Code / OpenCode invoke it as a native tool
   - Exposes: `search_code`, `assemble_context`, `orchestrate`, `set_profile`, `cache_get`, `cache_set`
   - On request: searches index → assembles context → optionally routes to local LLM or returns
   - Respects token budget and session state

2. **Janitor** (background, low priority)
   - Watches repo for file changes (debounced)
   - Periodically checks for new commits
   - Re-parses changed files → updates vector index (ChromaDB or LanceDB)
   - Backs off if MCP server is active (respects VRAM limits)
   - Summarizes git diffs into memory (token-budgeted)

### Context Assembly

When `assemble_context()` is called:

1. **Searches** for relevant code blocks (hybrid retrieval: lexical + semantic + structural)
2. **Reranks** with FlashRank cross-encoder; gates on confidence (default: ≥0.47)
3. **Loads** repository map (snapshot of all top-level functions/classes)
4. **Loads** recent sectioned memory (newest entries within token budget, split by kind)
5. **Assembles** a markdown prompt block with all three sources
6. **Compresses** losslessly (removes boilerplate, shortens identifiers)
7. **Injects** into your prompt (or returns as context string)

The system never modifies the actual user query or agent's output — it's transparent.

### Works Without a Local LLM

Cairn's indexing and search pipelines work **without any local LLM**:

- **Indexing:** AST parsing (tree-sitter) + structural block extraction are language-native, zero-LLM
- **Search:** Lexical retrieval (ripgrep/BM25) + structural matching (regex/AST graph) need no embeddings
- **Reranking:** FlashRank cross-encoder (CPU, ~millisecond) is optional and independent of embeddings

When embeddings are disabled (IaC profiles, `local_llm.embedder: none`), Cairn still indexes and retrieves using lexical + structural legs alone. At search time, FlashRank (if enabled) provides confidence gates without needing embeddings.

For embeddings *without* Ollama, use **fastembed mode-2**:
```yaml
local_llm:
  embedder: fastembed       # In-process ONNX, ~50MB, no Ollama needed
  fastembed_model: BAAI/bge-small-en-v1.5
```

This requires `pip install -e ".[local]"` (adds lancedb + fastembed). Embeddings then work on CPU with no external service.

### Repository Profiles

At `init`, Cairn auto-detects your repo type:

| Profile | File Types | Embedding | Strategy | Use Case |
|---------|-----------|-----------|----------|----------|
| **iac** | .tf, .hcl, .yaml | OFF | Structural + lexical | Terraform, Helm, Kubernetes |
| **dotnet** | .cs | ON (qwen3-embedding:0.6b) | Embeddings + lexical + structural | C#, MediatR, Roslyn |
| **python** | .py | ON (nomic-embed-text) | Embeddings + lexical + structural | Django, FastAPI, etc. |
| **shell** | .sh, .bash | OFF | Lexical + structural | Shell scripts |
| **code** | .js, .ts, .go, .rs, .java, .cpp, .rb, .toml | ON | Embeddings + lexical + structural | Generic: JS/TS, Go, Rust, Java, C++, Ruby |

Profiles can be overridden post-init with `cairn profile set <name>`.

### Per-Project Storage

All state lives in `.cairn/` within your repo (version-controlled `.gitignore` prevents committing):

- `config.yaml` — retrieval strategy, file patterns, profile, limits, embeddings config, token budgets
- `chroma/` or `~/.cache/cairn/<project-id>/chroma/` — vector DB directory (location controlled by `indexing.index_location`)
- `repo_map.json` — snapshot of functions/classes for context assembly
- `memory.md` — git-diff summaries (newest entries, token-budgeted)
- `metrics.json` — observability (query latency, token counts, indexing stats)
- `opencode.json` and `.mcp.json` — MCP server configs (auto-written at init)

## 📊 Observability

```bash
# Live dashboard with system health + retrieval stats
cairn dashboard

# Detailed metrics (indexing latency, token compression, etc.)
cairn metrics

# Raw metrics JSON
cat .cairn/metrics.json
```

## 🔒 Privacy & Security

- **Local-first by default**: Code stays on your machine. Queries are processed locally via MCP.
- **No telemetry**: Logs to stdout/file, never phones home.
- **Git-aware**: Uses your repo's `.git/` to track commits; memory entries include commit hashes.
- **MCP isolation**: Agents communicate with Cairn over stdio (not HTTP), with no external endpoints.

## 📖 Full Documentation

- **[RUNBOOK.md](RUNBOOK.md)** — First-time setup, commands reference, troubleshooting
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — Using with Claude Code & OpenCode via MCP
- **[AGENTS.md](AGENTS.md)** — MCP tool reference & usage patterns for AI agents
- **[docs/CONFIG.md](docs/CONFIG.md)** — Complete configuration schema reference

## 🛠 Development

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test
pytest
pytest tests/unit/test_ast_parser.py

# Lint
ruff check server/ pipeline/ cli/ core/ throttle/
black server/ pipeline/ cli/ core/ throttle/
mypy server/ pipeline/ cli/ core/ throttle/

# Benchmarks
python benchmarks/benchmark_ast_parser.py
```

---

_Internal project — not for public distribution._
