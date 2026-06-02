# Cairn

**Version 0.6.0** — A **local-first** semantic context engine for AI coding agents. It indexes your
codebase into a per-repo vector DB, retrieves the surgically-relevant functions for a query,
compresses them, and serves them to your agent — cutting the tokens you send to the cloud LLM while
keeping all your code on your machine. Works with **Claude Code and OpenCode out of the box via MCP**,
or as an OpenAI/Anthropic proxy.

```
Claude Code / OpenCode ──MCP tool── Gateway ──► retrieve + compress (local)
                                       ↓
                  per-repo .cairn/  (ChromaDB + BM25 + structural + cache + memory)
                       ↑ auto-synced on file edits & commits (never out of date)
```

## ⚡ Quick Start (ruflo-style: install once, init per repo)

```bash
# 1. Install and start Ollama + pull the models the gateway needs
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text qwen2.5-coder:1.5b
ollama serve &

# 2. Install the gateway CLI (once, globally — pipx keeps it isolated)
pipx install .            # or: pip install -e ".[dev]" from this repo

# 3. In the repo you want to work on:
cd /path/to/your-project
cairn init         # auto-detects profile → write config → build the index → ready
cairn doctor       # verify environment
```

Then point your agent at it (see **Use with Claude Code / OpenCode** below) — or run
`cairn run` to serve the proxy API.

## 🔌 Use with Claude Code / OpenCode (MCP — recommended)

The gateway runs as a native **MCP server**, so either agent can call it as a tool. No proxy wiring.

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
- **Compression on all paths** — MCP tools, CLI, and proxy all benefit from lossless context compression.
- **Configurable worker model** — choose the local Ollama model for optional reranking/summarization
  (default: qwen2.5-coder:1.5b).
- **Persistent session cache** — cache keys include git commit hash, auto-evicting stale results on commits.
- **MCP server** — `cairn mcp`; works with Claude Code & OpenCode as a native tool (no proxy needed).

> **Measured token reductions:** 95–99% across Python, Go, Rust, and IaC repos at good recall.
> **Measured retrieval:** Terraform top-1 recall improved from ~17% (BM25) to ~60% (IaC profile strategy).
> See [RUNBOOK](RUNBOOK.md#benchmarking) for methodology and real caveats.

## 🚀 One Command to Rule Them All

```bash
cairn start-all
```

Auto-checks health, indexes if stale, clears cache, rotates memory, starts gateway + janitor:

```
╔══════════════════════════════════════════════════════════════╗
║     Cairn — Smart Start                     ║
╚══════════════════════════════════════════════════════════════╝

[1/6] Health check... ✓ Ollama online
[2/6] Configuration... ✓ Found existing config
[3/6] Freshness... ✓ DB is up to date
[4/6] Cache... ✓ Cleared
[5/6] Memory... ✓ Rotated
[6/6] Starting processes...
  • Gateway on http://127.0.0.1:8000
  • Janitor: watching for changes

Ready. Press Ctrl+C to stop.
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
Use `pipx` or a venv. The gateway will fail if Ollama models can't be reached.

### Install via venv (for development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cairn --help
```

## 🎯 How It Works

### Architecture

Two concurrent processes:

1. **Gateway** (foreground, high priority)
   - Listens on `POST /v1/chat/completions` (OpenAI) + `POST /v1/messages` (Anthropic)
   - Also runs MCP server on stdio
   - On request: searches ChromaDB → assembles context → forwards to cloud (or returns locally)
   - Yields to janitor via VRAM lock

2. **Janitor** (background, low priority)
   - Watches repo for file changes (debounced)
   - Periodically checks for new commits
   - Re-parses changed files → updates ChromaDB vector index
   - Backs off if gateway is active (respects VRAM limits)
   - Summarizes git diffs into memory

### Context Assembly

When a query arrives, the gateway:

1. **Searches** for relevant code blocks (hybrid retrieval: lexical BM25 + semantic embeddings + AST)
2. **Reranks** with FlashRank cross-encoder; gates on confidence (default: ≥0.47)
3. **Loads** repository map (snapshot of all top-level functions/classes)
4. **Loads** recent git-diff memory (last 10 entries)
5. **Assembles** a markdown prompt block with all three sources
6. **Compresses** losslessly (removes boilerplate, shortens identifiers)
7. **Injects** as system message to cloud model (or returns directly if local-only)

The gateway never modifies the actual user query or agent's output — it's transparent.

### Repository Profiles

At `init`, the gateway auto-detects your repo type:

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

- `config.yaml` — retrieval strategy, file patterns, profile, limits
- `chroma/` — vector DB directory (auto-generated)
- `repo_map.json` — snapshot of functions/classes for context assembly
- `memory.md` — git-diff summaries, last 10 entries
- `metrics.json` — observability (query latency, token counts, indexing stats)
- `.mcp.json` — MCP server config (auto-written at init)

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

- **Local-first by default**: Code stays on your machine. The gateway only sends queries to the
  cloud (no code unless you explicitly enable cloud reranking).
- **No telemetry**: Gateway logs to stdout/file, never phones home.
- **Git-aware**: Uses your repo's `.git/` to track commits; memory entries include commit hashes.
- **Configurable routing**: `routing.mode` lets you choose "cloud_only" (always forward), "conservative" (local judgement),
  or "aggressive" (prefer local handling).

## 📖 Full Documentation

- **[RUNBOOK.md](RUNBOOK.md)** — First-time setup, commands reference, troubleshooting
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — Using with Claude Code, OpenCode, or as a proxy
- **[AGENTS.md](AGENTS.md)** — MCP tool reference & usage patterns for AI agents

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
