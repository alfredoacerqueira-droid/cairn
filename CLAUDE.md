# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A local-first semantic context engine for AI coding agents (Claude Code, OpenCode). It indexes your codebase into a vector DB, retrieves surgically-relevant functions for queries, and exposes them via CLI and MCP tools. Reduces tokens sent to cloud LLMs by 90%+ while keeping all code on your machine. Runs as two processes: an **MCP server** (on-demand, high priority) and a **background Janitor** (file watcher → indexer, low priority).

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest
pytest tests/unit/test_ast_parser.py          # single test file
pytest -k "test_function_name"                # single test by name
pytest --cov=server --cov=pipeline --cov=core --cov=throttle

# Lint / format / typecheck
ruff check server/ pipeline/ cli/ core/ throttle/
black server/ pipeline/ cli/ core/ throttle/ tests/
mypy server/ pipeline/ cli/ core/ throttle/

# Benchmarks
python benchmarks/benchmark_ast_parser.py

# CLI (from project root with .venv active)
cairn init                             # create .cairn/config.yaml + MCP configs
cairn reindex --mode quick             # index codebase into ChromaDB
cairn mcp                               # start MCP server (stdio) for agents
cairn janitor start                    # start background file watcher
cairn search "query" -k 5
cairn doctor                           # diagnose environment
cairn dashboard                        # live metrics
```

Optional: local LLM for embeddings/memory (disabled by default):
```bash
# Requires Ollama running (ollama serve &)
# Embeddings model: nomic-embed-text
# Memory summarizer: qwen2.5-coder:1.5b
# (Configured in .cairn/config.yaml local_llm section)
```

## Architecture

### Two-Process Model

```
MCP Server (on-demand, high priority, stdio)
  - Exposes: search_code, assemble_context, orchestrate, set_profile, cache_get, cache_set
  - Claude Code / OpenCode call these as native tools
  - Instantiates: ContextAssembler, SemanticCache, Orchestrator

Janitor (background, low priority, async)
  FileWatcher → PriorityJobQueue → ASTParser → IndexStore.upsert()
              (IndexStore = ChromaStore | LanceStore)
  MemorySummarizer → .cairn/memory.md
```

### Module Map

| Directory | Role |
|-----------|------|
| `server/` | MCP server (`mcp_server.py`), context assembly (`context_assembler.py`), orchestrator (`orchestrator.py`), workspace router (`workspace_router.py`), Ollama client (`ollama_client.py`) |
| `pipeline/` | AST parsing (`ast_parser.py`), index store backends (`store/chroma.py`, `store/lance.py`), file watcher (`watcher.py`), git-diff memory summarizer (`memory.py`), background job queue (`queue.py`) |
| `core/` | Pydantic config loader (`config.py`), semantic cache (`semantic_cache.py`), token budgeting (`tokens.py`), DB freshness detection (`freshness.py`), metrics (`metrics.py`), repo data manager (`repo.py`), profiles (`profiles.py`) |
| `throttle/` | CPU (`cpu.py`), RAM (`memory.py`), VRAM priority (`vram.py`) |
| `cli/` | Click CLI entry point (`main.py`) — all commands live here |

### Key Design Decisions

**MCP server** (`server/mcp_server.py`): Exposes six tools as native MCP resources. Claude Code / OpenCode invoke these directly over stdio. Supports SINGLE (one repo), WORKSPACE (multi-repo), and UNBOUND (error) binding modes.

**IndexStore abstraction** (`pipeline/store/`): Two backends (ChromaDB default, LanceDB optional via `[local]` extra) behind a common interface. Flips via `config.yaml` `indexing.store_backend`. Both achieve quality-equivalent retrieval; Chroma is leaner, Lance adds native hybrid + versioning.

**Index location strategy** (`core/config.py`): `indexing.index_location` can be `auto` (default), `native`, or `in_project`. On WSL with projects on `/mnt/*`, `auto` places the heavy DB on Linux native fs (`~/.cache/cairn/<project-id>/`) while source stays on `/mnt/c`. Elsewhere keeps index in-project (`.cairn/`). Config/memory/repo_map always stay in `.cairn/`.

**Semantic cache** (`core/semantic_cache.py`): Local, embedded, short TTL (~300s). Exact + semantic lookup (via embeddings). SET by local LLM, GET by cloud LLM via MCP `cache_set`/`cache_get` tools. No Redis.

**Token budgeting** (`core/tokens.py`): Real tiktoken (cl100k_base Sonnet proxy). `BudgetConfig`: session_window 200K, session_pct 0.18 (caps Cairn's session-start contribution to ~36K). Tool outputs capped at 8K each. Memory loading respects budget (newest entries survive).

**Orchestrator** (`server/orchestrator.py`): Routes work in tokens — context-only / one local-LLM call / map-reduce split for big jobs / defer-to-cloud. Respects token budget and session state.

**Per-project data** (`core/repo.py`, `core/config.py`): All state lives in `.cairn/` within the target project. Config is `config.yaml`, vector DB location in `indexing.index_location`, repo structure snapshot is `repo_map.json`, git-diff summaries are `memory.md`, metrics are `metrics.json`. None of these are version-controlled (see `.gitignore`).

**Stale DB detection** (`core/freshness.py`): Tracks last-indexed commit in memory (not persisted across restarts). Uses `git rev-list --count from..HEAD` to count how far behind the index is. Thresholds (quick: 1000 commits, full: 10000) trigger CLI warnings.

**AST parsing** (`pipeline/ast_parser.py`): Uses tree-sitter deterministically — no AI. Parses Python and other languages (via tree-sitter-language-pack). Extracts top-level functions, classes, and methods. Nested functions are recursed into. `diff_update()` does full re-parse (incremental not implemented).

**Background job queue** (`pipeline/queue.py`): `PriorityJobQueue` wraps Python's `queue.PriorityQueue`. Jobs that fail resource checks (CPU/RAM/VRAM) are re-queued with slightly lower priority and retried. Workers run in a single daemon thread.

### MCP Tool Integration

`cairn mcp` exposes these tools to Claude Code / OpenCode:
- `search_code(query, top_k)` — semantic + lexical search
- `assemble_context(query)` — full context assembly (search + repo map + memory)
- `orchestrate(query, instruction, payload)` — smart routing (local vs cloud)
- `set_profile(profile_name)` — switch profile (iac/python/dotnet/shell/code)
- `cache_get(query)` — retrieve cached responses
- `cache_set(query, value, ttl_seconds)` — store cached responses

## Code Style

- Line length: 100 characters (black + ruff both configured)
- Python 3.10+, uses `X | Y` union syntax and `match` statements where appropriate
- `asyncio_mode = "auto"` in pytest — async test functions work without decorators
- Ruff rules: E, F, I, N, W
