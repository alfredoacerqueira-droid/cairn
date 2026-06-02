# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A semantic context engine that sits between a coding agent (OpenCode) and a cloud LLM. It intercepts prompts, assembles surgical context from a local ChromaDB vector index, and forwards the enriched prompt to a cloud model — reducing token usage by 90%+. It runs as two concurrent processes: a **live Gateway** (high priority, sync) and a **background Janitor** (low priority, async).

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
cairn init                             # create .cairn/config.yaml
cairn reindex --mode quick             # index codebase into ChromaDB
cairn serve --port 8000                # start gateway API
cairn janitor start                    # start background file watcher
cairn search "query" -k 5
cairn dry-run "query" --show-prompt    # preview assembled context without sending
cairn dashboard                        # live metrics
```

Cloud forwarding requires env vars (optional — without them, gateway returns assembled context directly):
```bash
export CLOUD_API_KEY="..."
export CLOUD_API_BASE="https://api.deepseek.com/v1"   # default
export CLOUD_MODEL_NAME="deepseek-chat"                # default
```

## Architecture

### Two-Process Model

```
Janitor (background, low priority)
  FileWatcher → PriorityJobQueue → ASTParser → VectorIndexer → ChromaDB
  MemorySummarizer → .cairn/memory.md

Gateway (foreground, high priority)
  POST /v1/chat/completions → ContextAssembler → VectorIndexer.search()
                                               → RepoManager.load_repo_map()
                                               → RepoManager.load_memory()
                           → cloud API (or local-only if no key)
```

### Module Map

| Directory | Role |
|-----------|------|
| `server/` | FastAPI app (`api.py`), context assembly (`context_assembler.py`), Ollama HTTP client (`ollama_client.py`) |
| `pipeline/` | AST parsing (`ast_parser.py`), ChromaDB indexer (`indexer.py`), file watcher (`watcher.py`), git-diff memory summarizer (`memory.py`), background job queue (`queue.py`) |
| `core/` | Pydantic config loader (`config.py`), session cache (`cache.py`), DB freshness detection (`freshness.py`), metrics (`metrics.py`), repo data manager (`repo.py`) |
| `throttle/` | CPU (`cpu.py`), RAM (`memory.py`), VRAM priority (`vram.py`) |
| `cli/` | Click CLI entry point (`main.py`) — all commands live here |

### Key Design Decisions

**VRAM priority** (`throttle/vram.py`): `VRAMPriority` is a simple mutex-like object. Gateway calls `vram.request("gateway")` which always succeeds and blocks janitor. Janitor checks before running each job and backs off if gateway is active.

**Session cache** (`core/cache.py`): In-memory TTL cache. Cache keys include the current git commit hash (fetched via subprocess), so stale results are automatically evicted on each commit. Used for both raw embeddings and assembled prompts.

**ChromaDB IDs**: Each indexed function has the ID `filepath:function_name:line_start`. Methods are stored as `filepath:ClassName.method_name:line_start`. The collection uses cosine similarity.

**Context assembly** (`server/context_assembler.py`): `assemble()` pulls three sources — (1) top-K semantically similar functions, (2) repo map (up to 20 files of classes/functions), (3) last 10 memory entries — and formats them as a structured markdown prompt injected as the `system` message.

**Per-project data** (`core/repo.py`, `core/config.py`): All state lives in `.cairn/` within the target project. Config is `config.yaml`, vector DB is `chroma/`, repo structure snapshot is `repo_map.json`, git-diff summaries are `memory.md`, metrics are `metrics.json`. None of these are version-controlled (see `.gitignore`).

**Stale DB detection** (`core/freshness.py`): Tracks last-indexed commit in memory (not persisted across restarts). Uses `git rev-list --count from..HEAD` to count how far behind the index is. Thresholds (quick: 1000 commits, full: 10000) trigger CLI warnings.

**AST parsing** (`pipeline/ast_parser.py`): Uses tree-sitter deterministically — no AI. Parses Python only. Extracts top-level functions, classes, and methods (including decorated definitions). Nested functions are recursed into. `diff_update()` always does a full re-parse (incremental diffing is not implemented).

**Background job queue** (`pipeline/queue.py`): `PriorityJobQueue` wraps Python's `queue.PriorityQueue`. Jobs that fail resource checks (CPU/RAM/VRAM) are re-queued with slightly lower priority and retried. Workers run in a single daemon thread.

### OpenCode Integration

`opencode.json` at the project root configures OpenCode to use the gateway as an OpenAI-compatible provider at `http://127.0.0.1:8000/v1`. The model name `smart-context` maps to the gateway's `/v1/chat/completions` endpoint.

## Code Style

- Line length: 100 characters (black + ruff both configured)
- Python 3.10+, uses `X | Y` union syntax and `match` statements where appropriate
- `asyncio_mode = "auto"` in pytest — async test functions work without decorators
- Ruff rules: E, F, I, N, W
