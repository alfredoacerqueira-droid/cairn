# Changelog

## [0.6.0] — 2026-06-05 (Current)

Major architectural redesign: removed HTTP gateway, now pure CLI + MCP.

### Added
- **MCP server** (`cairn mcp`): pure Model Context Protocol on stdio
- **IndexStore abstraction**: ChromaDB (default) + LanceDB (optional, `[local]` extra)
- **Index location strategy**: auto-place heavy DB on native FS on WSL, in-project elsewhere
- **Token budgeting** (`core/tokens.py`): real tiktoken, session window + per-tool caps
- **Semantic cache** (`core/semantic_cache.py`): local, embedded, exact + semantic lookup
- **Smart orchestrator** (`server/orchestrator.py`): routes work (context-only / local-LLM / map-reduce / cloud)
- **Six MCP tools**: search_code, assemble_context, orchestrate, set_profile, cache_get, cache_set
- **Workspace binding**: SINGLE / WORKSPACE / UNBOUND modes for multi-repo setups
- **Offline mode**: pure structural + lexical (no embeddings, no Ollama required)

### Removed
- ❌ HTTP gateway (`cairn serve`, `/v1/chat/completions`, `/v1/messages`)
- ❌ Cloud API forwarding (CLOUD_API_KEY, CLOUD_API_BASE, CLOUD_MODEL_NAME)
- ❌ OpenAI/Anthropic proxy endpoints
- ❌ Streaming SSE responses

### Changed
- CLI commands now focus on local operations: init, reindex, search, assemble_context, janitor, mcp
- Config now includes IndexStore backend, index_location, token budgets, local_llm settings
- Profiles now support embeddings_enabled toggle (IaC = OFF by default)
- MCP is the sole agent integration point (no HTTP alternative)

## [0.2.0] — 2026-05-30

### Added
- Dual API format support: `/v1/chat/completions` (OpenAI) + `/v1/messages` (Anthropic)
- Full streaming SSE for both agents with tool-calling pass-through
- Hybrid retrieval: BM25 lexical + AST-graph PageRank + embeddings + RRF fusion
- Configurable retrieval strategy (embeddings | bm25 | ast | hybrid)
- `cairn doctor` pre-flight checks
- `cairn start-all` orchestrator
- `serve --background` PID-managed background server
- Benchmark suite: retrieval eval (Recall/MRR/nDCG), token reduction, latency
- Docker + docker-compose stack (Ollama + gateway + janitor)

### Changed
- Rewrote gateway core with canonical internal model
- Conversation history, tools, and system prompts are fully preserved
- Ollama URL and model names now configurable via env vars

## [0.1.0] — Initial

- OpenAI-compatible `/v1/chat/completions` endpoint
- Tree-sitter AST parsing (Python)
- ChromaDB vector indexing with Ollama embeddings
- Background janitor (file watcher + priority job queue)
- Git diff memory summarization
- Resource throttling (CPU, RAM, VRAM priority)
- CLI: init, serve, search, reindex, janitor, dashboard, metrics
