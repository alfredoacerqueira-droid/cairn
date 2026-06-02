# Changelog

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
- One-line install script
- Apache-2.0 license, CI workflow, contributing guide
- 52 new unit tests covering all major modules

### Changed
- Rewrote gateway core with canonical internal model
- Conversation history, tools, and system prompts are fully preserved
- Ollama URL and model names now configurable via env vars
- Context assembly injects as additional system block (no history discard)

## [0.1.0] — Initial

- OpenAI-compatible `/v1/chat/completions` endpoint
- Tree-sitter AST parsing (Python)
- ChromaDB vector indexing with Ollama embeddings
- Background janitor (file watcher + priority job queue)
- Git diff memory summarization
- Resource throttling (CPU, RAM, VRAM priority)
- CLI: init, serve, search, reindex, janitor, dashboard, metrics
