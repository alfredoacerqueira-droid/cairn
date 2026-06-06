# Integration Guide: Using Cairn with AI Agents

## Overview

Cairn integrates with AI agents via **MCP (Model Context Protocol)**. It exposes six tools as native MCP resources that Claude Code and OpenCode can invoke directly.

**Why MCP?** Agents have full control over when/how to use Cairn. No proxy wiring or HTTP forwarding needed.

---

## Setup (Both Claude Code & OpenCode)

### 1. Initialize the project

```bash
cd /path/to/your-project
cairn init
```

This auto-scaffolds:
- `.cairn/config.yaml` — Per-project retrieval strategy
- `opencode.json` — MCP config for OpenCode
- `.mcp.json` — MCP config for Claude Code

### 2. Verify MCP configs

```bash
cat opencode.json
cat .mcp.json
```

Both contain Cairn's MCP entry (auto-written by `init`):

**OpenCode (`opencode.json`):**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cairn": {
      "type": "local",
      "command": ["/path/to/cairn", "mcp"],
      "enabled": true,
      "env": { "CAIRN_PROJECT": "/absolute/path/to/your/project" }
    }
  }
}
```

**Claude Code (`.mcp.json`):**
```json
{
  "mcpServers": {
    "cairn": {
      "command": "/path/to/cairn",
      "args": ["mcp"],
      "env": { "CAIRN_PROJECT": "/absolute/path/to/your/project" }
    }
  }
}
```

### 3. (Optional) Start janitor

For background indexing on file changes:

```bash
cairn janitor start &
```

Agents invoke the MCP server on-demand; no need to start it manually.

---

## MCP Tools (Six Available)

When agents need context, they call Cairn's MCP tools:

- **`search_code(query, top_k=5)`** — Find relevant code blocks by natural language query
- **`assemble_context(query)`** — Get a compressed, ready-to-use context bundle
- **`orchestrate(query, instruction="", payload="")`** — Smart routing (local vs cloud)
- **`set_profile(name)`** — Override auto-detected repo profile
- **`cache_get(query)`** — Retrieve cached responses
- **`cache_set(query, value, ttl_seconds=300)`** — Store cached responses

### Example: Claude Code

**`.mcp.json`** (auto-written at init):
```json
{
  "mcpServers": {
    "cairn": {
      "command": "/path/to/cairn",
      "args": ["mcp"],
      "env": { "CAIRN_PROJECT": "/absolute/path/to/your/project" }
    }
  }
}
```

Claude Code will see `cairn` in its MCP tool list. It can call:

### Example Usage

**User prompt:**
```
"How does authentication work in this codebase?"
```

**Agent flow:**
```
Agent → calls search_code("authentication", top_k=5)
     → gets back 5 relevant functions with confidence scores
     → if more context needed, calls assemble_context()
     → gets a full prompt block (search + repo map + memory)
     → proceeds with the task using the context
```

---

## Configuration

Adjust retrieval behavior in `.cairn/config.yaml`:

```yaml
profile: python                    # Profile: iac, dotnet, python, shell, code

retrieval:
  mode: hybrid                      # Strategy: hybrid, embeddings, bm25, ast
  rerank_enabled: true              # Use FlashRank reranker
  reranker_type: cross_encoder      # Options: cross_encoder, llm, none
  rerank_min_score: 0.47            # Confidence threshold (0..1)
  min_confidence: 0.82              # Fallback (when reranker off)

compression:
  enabled: true                     # Enable token compression
  level: minimal                    # minimal (20-40%), aggressive (60-90%)

indexing:
  file_patterns: ["*.py", "*.go"]   # Files to index
  embedding_model: nomic-embed-text # Embedding model (when enabled)
```

### Tuning for Your Repo

1. **Check detected profile:**
   ```bash
   cairn profile
   ```

2. **If wrong, override:**
   ```bash
   cairn profile set iac
   cairn reindex --mode full
   ```

3. **Adjust confidence threshold** if you're getting too much/too little context:
   ```bash
   # Edit .cairn/config.yaml
   retrieval:
     rerank_min_score: 0.50  # Higher = stricter (less context)
   ```

4. **See the assembled context:**
   ```bash
   cairn dry-run "your query" --show-prompt
   ```

---

## Troubleshooting

### Agent doesn't see the tools

Check that `.mcp.json` or `opencode.json` exists and has the correct path:

```bash
cat .mcp.json
```

Run `cairn mcp` manually to test:
```bash
cairn mcp
# Should start the MCP server (waits on stdin)
# Ctrl+C to exit
```

### Context Not Injected / Off-Topic Results

Check the rerank confidence threshold:

```bash
cairn dry-run "your query" --show-prompt
```

If you're getting "No confident matches" but `search_code` found things, Cairn determined the results were off-topic. Adjust the threshold:

```yaml
retrieval:
  rerank_min_score: 0.30  # More lenient (was 0.47)
```

---

## Privacy & Security

**All code stays local.** MCP server runs on stdio (not HTTP); no network traffic from Cairn itself.

- ✓ Index stays on your machine (`.cairn/` or `~/.cache/cairn/`)
- ✓ Queries are processed locally
- ✓ Agents control which queries trigger searches
- ✓ Sensitive data should be excluded via `indexing.exclude_patterns`

**Recommendation:** Keep sensitive files out of indexing:

```yaml
indexing:
  exclude_patterns:
    - "**/.env"
    - "**/secrets/**"
    - "**/credentials.json"
    - "**/private_keys/**"
```

---

## Next Steps

- [RUNBOOK.md](RUNBOOK.md) — Full setup guide
- [README.md](README.md) — Architecture overview
- [AGENTS.md](AGENTS.md) — MCP tool reference (detailed)
- [docs/CONFIG.md](docs/CONFIG.md) — Full configuration schema

