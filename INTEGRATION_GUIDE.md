# Integration Guide: Using the Gateway with AI Agents

## Overview

The Cairn integrates with AI agents via **two complementary patterns**:

1. **MCP (Model Context Protocol) — RECOMMENDED**
   - Native tool integration: agents call `search_code()`, `assemble_context()`, `set_profile()` directly
   - No proxy configuration needed
   - Agents: Claude Code, OpenCode
   - How it works: Gateway runs as MCP server on stdio; agents invoke it as a native tool

2. **HTTP Proxy (OpenAI/Anthropic compatible) — OPTIONAL**
   - Agent thinks it's talking to the real API; gateway intercepts and enriches
   - Agents: Claude Code (Anthropic), OpenCode (OpenAI), any compatible client
   - How it works: Gateway runs as HTTP server; agent points `ANTHROPIC_BASE_URL` or `OPENAI_API_KEY` at it

---

## Option 1: MCP (Recommended)

MCP is the **primary, recommended integration**. Agents can use the gateway as a native tool with full control over when/how to call it.

### Setup

1. **Initialize the project** (MCP configs auto-scaffolded):
   ```bash
   cd /path/to/your-project
   cairn init
   ```

2. **Verify MCP configs were written:**
   ```bash
   cat opencode.json
   cat .mcp.json
   ```

   Both should contain the gateway's MCP entry (auto-written by `init`).

3. **Start the gateway** (no separate MCP server needed — agents will launch it):
   ```bash
   cairn run
   ```

   Or let the agent launch it on-demand.

### How It Works

When the agent needs context, it calls one of three MCP tools:

- **`search_code(query, top_k=5)`** — Find relevant code blocks by natural language query
- **`assemble_context(query)`** — Get a compressed, ready-to-use context bundle for a task
- **`set_profile(name)`** — Override the auto-detected repo profile if needed

### Example: Claude Code

**`.mcp.json`** (auto-written at init):
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

Claude Code will see `cairn` in its MCP tool list. It can call:

```
User: "How does authentication work in this codebase?"

Claude Code → calls search_code("authentication", top_k=5)
           → gets back 5 relevant functions with line numbers
           → if more context needed, calls assemble_context()
           → gets a full prompt block ready to inject
           → proceeds with the task
```

### Example: OpenCode

**`opencode.json`** (auto-written at init):
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

OpenCode will see `cairn` as an MCP server and can invoke the same three tools.

### Agent Behavior

Agents using MCP can be smart about context retrieval:

1. **On a vague query**, call `search_code()` to explore the codebase
2. **On a specific task**, call `assemble_context()` to get surgical context
3. **If retrieval seems wrong**, call `set_profile()` to switch strategies (e.g., if `init` misdetected a Terraform repo as generic)

---

## Option 2: HTTP Proxy (OpenAI/Anthropic Compatible)

Use the proxy when you want the agent to work **without knowing about** the gateway. The agent routes
through the gateway transparently; the gateway intercepts, enriches, and forwards to the real cloud API.

**Trade-offs:**
- ✓ Transparent to agent (no new tool calls)
- ✗ Less agent control (can't choose when to search)
- ✗ Requires cloud API key + forwarding setup

### Setup

#### Step 1: Start the Gateway Proxy

```bash
cd /path/to/your-project
cairn init              # config + index
cairn serve --port 8000 # start HTTP proxy
```

The gateway now listens on `http://127.0.0.1:8000` and exposes both OpenAI and Anthropic API endpoints.

#### Step 2a: Use with Claude Code (Anthropic)

Set environment variables before launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=dummy  # Gateway doesn't validate auth
export ANTHROPIC_API_KEY=sk-ant-...  # Your real API key for forwarding

claude
```

Claude Code will route `/v1/messages` to the gateway. The gateway will:
1. Intercept the request
2. Extract the user's query from the message
3. Search the codebase for relevant functions
4. Assemble context (search results + repo map + memory)
5. Inject as additional system message
6. Forward to real Anthropic API
7. Stream response back to Claude Code

**Example flow:**

```
Claude Code         Gateway                  Anthropic API
    |                  |                           |
    +--POST /v1/messages-->                        |
    |                  |                           |
    |             (search codebase)                |
    |             (assemble context)               |
    |                  |                           |
    |                  +--POST /v1/messages------->|
    |                  |  (with injected context)  |
    |                  |                           |
    |                  |<-----response (stream)----+
    |                  |                           |
    |<-------response (stream)-------+             |
    |                  |                           |
```

#### Step 2b: Use with OpenCode (OpenAI)

Set environment variables before launching OpenCode:

```bash
export OPENAI_API_KEY=sk-proj-...  # Your real API key for forwarding
export OPENAI_BASE_URL=http://127.0.0.1:8000
export OPENAI_API_VERSION=v1

opencode
```

OpenCode will route `/v1/chat/completions` to the gateway, which enriches and forwards.

### How the Proxy Works

The gateway listens on both endpoints and translates as needed:

| Endpoint | Protocol | Used By |
|----------|----------|---------|
| `POST /v1/messages` | Anthropic | Claude Code, Anthropic SDK |
| `POST /v1/chat/completions` | OpenAI | OpenCode, OpenAI SDK, others |

Both endpoints follow the same flow:

1. **Receive** the agent's request (messages + system prompt + tools)
2. **Extract** the latest user message as the search query
3. **Search** ChromaDB for semantically similar functions (hybrid: BM25 + embeddings + AST)
4. **Rerank** with FlashRank; gate on confidence (default: ≥0.47 cross-encoder score)
5. **Load** repo map (all top-level functions/classes)
6. **Load** recent memory (last 10 git-diff entries)
7. **Assemble** a markdown context block
8. **Compress** the block (lossless: removes boilerplate, shortens IDs)
9. **Inject** as additional system message to the original request
10. **Forward** to real cloud API (or return local-only if `routing.mode` is configured)
11. **Stream** response back to agent

The agent's prompt is **never modified**. Only the system message is enriched. All tool calls and
conversation history are preserved transparently.

### Example: Full Proxy Flow

**Terminal 1: Start gateway**
```bash
cd /path/to/myproject
cairn init
cairn serve --port 8000
```

**Terminal 2: Use Claude Code**
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=sk-ant-xyz...
claude
```

Now when you ask Claude Code a question, it:
1. Sends `/v1/messages` to `http://127.0.0.1:8000`
2. Gateway enriches with context from your codebase
3. Gateway forwards to real Anthropic API
4. Claude Code receives enriched response
5. Works normally — no prompting changes needed!

---

## Comparing MCP vs. Proxy

| Feature | MCP | Proxy |
|---------|-----|-------|
| **Transparency** | Agent is aware of tools | Agent unaware; works like normal |
| **Agent control** | Full: can choose when to search | None: gateway decides |
| **Setup** | Automatic (scaffolded at init) | Manual (configure env vars) |
| **Cloud API required?** | No | Yes (for forwarding) |
| **Works offline?** | Yes (local context only) | No (must forward) |
| **Privacy** | Code stays local | Code sent to cloud (in system message) |

**Recommendation:** Use **MCP** — it's simpler, more flexible, and doesn't require forwarding code to the cloud.

---

## Configuration

Both MCP and proxy share the same retrieval config. Adjust in `.cairn/config.yaml`:

```yaml
profile: python                    # Profile: iac, dotnet, python, shell, code

retrieval:
  mode: hybrid                      # Retrieval strategy: hybrid, embeddings, bm25, ast
  rerank_enabled: true              # Use FlashRank reranker
  reranker_type: cross_encoder      # Options: cross_encoder, llm, none
  rerank_min_score: 0.47            # Confidence threshold (0..1, 0=disabled)
  min_confidence: 0.82              # Fallback confidence (for non-reranked)

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

4. **Measure impact:**
   ```bash
   cairn dry-run "your query" --show-prompt
   ```

---

## Troubleshooting

### MCP: Agent doesn't see the tools

Check that `.mcp.json` or `opencode.json` exists and has the correct path:

```bash
cat .mcp.json
# Should show:
# {
#   "mcpServers": {
#     "cairn": {
#       "command": "cairn",
#       "args": ["mcp"],
#       "env": { "CAIRN_PROJECT": "/absolute/path" }
#     }
#   }
# }
```

Run `cairn mcp` manually to test:
```bash
cairn mcp
# Should start the MCP server (waits on stdin)
# Ctrl+C to exit
```

### Proxy: "Connection refused"

Check the gateway is running:
```bash
curl http://127.0.0.1:8000/health
# Should return {"status":"ok","version":"0.6.0"}
```

If not, start it:
```bash
cairn serve --port 8000
```

### Proxy: "Authorization failed" or "Invalid API key"

The gateway doesn't validate API keys. Use dummy values:

```bash
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=dummy  # Or your real key (gateway forwards it)
```

### Context Not Injected / Off-Topic Results

Check the rerank confidence threshold:

```bash
cairn dry-run "your query" --show-prompt
```

If similarity scores are low, increase the window or adjust:
```yaml
retrieval:
  rerank_min_score: 0.30  # More lenient
  top_k: 10               # Search deeper
```

---

## Privacy & Security

### MCP

- ✓ All code stays local. MCP server runs on stdio; no network traffic.
- ✓ No API forwarding needed.
- ✓ Agents control which queries trigger searches.

### Proxy

- ⚠ Your code is embedded in the system message and sent to the cloud API.
- ✓ System message is compressed (90%+ token reduction).
- ✓ Sensitive data should be excluded via `indexing.exclude_patterns`.

**Recommendation:** Keep sensitive files out of `file_patterns` or add to `exclude_patterns`:

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

