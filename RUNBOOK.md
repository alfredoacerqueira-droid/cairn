# Cairn — Runbook

End-to-end guide for setting up and using the Cairn on a fresh machine.

**Version:** 0.6.0
**Last Updated:** 2026-06-01

## What's New in v0.6.0

- **Repository profiles:** Auto-detects repo type (IaC/dotnet/python/generic) at init. Each profile optimizes
  retrieval strategy. IaC uses structural + lexical (embeddings OFF); dotnet/python use full hybrid.
- **Real tree-sitter parsing:** Native support for HCL, YAML, C#, bash (not just Python), with a regex fallback if a grammar fails to load.
- **Structural retrieval:** Block-level scope analysis for better function isolation in large files.
- **FlashRank reranker:** Cross-encoder confidence guard with measured thresholds (0.47 default on Django).
- **Compression on all paths:** MCP tools, CLI, and proxy all benefit from lossless context compression.
- **Configurable worker model:** Choose the Ollama model for reranking/summarization (qwen2.5-coder:1.5b default).
- **MCP server (`cairn mcp`):** Native integration with Claude Code & OpenCode — no proxy wiring needed.
- **`cairn init` ceremony:** One command auto-detects profile, writes config, scaffolds opencode.json/.mcp.json, builds index.

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | 0.20+ | Local LLM engine (optional; only for embeddings/summarization profiles) |
| Git | 2.30+ | Version control |
| GPU | 6GB+ VRAM | Local model inference (optional; falls back to CPU) |

**Note:** Ollama is **optional**. Indexing and search work without it via lexical+structural retrieval. If using embeddings, choose either Ollama OR fastembed (in-process ONNX).

---

## Step 1: Environment Setup

### 1.1 Install System Dependencies (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip build-essential
```

### 1.2 Install ripgrep (Optional but Recommended)

ripgrep enables fresh exact-match lexical search. Without it, Cairn falls back to in-memory BM25.

```bash
sudo apt install ripgrep
```

Or manually:
```bash
curl -LO https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/ripgrep_14.1.0_amd64.deb
sudo dpkg -i ripgrep_14.1.0_amd64.deb
```

### 1.3 Install Ollama (Optional — Only for Ollama-Based Embeddings)

Ollama is **optional**. Install only if:
- Your profile uses embeddings (python, dotnet, code, shell) AND you prefer Ollama over fastembed
- You want to use local LLM reranking or summarization
- IaC profiles (Terraform, Helm) disable embeddings by default and don't need Ollama

To use embeddings **without** Ollama, install fastembed instead (Section 1.3b).

#### 1.3a. Install Ollama (if preferred)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 1.4 Pull Required Models (Ollama only)

```bash
ollama pull nomic-embed-text qwen2.5-coder:1.5b
```

Verify:

```bash
ollama list
```

Expected output:
```
NAME                      ID              SIZE      MODIFIED
nomic-embed-text:latest   0a109f4668af    274 MB    2 minutes ago
qwen2.5-coder:1.5b        3cc9af7d0e38    943 MB    3 minutes ago
```

### 1.5 Start Ollama (if using Ollama)

```bash
ollama serve &
```

Verify:
```bash
curl http://127.0.0.1:11434/api/tags
```

#### 1.3b. Alternative: Install fastembed (No Ollama)

For in-process embeddings (CPU, no external service), install the local extras:

```bash
pip install -e ".[local]"  # Adds lancedb + fastembed (~100MB total)
```

Then in `.cairn/config.yaml` after init:
```yaml
local_llm:
  embedder: fastembed
  fastembed_model: BAAI/bge-small-en-v1.5
```

This eliminates the Ollama dependency entirely.

### 1.6 Small GPU Tip

If you have limited VRAM (6GB), set Ollama to keep only 2 models in memory:

```bash
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
ollama serve &
```

---

## Step 2: Install Cairn

### 2.1 Via pipx (Recommended — Global, Isolated)

```bash
# Install once globally
pipx install .
```

### 2.2 Via uv (Alternative — PEP 668 & Corporate CA Support)

For systems with PEP 668 restrictions or corporate root CA requirements:

```bash
# Install globally from local path, respecting system certificates
uv tool install . --force --system-certs
```

The `--system-certs` flag is essential when behind corporate proxies (e.g., Zscaler) that inject custom
CA certificates. This tells uv to use the system's certificate bundle for HTTPS validation.

### 2.3 Via venv (Development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**⚠️ DO NOT use `pip install --break-system-packages`** on system Python. Use `pipx`, `uv`, or a venv.

### 2.4 Verify Installation

```bash
cairn --help
```

---

## Step 3: Initialize a Project

Navigate to the project you want to index:

```bash
cd /path/to/your-project
cairn init
```

This command:
1. Runs pre-flight checks (Python version, Ollama, disk space, Git, ChromaDB writable)
2. Auto-detects source layout
3. **Auto-detects repository profile** (iac/dotnet/python/shell/code)
4. Creates `.cairn/config.yaml` with profile-driven settings
5. **Scaffolds opencode.json and .mcp.json** (auto-written)
6. Builds the vector index (unless `--no-index` is passed)

Expected output:

```
╔════════════════════════════════════════════════════════════╗
║         Cairn Initialization                         ║
╚════════════════════════════════════════════════════════════╝

1. Pre-flight checks:
  [✓] Python: 3.12.3 (need ≥3.10)
  [✓] Ollama: reachable
  [✓] Disk: 45GB free (need ≥2GB)
  [✓] Git: repository found
  [✓] ChromaDB: writable

2. Embedding model check:
  [✓] Embedding model found

3. Detecting source layout:
  Detected source roots: ['.']
  Detected languages: ['*.py', '*.go', '*.rs']

3b. Detecting repository profile:
  Detected profile: python
  Retrieval strategy: hybrid
  Embedding models: ON
  Embedding model: nomic-embed-text
  Description: Python (Django, FastAPI, etc.)...

4. Writing configuration:
  Created new config with profile settings
  ✓ Config saved to .cairn/config.yaml
  ✓ Created .cairn/.gitignore

4b. Scaffolding MCP client configs:
  ✓ Wrote opencode.json (MCP config)
  ✓ Wrote .mcp.json (MCP config)

5. Building index:
  Indexing (quick mode)  [##########] 100%  156/156  0:02:15
  ✓ Indexed 156 functions from 156 files

╔════════════════════════════════════════════════════════════╗
║               Initialization Complete!                      ║
╚════════════════════════════════════════════════════════════╝

Ready to go! Start serving with:
  cairn run
```

### 3.1 Override Auto-Detected Profile (Optional)

If auto-detection guessed wrong:

```bash
cairn profile set iac
cairn reindex --mode full
```

Available profiles: `iac`, `dotnet`, `python`, `shell`, `code`

---

## Step 4: Understand the Doctor Command

```bash
cairn doctor
```

This checks your environment and reports:

```
== Cairn Doctor ==

[i] Cairn code: /usr/local/lib/python3.12/site-packages/cairn
[i] Interpreter: /home/user/.venv/bin/python3.12

[✓] Python: 3.12.3 (need ≥3.10)
[✓] Ollama: reachable
[✓] Embedding model (nomic-embed-text or similar): found
[✓] Generation models: 2 available
[✓] Disk: 45GB free (need ≥2GB)
[✓] Git: repository found
[✓] ChromaDB: writable
[✓] ripgrep: found (fresh exact-match search)
[✓] Reranker (FlashRank): available

[i] Tip: on a small GPU set OLLAMA_MAX_LOADED_MODELS=2 (+ OLLAMA_KEEP_ALIVE=30m)
   → keeps embedder + worker resident, avoiding slow model-swap reloads

All checks passed. Ready to run!
```

**Key lines to understand:**
- **Cairn code:** Shows where Cairn is installed (helps debug editable installs)
- **Interpreter:** Shows which Python binary is being used (catches venv issues)
- **Embedding model:** Found only if Ollama is running and profile uses embeddings
- **ripgrep:** Optional; if missing, use in-memory BM25 fallback
- **Reranker:** FlashRank availability for confidence guard
- **OLLAMA_MAX_LOADED_MODELS:** Tip for small GPUs — keeps models resident to avoid reload latency

---

## Step 5: Repository Profiles

Cairn auto-detects your repo type and optimizes retrieval accordingly:

### Profile Strategies

| Profile | File Types | Embeddings | Retrieval Legs | Use Case |
|---------|-----------|-----------|--------|----------|
| **iac** | .tf, .hcl, .yaml, .yml, .sh, .bash | **OFF** | Structural + lexical | Terraform, Helm, Kubernetes |
| **dotnet** | .cs | **ON** (qwen3-embedding:0.6b) | Embeddings + lexical + structural | C#, MediatR, Roslyn |
| **python** | .py | **ON** (nomic-embed-text) | Embeddings + lexical + structural | Django, FastAPI |
| **shell** | .sh, .bash | **OFF** | Lexical + structural | Shell scripts |
| **code** | .js, .ts, .go, .rs, .java, .cpp, .rb, .toml | **ON** | Embeddings + lexical + structural | Generic: JS/TS, Go, Rust, Java, C++, Ruby |

### Why Embeddings OFF for IaC?

Terraform resources (aws_instance, google_compute_instance) can be semantically similar without being
equivalent in context. Structural + lexical retrieval (exact name matching + block boundaries) works
better. Embeddings add VRAM overhead without improving recall.

### Check Current Profile

```bash
cairn profile
```

### Change Profile

```bash
cairn profile set iac
cairn reindex --mode full  # Must rebuild to switch embedding strategy
```

---

## Step 6: Search and Preview Context

### 6.1 Search Semantically

```bash
cairn search "how does authentication work" -k 5
```

### 6.2 Preview Full Assembled Context

To see what would be sent to the cloud model (search results + repo map + memory + token compression):

```bash
cairn dry-run "how does auth work" --show-prompt
```

Omit `--show-prompt` to just see token savings without the full prompt.

---

## Step 7: Start the Janitor (Background Indexing)

### 7.1 Start the Janitor

```bash
cairn janitor start
```

Runs in the background, watching for file changes and commits. Logs to `.cairn/janitor.log`.

### 7.2 Stop the Janitor

```bash
cairn janitor stop
```

### 7.3 View Janitor Status

```bash
cairn status
```

Shows whether janitor is running, index freshness, and cache state.

---

## Step 8: Using with Claude Code or OpenCode (MCP)

Cairn automatically scaffolds MCP configs at init. Both agents can now use it as a native tool:

### OpenCode (opencode.json — auto-written)

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

### Claude Code (.mcp.json — auto-written)

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

Both agents can now call these MCP tools:
- `search_code(query, top_k=5)` — Find relevant code by query (semantic + lexical + structural)
- `assemble_context(query)` — Get full surgical context (search + repo map + memory)
- `orchestrate(query, instruction="", payload="")` — Smart routing (local vs cloud)
- `set_profile(name)` — Switch profile if auto-detect was wrong
- `cache_get(query)` — Retrieve cached responses
- `cache_set(query, value, ttl_seconds=300)` — Store cached responses

See [AGENTS.md](AGENTS.md) for detailed tool guidance.

---

## Step 9: Observability

### 9.1 Live Dashboard

```bash
cairn dashboard
```

Refresh every 2 seconds:
```bash
cairn dashboard -w 2
```

Shows: system health, indexing stats, search latency, token compression, recent activity.

### 9.2 Detailed Metrics

```bash
cairn metrics
```

Shows: indexing performance, search latency percentiles, compression stats, system snapshots.

### 9.3 Raw Metrics JSON

```bash
cat .cairn/metrics.json
```

---

## Troubleshooting

### Ollama Not Reachable (If Using Embeddings Profile)

```bash
curl http://127.0.0.1:11434/api/tags
```

If this fails:
1. Check Ollama is running: `ps aux | grep ollama`
2. Restart Ollama: `pkill ollama; sleep 2; ollama serve &`
3. Check logs: `cat ~/.ollama/logs/server.log` (if available)

Note: Only needed if your profile uses embeddings (python, dotnet, code, shell). IaC profiles don't require Ollama.

### Indexing Hangs

If using Ollama embeddings, it may be slow on first run. Check progress:

```bash
# In another terminal, watch memory usage
watch -n 1 'ps aux | grep ollama | head -3'
```

Give it 5–10 minutes for the first index.

### Vector DB Errors

```bash
# For ChromaDB
rm -rf .cairn/chroma
cairn reindex --mode full

# For LanceDB
rm -rf ~/.cache/cairn/<project-id>/lancedb
cairn reindex --mode full
```

### Stale Database

```bash
cairn status
```

If commits_behind is high, reindex:

```bash
cairn reindex --mode full
```

---

## Benchmarking

Real measured numbers (honest caveats):

- **Token reduction:** 95–99% across Python, Go, Rust, IaC repos at good recall
- **Terraform retrieval:** ~60% top-1 recall with IaC profile (structural + lexical)
  vs. ~17% with BM25-only. The 40% gap is because some resources require context (e.g., variable references).
- **Confidence guard:** FlashRank separates relevant queries (0.92 mean) from off-topic (0.007 mean)
  with perfect separation at 0.47 threshold (measured on Django, 8331 functions).
- **Search latency:** 5–50ms for semantic search, <10ms for lexical (ripgrep)

To reproduce on your repo:

```bash
python -m benchmarks.benchmark_ast_parser  # AST parsing speed
python -m benchmarks.benchmark_retrieval    # Recall@K, MRR, token reduction
```

---

## Next Steps

- Read [README.md](README.md) for architecture overview
- Read [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for proxy setup
- Read [AGENTS.md](AGENTS.md) for MCP tool reference

