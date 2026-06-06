# Work Environment Smoke Checklist

A manual checklist to validate Cairn on a real development machine (conditions CI cannot fully reproduce).

**Use this after initial setup to confirm the full workflow functions end-to-end.**

---

## Prerequisites

- macOS or Linux development machine (not WSL/virtualized if possible)
- At least one real project repo (not test fixtures)

---

## Checklist

### 1. Install via uv with --system-certs

```bash
uv tool install /path/to/cairn --force --system-certs
which cairn
```

**Expected:** `cairn` binary is in PATH; `cairn --help` works.

- [ ] Install successful
- [ ] Binary in PATH

---

### 2. Run cairn doctor

```bash
cairn doctor
```

**Expected:** All checks show `[✓]` or `[i]`. No `[✗]` critical failures.

Report should include:
- Python version (≥3.10)
- Ollama reachable
- Embedding model found
- Disk space adequate
- Git repo detected
- ChromaDB writable

- [ ] All critical checks pass
- [ ] Interpreter shown correctly
- [ ] GPU/VRAM info present (if available)

---

### 3. Initialize a Real Project

```bash
cd /path/to/real/project
cairn init
```

**Expected:**
- Profile auto-detected (iac/dotnet/python/shell/code — should match repo type)
- Config written to `.cairn/config.yaml`
- opencode.json and .mcp.json scaffolded
- Index built (unless `--no-index` used)

- [ ] Profile detected correctly
- [ ] Config file exists
- [ ] MCP configs created (opencode.json, .mcp.json)
- [ ] Indexing completed (if attempted)

---

### 4. Verify Fresh Reindex Completes

```bash
cairn reindex --mode full
```

**Expected:** Completes without hanging; shows progress; final count of indexed functions.

- [ ] Reindex completed
- [ ] No timeout/hang
- [ ] Functions indexed > 0

---

### 5. Verify OpenCode Integration

Open the project in OpenCode/Claude Code:

```bash
# Verify the MCP config loads without schema errors
cat opencode.json | jq .mcp
```

**Expected:**
- `cairn` entry present
- `command` is array or string
- `env.CAIRN_PROJECT` is absolute path

- [ ] opencode.json valid JSON
- [ ] cairn MCP entry present
- [ ] No schema validation errors in editor

---

### 6. Cross-Repo Query Test (if Workspace)

If working in a monorepo with multiple projects:

```bash
# From workspace root, initialize each project
cd project-a && cairn init
cd ../project-b && cairn init

# Query within each project
cd project-a
cairn search "some query" -k 5

cd ../project-b
cairn search "different topic" -k 5
```

**Expected:**
- Project A queries return only Project A files
- Project B queries return only Project B files
- No cross-contamination

- [ ] Project A isolated
- [ ] Project B isolated
- [ ] No cross-repo results

---

### 7. Verify Local LLM OFF Works (Default)

```bash
# Confirm local_llm.enabled is false (default)
grep -A 5 "local_llm:" .cairn/config.yaml
```

**Expected:**
- `enabled: false` (default)
- Embeddings work without local LLM

```bash
cairn search "test query" -k 3
```

- [ ] Config shows local_llm.enabled: false
- [ ] Search works without LLM
- [ ] No Ollama required for basic retrieval

---

### 8. Optional: Test with Local LLM Enabled

If Ollama is running:

```bash
# Edit config to enable local LLM
vim .cairn/config.yaml
# Set: local_llm.enabled: true

# Reindex with embeddings
cairn reindex --mode full

# Test semantic search
cairn search "function behavior" -k 5
```

**Expected:**
- Embeddings generated without error
- Semantic search returns relevant results
- No VRAM thrashing

- [ ] Local LLM enabled and working
- [ ] Embeddings generated
- [ ] Search quality acceptable

---

## Validation Summary

If all checkboxes are complete:
✓ Cairn is correctly installed and configured for real-world use.
✓ Profile detection and MCP integration work.
✓ Isolation between projects is maintained.
✓ Both offline (no local LLM) and online (with Ollama) modes function.

---

## Troubleshooting

### cairn doctor fails
- Ensure Python ≥3.10: `python3 --version`
- Ensure Ollama is running: `ollama serve` in another terminal
- Check disk space: `df -h` (need ≥2GB free)

### cairn init hangs
- Check Ollama is responsive: `curl http://127.0.0.1:11434/api/tags`
- Watch memory: `watch -n 1 'ps aux | grep ollama'`
- Allow 5–10 minutes for first index

### opencode.json schema errors
- Validate JSON: `cat opencode.json | jq .`
- Check command format: should be array for OpenCode 1.15+
- Ensure CAIRN_PROJECT is absolute path: `readlink -f .cairn/`

### Cross-repo contamination
- Confirm each project has its own `.cairn/` directory
- Verify `CAIRN_PROJECT` env var is set correctly in MCP config
- Check `.cairn/config.yaml` profile is project-specific

---

## Next Steps

- Integrate with OpenCode / Claude Code
- Run `cairn dashboard` for real-time metrics
- Customize profile and retrieval strategy as needed
