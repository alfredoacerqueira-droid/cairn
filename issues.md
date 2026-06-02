# Cairn Setup Issues Log

Captured during initial setup against `eks-migration` multirepo (dotrez-helm-charts, dotrez-ops, kubernetes-configs).

---

## Issue 1: Profile Auto-Detection Misclassifies Helm/Kubernetes Repos

**Severity:** High  

**Symptom:** `cairn init` detected profile `code` (embeddings ON, hybrid retrieval) for `dotrez-helm-charts` which contains only `.yaml`, `.sh`, `.json` files (Helm charts + Kubernetes manifests).  

**Expected:** Profile `iac` (embeddings OFF, structural + lexical retrieval) — explicitly listed in the runbook for Helm/Kubernetes.  

**Root cause:** Profile detection likely keys on `.tf`/`.hcl` extensions for IaC. Repos with only YAML/JSON/shell fall through to `code` profile.

**Example config written (wrong):**

```yaml

profile: code

embeddings_enabled: true

retrieval:

  mode: hybrid

```

**Expected config (correct for Helm/K8s):**

```yaml

profile: iac

embeddings_enabled: false

retrieval:

  mode: lexical

```

**Workaround applied:** Manually edited `.cairn/config.yaml` for each repo, then ran `cairn reindex --mode full`.

**Suggested fix:** Extend IaC profile detection to trigger on repos where >50% of indexed files are `.yaml`/`.yml` AND the repo contains `Chart.yaml`, `kustomization.yaml`, or Kubernetes `kind:` manifest patterns.

---

## Issue 2: Default Worker Model Too Small for Available Hardware

**Severity:** Medium  

**Symptom:** `cairn init` writes `compaction_model: qwen2.5-coder:1.5b` regardless of available VRAM.  

**Hardware:** RTX 2000 Ada, 8188 MiB VRAM — `qwen2.5-coder:7b-instruct-q4_K_M` (4.7GB) already installed and fits comfortably.  

**Impact:** Summaries and memory compaction use a significantly weaker model when better is available.

**Example default (written by init):**

```yaml

memory:

  compaction_model: qwen2.5-coder:1.5b

```

**Corrected value:**

```yaml

memory:

  compaction_model: qwen2.5-coder:7b-instruct-q4_K_M

```

**Workaround applied:** Manually edited `.cairn/config.yaml` on each repo.

**Suggested fix:** `cairn init` should probe Ollama for available models, check available VRAM via `nvidia-smi`, and auto-select the best fitting model (e.g. 7b if >6GB free, 1.5b otherwise).

---

## Issue 3: Large CRD YAML Files Silently Skipped (ChromaDB Batch Limit)

**Severity:** Medium  

**Symptom:** During `cairn reindex --mode full`, several large CRD YAML files are silently skipped with a ChromaDB batch size error.

**Example output:**

```

Skipped .../cert-manager-crds/templates/crds.yaml: ValueError: Batch size of 6712 is greater than max batch size of 5461

Skipped .../values/apps/argo-cd-dotrez/crds/applicationset-crd.yaml: ValueError: Batch size of 14162 is greater than max batch size of 5461

Skipped .../values/apps/argo-cd-hub/crds/applicationset-crd.yaml: ValueError: Batch size of 14162 is greater than max batch size of 5461

Skipped .../values/apps/cert-manager/CRDs/cert-manager.crds.yaml: ValueError: Batch size of 6511 is greater than max batch size of 5461

```

**Root cause:** ChromaDB's `max_batch_size` is 5461. Files parsed into >5461 chunks (e.g. large CRD manifests with hundreds of resources) cannot be inserted in one batch. The indexer does not chunk or split oversized batches.

**Impact:** CRD files (cert-manager, argo-cd) are entirely absent from the index. Queries about CRD schemas will return no results.

**Workaround:** Add these files to `exclude_patterns` in `.cairn/config.yaml` to suppress noisy skip messages (they won't index anyway):

```yaml

indexing:

  exclude_patterns:

    - '**/crds/**'

    - '**/*.crds.yaml'

    - '**/cert-manager-crds/**'

```

**Suggested fix:** In the indexer, split oversized batches into sub-batches of ≤5000 items before calling ChromaDB `add()`.

---

## Issue 4: Indexing Hangs When Embeddings Are ON for Large YAML Repos

**Severity:** Medium  

**Symptom:** `cairn init` on `dotrez-helm-charts` (419 files, 28210 parsed functions) ran for >10 minutes when profile was misdetected as `code` (embeddings ON). Each file required a separate Ollama embedding call.  

**Expected behaviour with `iac` profile:** No embedding calls — indexing completes in seconds via lexical/structural only.  

**Observed with correct `iac` profile:** `cairn reindex --mode full` completed instantly (no embedding calls).

**Impact:** If profile is wrong, init appears to hang. User may kill the process, leaving a partial index.

**Workaround applied:** Set `embeddings_enabled: false`, `profile: iac`, then reindex.

---

## Issue 5: pipx Not Available — uv tool Required

**Severity:** Low (setup friction)  

**Symptom:** RUNBOOK.md recommends `pipx install .` but `pipx` is not installed on this system.  

```

/bin/bash: pipx: command not found

```

System Python (3.12) has PEP 668 restrictions; `pip install pipx --user` also blocked.

**Workaround:** Use `uv tool install /path/to/cairn --force --system-certs` instead.  

Note: `--system-certs` is required due to Zscaler transparent proxy (corporate CA not trusted by default).

**Suggested fix:** Add `uv tool install` as the primary install path in RUNBOOK.md for environments with system Python restrictions or custom CA certs.

---

## Issue 6: opencode.json Written to Wrong Location (Repo Root vs. Workspace Root)

**Severity:** Medium  

**Symptom:** `cairn init` writes `opencode.json` inside each sub-repo (`dotrez-helm-charts/opencode.json`). OpenCode resolves MCP config from the **workspace root** (`eks-migration/`), not each sub-repo root.  

**Impact:** When opening the `eks-migration` workspace in OpenCode, the MCP server for each sub-repo is NOT auto-loaded. Manual config at the workspace root is required.

**Workaround applied:** A single `opencode.json` at the workspace root (`eks-migration/`) must reference each repo's MCP instance with separate names and `CAIRN_PROJECT` env vars.

**Example workspace-root `opencode.json`:**

```json

{

  "$schema": "https://opencode.ai/config.json",

  "mcp": {

    "cairn-helm-charts": {

      "type": "local",

      "command": "/home/meloa/.local/bin/cairn",

      "args": ["mcp"],

      "env": { "CAIRN_PROJECT": "/mnt/c/Users/meloa/Projects/eks-migration/dotrez-helm-charts" }

    },

    "cairn-dotrez-ops": {

      "type": "local",

      "command": "/home/meloa/.local/bin/cairn",

      "args": ["mcp"],

      "env": { "CAIRN_PROJECT": "/mnt/c/Users/meloa/Projects/eks-migration/dotrez-ops" }

    },

    "cairn-k8s-configs": {

      "type": "local",

      "command": "/home/meloa/.local/bin/cairn",

      "args": ["mcp"],

      "env": { "CAIRN_PROJECT": "/mnt/c/Users/meloa/Projects/eks-migration/kubernetes-configs" }

    }

  }

}

```

**Suggested fix:** `cairn init` should detect if it's running inside a sub-directory of a monorepo/workspace and warn the user that the scaffolded `opencode.json` may need to be placed at the workspace root instead.

---

## Issue 7: Indexing Hangs on Windows NTFS Filesystem (via WSL2 /mnt/c/)

**Severity:** High  

**Symptom:** `cairn reindex --mode full` on `kubernetes-configs` (780 YAML files, many >10KB, some up to 62KB, residing on `/mnt/c/` Windows NTFS mount) hangs indefinitely at ~17% CPU with zero output. Process never produces progress lines or completes.

**Failure example (observed verbatim):**

```

$ time cairn reindex --mode full

Re-indexing in full mode...

real    2m0.011s

user    0m5.363s

sys     0m2.666s

```

Only one line of output. The command was killed by a 2-minute timeout — it never printed another line, never indexed a single function, never returned on its own. The `cairn search` call immediately after returned nothing.

**Contrast — repos that succeeded on the same `/mnt/c/` mount:**

| Repo | Files | Avg file size | Outcome |

|------|-------|--------------|---------|

| `dotrez-helm-charts` | 419 YAML | ~1–5 KB | ✅ Indexed (28210 functions) |

| `dotrez-ops` | 362 `.tf`/YAML | ~2–8 KB | ✅ Indexed (957 functions) |

| `kubernetes-configs` | 780 YAML | many 10–62 KB | ❌ Hung, 0 functions indexed |

**Largest offending files (`kubernetes-configs`):**

```

62K  payment-api/argocd/uat/deployment-uat.yaml

62K  payment-api/argocd/sit/deployment-sit.yaml

62K  payment-api/argocd/prod/deployment-prod.yaml

61K  payment-api/argocd/dev/deployment-dev.yaml

58K  payment-api/argocd/lr/deployment-lr.yaml

53K  payment-service/argocd/uat/deployment-uat.yaml

53K  payment-service/argocd/prod/deployment-prod.yaml

```

These are multi-document YAML files containing full Kubernetes `Deployment` manifests with large `env:` arrays (hundreds of environment variables), expanded by ArgoCD at apply time. The tree-sitter YAML parser likely enters a slow or infinite parse on deeply-nested sequences of this size.

**Root cause hypothesis:** `/mnt/c/` WSL2 filesystem I/O is 5–20x slower than native Linux ext4. Combined with tree-sitter's lack of a parse timeout, a single 62KB deeply-nested YAML file can block the indexer thread indefinitely. The AST parser performs a full re-parse of every file on each indexing run with no per-file deadline.

**Workaround options:**

1. Copy `kubernetes-configs` to native Linux filesystem (`~/projects/`) and reindex from there — removes the I/O bottleneck

2. Add `exclude_patterns` to skip the largest files (partial index):

   ```yaml

   indexing:

     exclude_patterns:

       - '**/payment-api/argocd/**'

       - '**/payment-service/argocd/**'

   ```

3. Set a per-file size limit in `.cairn/config.yaml` (not yet supported — see suggested fix)

**Workaround applied:** None yet. `kubernetes-configs` left unindexed.

**Suggested fix:** Add two defences in `pipeline/ast_parser.py`:

1. **Per-file size limit:** Skip files above a configurable threshold (e.g. `max_file_kb: 20`) with a warning.

2. **Per-file parse timeout:** Wrap `tree_sitter.parse()` in a `concurrent.futures.ThreadPoolExecutor` with a timeout (e.g. 10s). If parsing exceeds the deadline, skip the file with a warning rather than blocking the indexer thread forever.

---

## Issue 8: FlashRank Reranker Fails to Download Behind Corporate Proxy (Zscaler SSL)

**Severity:** High  

**Symptom:** On first use of `cairn search` or `assemble_context()`, FlashRank attempts to download the `ms-marco-MiniLM-L-12-v2` cross-encoder model from HuggingFace. Behind a Zscaler transparent proxy, this fails with an SSL certificate verification error and reranking is silently disabled for the session.

**Failure example (observed verbatim):**

```

[WARNING] pipeline.retrieval.reranker: Failed to load FlashRank model:

  HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url:

  /prithivida/flashrank/resolve/main/ms-marco-MiniLM-L-12-v2.zip

  (Caused by SSLError(SSLCertVerificationError(1,

  '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:

  unable to get local issuer certificate (_ssl.c:1000)')));

  reranking disabled

```

**Impact:** The FlashRank cross-encoder is the confidence gating layer for `assemble_context()`. With reranking disabled:

- `assemble_context()` cannot apply the 0.47 confidence threshold

- All queries return "No confident matches" (or raw unfiltered results, depending on implementation)

- The primary MCP tool intended for agent use is effectively broken

**Root cause:** FlashRank's Python package downloads the model on first use from `huggingface.co`. The Zscaler CA certificate is installed at the OS level (`/usr/local/share/ca-certificates/Zscaler_Root_CA.crt`) but Python's `requests`/`httpx` libraries use their own bundled CA bundle (`certifi`) which does not include the Zscaler root CA.

**Workaround:**

Option 1 — Set `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` for the cairn process:

```bash

# In shell or in a wrapper script around cairn

export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/Zscaler_Root_CA.crt

export SSL_CERT_FILE=/usr/local/share/ca-certificates/Zscaler_Root_CA.crt

cairn serve --port 8100

```

Option 2 — Pre-download the model manually (offline install):

```bash

# On a machine with working HTTPS, download and copy to the FlashRank cache dir:

# ~/.cache/ms-marco-MiniLM-L-12-v2/

```

Option 3 — Disable reranking explicitly in config (search still works, confidence gating off):

```yaml

retrieval:

  rerank_enabled: false

```

**Workaround applied:** `REQUESTS_CA_BUNDLE` environment variable needs to be set. Not yet tested.

**Suggested fix:**

1. `cairn serve` / `cairn mcp` should accept `--ca-bundle` flag and/or read `CAIRN_CA_BUNDLE` env var, passed to FlashRank's HTTP client.

2. Alternatively, `cairn doctor` should test the FlashRank model download and warn if it fails, rather than silently disabling reranking at runtime.

3. `cairn init` should offer an `--offline` flag that skips the model download and configures `rerank_enabled: false`.

---

## Issue 9: `cairn search` CLI Hangs Even With Embeddings Disabled

**Severity:** Medium  

**Symptom:** `cairn search "<query>"` from the project directory hangs indefinitely with no output, even when `embeddings_enabled: false` and `retrieval.mode: lexical` are set in `.cairn/config.yaml`. The process spawns ~46 threads and sits at ~17% CPU but never produces output.

**Failure example (observed verbatim):**

```

$ timeout 15 cairn search "service port" -k 3

[no output]

exit: 124   # killed by timeout

```

**Contrast:** The gateway HTTP server starts successfully from the same repo and directory (`cairn serve --port 8100` returns `{"status":"ok","version":"0.6.0"}` at `/health`).

**Root cause (hypothesis):** The CLI search path may unconditionally call Ollama for embeddings regardless of the `embeddings_enabled` flag, or it may be blocked waiting for the FlashRank model to download (Issue 8) even in lexical-only mode. The 46 threads are consistent with FlashRank's `ThreadPoolExecutor` initializing and blocking on a failed network call.

**Impact:** The `cairn search` CLI cannot be used for quick ad-hoc queries. Only the gateway HTTP API and MCP server are reliable for IaC profiles.

**Workaround:** Use `cairn serve` and query via the gateway's `/v1/chat/completions` endpoint, or use the MCP tools via OpenCode.

**Suggested fix:**

1. Guard all embedding/reranking calls behind their respective feature flags in the CLI search path.

2. Apply the same FlashRank CA bundle fix from Issue 8 — if the model download times out or fails, the search should not block.

---

## Summary Table

| # | Issue | Severity | Workaround Available | Fix Needed |

|---|-------|----------|---------------------|------------|

| 1 | IaC profile not detected for YAML-only repos | High | Manual config edit | Profile detection heuristics |

| 2 | Default worker model ignores available VRAM | Medium | Manual config edit | VRAM-aware model selection at init |

| 3 | Large CRDs skipped (ChromaDB batch limit) | Medium | Exclude in config | Split oversized batches |

| 4 | Indexing hangs when embeddings ON for large YAML repos | Medium | Correct profile | Fix profile detection (#1) |

| 5 | pipx not available; uv tool + --system-certs needed | Low | Use uv tool | Update RUNBOOK install instructions |

| 6 | opencode.json written per-repo, not workspace root | Medium | Manual workspace config | Monorepo detection + warning |

| 7 | Indexing hangs on large YAML files via WSL2 /mnt/c/ | High | Copy to ~/projects/ | Per-file parse timeout in AST parser |

| 8 | FlashRank SSL failure behind Zscaler — reranking disabled | High | REQUESTS_CA_BUNDLE env var | `--ca-bundle` flag + `cairn doctor` check |

| 9 | `cairn search` CLI hangs even with embeddings disabled | Medium | Use gateway HTTP API | Guard embedding/rerank calls behind flags |

| 10 | `cairn init` writes invalid `opencode.json` — crashes OpenCode on startup | High | Manual fix (command array + enabled field) | Fix schema in `cairn init` output |

---

## Issue 10: `cairn init` Writes Invalid `opencode.json` Schema — Crashes OpenCode on Startup

**Severity:** High  

**Symptom:** OpenCode fails to start from any workspace where `cairn init` has written an `opencode.json`. Error at startup: `4 of 5 requests failed: Unexpected server error`.

**Failure example (OpenCode log verbatim):**

```

SchemaError: Expected array, got "/home/meloa/.local/bin/cairn"

  at ["mcp"]["cairn-helm-charts"]["command"]

Missing key

  at ["mcp"]["cairn-helm-charts"]["enabled"]

```

**Root cause:** `cairn init` scaffolds `opencode.json` with `command` as a string and a separate `args` array. OpenCode 1.15+ requires `command` to be a single array (command + args combined) and requires an explicit `enabled` boolean field.

**Generated (wrong):**

```json

{

  "mcp": {

    "cairn": {

      "type": "local",

      "command": "/home/meloa/.local/bin/cairn",

      "args": ["mcp"],

      "env": { "CAIRN_PROJECT": "..." }

    }

  }

}

```

**Required (correct):**

```json

{

  "mcp": {

    "cairn": {

      "type": "local",

      "command": ["/home/meloa/.local/bin/cairn", "mcp"],

      "enabled": true,

      "env": { "CAIRN_PROJECT": "..." }

    }

  }

}

```

**Workaround:** Manually edit every `opencode.json` written by `cairn init`:

1. Merge `command` string + `args` array into a single `command` array

2. Add `"enabled": true`

3. Remove the separate `"args"` key

**Suggested fix:** Update the `opencode.json` template in `cairn init` to use the array `command` format and include `"enabled": true`. Also validate against the OpenCode schema before writing.

---

## Benchmark Results — Phase 1 (Baseline) vs Phase 2 (Cairn MCP)

Benchmark run against `eks-migration` multirepo. Four representative IaC tasks.  

Environment: OpenCode 1.15.13, cairn v0.6.0, lexical-only profile (IaC), FlashRank reranker **disabled** (Zscaler SSL failure, Issue 8).

### Scoring Table

| Task | Baseline lines read | Cairn lines read | Baseline confidence | Cairn confidence | Cairn faster? |

|------|--------------------|-----------------|--------------------|-----------------|---------------|

| A — cert-manager TLS duration | ~250 (many files scanned) | ~1830 (MCP) + 0 extra | 8/10 | 8/10 | Marginal — avoided 1 extra file read, but buried in 1830 lines of CRD schema noise |

| B — EKS node group module | 95 (2 files) | ~2000 (MCP noise) + 95 (fallback) | 9/10 | 9/10 | No — 2x lines read, required full fallback |

| C — ArgoCD ApplicationSet | 142 (2 files) | ~1800 (wrong repo) + 142 (fallback) | 9/10 | 9/10 | No — wrong repo returned, required full fallback |

| D — payment-api env var | ~30 (grep) | ~1800 (noise) + 0 extra | 9/10 | 9/10 | No — large noisy context, no improvement |

### Verdict

**Did Cairn reduce files read?**  

No. For Tasks B and C it increased effective lines read because MCP context was large and wrong, requiring full fallback. Only Task A avoided one extra file read — but only by burying the answer in 1800 lines of CRD schema noise.

**Did Cairn improve confidence?**  

No. Confidence was identical (or indistinguishable) in all four tasks.

**Cases where Cairn made things worse:**

1. **Task B (EKS node group) — cross-repo contamination:** `cairn-dotrez-ops` returned results from `dotrez-helm-charts` and `kubernetes-configs`. Completely wrong repositories. Root cause: likely the MCP server ignored `CAIRN_PROJECT` and resolved the project from the process CWD (wherever OpenCode was launched), querying the wrong index. The ChromaDB indices themselves are correctly scoped (confirmed: `dotrez-ops` index contains only `dotrez-ops/` paths), so this is a runtime project resolution bug, not an indexing bug.

2. **Task C (ArgoCD ApplicationSet) — active misinformation from empty index:** `cairn-k8s-configs` is documented as not indexed (`AGENTS.md` says "Not indexed — read files directly"). Instead of returning "No confident matches" it returned `dotrez-ops/scripts/generate.py` with `similarity: 1.00`. An empty or misconfigured index should return nothing; returning confident wrong results is worse than returning nothing.

3. **All tasks — similarity scores of 1.00 on irrelevant files:** Every `assemble_context` call returned context with `similarity: 1.00` on clearly irrelevant files (CRD schemas, buildspec YAMLs, `.mcp.json`). These appear to be raw BM25 lexical scores on common YAML keys (`version`, `env`, `args`) rather than semantic relevance scores. The reranker that should filter these (FlashRank) was disabled due to Issue 8.

**Is the FlashRank reranker working?**  

No — disabled on first use due to Zscaler SSL failure (Issue 8). All "similarity" scores in the output are raw BM25 lexical hits, not cross-encoder rerank scores. This is the root cause of the noise: without the reranker, every file containing common YAML keys scores 1.00 and passes through unfiltered.

### Root Cause Analysis

Two compounding failures caused the poor benchmark results:

1. **Reranker disabled (Issue 8):** FlashRank failed to download on first use. Without the cross-encoder, `assemble_context` returns the top-K BM25 hits unfiltered. BM25 on YAML keys produces high false-positive rates (every YAML file has `version:`, `env:`, `name:`, etc.).

2. **CAIRN_PROJECT resolution bug (new — Issue 11):** The MCP server appears to resolve the project from CWD at runtime, not from the `CAIRN_PROJECT` env var injected by OpenCode. This causes all three named MCP servers to query the same index (whichever repo OpenCode was launched from), making cross-repo queries return wrong-repo results.

**Expected behaviour with reranker working:** The FlashRank cross-encoder would reject BM25 false positives, returning only genuinely relevant results. Tasks A and B would likely improve significantly. The reranker must be fixed before a meaningful re-benchmark.

### Phase 1 Task Answers (Baseline — for reference)

**Task A — cert-manager TLS duration:** Not configured. `values/apps/cert-manager/cert-manager-values-prod.yaml` only sets image tags (11 lines). No `duration` or `renewBefore` key anywhere. Falls back to cert-manager's 90-day default.

**Task B — EKS node group module:** `terragrunt/modules/nodes/main.tf:3` — source: S3 bucket (`eks/v20.37.1.zip///eks-managed-node-group`). Instance type: `instance_types` variable on `worker_nodes` map. Min/max: `min_size`, `max_size`, `desired_size` on same map.

**Task C — ArgoCD ApplicationSet:** Two patterns. Hub-level (`spoke-infra-applicationset`): `matchLabels: managed-by: fr-core-sandbox-dotrez-hub-eu-west-1-dev`. Per-cluster (`applicationset-dev`): no cluster selector — git generator reads `infra.yaml` from the cluster's own path.

**Task D — payment-api env var:** Chart does not exist in this workspace. For an analogous service: (1) `charts/<name>/templates/deployment.yaml`, (2) `charts/<name>/values.yaml`, (3) `values/apps/<name>/base.yaml`, (4) `values/apps/<name>/<env>.yaml`.

---

## Issue 11: MCP Server Ignores `CAIRN_PROJECT` Env Var — Resolves Project from CWD

**Severity:** Critical  

**Symptom:** When multiple named MCP servers are configured in `opencode.json` with different `CAIRN_PROJECT` values, all servers appear to query the same index — whichever repo OpenCode was launched from. The `CAIRN_PROJECT` env var injected per-server is not being used to select the project at query time.

**Observed:** In the benchmark, `cairn-dotrez-ops` (configured with `CAIRN_PROJECT=.../dotrez-ops`) returned results from `dotrez-helm-charts` and `kubernetes-configs`. ChromaDB inspection confirmed that each repo's index only contains its own files — so the cross-contamination happens at the MCP server level, not the index level.

**Impact:** With multiple Cairn MCP servers registered, all servers effectively become aliases for the same project. The per-repo context isolation — the primary value proposition for multirepo workspaces — does not work.

**Root cause hypothesis:** `cairn mcp` resolves the project directory from `os.getcwd()` at process start, or from a config file in the CWD, rather than reading `CAIRN_PROJECT` from the environment. Since all three MCP server processes share the same CWD (OpenCode's launch directory), all three query the same `.cairn/` directory.

**Workaround:** None available for multirepo use. For single-repo use, open OpenCode directly from the repo root — the single MCP server will then correctly resolve its own `.cairn/` index.

**Suggested fix:** In `cairn mcp` entry point, check `os.environ.get("CAIRN_PROJECT")` first. If set, use it as the project root. Fall through to CWD only if the env var is absent. Add a startup log line: `"Using project: <path> (from CAIRN_PROJECT)"` or `"Using project: <path> (from CWD)"` to make the resolution visible.

