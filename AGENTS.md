# Agent Integration Guide: MCP Tools for Claude Code & OpenCode

This guide is for AI agents (Claude Code, OpenCode, etc.) using the Cairn via MCP.

## What Is Cairn?

Cairn is a **local retrieval + compression engine** that sits between you and your codebase.
It indexes your code into a vector database, searches for relevant functions when you need them, compresses them,
and makes them available as MCP tools via `cairn mcp`. **Everything stays local.** Your code never leaves your machine.

**Key benefit:** Get surgical context for coding tasks without flooding the LLM with irrelevant code.

## How Profiles Work

Cairn auto-detects your repo type and chooses a retrieval strategy:

- **IaC** (Terraform, Helm, Kubernetes): structural + lexical search, embeddings OFF
- **.NET** (C#): full hybrid (embeddings + lexical + structural)
- **Python**: full hybrid (embeddings + lexical + structural)
- **Shell** scripts: lexical + structural, embeddings OFF
- **Generic code** (JS/TS, Go, Rust, Java, C++, Ruby): full hybrid

Different profiles optimize for different code patterns. For example, Terraform resources are disambiguated
by exact name matching and block boundaries, not semantic similarity.

**If the auto-detected profile is wrong**, use `set_profile()` to override.

---

## The Six MCP Tools

### 1. `search_code(query, top_k=5)`

**Use when:** You need to find specific code patterns or understand how something is implemented.

**What it does:**
- Takes your natural-language query (e.g., "how do we handle database connections?")
- Searches the codebase using the detected profile's strategy
- Returns up to `top_k` matching functions with:
  - File path and line number
  - Similarity score (0..1, higher = more relevant)
  - Code snippet

**Example:**
```
search_code("authentication middleware", top_k=3)
→ Returns:
  1. src/auth.py:validate_token (similarity: 0.89)
     def validate_token(token):
       ...
  2. src/middleware.py:AuthMiddleware (similarity: 0.85)
     class AuthMiddleware:
       ...
  3. src/decorators.py:require_auth (similarity: 0.78)
     def require_auth(func):
       ...
```

**When to use:**
- Exploring an unfamiliar codebase
- Finding examples of a pattern
- Locating related functionality

### 2. `assemble_context(query)`

**Use when:** You need ready-to-inject context for a specific task (implementing a feature, fixing a bug, etc.).

**What it does:**
- Runs the full context assembly pipeline:
  1. Searches for functions semantically similar to your query
  2. Reranks results with a confidence filter (default threshold: 0.47)
  3. Loads the repository map (all top-level functions/classes)
  4. Loads recent git-diff memory (last 10 entries)
  5. Assembles everything into a single markdown block
  6. Compresses it (removes boilerplate, shortens identifiers)
- Returns a **compressed, ready-to-use context string** with all three sources

**Example output:**
```markdown
## Semantic Context

### Search Results (top-K similar functions)

1. **src/payment.py:process_payment** (line 45)
   Handles charging the user's card. Validates amount > 0,
   integrates with Stripe API.
   [...]

2. **src/billing.py:calculate_total** (line 120)
   [...]

### Repository Map

**Functions:**
- auth.py: validate_token, require_auth
- payment.py: process_payment, refund_charge
- billing.py: calculate_total, apply_discount
- database.py: get_user, create_invoice

**Classes:**
- PaymentGateway (src/payment.py)
- BillingService (src/billing.py)

### Recent Memory (Git Diffs)

- 2024-06-01: Added Stripe webhook handling
- 2024-05-31: Fixed floating-point rounding in tax calculation
[...]
```

**When to use:**
- Starting work on a feature or bug fix
- Need to understand dependencies across multiple files
- Want all context in one place (search + map + history)

### 3. `set_profile(name)`

**Use when:** The auto-detected profile is wrong.

**What it does:**
- Changes the repo's retrieval strategy to a different profile
- Options: `iac`, `dotnet`, `python`, `shell`, `code`
- Takes effect immediately (no reindexing required)

**Example:**
```
# Auto-detect guessed "python" but it's actually Terraform
set_profile("iac")
→ Cairn now uses structural + lexical retrieval (no embeddings)
```

**When to use:**
- Cairn is returning irrelevant results
- You know the correct profile but init misdetected it

### 4. `orchestrate(query, instruction="", payload="")`

**Use when:** You want Cairn to decide whether to handle work locally or defer to the cloud LLM.

**What it does:**
- Sizes work in tokens
- Routes intelligently:
  - Context-only (just return assembled context)
  - One local-LLM call (if work fits the small window)
  - Map-reduce split (for large work)
  - Defer-to-cloud (if local can't handle it)

**When to use:**
- Complex reasoning tasks that might need local LLM assistance
- Work that could be too large for a single call

### 5. `cache_get(query)`

**Use when:** You want to retrieve a cached response.

**What it does:**
- Looks up query in the local semantic cache
- Returns cached value if found and not expired

**When to use:**
- Repeated queries on the same codebase (same session)

### 6. `cache_set(query, value, ttl_seconds=300)`

**Use when:** You want to store a computed response for later retrieval.

**What it does:**
- Stores value in the local semantic cache
- Auto-expires after `ttl_seconds` (default 300s)

**When to use:**
- Caching results of expensive computations
- Sharing context between tool calls

---

## Usage Patterns

### Pattern 1: Exploring an Unfamiliar Codebase

**Task:** "I'm new to this codebase. How does authentication work?"

1. Call `search_code("authentication", top_k=5)`
2. Review results to find key files
3. Call `assemble_context("authentication")` to get full context
4. Ask follow-up questions with the context in mind

### Pattern 2: Implementing a Feature

**Task:** "Add a new payment method to the checkout flow"

1. Call `assemble_context("payment checkout flow")`
2. Use the returned context (search results + repo map + memory) to understand:
   - Where payment processing happens
   - What classes/functions you'll need to modify
   - Recent changes to the billing system
3. Write code with full understanding of existing patterns

### Pattern 3: Debugging

**Task:** "Why are refunds failing?"

1. Call `search_code("refund logic", top_k=10)` to find relevant functions
2. Call `assemble_context("refund processing")` to get full context including recent changes
3. Use the context to understand the flow and identify the bug

### Pattern 4: Code Review

**Task:** "Review this PR that adds caching"

1. Call `assemble_context("cache invalidation")`
2. Use context to understand existing cache patterns
3. Review the PR against established patterns

---

## How to Interpret Results

### Similarity Scores

When using `search_code()`, each result has a similarity score (0..1).

- **0.85+:** Very relevant — should be included
- **0.70–0.85:** Relevant — likely useful
- **0.50–0.70:** Tangentially related — might be useful
- **<0.50:** Probably noise

Scores depend on:
- The profile's retrieval strategy
- Whether the query matches exact keywords (lexical) or semantic meaning (embeddings)

### Confidence Filter

`assemble_context()` filters results with a confidence threshold (default: 0.47 using FlashRank cross-encoder).
This means:

- Off-topic queries → no context injected (returns "No confident matches")
- On-topic queries → full context injected
- You never get irrelevant noise in `assemble_context()`

If `assemble_context()` returns "No confident matches" but `search_code()` found things, Cairn
decided those results were off-topic. This is **intentional** — better to have no context than wrong context.

---

## Tips for Best Results

### 1. Be Specific with Queries

❌ Bad: "How does this work?"
✓ Good: "How does the user authentication system validate credentials?"

Specific queries match better to relevant code.

### 2. Use Natural Language

Cairn understands English descriptions of what you want to do.

✓ Good: "database connection pooling"
✓ Good: "how do we serialize JSON responses?"
✓ Good: "middleware that logs HTTP requests"

### 3. Leverage Different Tools for Different Tasks

- **`search_code()`** → Exploring / finding examples
- **`assemble_context()`** → Ready to work / have all context
- **`set_profile()`** → Fix auto-detection if needed

### 4. Check Recent Memory

The context from `assemble_context()` includes recent git diffs (last 10 entries). This tells you:
- What changed recently
- What the team was working on
- Patterns being established

Use this to avoid contradicting recent changes.

### 5. Profiles Matter for Specialized Codebases

If you're working on:
- **Terraform/Helm/Kubernetes** → IaC profile (exact name matching)
- **.NET/C#** → dotnet profile (embeddings for type disambiguation)
- **Python web framework** → python profile
- **Shell scripts** → shell profile

The wrong profile gives noisy results. Use `set_profile()` if auto-detection was wrong.

---

## Privacy & Security

- **Local first:** Code stays on your machine. Cairn runs locally via MCP (stdio).
- **No telemetry:** Cairn doesn't phone home.
- **No cloud forwarding:** MCP tools don't send code to the cloud (only your natural-language query is processed locally).

---

## Troubleshooting

### "No confident matches" from `assemble_context()`

Cairn thinks your query is off-topic. Try:
1. Make the query more specific
2. Use `search_code()` to see what's available
3. Check that the right profile is detected: `set_profile()` if not

### Results from `search_code()` look irrelevant

Cairn may have the wrong profile. Check:
1. The repo type (IaC? .NET? Python?)
2. Call `set_profile()` to override if wrong
3. Try a different query (more specific, different keywords)

### Slow search latency

Likely causes:
- First indexing (takes 2–5 minutes)
- Ollama model loading on first query (can take 10–30 seconds)
- Large codebase with many results

Subsequent queries should be fast (<100ms).

---

## FAQ

**Q: Does Cairn send my code to the cloud?**

A: No. Code stays local. Only your natural-language query is processed locally via MCP. Nothing leaves your machine.

**Q: What happens if the profile is wrong?**

A: Results will be noisy. Use `set_profile()` to fix it immediately.

**Q: Can I exclude sensitive files from indexing?**

A: Yes. Edit `.cairn/config.yaml` and adjust `exclude_patterns`:
```yaml
indexing:
  exclude_patterns:
    - "**/.env"
    - "**/secrets/**"
    - "**/private_keys/**"
```

**Q: How often does the index update?**

A: Automatically as you save files (debounced), and on commits (via background janitor). The index
is never stale.

**Q: What if I want to use the HTTP proxy instead?**

A: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#option-2-http-proxy-openaianthropiccompatible).

---

## Next Steps

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) — Setup and integration options
- [RUNBOOK.md](RUNBOOK.md) — Full reference for all CLI commands
- [README.md](README.md) — Architecture & design overview

