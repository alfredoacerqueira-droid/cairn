"""Dev workflow token/cost/speed simulator — measures Cairn vs baseline token usage.

Run:
    python3 -m scripts.dev_workflow_sim --repo django

Writes to docs/DEV_WORKFLOW_TOKEN_REPORT.md
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

# ── tasks table ──────────────────────────────────────────────────────────
TASKS = [
    (
        "how does QuerySet.filter build the query",
        "django/db/models/query.py",
        "def filter",
    ),
    (
        "where and how is the CSRF token validated",
        "django/middleware/csrf.py",
        "class CsrfViewMiddleware",
    ),
    (
        "how does Paginator compute the number of pages",
        "django/core/paginator.py",
        "class Paginator",
    ),
    (
        "how are URL patterns resolved to a view",
        "django/urls/resolvers.py",
        "class URLResolver",
    ),
    (
        "how does the model save() persist a row",
        "django/db/models/base.py",
        "def save",
    ),
    (
        "how is a QuerySet turned into SQL",
        "django/db/models/sql/compiler.py",
        "class SQLCompiler",
    ),
    (
        "how does the template engine render a template",
        "django/template/base.py",
        "class Template",
    ),
    ("how are forms validated", "django/forms/forms.py", "def full_clean"),
    (
        "how does the cache framework get/set values",
        "django/core/cache/backends/locmem.py",
        "class LocMemCache",
    ),
    (
        "how does signing protect a value",
        "django/core/signing.py",
        "def dumps",
    ),
]


def _key_words(query: str) -> list[str]:
    """Extract 2-3 key content words from a query for rg -l search."""
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "about",
        "also",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "this",
        "that",
        "these",
        "those",
    }
    words = query.lower().split()
    content = [w.strip("?.,!;:()[]{}'\"") for w in words]
    content = [w for w in content if len(w) >= 3 and w not in stop]
    return content[:3]


def count_tokens(text: str) -> int:
    from core.tokens import count_tokens as _ct

    return _ct(text)


def read_file_tokens(path: Path) -> int:
    try:
        return count_tokens(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def rg_find_files(repo: Path, query: str) -> list[Path]:
    """rg -l for content words; return up to 10 source files."""
    words = _key_words(query)
    if not words:
        return []
    pattern = "|".join(words)
    try:
        result = subprocess.run(
            ["rg", "-l", pattern],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in (0, 1):
            return []
        lines = result.stdout.strip().split("\n")
        paths = [ln.strip() for ln in lines if ln.strip()]
        source_exts = {
            ".py",
            ".rs",
            ".go",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cs",
            ".java",
            ".rb",
            ".sh",
            ".bash",
            ".tf",
            ".toml",
        }
        paths = [p for p in paths if Path(p).suffix in source_exts]
        return [repo / p for p in paths[:10]]
    except Exception:
        return []


def fmt_tok(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def fmt_pct(v: float) -> str:
    if v in (float("inf"), float("-inf")):
        return "N/A"
    return f"{v:.1f}%"


def fmt_usd(v: float) -> str:
    return f"${v:.6f}"


def fmt_lat(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def median(vals):
    if not vals:
        return 0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _index_with_llm(repo_path: Path) -> float:
    """Cold-index with embeddings enabled via real Ollama nomic-embed-text.

    Mirrors tests.fixtures.harness.fresh_index but uses a REAL OllamaClient
    (not _ExplodingOllama) with embeddings_enabled=True so every code block
    gets a semantic embedding vector.

    Returns indexing time in seconds.
    """
    import shutil

    from core.config import Config, save_config
    from core.freshness import DBFreshness
    from core.repo import RepoManager, collect_source_files, detect_source_layout
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer
    from server.ollama_client import OllamaClient

    print("Indexing django (WITH-LLM, embed=nomic-embed-text)...", flush=True)
    t0 = time.perf_counter()

    # Step 1: Detect layout
    detected_roots, detected_patterns = detect_source_layout(repo_path)

    # Step 2: Delete existing .cairn
    cairn_dir = repo_path / ".cairn"
    if cairn_dir.exists():
        shutil.rmtree(cairn_dir)

    # Step 3: Write fresh config with embeddings enabled + local_llm settings
    cfg = Config()
    cfg.indexing.source_roots = detected_roots
    cfg.indexing.file_patterns = detected_patterns
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = True
    cfg.local_llm.model = "gemma4:latest"
    cfg.local_llm.embed_model = "nomic-embed-text"
    cfg.local_llm.embedder = "ollama"
    save_config(cfg, repo_path)

    # Step 4: Collect source files
    files = collect_source_files(
        repo_path,
        cfg.indexing.file_patterns,
        cfg.indexing.exclude_patterns,
        cfg.indexing.source_roots,
    )

    # Step 5: Index via VectorIndexer with REAL OllamaClient
    repo = RepoManager(repo_path)
    ollama_client = OllamaClient(embed_model="nomic-embed-text")
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(),
        ollama_client=ollama_client,
        embeddings_enabled=True,
        project_root=repo_path,
    )

    parser = ASTParser()
    for filepath in files:
        try:
            ast = parser.parse_file(filepath)
            indexer.index_ast(ast)
        except Exception:
            pass

    # Step 6: Mark freshness
    freshness = DBFreshness(
        repo_path,
        quick_threshold=cfg.stale_db.quick_reindex_threshold,
        full_threshold=cfg.stale_db.full_reindex_threshold,
    )
    freshness.mark_indexed(freshness.get_current_commit())
    repo.write_index_meta()

    index_sec = time.perf_counter() - t0
    print(f"Indexed (WITH-LLM, embed=nomic-embed-text) in {index_sec:.1f}s", flush=True)
    return index_sec


def main():
    parser = argparse.ArgumentParser(description="Dev workflow token/cost simulator")
    parser.add_argument("--repo", default="django", help="Repo key (default: django)")
    parser.add_argument(
        "--price-per-mtok",
        type=float,
        default=3.0,
        help="USD per million input tokens (default: 3.0)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Report output path (default: docs/DEV_WORKFLOW_TOKEN_REPORT.md "
        "or ..._WITH_LLM.md)",
    )
    parser.add_argument("--list", action="store_true", help="List tasks and exit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-llm",
        action="store_true",
        default=True,
        help="No-LLM mode (lexical+structural only, default)",
    )
    mode.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="WITH-LLM mode (embeddings via ollama nomic-embed-text)",
    )
    args = parser.parse_args()

    if args.list:
        for i, (q, f, s) in enumerate(TASKS):
            print(f"{i + 1:2d}. {q[:60]}…  [{f}:{s}]")
        return

    use_llm = args.with_llm

    repo_path = Path("/mnt/c/Users/alfre/Projects/cairn-hardtest/corpus/django")
    if not repo_path.exists():
        print(f"ERROR: django repo not found at {repo_path}")
        sys.exit(1)

    price_per_mtok = args.price_per_mtok

    # ── Report path ────────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
    elif use_llm:
        report_path = (
            Path(__file__).parent.parent / "docs" / "DEV_WORKFLOW_TOKEN_REPORT_WITH_LLM.md"
        )
    else:
        report_path = Path(__file__).parent.parent / "docs" / "DEV_WORKFLOW_TOKEN_REPORT.md"

    # ── Cold index once ───────────────────────────────────────────────
    from tests.fixtures.harness import fresh_index

    if use_llm:
        index_sec = _index_with_llm(repo_path)
    else:
        print("Indexing django (no-LLM, embeddings=False)...", flush=True)

        t0 = time.perf_counter()
        fresh_index(repo_path, embeddings=False)
        index_sec = time.perf_counter() - t0
        print(f"Indexed in {index_sec:.1f}s", flush=True)

    # ── Assembler ─────────────────────────────────────────────────────
    from server.context_assembler import ContextAssembler

    asm = ContextAssembler(project_path=repo_path)

    # ── BASELINE_REPO (compute once) ──────────────────────────────────
    from core.config import Config
    from core.repo import collect_source_files, detect_source_layout

    roots, patterns = detect_source_layout(repo_path)
    excl = Config().indexing.exclude_patterns
    all_files = collect_source_files(repo_path, patterns, excl, roots)
    baseline_repo_tokens = 0
    for f in all_files:
        baseline_repo_tokens += read_file_tokens(f)
    baseline_repo_blocks = len(all_files)
    print(
        f"BASELINE_REPO: {baseline_repo_tokens} tokens " f"in {baseline_repo_blocks} files",
        flush=True,
    )

    # ── System resources ──────────────────────────────────────────────
    try:
        from core.resources import get_system_resources

        sys_res = get_system_resources()
    except Exception:
        sys_res = {"ram_total_gb": 0, "cpu_count": 0}

    # ── Run tasks ────────────────────────────────────────────────────
    rows: list[dict] = []
    needed_present_count = 0

    for qi, (query, gt_file_rel, gt_symbol) in enumerate(TASKS):
        print(f"\n[{qi + 1}/{len(TASKS)}] {query[:70]}...", flush=True)
        row = {
            "query": query,
            "gt_file": gt_file_rel,
            "gt_symbol": gt_symbol,
        }

        gt_path = repo_path / gt_file_rel

        # CAIRN
        try:
            t1 = time.perf_counter()
            ctx = asm.assemble(query)
            cairn_ms = (time.perf_counter() - t1) * 1000
            cairn_tok = count_tokens(ctx)
            row["cairn_tok"] = cairn_tok
            row["cairn_ms"] = cairn_ms
            row["cairn_ok"] = True
            print(
                f"  CAIRN: {fmt_tok(cairn_tok)} tok, " f"{fmt_lat(cairn_ms)}",
                flush=True,
            )
        except Exception as e:
            print(f"  CAIRN FAIL: {e}", flush=True)
            row["cairn_tok"] = 0
            row["cairn_ms"] = 0
            row["cairn_ok"] = False

        # Recall gate
        if row.get("cairn_ok"):
            needed_present = gt_symbol.lower() in ctx.lower()
            row["needed_present"] = needed_present
            if needed_present:
                needed_present_count += 1
            else:
                print(
                    f"  ** QUALITY FAIL: gt_symbol '{gt_symbol}' " "not in CAIRN context",
                    flush=True,
                )
        else:
            row["needed_present"] = False

        # BASELINE_FILES
        try:
            base_tok = read_file_tokens(gt_path)
            row["base_files_tok"] = base_tok
            print(
                f"  BASELINE_FILES: {fmt_tok(base_tok)} tok " f"({gt_file_rel})",
                flush=True,
            )
        except Exception as e:
            print(f"  BASELINE_FILES FAIL: {e}", flush=True)
            row["base_files_tok"] = 0

        # BASELINE_RGDUMP
        try:
            rg_files = rg_find_files(repo_path, query)
            rg_tok = 0
            for rf in rg_files:
                rg_tok += read_file_tokens(rf)
            row["rgdump_tok"] = rg_tok
            row["rgdump_nfiles"] = len(rg_files)
            print(
                f"  RGDUMP: {fmt_tok(rg_tok)} tok " f"in {len(rg_files)} files",
                flush=True,
            )
        except Exception as e:
            print(f"  RGDUMP FAIL: {e}", flush=True)
            row["rgdump_tok"] = 0
            row["rgdump_nfiles"] = 0

        # Reduction%
        if row["base_files_tok"] > 0 and row["cairn_tok"] > 0:
            row["reduction_pct"] = (1 - row["cairn_tok"] / row["base_files_tok"]) * 100
        else:
            row["reduction_pct"] = 0

        # Cost
        row["cost_cairn"] = row["cairn_tok"] / 1_000_000 * price_per_mtok
        row["cost_base"] = row["base_files_tok"] / 1_000_000 * price_per_mtok
        row["cost_saved"] = row["cost_base"] - row["cost_cairn"]

        rows.append(row)

    # ── Freshness check ───────────────────────────────────────────────
    print("\n── Freshness check ──", flush=True)
    paginator_path = repo_path / "django/core/paginator.py"
    marker_func = (
        "\n\ndef zzfreshmarker_helper():\n"
        '    """Freshness marker for dev workflow sim."""\n'
        "    return 42\n"
    )
    try:
        original = paginator_path.read_text(encoding="utf-8")
        paginator_path.write_text(original + marker_func, encoding="utf-8")
        print(
            "  Appended zzfreshmarker_helper to paginator.py",
            flush=True,
        )

        if use_llm:
            _index_with_llm(repo_path)
        else:
            fresh_index(repo_path, embeddings=False)
        print("  Re-indexed after freshness marker", flush=True)

        asm2 = ContextAssembler(project_path=repo_path)
        fresh_ctx = asm2.assemble("zzfreshmarker_helper")
        fresh_present = "zzfreshmarker" in fresh_ctx.lower()
        print(
            f"  zzfreshmarker in CAIRN context: {fresh_present}",
            flush=True,
        )

        # restore file via git checkout
        subprocess.run(
            ["git", "checkout", "--", str(paginator_path)],
            cwd=str(repo_path),
            capture_output=True,
            timeout=30,
        )
        print("  Restored paginator.py via git checkout", flush=True)

        # re-index clean
        if use_llm:
            _index_with_llm(repo_path)
        else:
            fresh_index(repo_path, embeddings=False)
        print("  Re-indexed to clean state", flush=True)

        if not fresh_present:
            print(
                "  WARNING: Freshness marker NOT found in context " "after re-index!",
                flush=True,
            )
    except Exception as e:
        print(f"  Freshness check FAIL: {e}", flush=True)
        fresh_present = False
        try:
            subprocess.run(
                ["git", "checkout", "--", str(paginator_path)],
                cwd=str(repo_path),
                capture_output=True,
                timeout=30,
            )
            if use_llm:
                _index_with_llm(repo_path)
            else:
                fresh_index(repo_path, embeddings=False)
        except Exception:
            pass

    # ── Aggregate ────────────────────────────────────────────────────
    reductions = [r["reduction_pct"] for r in rows if r.get("cairn_ok") and r["base_files_tok"] > 0]
    cairn_toks = [r["cairn_tok"] for r in rows if r.get("cairn_ok")]
    base_toks = [r["base_files_tok"] for r in rows if r["base_files_tok"] > 0]
    latencies = [r["cairn_ms"] for r in rows if r.get("cairn_ok") and r["cairn_ms"] > 0]

    total_cairn = sum(cairn_toks)
    total_base = sum(base_toks)
    total_cost_cairn = total_cairn / 1_000_000 * price_per_mtok
    total_cost_base = total_base / 1_000_000 * price_per_mtok
    total_cost_saved = total_cost_base - total_cost_cairn

    avg_reduction = sum(reductions) / len(reductions) if reductions else 0
    med_reduction = median(reductions)
    med_latency = median(latencies)
    vs_repo_ratio = (
        f"{total_cairn / baseline_repo_tokens * 100:.1f}%" if baseline_repo_tokens > 0 else "N/A"
    )

    # ── Report ────────────────────────────────────────────────────────
    report = []
    report.append("# Cairn Dev Workflow Token/Cost/Speed Report\n\n")
    report.append(f"**Date:** " f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(
        "**Mode:** "
        + (
            "WITH-LLM (embeddings enabled, embed=nomic-embed-text)"
            if use_llm
            else "NO-LLM (lexical+structural only)"
        )
        + "\n\n"
    )
    report.append(f"**Indexing time:** {fmt_lat(index_sec * 1000)}\n\n")
    report.append(
        "**Host resources:** "
        f"CPU={sys_res.get('cpu_count', '?')}, "
        f"RAM total={sys_res.get('ram_total_gb', '?')}GB, "
        f"RAM avail={sys_res.get('ram_available_gb', '?')}GB, "
        f"GPU={sys_res.get('gpu_name') or 'none'}, "
        f"VRAM={sys_res.get('vram_total_gb') or '0'}GB\n\n"
    )
    report.append(f"**Repo:** django ({repo_path})\n\n")
    report.append(
        f"**Total indexed source tokens:** {fmt_tok(baseline_repo_tokens)} "
        f"({baseline_repo_blocks} files)\n\n"
    )
    report.append(f"**Price per 1M input tokens:** \\${price_per_mtok:.1f} (ESTIMATE)\n\n")

    report.append("## Methodology\n\n")
    report.append(
        "Four baselines per task:\n"
        "- **CAIRN**: tokens in `ContextAssembler.assemble(query)` — "
        "compressed context the agent receives\n"
        "- **BASELINE_FILES**: tokens in the single ground-truth source "
        "file (what an agent reads in full)\n"
        "- **BASELINE_RGDUMP**: tokens in up to 10 files found by `rg -l` "
        "with query content words\n"
        "- **BASELINE_REPO**: tokens in ALL indexed source files "
        "(whole-repo dump)\n\n"
        "Token counting uses `tiktoken` cl100k_base (proxy for "
        "Claude/Sonnet). Cost estimate: input tokens $ per 1M "
        "(configurable via `--price-per-mtok`). The **recall gate** "
        "checks whether the ground-truth symbol (e.g. `class Paginator`) "
        "appears in the CAIRN context; a miss is a quality failure.\n\n"
    )

    report.append("## Per-Task Results\n\n")
    report.append(
        "| # | Task | CAIRN tok | Base tok | Reduction% | "
        "RGDUMP tok (#files) | Recall | Latency | "
        "$ CAIRN | $ Base | $ Saved |\n"
    )
    report.append(
        "|---|------|-----------|----------|------------|"
        "----------------------|--------|---------|"
        "---------|--------|---------|\n"
    )

    for i, r in enumerate(rows):
        if r.get("cairn_ok"):
            recall = "YES" if r.get("needed_present") else "NO"
        else:
            recall = "FAIL"

        task_short = r["query"][:55] + "…" if len(r["query"]) > 55 else r["query"]
        report.append(
            f"| {i + 1} | {task_short} | {fmt_tok(r['cairn_tok'])} | "
            f"{fmt_tok(r['base_files_tok'])} | "
            f"{fmt_pct(r['reduction_pct'])} | "
            f"{fmt_tok(r['rgdump_tok'])} ({r['rgdump_nfiles']}) | "
            f"{recall} | "
            f"{fmt_lat(r['cairn_ms'])} | "
            f"{fmt_usd(r['cost_cairn'])} | "
            f"{fmt_usd(r['cost_base'])} | "
            f"{fmt_usd(r['cost_saved'])} |\n"
        )

    report.append("\n## Aggregate\n\n")
    report.append(
        f"- **Reduction vs BASELINE_FILES:** mean={fmt_pct(avg_reduction)}, "
        f"median={fmt_pct(med_reduction)}\n"
    )
    report.append(
        f"- **Recall:** {needed_present_count}/{len(TASKS)} tasks " "passed the recall gate\n"
    )
    report.append(
        f"- **Total CAIRN tokens:** {fmt_tok(total_cairn)} vs "
        f"**total BASELINE_FILES tokens:** {fmt_tok(total_base)}\n"
    )
    report.append(
        f"- **Estimated $ saved (vs BASELINE_FILES):** "
        f"{fmt_usd(total_cost_saved)} "
        f"(CAIRN cost: {fmt_usd(total_cost_cairn)} vs "
        f"base: {fmt_usd(total_cost_base)})\n"
    )
    report.append(f"- **Median latency per CAIRN call:** " f"{fmt_lat(med_latency)}\n")
    report.append(
        f"- **CAIRN vs BASELINE_REPO ratio:** {vs_repo_ratio} "
        f"(CAIRN uses {fmt_tok(total_cairn)} of "
        f"{fmt_tok(baseline_repo_tokens)} repo tokens)\n"
    )
    report.append(
        f"- **Freshness:** {'PASS' if fresh_present else 'FAIL'} "
        "(zzfreshmarker appeared in context after re-index)\n\n"
    )

    report.append("## Findings & Limitations\n\n")
    report.append(
        "- **Baseline fairness:** BASELINE_FILES is a single-file read — "
        "generous to the baseline, since a real agent would likely need "
        "2-5 files to understand surrounding context. Cairn's reduction "
        "vs BASELINE_FILES is therefore a *conservative* lower bound on "
        "actual savings.\n\n"
    )
    report.append(
        "- **RGDUMP overestimates:** `rg -l` returns files containing "
        "query keywords but many are tangential (e.g. test files, "
        "docstrings mentioning the symbol). A real agent would not read "
        "all 10 files in full, making RGDUMP an upper bound on naive "
        "file-based retrieval cost.\n\n"
    )
    if use_llm:
        report.append(
            "- **WITH-LLM indexing:** This benchmark uses "
            "embeddings via ollama nomic-embed-text for semantic retrieval. "
            "Embedding-augmented retrieval typically improves recall "
            "but adds indexing cost.\n\n"
        )
    else:
        report.append(
            "- **No-LLM indexing:** This benchmark uses "
            "`fresh_index(embeddings=False)` — lexical+structural retrieval "
            "only. Embedding-augmented retrieval typically improves recall "
            "further but adds cost.\n\n"
        )
    recall_fail_tasks = [i + 1 for i, r in enumerate(rows) if not r.get("needed_present")]
    if recall_fail_tasks:
        report.append(
            f"- **Recall misses:** Tasks {recall_fail_tasks} did not "
            "have the gt_symbol in CAIRN context. "
            + (
                "With embeddings the confidence guard may still reject "
                "if the semantic match is weak."
                if use_llm
                else "Without embeddings the confidence guard may reject "
                "structurally-mismatched retrievals."
            )
            + "\n\n"
        )
    if use_llm:
        report.append(
            "- **Ranking:** WITH-LLM mode uses lexical+structural retrieval "
            "+ embedding-based retrieval + cross-encoder reranker. "
            "Semantic similarity improves query-concept-to-code mapping.\n\n"
        )
    else:
        report.append(
            "- **Ranking caveats:** No-LLM mode uses lexical+structural "
            "retrieval + cross-encoder reranker. Query-concept-to-code "
            "mapping may miss or deprioritize the target symbol if the "
            "structure doesn't match query terms. Embeddings-enabled mode "
            "would improve semantic matching.\n\n"
        )

    report.append("## Headline\n\n")
    headline = (
        f"Cairn reduces agent input context by "
        f"**{fmt_pct(avg_reduction)} on average** "
        f"(median {fmt_pct(med_reduction)}) vs reading the single "
        "source file in full, across "
        f"{len(TASKS)} representative Django development tasks. "
        f"Recall: **{needed_present_count}/{len(TASKS)}**. "
        "Estimated cost saving: "
        f"**{fmt_usd(total_cost_saved)}** per task set "
        f"(at \\${price_per_mtok:.1f}/1M input tokens). "
        f"Median assembly latency: **{fmt_lat(med_latency)}**.\n\n"
    )
    report.append(headline)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(report), encoding="utf-8")
    print(f"\nReport written to {report_path}", flush=True)

    # ── Print summary to stdout ──────────────────────────────────────
    sep = "-" * 76
    print(f"\n{'=' * 76}")
    print("PER-TASK TABLE")
    print(f"{'=' * 76}")
    hdr = (
        f"{'#':<3} {'Task':<32} {'CAIRN':>7} {'Base':>7} "
        f"{'Red%':>7} {'RGDump':>9} {'Recall':>6} {'Lat':>7}"
    )
    print(hdr)
    print(sep)
    for i, r in enumerate(rows):
        if r.get("cairn_ok"):
            recall = "YES" if r.get("needed_present") else "NO"
        else:
            recall = "FAIL"
        task = r["query"][:30] + "…" if len(r["query"]) > 30 else r["query"]
        print(
            f"{i + 1:<3} {task:<32} {fmt_tok(r['cairn_tok']):>7} "
            f"{fmt_tok(r['base_files_tok']):>7} "
            f"{fmt_pct(r['reduction_pct']):>7} "
            f"{fmt_tok(r['rgdump_tok']):>9} {recall:>6} "
            f"{fmt_lat(r['cairn_ms']):>7}"
        )
    print(sep)
    print("\nAGGREGATE:")
    print(
        f"  Reduction vs BASELINE_FILES: "
        f"mean={fmt_pct(avg_reduction)} "
        f"median={fmt_pct(med_reduction)}"
    )
    print(f"  Recall: {needed_present_count}/{len(TASKS)}")
    print(f"  Total CAIRN: {fmt_tok(total_cairn)} vs " f"Total BASE: {fmt_tok(total_base)}")
    print(f"  Est $ saved: {fmt_usd(total_cost_saved)}")
    print(f"  Median latency: {fmt_lat(med_latency)}")
    print(f"  vs BASELINE_REPO: {vs_repo_ratio}")
    print(f"\nHEADLINE: {headline}")


if __name__ == "__main__":
    main()
