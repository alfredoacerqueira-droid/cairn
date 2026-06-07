#!/usr/bin/env python3
"""Retrieval-quality harness: measures hit-rate@k and MRR on indexed repos.

Drives the REAL hybrid+rerank search path via ContextAssembler.semantic_search(),
NOT a raw single-leg store.search().

Metrics:
  A) Self-retrieval: docstring/comment → function recall (objective, no labels).
  B) Curated conceptual queries (hand-written, per-repo).
"""

from __future__ import annotations

import argparse
import ast as pyast
import logging
import os
import random
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup: ensure the cairn package is importable
# ---------------------------------------------------------------------------
_CAIRN_ROOT = Path(__file__).resolve().parent.parent
if str(_CAIRN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAIRN_ROOT))

# Quiet down cairn's own logging (the assembler logs at INFO by default).
logging.basicConfig(level=logging.WARNING, format="%(message)s")
# Also squelch noisy sub-loggers.
for _name in (
    "chromadb",
    "sentence_transformers",
    "pipeline.retrieval.reranker",
    "pipeline.retrieval.embeddings",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ruff: noqa: E402, I001  (cairn imports must follow sys.path setup)
from core.config import embeddings_available, load_config  # noqa: E402
from core.repo import project_id  # noqa: E402
from pipeline.ast_parser import ASTParser  # noqa: E402
from server.context_assembler import ContextAssembler  # noqa: E402

# ---------------------------------------------------------------------------
# Docstring extraction helpers
#   ASTParser does NOT capture docstrings/comments, so we extract them from
#   the .code field of each FunctionDef / ClassDef.
# ---------------------------------------------------------------------------

_TRIPLE_Q = re.compile(
    r'^(?:\s*(?:\"\"\"|\'\'\')(.+?)(?:\"\"\"|\'\'\'))', re.DOTALL
)


def _extract_python_docstring(code: str) -> str | None:
    """Extract the docstring of the first function/class body in *code*."""
    try:
        tree = pyast.parse(textwrap.dedent(code))
        for node in pyast.walk(tree):
            if isinstance(
                node,
                (
                    pyast.FunctionDef,
                    pyast.AsyncFunctionDef,
                    pyast.ClassDef,
                ),
            ):
                doc = pyast.get_docstring(node)
                if doc:
                    return doc
    except SyntaxError:
        pass
    return None


def _extract_generic_docstring(code: str) -> str | None:
    """Extract first triple-quoted string or leading-comment block."""
    # Triple-quoted string near the top
    m = _TRIPLE_Q.search(code)
    if m:
        return m.group(1).strip()
    # Leading comment lines (# or //)
    lines = code.strip().split("\n")
    comments: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            comments.append(s[1:].strip())
        elif s.startswith("//"):
            comments.append(s[2:].strip())
        elif s.startswith("/*"):
            inner = s[2:]
            if "*/" in inner:
                inner = inner[: inner.index("*/")]
            comments.append(inner.strip())
        elif comments and not s:
            continue  # blank line inside comment block
        else:
            break
    if comments:
        return " ".join(comments)
    return None


def extract_docstring(code: str) -> str | None:
    """Best-effort docstring/leading-comment extraction."""
    doc = _extract_python_docstring(code)
    if doc:
        return doc
    return _extract_generic_docstring(code)


# ---------------------------------------------------------------------------
# Symbol-name removal from query
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")


def _sanitize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def remove_symbol_from_text(text: str, name: str) -> tuple[str, bool]:
    """Remove occurrences of *name* (case-insensitive, word boundaries) from
    *text*.  Returns (cleaned_text, was_empty_after_removal).

    If the cleaned text becomes empty, we return the original text and flag it.
    """
    # Split compound names like "ClassName.method_name" and remove each part.
    parts = re.split(r"[._]", name)
    cleaned = text
    for part in parts:
        if len(part) < 3:
            continue  # skip very short tokens — they'd nuke common words
        pattern = re.compile(r"\b" + re.escape(part) + r"\b", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)
    cleaned = _sanitize(cleaned)
    was_empty = len(cleaned) == 0
    if was_empty:
        return _sanitize(text), True
    return cleaned, False


# ---------------------------------------------------------------------------
# Metric A: self-retrieval
# ---------------------------------------------------------------------------

_SKIP_EXTS = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".md",
    ".rst",
    ".txt",
    ".csv",
    ".svg",
    ".png",
    ".jpg",
    ".lock",
}
_CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cc",
    ".cs",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".tf",
    ".sol",
    ".sql",
}


def _is_code_file(filepath: Path) -> bool:
    return filepath.suffix.lower() in _CODE_EXTS


def collect_documented_blocks(
    repo_path: Path, min_doc_chars: int, parser: ASTParser
) -> list[dict]:
    """Walk the repo, parse code files, return blocks with docstrings."""
    blocks: list[dict] = []
    seen: set[tuple[str, str, int]] = set()

    for fp in sorted(repo_path.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in _SKIP_EXTS:
            continue
        if not _is_code_file(fp):
            continue
        # Skip hidden dirs
        parts = fp.parts
        if any(p.startswith(".") and p != "." for p in parts):
            continue
        # Skip common vendor / generated dirs
        skip_dirs = {
            "node_modules",
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "dist",
            "build",
            "target",
            ".tox",
            ".eggs",
            "migrations",
        }
        if skip_dirs.intersection(parts):
            continue

        try:
            ast_result = parser.parse_file(fp)
        except Exception:
            continue

        rel = str(fp.relative_to(repo_path))

        # Top-level functions
        for func in ast_result.functions:
            doc = extract_docstring(func.code)
            if doc and len(doc) >= min_doc_chars:
                key = (rel, func.name, func.line_start)
                if key not in seen:
                    seen.add(key)
                    blocks.append(
                        {
                            "filepath": rel,
                            "name": func.name,
                            "line_start": func.line_start,
                            "line_end": func.line_end,
                            "code": func.code,
                            "docstring": doc,
                        }
                    )

        # Classes + methods
        for cls in ast_result.classes:
            # Class-level docstring
            doc = extract_docstring(cls.code)
            if doc and len(doc) >= min_doc_chars:
                key = (rel, cls.name, cls.line_start)
                if key not in seen:
                    seen.add(key)
                    blocks.append(
                        {
                            "filepath": rel,
                            "name": cls.name,
                            "line_start": cls.line_start,
                            "line_end": cls.line_end,
                            "code": cls.code,
                            "docstring": doc,
                        }
                    )
            # Method docstrings
            for method in cls.methods:
                doc = extract_docstring(method.code)
                if doc and len(doc) >= min_doc_chars:
                    indexed_name = f"{cls.name}.{method.name}"
                    key = (rel, indexed_name, method.line_start)
                    if key not in seen:
                        seen.add(key)
                        blocks.append(
                            {
                                "filepath": rel,
                                "name": indexed_name,
                                "line_start": method.line_start,
                                "line_end": method.line_end,
                                "code": method.code,
                                "docstring": doc,
                            }
                        )

    return blocks


def _result_matches_block(result: dict, block: dict) -> bool:
    """Check if a search result identifies the same block."""
    r_file = str(result.get("filepath", ""))
    r_func = str(result.get("function", ""))

    b_file = block["filepath"]
    b_name = block["name"]

    # File: accept suffix match (result may use relative path, block uses rel)
    if not (r_file.endswith(b_file) or b_file.endswith(r_file)):
        # Also try matching just the filename part
        r_fname = os.path.basename(r_file)
        b_fname = os.path.basename(b_file)
        if r_fname != b_fname:
            return False

    # Symbol name: exact match
    if r_func == b_name:
        return True
    # Accept when result function is a method matching the block name
    if b_name in r_func or r_func in b_name:
        return True
    return False


def _find_rank(
    results: list[dict], block: dict, k: int
) -> int | None:
    """Return 1-based rank of result matching *block*, or None if not in top-k."""
    for rank, result in enumerate(results[:k], 1):
        if _result_matches_block(result, block):
            return rank
    return None


def run_metric_a(
    assembler: ContextAssembler,
    repo_path: Path,
    n: int,
    k: int,
    min_doc_chars: int,
    seed: int,
) -> dict:
    """Self-retrieval metric: how well do docstrings retrieve their own block?"""
    parser = ASTParser()
    all_blocks = collect_documented_blocks(repo_path, min_doc_chars, parser)

    if len(all_blocks) == 0:
        return {
            "n_total_documented": 0,
            "n_evaluated": 0,
            "hit_at_1": 0.0,
            "hit_at_5": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "flagged_count": 0,
        }

    rng = random.Random(seed)
    sample = rng.sample(all_blocks, min(n, len(all_blocks)))

    ranks: list[int | None] = []
    flagged = 0

    for block in sample:
        query, was_flagged = remove_symbol_from_text(
            block["docstring"], block["name"]
        )
        if was_flagged:
            flagged += 1

        try:
            results = assembler.semantic_search(
                query, top_k=k, apply_guard=False
            )
        except Exception:
            ranks.append(None)
            continue

        rank = _find_rank(results, block, k)
        ranks.append(rank)

    # Compute metrics
    hits = [r for r in ranks if r is not None]
    reciprocal_sum = sum(1.0 / r for r in hits)

    hn = len(ranks)
    return {
        "n_total_documented": len(all_blocks),
        "n_evaluated": hn,
        "hit_at_1": sum(1 for r in ranks if r is not None and r <= 1) / hn,
        "hit_at_5": sum(1 for r in ranks if r is not None and r <= min(5, k)) / hn,
        "hit_at_k": len(hits) / hn,
        "mrr": reciprocal_sum / hn if hn else 0.0,
        "flagged_count": flagged,
    }


# ---------------------------------------------------------------------------
# Metric B: curated conceptual queries
# ---------------------------------------------------------------------------

CURATED_QUERIES: dict[str, list[tuple[str, str]]] = {
    "cairn": [
        ("semantic cache get", "semantic_cache"),
        ("token budget for context assembly", "token_budget"),
        ("rerank search candidates", "reranker"),
        ("parse a function with tree-sitter", "ast_parser"),
        ("chroma upsert or index blocks", "chroma"),
        ("cross-project isolation filter", "project_id"),
        ("embedding retrieval with fastembed", "embeddings"),
    ],
    "django": [
        ("filter a queryset with ORM", "query"),
        ("render a template to string", "template"),
        ("validate a form field input", "forms"),
        ("url routing resolve a path to view", "urls"),
        ("database migration schema editor", "migration"),
    ],
}


def run_metric_b(
    assembler: ContextAssembler, repo_name: str, k: int
) -> dict:
    """Curated conceptual queries: does ANY top-k result match expected file?"""
    queries = CURATED_QUERIES.get(repo_name, [])
    if not queries:
        return {"curated_total": 0, "curated_hits": 0, "curated_details": []}

    details: list[dict] = []
    hits = 0

    for query, expected_sub in queries:
        try:
            results = assembler.semantic_search(
                query, top_k=k, apply_guard=False
            )
        except Exception:
            details.append(
                {"query": query, "expected": expected_sub, "pass": False, "error": True}
            )
            continue

        matched = any(
            expected_sub.lower() in str(r.get("filepath", "")).lower()
            for r in results[:k]
        )
        details.append(
            {
                "query": query,
                "expected": expected_sub,
                "pass": matched,
                "top_files": [
                    str(r.get("filepath", "")) for r in results[:3]
                ],
            }
        )
        if matched:
            hits += 1

    return {
        "curated_total": len(queries),
        "curated_hits": hits,
        "curated_details": details,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def format_report(
    repo_path: Path,
    k: int,
    metric_a: dict,
    metric_b: dict,
    seed: int,
    min_doc_chars: int,
) -> str:
    cfg = load_config(repo_path)
    pid = project_id(repo_path)

    _, embedder_name = embeddings_available(cfg)
    embedder_dim = 384  # fastembed default (BAAI/bge-small-en-v1.5)

    lines: list[str] = []
    lines.append(f"## Repository: `{repo_path}`")
    lines.append("")
    lines.append(f"- **Project ID**: `{pid}`")
    lines.append(f"- **Profile**: `{cfg.profile}`")
    lines.append(
        f"- **Retrieval mode**: `{cfg.retrieval.mode}` "
        f"(rerank: {cfg.retrieval.rerank_enabled})"
    )
    lines.append(
        f"- **Embedder**: {embedder_name or 'N/A'} (dim={embedder_dim})"
    )
    lines.append(f"- **Sample seed**: {seed}, min doc chars: {min_doc_chars}")
    lines.append("")

    # Metric A
    lines.append("### Metric A — Self-Retrieval (docstring → function)")
    lines.append("")
    ma = metric_a
    lines.append(
        f"- Total documented blocks found: {ma['n_total_documented']}"
    )
    lines.append(f"- Sampled & evaluated: {ma['n_evaluated']}")
    lines.append(
        f"- Flagged (name removal emptied query): {ma['flagged_count']}"
    )
    lines.append("")
    lines.append(
        f"| hit@1 | hit@5 | hit@{k} | MRR |"
    )
    lines.append("|-------|-------|------|-----|")
    lines.append(
        f"| {_fmt_pct(ma['hit_at_1'])} "
        f"| {_fmt_pct(ma['hit_at_5'])} "
        f"| {_fmt_pct(ma['hit_at_k'])} "
        f"| {ma['mrr']:.4f} |"
    )
    lines.append("")

    # Method note
    lines.append(
        "_Method: docstring/leading-comment used as query, with symbol name "
        "removed to reduce lexical echo. Query drives the full hybrid+rerank "
        "search pipeline via ContextAssembler.semantic_search()._"
    )
    lines.append("")

    # Metric B
    mb = metric_b
    lines.append("### Metric B — Curated Conceptual Queries")
    lines.append("")
    if mb.get("curated_total", 0) == 0:
        lines.append("_(No curated queries defined for this repo.)_")
    else:
        lines.append(
            f"**Score**: {mb['curated_hits']}/{mb['curated_total']} "
            f"({_fmt_pct(mb['curated_hits'] / max(1, mb['curated_total']))})"
        )
        lines.append("")
        for d in mb.get("curated_details", []):
            status = "PASS" if d["pass"] else "FAIL"
            lines.append(
                f"- **{status}** `{d['query']}` → expected `{d['expected']}`"
            )
            if "top_files" in d:
                for tf in d["top_files"]:
                    lines.append(f"  - {tf}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retrieval-quality harness (hybrid+rerank pipeline)"
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("/mnt/c/Users/alfre/Projects/django"),
        help="Absolute path to the indexed repo (default: django)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=60,
        help="Sample size for Metric A (default: 60)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k for search (default: 10)",
    )
    ap.add_argument(
        "--min-doc-chars",
        type=int,
        default=40,
        dest="min_doc_chars",
        help="Minimum docstring/comment length in chars (default: 40)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("scripts/retrieval_quality_report.md"),
        help="Markdown report path (default: scripts/retrieval_quality_report.md)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling (default: 0)",
    )
    args = ap.parse_args()

    repo_path = args.repo.resolve()
    if not repo_path.is_dir():
        print(f"ERROR: repo not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    repo_name = repo_path.name

    print(f"Repo: {repo_path}")
    print(f"Sample size: {args.n}, top-k: {args.k}")
    print(f"Min doc chars: {args.min_doc_chars}, seed: {args.seed}")
    print()

    # Build the assembler once — this loads the index, reranker, etc.
    print("Initializing context assembler (loading index, reranker, etc.) ...")
    assembler = ContextAssembler(
        project_path=repo_path, top_k=args.k
    )
    print("Assembler ready.\n")

    # --- Metric A ---
    print("=== Metric A: Self-Retrieval ===")
    print(f"Parsing repo & collecting documented blocks (min {args.min_doc_chars} chars)...")
    metric_a = run_metric_a(
        assembler,
        repo_path,
        n=args.n,
        k=args.k,
        min_doc_chars=args.min_doc_chars,
        seed=args.seed,
    )
    print(f"  Total documented blocks: {metric_a['n_total_documented']}")
    print(f"  Evaluated: {metric_a['n_evaluated']}")
    print(f"  Flagged (name removal emptied query): {metric_a['flagged_count']}")
    print(f"  hit@1:  {_fmt_pct(metric_a['hit_at_1'])}")
    print(f"  hit@5:  {_fmt_pct(metric_a['hit_at_5'])}")
    print(f"  hit@{args.k}: {_fmt_pct(metric_a['hit_at_k'])}")
    print(f"  MRR:    {metric_a['mrr']:.4f}")
    print()

    # --- Metric B ---
    print("=== Metric B: Curated Conceptual Queries ===")
    metric_b = run_metric_b(assembler, repo_name, k=args.k)
    if metric_b["curated_total"] == 0:
        print("  (no curated queries for this repo)")
    else:
        print(
            f"  Score: {metric_b['curated_hits']}/{metric_b['curated_total']}"
        )
        for d in metric_b["curated_details"]:
            status = "PASS" if d["pass"] else "FAIL"
            if d.get("error"):
                print(f"  [{status}] {d['query']} (error)")
            else:
                print(f"  [{status}] {d['query']}")
                for tf in d.get("top_files", [])[:2]:
                    print(f"      -> {tf}")
    print()

    # --- Write report ---
    report_text = format_report(
        repo_path,
        args.k,
        metric_a,
        metric_b,
        args.seed,
        args.min_doc_chars,
    )
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report appended to: {report_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
