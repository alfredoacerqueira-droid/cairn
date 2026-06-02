"""Token reduction benchmarks — measure prompt compression.

Computes token counts for different context strategies and reports reduction
relative to a baseline (whole-repo or file-tree + opened files).  Always paired
with retrieval recall so the quality/cost tradeoff is visible.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def approximate_tokens(text: str) -> int:
    """Approximate token count: ~4 chars per token (code text)."""
    return max(1, len(text) // 4)


def whole_repo_baseline(project_path: Path, patterns: list[str] | None = None) -> int:
    """Token count for dumping the entire codebase."""
    if patterns is None:
        patterns = ["**/*.py"]
    total = 0
    for pattern in patterns:
        for f in project_path.glob(pattern):
            if ".venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
                continue
            try:
                total += approximate_tokens(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return total


def file_tree_baseline(project_path: Path, max_files: int = 50) -> int:
    """Token count for file-tree listing + first N files."""
    total = 0
    files = sorted(
        [
            f
            for f in project_path.rglob("*.py")
            if ".venv" not in str(f) and "__pycache__" not in str(f)
        ]
    )
    for f in files[:max_files]:
        try:
            total += approximate_tokens(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return total


def bm25_at_budget(bm25: Any, queries: list[str], budget: int) -> list[int]:
    """Token count for BM25 retrieval at a target token budget."""
    counts: list[int] = []
    for q in queries:
        results = bm25.search(q, top_k=30)
        tokens = 0
        for r in results:
            t = approximate_tokens(r.get("text", ""))
            if tokens + t > budget:
                break
            tokens += t
        counts.append(tokens)
    return counts


def compute_reduction_stats(
    baseline_tokens: int,
    per_query_tokens: list[int],
) -> dict[str, Any]:
    """Compute median, p25, p75 token reduction stats."""
    if not per_query_tokens:
        return {"baseline_tokens": baseline_tokens, "median_tokens": 0, "reduction_pct": 0.0}

    sorted_tokens = sorted(per_query_tokens)
    n = len(sorted_tokens)
    median = sorted_tokens[n // 2]
    p25 = sorted_tokens[n // 4]
    p75 = sorted_tokens[3 * n // 4] if 3 * n // 4 < n else sorted_tokens[-1]

    reduction_pct = round(100 * (1 - median / max(baseline_tokens, 1)), 1)

    return {
        "baseline_tokens": baseline_tokens,
        "median_tokens": median,
        "p25_tokens": p25,
        "p75_tokens": p75,
        "reduction_pct": reduction_pct,
        "per_query_tokens": per_query_tokens,
    }


def run_token_benchmark(
    project_path: Path,
    retriever_fn: Callable[[str, int], list[dict[str, Any]]],
    queries: list[str],
    budget: int = 3000,
) -> dict[str, Any]:
    """Run a full token reduction benchmark comparing three baselines."""
    whole_repo = whole_repo_baseline(project_path)
    file_tree = file_tree_baseline(project_path)

    per_query: list[int] = []
    for q in queries:
        results = retriever_fn(q, 10)
        tokens = sum(approximate_tokens(r.get("text", "")) for r in results)
        per_query.append(tokens)

    return {
        "baselines": {
            "whole_repo_tokens": whole_repo,
            "file_tree_50_files": file_tree,
        },
        "retrieval": compute_reduction_stats(whole_repo, per_query),
    }
