"""Demo dataset loader using the testAPI nail marketplace as a benchmark target.

Each query is a bug description; the ground truth is the set of function IDs
that should appear in top-K results.

In production, replace with SWE-bench-Lite loaders (flask) and docstring-based
loaders (requests).
"""

from __future__ import annotations

BenchmarkQuery = tuple[str, set[str]]


def load_demo_queries() -> list[BenchmarkQuery]:
    """Return (query_text, {relevant_function_ids}) pairs."""
    prefix = "app/main.py"
    return [
        (
            "Race condition in ID generation with sleep between read and increment",
            {f"{prefix}:create_nail:10"},
        ),
        (
            "Memory leak in request logging unbounded list",
            {f"{prefix}:log_requests:120"},
        ),
        (
            "Missing tax calculation in order total",
            {f"{prefix}:place_order:80"},
        ),
        (
            "Inefficient linear search no pagination O(n)",
            {f"{prefix}:search_nails:50", f"{prefix}:list_nails:5"},
        ),
        (
            "Stock deduction rollback transaction",
            {f"{prefix}:place_order:80"},
        ),
        (
            "Admin endpoint missing authentication",
            {f"{prefix}:admin_cancel_all_orders:140"},
        ),
    ]
