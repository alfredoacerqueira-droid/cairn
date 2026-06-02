"""Retrieval evaluation — Recall@k, MRR, nDCG@k across strategies.

Computes standard IR metrics over a list of (query, ground_truth) pairs,
comparing retrieval strategies such as embeddings-only vs hybrid.

NOTE: ID matching is normalized to handle both absolute and relative paths.
Indexed IDs look like '/absolute/path/file.py:Function:62' while dataset IDs
are 'relative/path/file.py:Function'. Matching strips line numbers and uses
endswith comparison for path robustness.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def _normalize_id(s: str) -> str:
    """Normalize an ID for comparison.

    Strips trailing ':<digits>' line-number suffix if present.
    Returns the normalized ID for endswith-based matching.
    """
    # Strip line number suffix (e.g. ':62')
    parts = s.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return s


def _matches(retrieved_id: str, relevant_id: str) -> bool:
    """Check if a retrieved ID matches a relevant ID.

    Uses endswith comparison after normalization to handle absolute/relative
    path mismatches.
    """
    retrieved_norm = _normalize_id(retrieved_id)
    relevant_norm = _normalize_id(relevant_id)

    # Match if one endswith the other, or if they're equal
    return (
        retrieved_norm == relevant_norm
        or retrieved_norm.endswith(relevant_norm)
        or relevant_norm.endswith(retrieved_norm)
    )


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k results."""
    if not relevant_ids:
        return 0.0

    matched = 0
    for rid in retrieved_ids[:k]:
        for rel_id in relevant_ids:
            if _matches(rid, rel_id):
                matched += 1
                break

    return matched / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant result."""
    for i, rid in enumerate(retrieved_ids, start=1):
        for rel_id in relevant_ids:
            if _matches(rid, rel_id):
                return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG@k using binary relevance (1 if relevant, 0 otherwise)."""
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        for rel_id in relevant_ids:
            if _matches(rid, rel_id):
                dcg += 1.0 / math.log2(i + 1)
                break

    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retriever_fn: Callable[[str, int], list[dict[str, Any]]],
    queries: list[tuple[str, set[str]]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@k, MRR, and nDCG@k across all queries.

    Args:
        retriever_fn: callable(query_text, top_k) -> [{id, score, text, source}]
        queries: list of (query_text, {relevant_ids})
        k_values: tuple of k values for Recall/nDCG (e.g. (1, 5, 10))

    Returns:
        dict with keys like 'recall@1', 'recall@5', 'nDCG@10', 'MRR', etc.
    """
    if not queries:
        return {}

    mrr_total = 0.0
    recall_acc: dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_acc: dict[int, float] = {k: 0.0 for k in k_values}

    for query_text, relevant in queries:
        results = retriever_fn(query_text, max(k_values))
        retrieved_ids = [r["id"] for r in results]

        mrr_total += mrr(retrieved_ids, relevant)
        for k in k_values:
            recall_acc[k] += recall_at_k(retrieved_ids, relevant, k)
            ndcg_acc[k] += ndcg_at_k(retrieved_ids, relevant, k)

    n = len(queries)
    metrics: dict[str, float] = {"MRR": round(mrr_total / n, 4)}
    for k in k_values:
        metrics[f"recall@{k}"] = round(recall_acc[k] / n, 4)
        metrics[f"nDCG@{k}"] = round(ndcg_acc[k] / n, 4)

    return metrics


def compare_strategies(
    strategies: dict[str, Callable[[str, int], list[dict[str, Any]]]],
    queries: list[tuple[str, set[str]]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, Any]]:
    """Evaluate multiple retrieval strategies and return a comparison table."""
    rows: list[dict[str, Any]] = []
    for name, fn in strategies.items():
        metrics = evaluate_retrieval(fn, queries, k_values)
        row: dict[str, Any] = {"strategy": name}
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)
    return rows
