"""AST-graph PageRank retriever — structural code ranking.

Builds a symbol reference graph from parsed AST data using pipeline/ast_parser.py,
then scores functions via PageRank and keyword-based relevance.
"""

from __future__ import annotations

from typing import Any


def _extract_keywords(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z_][a-z0-9_]*", text.lower()))


class ASTRankRetriever:
    """Scores functions by a mix of PageRank in the call graph and keyword relevance."""

    def __init__(self, damping: float = 0.85, max_iter: int = 100):
        self.damping = damping
        self.max_iter = max_iter
        self._functions: list[dict[str, Any]] = []
        self._name_to_idx: dict[str, int] = {}
        self._pagerank_scores: list[float] = []

    def index(self, items: list[dict[str, Any]], repo_map: dict | None = None) -> None:
        """Load function metadata and optional repo map for the call graph.

        Each item should have: id, text, name, filepath.
        """
        self._functions = items
        self._name_to_idx = {item["id"]: i for i, item in enumerate(items)}

        n = len(items)
        if n == 0:
            self._pagerank_scores = []
            return

        graph: list[list[int]] = [[] for _ in range(n)]

        for i, item in enumerate(items):
            text = item.get("text", "")
            keywords = _extract_keywords(text)

            for j, other in enumerate(items):
                if i == j:
                    continue
                other_name = other.get("name", "")
                if other_name and other_name in keywords:
                    graph[i].append(j)

        self._pagerank_scores = self._compute_pagerank(graph)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Score by PageRank + keyword relevance to query."""
        if not self._functions:
            return []

        query_kw = _extract_keywords(query.lower())
        n = len(self._functions)

        scores: list[tuple[float, str]] = []
        for i, func in enumerate(self._functions):
            text = func.get("text", "")
            func_kw = _extract_keywords(text)

            keyword_score = 0.0
            for qk in query_kw:
                if qk in func_kw:
                    keyword_score += 1.0
            keyword_score = keyword_score / max(len(query_kw), 1)

            pr = self._pagerank_scores[i] if i < len(self._pagerank_scores) else 0.0
            combined = 0.7 * keyword_score + 0.3 * (pr * n)

            scores.append((combined, func["id"]))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, doc_id in scores[:top_k]:
            func = self._functions[self._name_to_idx.get(doc_id, 0)]
            results.append(
                {
                    "id": doc_id,
                    "text": func.get("text", ""),
                    "score": round(score, 4),
                    "source": "ast_rank",
                }
            )
        return results

    def _compute_pagerank(self, graph: list[list[int]]) -> list[float]:
        n = len(graph)
        if n == 0:
            return []

        pr = [1.0 / n] * n
        for _ in range(self.max_iter):
            new_pr = [(1.0 - self.damping) / n] * n
            for i in range(n):
                if not graph[i]:
                    continue
                share = pr[i] / len(graph[i])
                for j in graph[i]:
                    new_pr[j] += self.damping * share
            pr = new_pr
        return pr
