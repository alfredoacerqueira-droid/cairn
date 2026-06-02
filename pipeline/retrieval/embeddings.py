"""Embedding-based retriever — wraps existing ChromaDB semantic search."""

from __future__ import annotations

from typing import Any, Optional

from core.cache import SessionCache
from pipeline.indexer import VectorIndexer


class EmbeddingRetriever:
    """Thin wrapper around VectorIndexer.search() with the retriever interface."""

    def __init__(
        self,
        vector_indexer: VectorIndexer,
        cache: Optional[SessionCache] = None,
    ):
        self._indexer = vector_indexer
        self._cache = cache

    def search(
        self,
        query: str,
        top_k: int = 10,
        commit: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Semantic search via ChromaDB cosine similarity, with optional cache."""
        cache_key = ("emb_retrieve", query, str(top_k), commit)

        if self._cache:
            cached = self._cache.get(*cache_key)
            if cached is not None:
                return cached

        raw = self._indexer.search(query, top_k=top_k)
        results: list[dict[str, Any]] = []
        for r in raw:
            results.append(
                {
                    "id": (
                        f"{r.get('filepath', '')}:{r.get('function', '')}"
                        f":{r.get('line_start', 0)}"
                    ),
                    "text": r.get("code", ""),
                    "score": round(r.get("similarity", 0.0), 4),
                    "source": "embeddings",
                }
            )

        if self._cache:
            self._cache.set(results, *cache_key)

        return results
