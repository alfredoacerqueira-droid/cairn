"""Cross-encoder reranker using FlashRank.

Wraps the FlashRank library to reorder and rescore retrieval results
by true query-document relevance. Designed to improve precision by
ranking on semantic similarity rather than lexical/structural signals.

Uses a singleton Ranker instance for efficiency (model loading is slow).
Gracefully degrades to a no-op if flashrank is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Singleton Ranker instance — loaded on first use
_ranker: Any = None
_ranker_failed: bool = False


def _get_ranker() -> Any | None:
    """Lazy-load the FlashRank Ranker as a singleton.

    Returns None if flashrank is not available or the model fails to load.
    Sets a flag to avoid repeated attempts after failure.
    """
    global _ranker, _ranker_failed

    if _ranker_failed:
        return None

    if _ranker is not None:
        return _ranker

    try:
        from flashrank import Ranker

        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        logger.info("FlashRank reranker initialized (ms-marco-MiniLM-L-12-v2)")
        return _ranker
    except ImportError:
        logger.warning("flashrank not available; reranking disabled (install via pip)")
        _ranker_failed = True
        return None
    except Exception as e:
        logger.warning(f"Failed to load FlashRank model: {e}; reranking disabled")
        _ranker_failed = True
        return None


class Reranker:
    """Rerank retrieval results using FlashRank cross-encoder.

    The Reranker wraps the FlashRank library to rescore candidate documents
    by true query-document relevance. It reorders candidates by the
    cross-encoder's absolute relevance score and attaches the score to
    each result dict as 'rerank_score'.

    If flashrank is unavailable or model loading fails, the reranker
    gracefully returns candidates unchanged (no-op fallback).
    """

    def __init__(self):
        """Initialize the reranker (lazy-loads model on first rerank call)."""
        self.ranker = None

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerank candidates by cross-encoder relevance.

        Args:
            query: The search query string.
            candidates: List of dicts with at least 'id' and 'text' keys.
            top_k: If specified, return top_k results. Otherwise return all.

        Returns:
            Reranked list of dicts, sorted descending by 'rerank_score'.
            Each dict includes the new 'rerank_score' field (0..1 scale).
            If reranking fails, returns candidates unchanged with
            'rerank_score' set from existing 'similarity' or 0.0.

        Design notes:
            - Truncates code to ~2000 chars to bound latency.
            - Maps passages back to original dicts by index.
            - Gracefully degrades if flashrank is unavailable.
            - Never raises to caller.
        """
        if not candidates:
            return []

        ranker = _get_ranker()
        if ranker is None:
            # Graceful no-op: attach rerank_score from existing similarity
            for c in candidates:
                c.setdefault("rerank_score", c.get("similarity", 0.0))
            return candidates

        try:
            # Build passages. FlashRank maps results back via the passage "id" it
            # echoes in each result — so we use the candidate's list index as the id.
            passages = []
            for i, candidate in enumerate(candidates):
                text = candidate.get("text", "")
                # Truncate very long code (~2000 chars) to bound cross-encoder latency
                if len(text) > 2000:
                    text = text[:2000] + "..."
                passages.append({"id": i, "text": text, "meta": {}})

            # Run the cross-encoder
            from flashrank import RerankRequest

            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            # FlashRank returns results sorted by score desc, each echoing the passage
            # "id" we supplied. Map each back to its original candidate by that id.
            reranked = []
            seen: set[int] = set()
            for result in results:
                idx = int(result.get("id", -1))
                if idx in seen or not (0 <= idx < len(candidates)):
                    continue
                seen.add(idx)
                candidate = dict(candidates[idx])
                candidate["rerank_score"] = round(float(result.get("score", 0.0)), 4)
                reranked.append(candidate)

            # Defensive: if mapping produced nothing, fall back to input order.
            if not reranked:
                for c in candidates:
                    c.setdefault("rerank_score", c.get("similarity", 0.0))
                reranked = list(candidates)

            # Return top_k if specified, else all reranked results
            if top_k is not None:
                reranked = reranked[:top_k]

            return reranked

        except Exception as e:
            # Defensive: never raise to caller. Log and return with fallback scores.
            logger.warning(f"Reranking failed: {e}; returning candidates unchanged")
            for c in candidates:
                c.setdefault("rerank_score", c.get("similarity", 0.0))
            return candidates
