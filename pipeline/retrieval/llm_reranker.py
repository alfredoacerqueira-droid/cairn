"""LLM-based reranker — scores candidates with the local Ollama worker model.

OPT-IN ONLY. Drop-in for the FlashRank `Reranker` (same `.rerank` interface),
but it asks a local generative model to score each candidate's relevance.

Honest performance note: on a 6GB desktop GPU the local worker model takes
~19-39s PER generation, so reranking even 10 candidates is several minutes per
query — impractical on the request path. FlashRank (cross-encoder, CPU, ms) is
the default. Enable this only on fast local inference (Apple Silicon / big VRAM).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cap how many candidates we send to the LLM — each one is a slow generation.
_MAX_CANDIDATES = 10
# Truncate each block so the prompt stays small.
_MAX_BLOCK_CHARS = 800


class LLMReranker:
    """Rerank candidates by asking a local LLM to score relevance 0-10."""

    def __init__(self, ollama_client: Any = None, model: Optional[str] = None):
        # Lazy import to avoid a hard dependency at module load.
        if ollama_client is None:
            from server.ollama_client import OllamaClient

            ollama_client = OllamaClient()
        self.ollama = ollama_client
        self.model = model  # None → OllamaClient's configured generate model

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Score candidates with the LLM; sort desc by rerank_score (0..1).

        Graceful: on any error the candidate keeps its existing similarity as the
        score; the reranker never raises to the caller. Only the first
        _MAX_CANDIDATES are LLM-scored (latency cap); the rest keep similarity.
        """
        if not candidates:
            return []

        scored: list[dict] = []
        for i, cand in enumerate(candidates):
            new = dict(cand)
            if i >= _MAX_CANDIDATES:
                # Beyond the cap: don't pay the LLM cost; fall back to similarity.
                new["rerank_score"] = float(new.get("similarity", 0.0))
                scored.append(new)
                continue
            new["rerank_score"] = self._score_one(query, new)
            scored.append(new)

        scored.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k] if top_k is not None else scored

    def _score_one(self, query: str, candidate: dict) -> float:
        """Ask the LLM for a 0-10 relevance score; normalize to 0..1."""
        text = str(candidate.get("text", ""))[:_MAX_BLOCK_CHARS]
        prompt = (
            "Score from 0 to 10 how relevant this code block is to the query. "
            "Answer with ONLY a single integer 0-10, nothing else.\n\n"
            f"Query: {query}\n\nBlock:\n{text}\n\nScore:"
        )
        try:
            out = self.ollama.generate(prompt, model=self.model)
            m = re.search(r"\d+", out or "")
            if not m:
                return float(candidate.get("similarity", 0.0))
            score = max(0, min(10, int(m.group(0))))
            return round(score / 10.0, 4)
        except Exception as e:  # noqa: BLE001 — never break retrieval
            logger.debug("LLM rerank failed for one candidate: %s", e)
            return float(candidate.get("similarity", 0.0))
