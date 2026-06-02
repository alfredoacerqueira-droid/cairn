"""Hybrid retriever — fuses scores from multiple retrievers via reciprocal-rank fusion.

Combines BM25 (lexical), AST-graph PageRank (structural), and embeddings
(semantic), then optionally applies a cross-encoder reranker.

Method: Reciprocal Rank Fusion (RRF) produces a single ranking from multiple
rankers while respecting that each score distribution differs.
"""

from __future__ import annotations

from typing import Any

from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.embeddings import EmbeddingRetriever


def _normalize_scores(results: list[dict], raw_key: str = "score") -> list[dict]:
    """Min-max normalize scores to [0, 1] range.

    Takes raw scores under raw_key and produces normalized 'similarity' in [0,1].
    If all scores are equal or only one result, similarity=1.0 for all.
    Preserves the raw value under 'rrf_score'.
    Preserves ranking order and all other fields.
    """
    if not results:
        return results

    # Extract raw scores
    raw_scores = [float(r.get(raw_key, 0.0)) for r in results]

    min_score = min(raw_scores) if raw_scores else 0.0
    max_score = max(raw_scores) if raw_scores else 0.0

    # Handle edge cases: all equal or single result
    if min_score == max_score or len(results) == 1:
        normalized = [1.0] * len(results)
    else:
        # Min-max normalization: (x - min) / (max - min)
        normalized = [(score - min_score) / (max_score - min_score) for score in raw_scores]

    # Build result dicts with normalized similarity and preserved rrf_score
    normalized_results = []
    for i, result in enumerate(results):
        new_result = dict(result)
        new_result["rrf_score"] = round(raw_scores[i], 4)
        new_result["similarity"] = round(normalized[i], 4)
        # Remove the old raw_key if it's not already rrf_score
        if raw_key != "rrf_score":
            new_result.pop(raw_key, None)
        normalized_results.append(new_result)

    return normalized_results


def reciprocal_rank_fusion(
    result_sets: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists into one via RRF.

    Each doc gets score = sum(w_i / (k + rank_i)) across all lists.
    When weights=None, all lists are weighted equally.

    Note: The raw 'score' returned is an RRF artifact (w/(k+rank)), not a
    comparable similarity. Consumers should normalize via _normalize_scores()
    to get 'similarity' in [0,1].
    """
    if weights is None:
        weights = [1.0] * len(result_sets)

    scores: dict[str, float] = {}
    best_text: dict[str, str] = {}
    # Preserve the raw embedding cosine per doc — this is the ABSOLUTE quality
    # signal the confidence guard needs (RRF scores are relative-rank artifacts).
    raw_cosine: dict[str, float] = {}

    for set_idx, results in enumerate(result_sets):
        w = weights[set_idx] if set_idx < len(weights) else 1.0
        for rank, r in enumerate(results):
            doc_id = r["id"]
            rrf_score = w / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in best_text:
                best_text[doc_id] = r.get("text", "")
            # Capture the raw embedding cosine for docs that appear in embeddings results.
            # Docs not in embeddings results will have raw_cosine=0.0 (no embedding signal).
            # The confidence guard uses this to evaluate result quality.
            if r.get("source") == "embeddings":
                raw_cosine[doc_id] = float(r.get("score", 0.0))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "id": doc_id,
            "text": best_text.get(doc_id, ""),
            "score": round(score, 4),
            "raw_cosine": round(raw_cosine.get(doc_id, 0.0), 4),
            "source": "hybrid",
        }
        for doc_id, score in ranked
    ]


class HybridRetriever:
    """Combines multiple retrievers via RRF fusion, optionally reranked."""

    def __init__(
        self,
        bm25: BM25Retriever,
        ast_rank: ASTRankRetriever,
        embeddings: EmbeddingRetriever | None,
        weights: list[float] | None = None,
        mode: str = "hybrid",
        reranker: Any = None,  # FlashRank Reranker OR LLMReranker (duck-typed .rerank)
        rerank_enabled: bool = False,
        lexical: Any = None,
        structural: Any = None,
        profile_legs: list[str] | None = None,
    ):
        self.bm25 = bm25
        self.ast_rank = ast_rank
        self.embeddings = embeddings
        self.mode = mode
        self.weights = weights or [0.4, 0.3, 0.3]  # bm25, ast, embeddings
        self.reranker = reranker
        self.rerank_enabled = rerank_enabled
        # Optional lexical leg (RipgrepRetriever). When set, it REPLACES the
        # bm25+ast legs in hybrid fusion: lexical (fresh, exact) + embeddings,
        # then the cross-encoder reranks. The AST keyword-graph leg degraded
        # ranking on large repos (Django) so it is no longer a default leg.
        self.lexical = lexical
        # Optional structural leg (StructuralRetriever). When set, it is added
        # as an additional retrieval signal in hybrid fusion alongside lexical
        # and embeddings. Excels at exact block-identity matching for config
        # (Terraform, Kubernetes, etc.) where embeddings fail.
        self.structural = structural
        # Profile-driven retrieval legs from the repo profile (e.g., ['structural',
        # 'lexical'] for iac). If set, OVERRIDES the default leg selection logic.
        self.profile_legs = profile_legs or []

    def search(
        self,
        query: str,
        top_k: int = 10,
        commit: str = "unknown",
    ) -> list[dict[str, Any]]:
        """Execute retrieval based on config mode.

        Returns results with normalized 'similarity' in [0,1] and 'rrf_score'
        containing the raw source score (RRF fusion value or original score).
        """
        if self.mode == "bm25":
            results = self.bm25.search(query, top_k=top_k)
            return self._normalize_single_mode(results, "score")
        if self.mode == "ast":
            results = self.ast_rank.search(query, top_k=top_k)
            return self._normalize_single_mode(results, "score")
        if self.mode == "embeddings" and self.embeddings:
            results = self.embeddings.search(query, top_k=top_k, commit=commit)
            return self._normalize_single_mode(results, "score")
        if self.mode == "hybrid":
            result_sets: list[list[dict[str, Any]]] = []
            active_weights: list[float] = []

            # Lexical leg: ripgrep (fresh, exact-match) if available, else BM25.
            if self.lexical is not None:
                result_sets.append(self.lexical.search(query, top_k=top_k * 2))
                active_weights.append(self.weights[0])
            else:
                result_sets.append(self.bm25.search(query, top_k=top_k * 2))
                active_weights.append(self.weights[0])

            if self.embeddings:
                result_sets.append(self.embeddings.search(query, top_k=top_k * 2, commit=commit))
                active_weights.append(self.weights[2])

            # Optional structural leg: exact block-identity + reference matching.
            # Complements lexical and embeddings with deterministic struct
            if self.structural is not None:
                result_sets.append(self.structural.search(query, top_k=top_k * 2))
                # Add structural weight (default same as embeddings)
                if len(active_weights) >= 2:
                    # weights[2] is embeddings weight, use same for structural
                    active_weights.append(self.weights[2])
                else:
                    active_weights.append(0.3)

            fused = reciprocal_rank_fusion(result_sets, k=60, weights=active_weights)
            normalized = _normalize_scores(fused, raw_key="score")

            # If reranking enabled, expand candidates and rerank by cross-encoder
            if self.rerank_enabled and self.reranker:
                # Use a wider candidate pool for reranking (e.g. top ~40)
                candidates = normalized[: max(top_k * 4, 40)]
                reranked = self.reranker.rerank(query, candidates, top_k=top_k)
                return reranked

            return normalized[:top_k]

        # Fallback to bm25
        results = self.bm25.search(query, top_k=top_k)
        return self._normalize_single_mode(results, "score")

    def _normalize_single_mode(
        self, results: list[dict[str, Any]], raw_key: str = "score"
    ) -> list[dict[str, Any]]:
        """Normalize single-retriever results to [0,1] similarity scale."""
        if not results:
            return results

        # Extract raw scores
        raw_scores = [float(r.get(raw_key, 0.0)) for r in results]
        min_score = min(raw_scores) if raw_scores else 0.0
        max_score = max(raw_scores) if raw_scores else 0.0

        # Handle edge cases
        if min_score == max_score or len(results) == 1:
            normalized = [1.0] * len(results)
        else:
            normalized = [(score - min_score) / (max_score - min_score) for score in raw_scores]

        # Build normalized results
        normalized_results = []
        for i, result in enumerate(results):
            new_result = dict(result)
            new_result["rrf_score"] = round(raw_scores[i], 4)
            new_result["similarity"] = round(normalized[i], 4)
            # For embeddings results the raw score IS cosine similarity — preserve
            # it as the absolute quality signal the confidence guard uses.
            if "raw_cosine" not in new_result:
                if result.get("source") == "embeddings":
                    new_result["raw_cosine"] = round(raw_scores[i], 4)
                else:
                    new_result["raw_cosine"] = 0.0
            # Remove old raw_key if not already rrf_score
            if raw_key != "rrf_score":
                new_result.pop(raw_key, None)
            normalized_results.append(new_result)

        return normalized_results
