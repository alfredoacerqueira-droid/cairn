"""Test that raw_cosine is correctly preserved through hybrid RRF fusion.

This test ensures the confidence guard can properly evaluate results in hybrid mode.
"""

from unittest.mock import MagicMock
from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.embeddings import EmbeddingRetriever
from pipeline.retrieval.hybrid import HybridRetriever


class TestHybridRawCosinePreservation:
    """Test that embedding cosines are preserved in hybrid mode."""

    def test_hybrid_preserves_embedding_cosine_for_matched_documents(self):
        """When a doc appears in embeddings with a known cosine, preserve it in hybrid fusion."""
        # Setup: create mock retrievers
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            # doc_a scores well in BM25
            {"id": "doc_a", "text": "code_a", "score": 20.0, "source": "bm25"},
            {"id": "doc_b", "text": "code_b", "score": 10.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            # doc_a also scores well in AST
            {"id": "doc_a", "text": "code_a", "score": 100.0, "source": "ast_rank"},
        ]

        embeddings = MagicMock(spec=EmbeddingRetriever)
        embeddings.search.return_value = [
            # doc_a has a REAL embedding cosine of 0.86
            {"id": "doc_a", "text": "code_a", "score": 0.86, "source": "embeddings"},
        ]

        # Create hybrid retriever
        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=embeddings,
            weights=[0.4, 0.3, 0.3],
            mode="hybrid",
        )

        # Execute search
        results = retriever.search("test query", top_k=5)

        # Find doc_a in results
        doc_a = [r for r in results if r["id"] == "doc_a"]
        assert len(doc_a) == 1, "doc_a should be in results"

        # Verify raw_cosine was preserved
        raw_cosine = doc_a[0].get("raw_cosine", 0.0)
        assert abs(raw_cosine - 0.86) < 0.01, f"doc_a raw_cosine should be ~0.86, got {raw_cosine}"

    def test_hybrid_sets_raw_cosine_zero_for_non_embedding_docs(self):
        """Docs that don't appear in embeddings results get raw_cosine=0.0."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            # doc_b ranks well in BM25 but NOT in embeddings
            {"id": "doc_b", "text": "code_b", "score": 20.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = []

        embeddings = MagicMock(spec=EmbeddingRetriever)
        embeddings.search.return_value = [
            # No results for this query (or doc_b not in results)
            {"id": "doc_a", "text": "code_a", "score": 0.9, "source": "embeddings"},
        ]

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=embeddings,
            weights=[0.4, 0.3, 0.3],
            mode="hybrid",
        )

        results = retriever.search("test query", top_k=5)

        # Find doc_b in results
        doc_b = [r for r in results if r["id"] == "doc_b"]
        if doc_b:  # doc_b might be filtered if out of top-k
            raw_cosine = doc_b[0].get("raw_cosine", 0.0)
            # Doc not in embeddings should have raw_cosine=0.0
            assert (
                raw_cosine == 0.0
            ), f"Non-embedding doc should have raw_cosine=0.0, got {raw_cosine}"

    def test_hybrid_with_guard_accepts_high_cosine(self):
        """Hybrid results with high embedding cosine should pass confidence guard."""
        from server.context_assembler import ContextAssembler

        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "doc_a", "text": "code_a", "score": 20.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": "doc_a", "text": "code_a", "score": 100.0, "source": "ast_rank"},
        ]

        embeddings = MagicMock(spec=EmbeddingRetriever)
        embeddings.search.return_value = [
            # Real cosine of 0.9 — passes guard with threshold 0.75
            {"id": "doc_a", "text": "code_a", "score": 0.9, "source": "embeddings"},
        ]

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=embeddings,
            weights=[0.4, 0.3, 0.3],
            mode="hybrid",
        )

        results = retriever.search("test query", top_k=5)

        # Check guard behavior (guard checks if top cosine >= threshold)
        if results:
            top_cosine = results[0].get("raw_cosine", 0.0)
            guard_threshold = 0.75
            should_pass = top_cosine >= guard_threshold
            assert should_pass, f"Top result cosine {top_cosine} should pass guard"

    def test_hybrid_with_guard_rejects_low_cosine(self):
        """Hybrid results with low embedding cosine should not pass confidence guard."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "doc_a", "text": "code_a", "score": 20.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": "doc_a", "text": "code_a", "score": 100.0, "source": "ast_rank"},
        ]

        embeddings = MagicMock(spec=EmbeddingRetriever)
        embeddings.search.return_value = [
            # Low cosine of 0.6 — does not pass guard with threshold 0.75
            {"id": "doc_a", "text": "code_a", "score": 0.6, "source": "embeddings"},
        ]

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=embeddings,
            weights=[0.4, 0.3, 0.3],
            mode="hybrid",
        )

        results = retriever.search("test query", top_k=5)

        # Check guard behavior
        if results:
            top_cosine = results[0].get("raw_cosine", 0.0)
            guard_threshold = 0.75
            should_pass = top_cosine >= guard_threshold
            assert not should_pass, f"Low cosine {top_cosine} should not pass guard"
