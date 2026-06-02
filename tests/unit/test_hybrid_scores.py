"""Unit tests for score normalization in HybridRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock

from pipeline.retrieval.ast_rank import ASTRankRetriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.embeddings import EmbeddingRetriever
from pipeline.retrieval.hybrid import HybridRetriever, _normalize_scores


class TestNormalizeScores:
    """Test the _normalize_scores helper function."""

    def test_normalize_empty_list(self):
        """Empty input should return empty list."""
        result = _normalize_scores([], raw_key="score")
        assert result == []

    def test_normalize_single_result(self):
        """Single result should have similarity=1.0."""
        results = [{"id": "a", "text": "hello", "score": 5.0}]
        normalized = _normalize_scores(results, raw_key="score")

        assert len(normalized) == 1
        assert normalized[0]["id"] == "a"
        assert normalized[0]["similarity"] == 1.0
        assert normalized[0]["rrf_score"] == 5.0
        assert "score" not in normalized[0]  # raw key removed

    def test_normalize_equal_scores(self):
        """All equal scores should all get similarity=1.0."""
        results = [
            {"id": "a", "text": "hello", "score": 3.0},
            {"id": "b", "text": "world", "score": 3.0},
            {"id": "c", "text": "test", "score": 3.0},
        ]
        normalized = _normalize_scores(results, raw_key="score")

        assert len(normalized) == 3
        for i, r in enumerate(normalized):
            assert r["similarity"] == 1.0
            assert r["rrf_score"] == 3.0

    def test_normalize_range_minmax(self):
        """Min-max normalization should scale to [0, 1]."""
        results = [
            {"id": "a", "text": "hello", "score": 10.0},
            {"id": "b", "text": "world", "score": 5.0},
            {"id": "c", "text": "test", "score": 0.0},
        ]
        normalized = _normalize_scores(results, raw_key="score")

        assert len(normalized) == 3
        assert normalized[0]["similarity"] == 1.0  # (10-0)/(10-0)
        assert normalized[1]["similarity"] == 0.5  # (5-0)/(10-0)
        assert normalized[2]["similarity"] == 0.0  # (0-0)/(10-0)

    def test_normalize_preserves_order(self):
        """Normalization should preserve ranking order."""
        results = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
            {"id": "c", "text": "test", "score": 0.0, "source": "bm25"},
        ]
        normalized = _normalize_scores(results, raw_key="score")

        assert [r["id"] for r in normalized] == ["a", "b", "c"]

    def test_normalize_preserves_source_field(self):
        """Other fields should be preserved."""
        results = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
        ]
        normalized = _normalize_scores(results, raw_key="score")

        assert normalized[0]["source"] == "bm25"
        assert normalized[0]["text"] == "hello"

    def test_normalize_rounding(self):
        """Normalized scores should be rounded to 4 decimals."""
        results = [
            {"id": "a", "text": "hello", "score": 1.0},
            {"id": "b", "text": "world", "score": 0.0},
        ]
        normalized = _normalize_scores(results, raw_key="score")

        # (1.0 - 0) / (1.0 - 0) = 1.0, (0 - 0) / (1.0 - 0) = 0.0
        assert normalized[0]["similarity"] == 1.0
        assert normalized[1]["similarity"] == 0.0


class TestHybridRetrieverBM25Mode:
    """Test HybridRetriever in bm25 mode."""

    def test_bm25_mode_normalizes_scores(self):
        """BM25 mode should normalize scores to [0, 1]."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 0.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        retriever = HybridRetriever(bm25=bm25, ast_rank=ast_rank, embeddings=None, mode="bm25")

        results = retriever.search("test", top_k=10)

        assert len(results) == 2
        assert results[0]["similarity"] == 1.0
        assert results[1]["similarity"] == 0.0
        assert results[0]["rrf_score"] == 10.0
        assert results[1]["rrf_score"] == 0.0
        assert "score" not in results[0]
        assert "score" not in results[1]

    def test_bm25_mode_preserves_order(self):
        """BM25 mode should preserve ranking order."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
            {"id": "c", "text": "test", "score": 2.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        retriever = HybridRetriever(bm25=bm25, ast_rank=ast_rank, embeddings=None, mode="bm25")

        results = retriever.search("test", top_k=10)

        assert [r["id"] for r in results] == ["a", "b", "c"]
        # Verify decreasing similarity
        assert results[0]["similarity"] > results[1]["similarity"]
        assert results[1]["similarity"] > results[2]["similarity"]

    def test_bm25_mode_empty_results(self):
        """BM25 mode should handle empty results."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = []

        ast_rank = MagicMock(spec=ASTRankRetriever)
        retriever = HybridRetriever(bm25=bm25, ast_rank=ast_rank, embeddings=None, mode="bm25")

        results = retriever.search("test", top_k=10)
        assert results == []


class TestHybridRetrieverASTMode:
    """Test HybridRetriever in ast mode."""

    def test_ast_mode_normalizes_scores(self):
        """AST mode should normalize scores to [0, 1]."""
        bm25 = MagicMock(spec=BM25Retriever)
        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": "a", "text": "func_a", "score": 0.8, "source": "ast_rank"},
            {"id": "b", "text": "func_b", "score": 0.4, "source": "ast_rank"},
        ]

        retriever = HybridRetriever(bm25=bm25, ast_rank=ast_rank, embeddings=None, mode="ast")

        results = retriever.search("test", top_k=10)

        assert len(results) == 2
        assert results[0]["similarity"] == 1.0
        assert results[1]["similarity"] == 0.0
        assert results[0]["rrf_score"] == 0.8
        assert results[1]["rrf_score"] == 0.4


class TestHybridRetrieverEmbeddingsMode:
    """Test HybridRetriever in embeddings mode."""

    def test_embeddings_mode_normalizes_scores(self):
        """Embeddings mode should normalize scores to [0, 1]."""
        bm25 = MagicMock(spec=BM25Retriever)
        ast_rank = MagicMock(spec=ASTRankRetriever)
        embeddings = MagicMock(spec=EmbeddingRetriever)
        embeddings.search.return_value = [
            {"id": "a", "text": "hello", "score": 0.9, "source": "embeddings"},
            {"id": "b", "text": "world", "score": 0.1, "source": "embeddings"},
        ]

        retriever = HybridRetriever(
            bm25=bm25, ast_rank=ast_rank, embeddings=embeddings, mode="embeddings"
        )

        results = retriever.search("test", top_k=10, commit="abc123")

        assert len(results) == 2
        assert results[0]["similarity"] == 1.0
        assert results[1]["similarity"] == 0.0
        assert results[0]["rrf_score"] == 0.9
        assert results[1]["rrf_score"] == 0.1


class TestHybridRetrieverHybridMode:
    """Test HybridRetriever in hybrid mode."""

    def test_hybrid_mode_normalizes_fusion_scores(self):
        """Hybrid mode should normalize RRF scores to [0, 1]."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": "a", "text": "hello", "score": 0.8, "source": "ast_rank"},
            {"id": "c", "text": "test", "score": 0.4, "source": "ast_rank"},
        ]

        embeddings = None

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=embeddings,
            weights=[0.4, 0.3, 0.3],
            mode="hybrid",
        )

        results = retriever.search("test", top_k=10)

        # Results should be normalized
        assert len(results) > 0
        for result in results:
            assert "similarity" in result
            assert "rrf_score" in result
            assert 0.0 <= result["similarity"] <= 1.0
            assert result["source"] == "hybrid"
            assert "score" not in result  # raw score removed

    def test_hybrid_mode_top_k_limit(self):
        """Hybrid mode should respect top_k."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": f"a{i}", "text": f"text{i}", "score": 10.0 - i, "source": "bm25"}
            for i in range(20)
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": f"b{i}", "text": f"text{i}", "score": 8.0 - i, "source": "ast_rank"}
            for i in range(20)
        ]

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=None,
            weights=[0.4, 0.3],
            mode="hybrid",
        )

        results = retriever.search("test", top_k=5)
        assert len(results) == 5

    def test_hybrid_mode_preserves_ranking_order(self):
        """Hybrid mode should preserve RRF ranking order."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        ast_rank.search.return_value = [
            {"id": "b", "text": "world", "score": 10.0, "source": "ast_rank"},
            {"id": "a", "text": "hello", "score": 5.0, "source": "ast_rank"},
        ]

        retriever = HybridRetriever(
            bm25=bm25,
            ast_rank=ast_rank,
            embeddings=None,
            weights=[0.5, 0.5],
            mode="hybrid",
        )

        results = retriever.search("test", top_k=10)

        # Both a and b should be present
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "a" in ids and "b" in ids


class TestHybridRetrieverContract:
    """Test the output contract for all modes."""

    def test_all_modes_return_required_fields(self):
        """All modes should return results with id, text, similarity, rrf_score."""
        for mode in ["bm25", "ast", "embeddings", "hybrid"]:
            bm25 = MagicMock(spec=BM25Retriever)
            bm25.search.return_value = [
                {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
                {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
            ]

            ast_rank = MagicMock(spec=ASTRankRetriever)
            ast_rank.search.return_value = [
                {"id": "a", "text": "hello", "score": 0.8, "source": "ast_rank"},
                {"id": "b", "text": "world", "score": 0.4, "source": "ast_rank"},
            ]

            embeddings = MagicMock(spec=EmbeddingRetriever)
            embeddings.search.return_value = [
                {"id": "a", "text": "hello", "score": 0.9, "source": "embeddings"},
                {"id": "b", "text": "world", "score": 0.6, "source": "embeddings"},
            ]

            retriever = HybridRetriever(
                bm25=bm25,
                ast_rank=ast_rank,
                embeddings=embeddings if mode == "embeddings" else None,
                mode=mode,
            )

            results = retriever.search("test", top_k=10)

            # Check required fields
            assert len(results) > 0
            for result in results:
                assert "id" in result
                assert "text" in result
                assert "similarity" in result
                assert "rrf_score" in result
                assert isinstance(result["similarity"], (int, float))
                assert isinstance(result["rrf_score"], (int, float))
                assert 0.0 <= result["similarity"] <= 1.0

    def test_top_result_similarity_is_max(self):
        """Top-ranked result should have highest similarity."""
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = [
            {"id": "a", "text": "hello", "score": 10.0, "source": "bm25"},
            {"id": "b", "text": "world", "score": 5.0, "source": "bm25"},
            {"id": "c", "text": "test", "score": 0.0, "source": "bm25"},
        ]

        ast_rank = MagicMock(spec=ASTRankRetriever)
        retriever = HybridRetriever(bm25=bm25, ast_rank=ast_rank, embeddings=None, mode="bm25")

        results = retriever.search("test", top_k=10)

        assert results[0]["similarity"] >= results[1]["similarity"]
        if len(results) > 2:
            assert results[1]["similarity"] >= results[2]["similarity"]
