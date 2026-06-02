"""Tests for retrieval consistency across cache and normalization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.cache import SessionCache
from server.context_assembler import ContextAssembler


class TestSearchConsistency:
    """Verify that search results are consistent across cold and warm cache hits."""

    def setup_method(self):
        """Create a test assembler with minimal dependencies."""
        self.test_project = Path.cwd()
        self.cache = SessionCache(max_entries=10, ttl_seconds=300)

    def _make_hybrid_results(self, similarities: list[float]) -> list[dict]:
        """Generate mock hybrid retriever results with normalized similarities."""
        return [
            {
                "id": f"app.py:func{i}:10",
                "text": f"def func{i}():\n    pass",
                "similarity": round(sim, 4),
                "rrf_score": round(0.1 * (i + 1), 4),
                "source": "hybrid",
            }
            for i, sim in enumerate(similarities)
        ]

    def _make_raw_score_results(self, scores: list[float]) -> list[dict]:
        """Generate mock results with raw 'score' field (pre-normalization)."""
        return [
            {
                "id": f"app.py:func{i}:10",
                "text": f"def func{i}():\n    pass",
                "score": round(score, 4),
                "source": "bm25",
            }
            for i, score in enumerate(scores)
        ]

    def test_hybrid_results_to_legacy_uses_normalized_similarity(self):
        """TASK 1: When upstream provides normalized similarity, use it."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(mode="hybrid", weights=[0.4, 0.3, 0.3]),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # Upstream provides normalized [0,1] similarities
            hybrid_results = self._make_hybrid_results([0.95, 0.72, 0.45])

            legacy = assembler._hybrid_results_to_legacy(hybrid_results)

            assert len(legacy) == 3
            assert legacy[0]["similarity"] == 0.95
            assert legacy[1]["similarity"] == 0.72
            assert legacy[2]["similarity"] == 0.45
            # All in [0,1] range
            for result in legacy:
                assert 0.0 <= result["similarity"] <= 1.0

    def test_hybrid_results_to_legacy_self_normalizes_raw_scores(self):
        """TASK 1: When upstream does NOT provide normalized similarity, self-normalize."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(mode="hybrid", weights=[0.4, 0.3, 0.3]),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # Raw scores (e.g., RRF scores: 0.05, 0.03, 0.01)
            raw_results = self._make_raw_score_results([0.05, 0.03, 0.01])

            legacy = assembler._hybrid_results_to_legacy(raw_results)

            assert len(legacy) == 3
            # Min-max normalization: (0.05-0.01)/(0.05-0.01) = 1.0
            # (0.03-0.01)/(0.05-0.01) = 0.5, (0.01-0.01)/(0.05-0.01) = 0.0
            assert legacy[0]["similarity"] == 1.0
            assert abs(legacy[1]["similarity"] - 0.5) < 0.0001
            assert legacy[2]["similarity"] == 0.0
            # All in [0,1] range
            for result in legacy:
                assert 0.0 <= result["similarity"] <= 1.0

    def test_hybrid_results_to_legacy_all_equal_scores(self):
        """TASK 1: Handle edge case where all raw scores are equal."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(mode="hybrid", weights=[0.4, 0.3, 0.3]),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # All scores equal
            raw_results = self._make_raw_score_results([0.05, 0.05, 0.05])

            legacy = assembler._hybrid_results_to_legacy(raw_results)

            # All should normalize to 1.0 when equal
            assert all(result["similarity"] == 1.0 for result in legacy)

    def test_semantic_search_consistent_cold_and_warm(self):
        """TASK 2: Same query yields identical results (cold cache vs warm cache)."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=True, max_entries=100, ttl_seconds=300),
                retrieval=MagicMock(mode="hybrid", weights=[0.4, 0.3, 0.3]),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=self.cache)

            # Mock the git commit to return a fixed value
            with patch.object(assembler, "_git_commit", return_value="abc123"):
                # Mock _get_retriever to return consistent results
                mock_retriever = MagicMock()
                mock_retriever.search.return_value = self._make_hybrid_results([0.95, 0.72, 0.45])

                with patch.object(assembler, "_get_retriever", return_value=mock_retriever):
                    # Cold call (cache miss)
                    results_cold = assembler.semantic_search("test query", top_k=3)

                    # Warm call (cache hit)
                    results_warm = assembler.semantic_search("test query", top_k=3)

                    # Both should have identical similarity values in [0,1]
                    assert len(results_cold) == len(results_warm)
                    for cold, warm in zip(results_cold, results_warm):
                        assert cold["similarity"] == warm["similarity"]
                        assert 0.0 <= cold["similarity"] <= 1.0
                        assert 0.0 <= warm["similarity"] <= 1.0

    def test_confidence_guard_high_similarity_passes(self):
        """TASK 3: High similarity result passes the confidence guard."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(
                    mode="hybrid",
                    weights=[0.4, 0.3, 0.3],
                    min_confidence=0.2,
                    rerank_enabled=False,
                    rerank_min_score=0.0,
                ),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # Mock semantic_search to return high-similarity results
            with patch.object(
                assembler,
                "semantic_search",
                return_value=[
                    {
                        "filepath": "app.py",
                        "function": "handler",
                        "line_start": 10,
                        "line_end": 20,
                        "code": "def handler(): pass",
                        "similarity": 0.9,
                        "raw_cosine": 0.9,  # High cosine, above 0.2 threshold
                    }
                ],
            ):
                with patch.object(assembler, "get_repo_map", return_value={}):
                    with patch.object(assembler, "get_memory", return_value=""):
                        result = assembler.assemble_context("test query")

                        # Should NOT be suppressed
                        assert "No confident matches" not in result
                        assert "Relevant Functions" in result

    def test_confidence_guard_low_similarity_suppressed(self):
        """TASK 3: Low similarity result is suppressed by confidence guard."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(
                    mode="hybrid",
                    weights=[0.4, 0.3, 0.3],
                    min_confidence=0.2,
                    rerank_enabled=False,
                    rerank_min_score=0.0,
                ),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # Mock semantic_search to return low-similarity results
            with patch.object(
                assembler,
                "semantic_search",
                return_value=[
                    {
                        "filepath": "app.py",
                        "function": "helper",
                        "line_start": 30,
                        "line_end": 35,
                        "code": "def helper(): pass",
                        "similarity": 1.0,  # normalized top is always ~1.0
                        "raw_cosine": 0.1,  # Low cosine, below 0.2 threshold
                    }
                ],
            ):
                with patch.object(assembler, "get_repo_map", return_value={}):
                    with patch.object(assembler, "get_memory", return_value=""):
                        result = assembler.assemble_context("test query")

                        # Should be suppressed
                        assert "No confident matches found" in result

    def test_confidence_guard_disabled_with_zero_threshold(self):
        """TASK 3: Setting min_confidence=0.0 disables the guard."""
        with patch(
            "server.context_assembler.load_config",
            return_value=MagicMock(
                cache=MagicMock(enabled=False),
                retrieval=MagicMock(
                    mode="hybrid",
                    weights=[0.4, 0.3, 0.3],
                    min_confidence=0.0,
                    rerank_enabled=False,
                    rerank_min_score=0.0,
                ),
            ),
        ):
            assembler = ContextAssembler(project_path=self.test_project, cache=None)

            # Mock semantic_search to return very low-similarity results
            with patch.object(
                assembler,
                "semantic_search",
                return_value=[
                    {
                        "filepath": "app.py",
                        "function": "rare",
                        "line_start": 50,
                        "line_end": 55,
                        "code": "def rare(): pass",
                        "similarity": 1.0,
                        "raw_cosine": 0.01,  # Very low, but guard disabled
                    }
                ],
            ):
                with patch.object(assembler, "get_repo_map", return_value={}):
                    with patch.object(assembler, "get_memory", return_value=""):
                        result = assembler.assemble_context("test query")

                        # Should NOT be suppressed (guard is disabled)
                        assert "No confident matches" not in result
                        assert "Relevant Functions" in result
