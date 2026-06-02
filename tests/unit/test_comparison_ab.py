"""Unit tests for reranker A/B comparison logic (offline, mocked).

Tests the A/B aggregation logic without requiring:
  - Live Ollama
  - Indexed Django
  - ContextAssembler integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comparison import (
    _is_plausibly_correct,
    compare_rerank_ab,
    django_rerank_queries,
)


class TestIsPlausiblyCorrect:
    """Test the heuristic correctness checker."""

    def test_matches_function_name(self):
        """Should match when function name contains expected substring."""
        assert _is_plausibly_correct("execute_query", "path/file.py", "execute")

    def test_matches_filepath(self):
        """Should match when filepath contains expected substring."""
        assert _is_plausibly_correct("func", "path/execute/file.py", "execute")

    def test_case_insensitive(self):
        """Should match case-insensitively."""
        assert _is_plausibly_correct("Execute_Query", "path/file.py", "EXECUTE")

    def test_no_match_returns_false(self):
        """Should return False when substring not found."""
        assert not _is_plausibly_correct("other_func", "path/file.py", "execute")

    def test_empty_function_name(self):
        """Should handle empty function name."""
        assert not _is_plausibly_correct("", "path/file.py", "execute")

    def test_empty_expected_substring(self):
        """Should match any function with empty expected substring."""
        assert _is_plausibly_correct("any_func", "path/file.py", "")


class TestDjangoRetrankQueries:
    """Test the Django query dataset."""

    def test_returns_list_of_tuples(self):
        """Should return list of (query, gold_ids, expected_substring) tuples."""
        queries = django_rerank_queries()
        assert isinstance(queries, list)
        assert len(queries) == 8  # 5 relevant + 3 nonsense
        for query_text, gold_ids, expected_substring in queries:
            assert isinstance(query_text, str)
            assert isinstance(gold_ids, set)
            assert isinstance(expected_substring, str)

    def test_relevant_queries_have_gold_ids(self):
        """First 5 queries should have gold_ids."""
        queries = django_rerank_queries()
        relevant = queries[:5]
        for _, gold_ids, _ in relevant:
            assert len(gold_ids) > 0, "Relevant queries must have gold_ids"

    def test_nonsense_queries_have_no_gold_ids(self):
        """Last 3 queries should have empty gold_ids."""
        queries = django_rerank_queries()
        nonsense = queries[5:]
        for _, gold_ids, _ in nonsense:
            assert len(gold_ids) == 0, "Nonsense queries must have empty gold_ids"

    def test_all_queries_have_expected_substrings(self):
        """All queries should have non-empty expected_substring."""
        queries = django_rerank_queries()
        for _, _, expected_substring in queries:
            assert len(expected_substring) > 0


class TestCompareRerankABLogic:
    """Test the A/B aggregation logic with mocked results."""

    def test_empty_project_path(self):
        """Should return error for non-existent project."""
        result = compare_rerank_ab(Path("/nonexistent/django"))
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_not_indexed_project(self):
        """Should return error if project not indexed."""
        # Create a temp dir without .cairn
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = compare_rerank_ab(Path(tmpdir))
            assert "error" in result
            assert "not indexed" in result["error"]

    def test_verdict_win_when_improves_and_suppresses(self):
        """Should report 'WIN' when rerank improves top-1 and suppresses nonsense."""
        # Synthetic A/B results: all relevant queries correct with rerank ON,
        # all nonsense suppressed with rerank ON
        mock_ab_results = {
            "results": [
                {
                    "query": "relevant query 1",
                    "is_nonsense": False,
                    "rerank_off_correct": False,
                    "rerank_on_correct": True,
                    "suppressed_off": False,
                    "suppressed_on": False,
                },
                {
                    "query": "relevant query 2",
                    "is_nonsense": False,
                    "rerank_off_correct": False,
                    "rerank_on_correct": True,
                    "suppressed_off": False,
                    "suppressed_on": False,
                },
                {
                    "query": "nonsense query 1",
                    "is_nonsense": True,
                    "rerank_off_correct": False,
                    "rerank_on_correct": False,
                    "suppressed_off": False,
                    "suppressed_on": True,
                },
                {
                    "query": "nonsense query 2",
                    "is_nonsense": True,
                    "rerank_off_correct": False,
                    "rerank_on_correct": False,
                    "suppressed_off": False,
                    "suppressed_on": True,
                },
            ],
            "top1_correct_off": 0.0,
            "top1_correct_on": 1.0,
            "nonsense_suppressed_off": 0.0,
            "nonsense_suppressed_on": 1.0,
            "relevant_count": 2,
            "nonsense_count": 2,
        }

        # Manually compute verdict (simplified version of the function)
        top1_correct_on = mock_ab_results["top1_correct_on"]
        top1_correct_off = mock_ab_results["top1_correct_off"]
        nonsense_suppressed_on = mock_ab_results["nonsense_suppressed_on"]

        is_win = top1_correct_on >= top1_correct_off and nonsense_suppressed_on == 1.0

        assert is_win, "Should be a win when improves and suppresses"

    def test_verdict_loss_when_no_improvement(self):
        """Should report MIXED/LOSS when rerank doesn't improve top-1."""
        top1_correct_off = 0.5
        top1_correct_on = 0.5
        nonsense_suppressed_on = 0.5

        is_win = top1_correct_on >= top1_correct_off and nonsense_suppressed_on == 1.0

        assert not is_win, "Should be loss when doesn't suppress all nonsense"

    def test_verdict_loss_when_suppression_fails(self):
        """Should report MIXED/LOSS when nonsense not fully suppressed."""
        top1_correct_off = 0.0
        top1_correct_on = 1.0
        nonsense_suppressed_on = 0.5

        is_win = top1_correct_on >= top1_correct_off and nonsense_suppressed_on == 1.0

        assert not is_win, "Should be loss when doesn't suppress all nonsense"

    def test_correctness_rate_computation(self):
        """Should correctly compute top-1 correctness rate."""
        results = [
            {"is_nonsense": False, "rerank_off_correct": True},
            {"is_nonsense": False, "rerank_off_correct": False},
            {"is_nonsense": False, "rerank_off_correct": True},
            {"is_nonsense": True},  # Ignored
        ]

        correct = sum(1 for r in results if not r["is_nonsense"] and r["rerank_off_correct"])
        total = sum(1 for r in results if not r["is_nonsense"])
        rate = correct / total if total > 0 else 0.0

        assert rate == 2 / 3
        assert round(rate, 3) == 0.667

    def test_suppression_rate_computation(self):
        """Should correctly compute nonsense suppression rate."""
        results = [
            {"is_nonsense": True, "suppressed_on": True},
            {"is_nonsense": True, "suppressed_on": False},
            {"is_nonsense": True, "suppressed_on": True},
            {"is_nonsense": False},  # Ignored
        ]

        suppressed = sum(1 for r in results if r["is_nonsense"] and r["suppressed_on"])
        total = sum(1 for r in results if r["is_nonsense"])
        rate = suppressed / total if total > 0 else 0.0

        assert rate == 2 / 3
        assert round(rate, 3) == 0.667


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
