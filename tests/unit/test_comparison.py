"""Unit tests for comparison benchmark helper functions.

Tests pure logic: reduction-% calc, recall-gating, gap computation.
No Ollama or live repo required.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestReductionCalculation:
    """Test token reduction percentage calculation."""

    def test_zero_baseline_returns_zero(self):
        """When baseline is 0, reduction is 0%."""
        from benchmarks.comparison import compute_reduction_pct

        result = compute_reduction_pct(100, 0)
        assert result == 0.0

    def test_half_tokens_gives_fifty_percent(self):
        """Reducing from 1000 to 500 tokens gives 50% reduction."""
        from benchmarks.comparison import compute_reduction_pct

        result = compute_reduction_pct(500, 1000)
        assert result == 50.0

    def test_quarter_tokens_gives_seventyfive_percent(self):
        """Reducing from 1000 to 250 tokens gives 75% reduction."""
        from benchmarks.comparison import compute_reduction_pct

        result = compute_reduction_pct(250, 1000)
        assert result == 75.0

    def test_no_reduction_gives_zero(self):
        """No reduction gives 0%."""
        from benchmarks.comparison import compute_reduction_pct

        result = compute_reduction_pct(1000, 1000)
        assert result == 0.0

    def test_increase_gives_negative(self):
        """Increase in tokens gives negative reduction."""
        from benchmarks.comparison import compute_reduction_pct

        result = compute_reduction_pct(1200, 1000)
        assert result == -20.0


class TestRecallGating:
    """Test the recall-based honesty gate."""

    def test_recall_at_threshold_is_claimable(self):
        """Recall exactly at 0.8 threshold is claimable."""
        from benchmarks.comparison import reduction_claimable

        assert reduction_claimable(0.8) is True

    def test_recall_above_threshold_is_claimable(self):
        """Recall above 0.8 is claimable."""
        from benchmarks.comparison import reduction_claimable

        assert reduction_claimable(0.9) is True
        assert reduction_claimable(1.0) is True

    def test_recall_below_threshold_not_claimable(self):
        """Recall below 0.8 is not claimable."""
        from benchmarks.comparison import reduction_claimable

        assert reduction_claimable(0.79) is False
        assert reduction_claimable(0.5) is False
        assert reduction_claimable(0.0) is False

    def test_custom_threshold(self):
        """Can override the default 0.8 threshold."""
        from benchmarks.comparison import reduction_claimable

        assert reduction_claimable(0.7, threshold=0.7) is True
        assert reduction_claimable(0.69, threshold=0.7) is False


class TestApproxTokens:
    """Test the ~4-char/token approximation."""

    def test_empty_string(self):
        """Empty string gives 1 token (minimum)."""
        from benchmarks.comparison import approx_tokens

        assert approx_tokens("") == 1

    def test_four_chars_one_token(self):
        """4 characters = 1 token."""
        from benchmarks.comparison import approx_tokens

        assert approx_tokens("abcd") == 1

    def test_eight_chars_two_tokens(self):
        """8 characters = 2 tokens."""
        from benchmarks.comparison import approx_tokens

        assert approx_tokens("abcdefgh") == 2

    def test_typical_code_line(self):
        """A typical code line."""
        # "def foo():" is 10 chars -> 2-3 tokens
        from benchmarks.comparison import approx_tokens

        code = "def foo():"
        assert approx_tokens(code) == 2  # 10 // 4 = 2

    def test_large_text(self):
        """Large text block."""
        from benchmarks.comparison import approx_tokens

        text = "x" * 1000
        assert approx_tokens(text) == 250  # 1000 // 4


class TestGoldFunctionHit:
    """Test gold function hit detection."""

    def test_rejection_message_no_hit(self):
        """Rejection message means no hit."""
        from benchmarks.comparison import gold_function_hit

        context = "No confident matches found for your query."
        gold_ids = {"app/main.py:create_nail:10"}
        assert gold_function_hit(context, gold_ids) is False

    def test_minimal_context_no_hit(self):
        """Minimal context (<50 chars) means no hit."""
        from benchmarks.comparison import gold_function_hit

        context = "x" * 40
        gold_ids = {"app/main.py:create_nail:10"}
        assert gold_function_hit(context, gold_ids) is False

    def test_function_name_in_context_is_hit(self):
        """Function name in context is a hit."""
        from benchmarks.comparison import gold_function_hit

        context = "The function create_nail is called here. " + "x" * 50
        gold_ids = {"app/main.py:create_nail:10"}
        assert gold_function_hit(context, gold_ids) is True

    def test_partial_match_is_hit(self):
        """Partial function name match is a hit."""
        from benchmarks.comparison import gold_function_hit

        context = "In create_nail we do stuff. " + "x" * 50
        gold_ids = {"app/main.py:create_nail:10"}
        assert gold_function_hit(context, gold_ids) is True

    def test_no_matching_function_is_no_hit(self):
        """Function name not in context means no hit."""
        from benchmarks.comparison import gold_function_hit

        context = "This is a function about something else. " + "x" * 50
        gold_ids = {"app/main.py:create_nail:10"}
        assert gold_function_hit(context, gold_ids) is False

    def test_multiple_gold_ids_any_match_is_hit(self):
        """Any of multiple gold IDs matching is a hit."""
        from benchmarks.comparison import gold_function_hit

        context = "Found place_order function here. " + "x" * 50
        gold_ids = {
            "app/main.py:create_nail:10",
            "app/main.py:place_order:80",
        }
        assert gold_function_hit(context, gold_ids) is True
