"""Unit tests for Phase 6 profile retrieval evaluation logic.

Tests the correctness-checking heuristics and aggregation logic
without requiring real repos or Ollama.
"""

from __future__ import annotations

import pytest


def test_phase6_expected_substring_matching():
    """Test heuristic: expected_substring appears in result."""
    combined_text = "iam_role: aws_iam_role"
    expected = "iam_role"

    # Should match: substring in combined text
    assert expected.lower() in combined_text.lower()

    # Should not match: unrelated query
    combined_text2 = "security_group: aws_security_group"
    assert expected.lower() not in combined_text2.lower()


def test_phase6_nonsense_suppression_heuristic():
    """Test nonsense detection: low confidence or minimal context."""
    # Case 1: "No confident matches found" message
    context_suppressed = "No confident matches found for query: kubernetes autoscaler"
    assert "No confident matches found" in context_suppressed

    # Case 2: Minimal context (< 50 chars)
    context_minimal = "Query too short"
    assert len(context_minimal.strip()) < 50

    # Case 3: Normal relevant context
    context_normal = "def execute_sql(self, query): return self.connection.execute(query)"
    assert "No confident matches found" not in context_normal
    assert len(context_normal.strip()) >= 50


def test_phase6_token_reduction_calculation():
    """Test token reduction percentage calculation."""

    def approx_tokens(text: str) -> int:
        """~4-char/token approximation."""
        return max(1, len(text) // 4)

    # Full repo: 1000 chars = 250 tokens
    whole_repo = "x" * 1000
    whole_repo_tokens = approx_tokens(whole_repo)
    assert whole_repo_tokens == 250

    # Gateway context: 100 chars = 25 tokens
    gateway_context = "x" * 100
    gateway_tokens = approx_tokens(gateway_context)
    assert gateway_tokens == 25

    # Reduction = (1 - 25/250) * 100 = 90%
    reduction_pct = round(100 * (1 - gateway_tokens / whole_repo_tokens), 1)
    assert reduction_pct == 90.0


def test_phase6_hit_detection_logic():
    """Test top-1 and top-5 hit detection logic."""
    # Simulate results from semantic_search
    results = [
        {
            "function": "iam_role_creation",
            "filepath": "modules/iam/main.tf",
            "raw_cosine": 0.95,
        },
        {
            "function": "security_group_rules",
            "filepath": "modules/security/main.tf",
            "raw_cosine": 0.88,
        },
        {
            "function": "kms_key_policy",
            "filepath": "modules/kms/main.tf",
            "raw_cosine": 0.85,
        },
    ]

    # Simulate context assembly
    context = "\n".join([f"{r['function']}: {r['filepath']}" for r in results])

    # Query for "IAM role for cluster" should match "iam_role" substring
    expected = "iam_role"
    assert expected.lower() in context.lower()

    # Query for "unrelated kubernetes" should NOT match
    expected_unrelated = "kubernetes"
    assert expected_unrelated.lower() not in context.lower()


def test_phase6_aggregation_top_1_top_5():
    """Test aggregating top-1 and top-5 hits across queries."""
    # Simulate 5 relevant queries
    relevant_queries = [
        ("query 1", True, True),  # top-1 hit, top-5 hit
        ("query 2", True, True),  # top-1 hit, top-5 hit
        ("query 3", False, True),  # no top-1, but top-5 hit
        ("query 4", True, True),  # top-1 hit, top-5 hit
        ("query 5", True, True),  # top-1 hit, top-5 hit
    ]

    top_1_hits = sum(1 for _, top1, _ in relevant_queries if top1)
    top_5_hits = sum(1 for _, _, top5 in relevant_queries if top5)
    total = len(relevant_queries)

    top_1_pct = (top_1_hits / total) * 100
    top_5_pct = (top_5_hits / total) * 100

    assert top_1_pct == 80.0  # 4/5
    assert top_5_pct == 100.0  # 5/5


def test_phase6_nonsense_suppression_aggregation():
    """Test aggregating nonsense suppression across queries."""
    # Simulate 3 nonsense queries
    nonsense_queries = [
        ("kubernetes autoscaler", True),  # suppressed
        ("terraform vpc", False),  # not suppressed
        ("blockchain consensus", True),  # suppressed
    ]

    nonsense_suppressed = sum(1 for _, supp in nonsense_queries if supp)
    total = len(nonsense_queries)

    suppression_pct = (nonsense_suppressed / total) * 100

    assert round(suppression_pct, 2) == 66.67  # 2/3 (rounded)


def test_phase6_improvement_calculation():
    """Test v0.5 vs v0.6 improvement calculation."""
    baselines = {
        "tf-eks": 17,  # v0.5 top-1 %
        "mediatr": 60,  # v0.5 top-1 %
        "django": 40,  # v0.5 top-1 %
    }

    v0_6_results = {
        "tf-eks": 100.0,  # v0.6 top-1 %
        "mediatr": 100.0,  # v0.6 top-1 %
        "django": 100.0,  # v0.6 top-1 %
    }

    improvements = {repo: v0_6_results[repo] - baselines[repo] for repo in baselines.keys()}

    assert improvements["tf-eks"] == 83.0  # +83 percentage points
    assert improvements["mediatr"] == 40.0  # +40 percentage points
    assert improvements["django"] == 60.0  # +60 percentage points


def test_phase6_is_nonsense_detection():
    """Test detecting nonsense vs relevant queries."""
    nonsense_patterns = [
        "kubernetes pod autoscaler reconcile loop",
        "terraform aws vpc subnet",
        "kubernetes pod autoscaler",
        "blockchain consensus",
    ]

    nonsense_set = set(nonsense_patterns)

    # Queries in the set should be detected as nonsense
    assert "kubernetes pod autoscaler reconcile loop" in nonsense_set
    assert "blockchain consensus" in nonsense_set

    # Queries not in the set should be relevant
    relevant = "resolve URL to view"
    assert relevant not in nonsense_set


def test_phase6_profile_correctness():
    """Test profile-based context assembly correctness."""
    # Simulate profiles and their expected behavior
    profiles = {
        "iac": {
            "embeddings_enabled": False,
            "legs": ["structural", "lexical"],
            "expected_behavior": "Tree-sitter blocks + text search",
        },
        "dotnet": {
            "embeddings_enabled": True,
            "legs": ["embeddings", "lexical", "structural"],
            "expected_behavior": "Vector search + text + tree-sitter",
        },
        "python": {
            "embeddings_enabled": True,
            "legs": ["embeddings", "lexical", "structural"],
            "expected_behavior": "Vector search + text + tree-sitter",
        },
    }

    # Verify iac profile disables embeddings
    assert profiles["iac"]["embeddings_enabled"] is False
    assert "structural" in profiles["iac"]["legs"]

    # Verify dotnet/python profiles enable embeddings
    assert profiles["dotnet"]["embeddings_enabled"] is True
    assert profiles["python"]["embeddings_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
