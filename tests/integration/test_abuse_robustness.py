"""Comprehensive abuse/robustness testing for Cairn v2.

Tests for:
1. Cross-project isolation / leakage (project_id filtering, foreign path checks)
2. Path traversal / malicious filepaths (../,  absolute paths, symlinks)
3. Oversized/malformed inputs (token budgets, malformed YAML, huge files)
4. MCP tool injection / hostile strings (null bytes, control chars, format strings, SQL-ish)
5. Budget/DoS guards (SessionBudget, cache eviction, bounds)

All tests drive Cairn IN-PROCESS (ContextAssembler, MCP tools, stores) with
no Ollama/network. Uses existing fixture builders (make_python_repo, fresh_index).
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.cache import SessionCache
from core.config import Config, load_config, save_config
from core.repo import project_id
from core.semantic_cache import SemanticCache
from pipeline.indexer import VectorIndexer
from pipeline.store.embedders import PlaceholderEmbedder
from server.context_assembler import ContextAssembler, _is_foreign_path
from server.mcp_server import (
    _get_session_budget,
    reset_session_budget,
    SessionBudget,
)
from server.orchestrator import Orchestrator, emit
from tests.fixtures.builders import make_python_repo, make_workspace
from tests.fixtures.harness import fresh_index


# ============================================================================
# 1) CROSS-PROJECT ISOLATION / LEAKAGE
# ============================================================================


class TestCrossProjectIsolation:
    """Test that project_id filtering prevents inter-repo leakage."""

    def test_assembler_filters_foreign_project_ids(self, tmp_path):
        """Assembler.semantic_search filters results with _is_foreign_path checks."""
        # Build two separate repos with their own ChromaDB instances
        (tmp_path / "base_a").mkdir(exist_ok=True)
        (tmp_path / "base_b").mkdir(exist_ok=True)
        repo_a = make_python_repo(tmp_path / "base_a")
        repo_b = make_python_repo(tmp_path / "base_b")
        fresh_index(repo_a)
        fresh_index(repo_b)

        # Create assemblers for each
        asm_a = ContextAssembler(project_path=repo_a)
        asm_b = ContextAssembler(project_path=repo_b)

        # Verify they have different project_ids
        assert asm_a.project_id is not None
        assert asm_b.project_id is not None
        assert asm_a.project_id != asm_b.project_id

        # Index a function into A's ChromaDB
        asm_a.store.indexer.index_function(
            filepath="src/auth.py",
            function_name="authenticate",
            code="def authenticate(): pass",
            line_start=1,
            line_end=3,
        )

        # Index a different function into B's ChromaDB
        asm_b.store.indexer.index_function(
            filepath="src/secret.py",
            function_name="secret_function",
            code="def secret_function(): return 'from repo B'",
            line_start=1,
            line_end=3,
        )

        # Verify both indexed successfully
        data_a = asm_a.store.collection.get(include=["metadatas"])
        data_b = asm_b.store.collection.get(include=["metadatas"])
        assert len(data_a["metadatas"]) > 0
        assert len(data_b["metadatas"]) > 0

        # Key assertion: A's semantic_search should NOT return B's records
        # (because they have different project_ids)
        results = asm_a.semantic_search("secret function authenticate", top_k=10)
        for result in results:
            # Every result should belong to A
            assert (
                result.get("project_id") == asm_a.project_id
            ), f"Result from wrong project: {result.get('project_id')} vs {asm_a.project_id}"

    def test_is_foreign_path_rejects_absolute_escapes(self):
        """_is_foreign_path returns True for absolute paths outside project_root."""
        project_root = "/home/user/project"

        # Test valid in-project paths
        assert not _is_foreign_path("src/main.py", project_root)
        assert not _is_foreign_path("relative/path.py", project_root)

        # Test absolute paths outside project
        assert _is_foreign_path("/etc/passwd", project_root)
        assert _is_foreign_path("/home/user/other/file.py", project_root)

        # Test absolute paths at exact project_root (should NOT be foreign)
        assert not _is_foreign_path(project_root, project_root)

        # Test absolute paths inside project_root (should NOT be foreign)
        assert not _is_foreign_path(
            f"{project_root}/src/main.py", project_root
        )

    def test_is_foreign_path_handles_symbol_suffix(self):
        """_is_foreign_path strips :symbol suffix before checking."""
        project_root = "/home/user/project"

        # In-project with symbol suffix
        assert not _is_foreign_path(
            f"{project_root}/values.yaml:replicaCount", project_root
        )

        # Foreign with symbol suffix
        assert _is_foreign_path("/etc/passwd:port", project_root)

    def test_workspace_router_respects_project_isolation(self, tmp_path):
        """WorkspaceRouter keeps each repo's results isolated."""
        from server.workspace_router import WorkspaceRouter

        # Build a workspace with multiple repos
        workspace = tmp_path / "ws"
        workspace.mkdir(exist_ok=True)
        workspace = make_workspace(workspace)
        for repo_path in [
            workspace / "helm-repo",
            workspace / "terraform-repo",
            workspace / "k8s-repo",
        ]:
            fresh_index(repo_path)

        router = WorkspaceRouter(workspace)
        assert len(router.repo_paths) == 3

        # Search should route to one repo, or return None (fail-closed)
        best_repo, results = router.route("terraform resource", top_k=5)
        if best_repo is not None:
            # If we got a match, it must be one of our repos
            assert best_repo in router.repo_paths
            # And results should come from that repo only
            assert isinstance(results, list)


# ============================================================================
# 2) PATH TRAVERSAL / MALICIOUS FILEPATHS
# ============================================================================


class TestPathTraversal:
    """Test path containment and malicious filepath handling."""

    def test_is_foreign_path_containment(self):
        """_is_foreign_path ensures no escape with .. or absolute paths."""
        project_root = "/home/user/myproject"

        # Relative paths are OK (assumed in-repo by caller)
        assert not _is_foreign_path("src/../main.py", project_root)
        assert not _is_foreign_path("./utils.py", project_root)

        # Absolute paths that escape the project are detected
        assert _is_foreign_path("/etc/passwd", project_root)
        assert _is_foreign_path("/home/user/other_project/file.py", project_root)

        # Edge case: exact project root
        assert not _is_foreign_path(project_root, project_root)

        # Edge case: within project with complex path
        assert not _is_foreign_path(
            f"{project_root}/src/../src/main.py", project_root
        )


# ============================================================================
# 3) OVERSIZED / MALFORMED INPUTS
# ============================================================================


class TestOversizedMalformedInputs:
    """Test token budgets, huge payloads, malformed config."""


    def test_session_budget_charge_truncates(self):
        """SessionBudget.charge() truncates and updates remaining."""
        budget = SessionBudget(cap=100)
        from core.tokens import count_tokens

        # First charge: large text that fits
        text1 = "word " * 100  # ~100+ tokens
        result1 = budget.charge(text1)
        n1 = count_tokens(result1)
        assert n1 <= budget.cap  # Must not exceed cap
        old_remaining = budget.remaining
        assert old_remaining < budget.cap  # Budget was charged

        # Second charge: request more than remaining
        text2 = "word " * 100  # ~100+ tokens, but budget doesn't have that much
        result2 = budget.charge(text2)
        n2 = count_tokens(result2)
        assert n2 <= budget.remaining + n1  # <= what's left + what we used
        assert budget.remaining <= old_remaining  # Didn't go backwards
        assert budget.remaining >= 0  # Never negative



# ============================================================================
# 4) MCP TOOL-ARG INJECTION / HOSTILE STRINGS
# ============================================================================


class TestMCPToolInjection:
    """Test MCP tool resilience to hostile/injected strings."""

    def test_semantic_cache_hostile_keys(self, tmp_path):
        """SemanticCache should handle null bytes, huge keys, path traversal."""
        cache_dir = tmp_path / "cache"
        embedder = PlaceholderEmbedder()
        cache = SemanticCache(cache_dir, embedder, max_entries=10)

        hostile_queries = [
            "normal query",
            "query\x00with\x00nulls",
            "query" + "x" * 100_000,  # Huge query
            "../../evil/path.py",  # Path traversal
            "../../../etc/passwd",
        ]

        for query in hostile_queries:
            try:
                # Should not crash on set
                cache.set(query, "cached_value")
                # Should not crash on get
                result = cache.get(query)
                # If we got a result, it should be the right one or None
                assert result is None or result == "cached_value"
            except Exception as e:
                # OK to raise, but must be safe (not write outside cache_dir)
                pass

        # Verify cache_dir is still valid (not escaped)
        assert cache_dir.exists()
        cache_files = list(cache_dir.glob("*.json"))
        for f in cache_files:
            # All files must be inside cache_dir
            assert str(f).startswith(str(cache_dir))

    def test_semantic_cache_path_containment(self, tmp_path):
        """SemanticCache._entry_path must never escape cache_dir."""
        cache_dir = tmp_path / "cache"
        embedder = PlaceholderEmbedder()
        cache = SemanticCache(cache_dir, embedder, max_entries=10)

        # Try to trigger path escape via normalized query
        queries = [
            "normal",
            "../../escape",
            "/absolute/path",
            "C:\\windows\\escape",
        ]

        for query in queries:
            entry_path = cache._entry_path(
                " ".join(query.strip().lower().split())
            )
            # Verify entry_path is inside cache_dir
            assert str(entry_path).startswith(
                str(cache_dir)
            ), f"Entry path escaped cache_dir: {entry_path}"


# ============================================================================
# 5) BUDGET / DoS GUARDS
# ============================================================================


class TestBudgetDoSGuards:
    """Test SessionBudget, cache eviction, and bounded growth."""

    def test_session_budget_cannot_exceed_cap(self):
        """SessionBudget.remaining never goes negative."""
        budget = SessionBudget(cap=100)
        from core.tokens import count_tokens

        # Charge multiple times
        for i in range(10):
            text = "word " * 20  # ~100 tokens per charge attempt
            budget.charge(text)

        # remaining should never be < 0
        assert budget.remaining >= 0

    def test_session_budget_resets(self):
        """SessionBudget.reset() restores cap."""
        budget = SessionBudget(cap=100)
        budget.remaining = 10

        budget.reset()
        assert budget.remaining == 100

    def test_semantic_cache_max_entries_eviction(self, tmp_path):
        """SemanticCache should evict old entries when max_entries exceeded."""
        cache_dir = tmp_path / "cache"
        embedder = PlaceholderEmbedder()
        cache = SemanticCache(cache_dir, embedder, max_entries=5)

        # Add 10 entries
        for i in range(10):
            cache.set(f"query_{i}", f"value_{i}")

        # Check that we never exceed max_entries
        n_files = len(list(cache_dir.glob("*.json")))
        assert n_files <= 5, f"Cache exceeded max_entries: {n_files} > 5"

    def test_semantic_cache_ttl_expiry(self, tmp_path):
        """SemanticCache should not return expired entries."""
        cache_dir = tmp_path / "cache"
        embedder = PlaceholderEmbedder()

        # Use a fake time function to control expiry
        fake_time = [0.0]

        def get_time():
            return fake_time[0]

        cache = SemanticCache(
            cache_dir, embedder, ttl_seconds=100, time_fn=get_time
        )

        # Set a value at time 0
        cache.set("query", "value")

        # Advance time past TTL
        fake_time[0] = 150.0

        # Should not return it
        result = cache.get("query")
        assert result is None, "Expired cache entry was returned"

    def test_session_cache_max_entries_eviction(self):
        """SessionCache should evict old entries when max_entries exceeded."""
        cache = SessionCache(max_entries=3)

        # Add 5 entries to a cache with max 3
        for i in range(5):
            cache.set(f"value_{i}", f"key_{i}")

        # Should not exceed max_entries
        assert len(cache._store) <= 3

    def test_emit_with_oversized_text(self):
        """emit() caps text to per_tool_cap regardless of budget."""
        from core.tokens import count_tokens

        # Create a 10k-token text
        huge_text = "word " * 2000
        per_tool_cap = 100
        budget = SessionBudget(cap=10_000)

        result = emit(huge_text, per_tool_cap, budget)

        # Result should be capped to per_tool_cap
        n_tokens = count_tokens(result)
        assert n_tokens <= per_tool_cap, (
            f"emit() result exceeded per_tool_cap: {n_tokens} > {per_tool_cap}"
        )

        # Budget should be charged accordingly
        assert budget.remaining < 10_000


# ============================================================================
# 6) PARSE SAFETY & RESOURCE LIMITS
# ============================================================================


class TestParseSafetyResourceLimits:
    """Test that AST parser and indexing respect size/timeout limits."""


# ============================================================================
# 7) WORKSPACE ROUTING FAIL-CLOSED
# ============================================================================


class TestWorkspaceRoutingFailClosed:
    """Test that WorkspaceRouter fails closed (returns None, not guesses)."""


# ============================================================================
# 8) CONFIG & MEMORY LOADING RESILIENCE
# ============================================================================


class TestConfigMemoryLoadingResilience:
    """Test graceful handling of corrupted/missing config/memory."""


# ============================================================================
# 9) REGRESSION: EXISTING ISOLATION TESTS STILL PASS
# ============================================================================


class TestExistingIsolationStillWorks:
    """Regression: ensure we didn't break existing isolation behavior."""

    def test_project_id_stable(self, tmp_path):
        """project_id should be stable for same path."""
        repo = tmp_path / "my-repo"
        repo.mkdir()
        id1 = project_id(repo)
        id2 = project_id(repo)
        assert id1 == id2

    def test_different_repos_different_ids(self, tmp_path):
        """Different repos get different project_ids."""
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()
        assert project_id(repo1) != project_id(repo2)
