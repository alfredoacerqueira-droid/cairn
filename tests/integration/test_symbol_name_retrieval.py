"""Integration tests for symbol-name retrieval rescue.

Verifies that exact symbol-name queries (class names, function names) survive
the FlashRank cross-encoder confidence guard via the _name_matches_query bypass,
while genuinely off-topic queries remain fail-closed (return []).
"""

import subprocess
from pathlib import Path

from server.context_assembler import ContextAssembler
from tests.fixtures.harness import fresh_index


def _make_python_repo_with_symbols(base: Path) -> Path:
    """Create a Python repo with a class and a function for retrieval tests."""
    repo = base / "py-repo"
    repo.mkdir(exist_ok=True)

    (repo / "orders.py").write_text(
        '''"""Order management module."""


class OrderService:
    """Manage orders."""

    def settle_payment(self, amount: float) -> bool:
        return True

    def process(self, order_id: str) -> str:
        return f"processing {order_id}"


def reconcile_ledger(entries: list[dict]) -> dict:
    """Reconcile ledger entries."""
    result = {"balanced": True}
    for entry in entries:
        result[entry["id"]] = entry["amount"]
    return result


def helper_validate(value: str) -> bool:
    return bool(value)
'''
    )

    # Initialize git
    subprocess.run(
        ["git", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


class TestSymbolNameRetrieval:
    """Symbol-name queries survive the confidence guard; fail-closed preserved."""

    def test_class_name_retrieval(self, tmp_path):
        """class OrderService is retrievable via semantic_search(apply_guard=True)."""
        repo = _make_python_repo_with_symbols(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("OrderService", apply_guard=True)
        assert len(results) >= 1, (
            f"Expected >=1 results for 'OrderService', got {len(results)}"
        )
        found = any(
            "OrderService" in r.get("function", "") for r in results
        )
        assert found, f"OrderService not in results: {[r.get('function') for r in results]}"

    def test_function_name_retrieval(self, tmp_path):
        """def reconcile_ledger is retrievable via semantic_search(apply_guard=True)."""
        repo = _make_python_repo_with_symbols(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("reconcile_ledger", apply_guard=True)
        assert len(results) >= 1, (
            f"Expected >=1 results for 'reconcile_ledger', got {len(results)}"
        )
        found = any(
            "reconcile_ledger" in r.get("function", "") for r in results
        )
        assert found, f"reconcile_ledger not in results: {[r.get('function') for r in results]}"

    def test_nonsense_query_fail_closed(self, tmp_path):
        """Off-topic queries still return [] (fail-closed preserved)."""
        repo = _make_python_repo_with_symbols(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search(
            "zzqwerty_nonsense_xyz", apply_guard=True
        )
        assert results == [], (
            f"Expected [] for nonsense query, got {len(results)} results"
        )

    def test_assemble_context_includes_class_name(self, tmp_path):
        """assemble_context('OrderService') does NOT return 'No confident matches'."""
        repo = _make_python_repo_with_symbols(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        ctx = assembler.assemble_context("OrderService")
        assert "No confident matches found" not in ctx, (
            f"assemble_context returned guard-rejection string:\n{ctx[:200]}"
        )
        assert "OrderService" in ctx, (
            f"assemble_context missing OrderService:\n{ctx[:200]}"
        )

    def test_name_matches_query_helper(self, tmp_path):
        """_name_matches_query tokenisation logic."""
        repo = _make_python_repo_with_symbols(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        assert assembler._name_matches_query("OrderService", "OrderService")
        assert assembler._name_matches_query("OrderService", "OrderService.settle_payment")
        assert assembler._name_matches_query("order", "OrderService")
        assert assembler._name_matches_query("reconcile_ledger", "reconcile_ledger")
        assert assembler._name_matches_query("reconcile", "reconcile_ledger")
        assert assembler._name_matches_query("HttpClient", "HttpClient")
        assert assembler._name_matches_query("http", "HttpClient")
        assert not assembler._name_matches_query("zzqwerty", "OrderService")
        assert not assembler._name_matches_query("", "OrderService")
        assert not assembler._name_matches_query("OrderService", "")
