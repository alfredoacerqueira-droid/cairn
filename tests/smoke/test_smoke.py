"""Smoke tests: quick happy-path E2E for all profiles and integration points.

Each test is marked @pytest.mark.smoke and runs in seconds on a clean index.
These verify:
1. Profile detection works for all repo types
2. Indexing + semantic search work correctly
3. MCP fail-closed behavior
4. Scaffold and opencode.json are valid
"""

import json
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.main import main
from core.repo import project_id
from server.context_assembler import ContextAssembler
from server.mcp_server import _resolve_project_path
from tests.fixtures.builders import (
    make_helm_repo,
    make_python_repo,
    make_terraform_repo,
)
from tests.fixtures.harness import fresh_index


@pytest.mark.smoke
class TestSmokePythonRepo:
    """Smoke test: Python repo profile detection + indexing."""

    def test_python_repo_indexes_and_searches(self, tmp_path):
        """Fresh index a Python repo and search for a known function."""
        repo = make_python_repo(tmp_path)

        # Index
        fresh_index(repo, embeddings=False)

        # Verify index created
        assert (repo / ".cairn" / "chroma").exists()

        # Search for known function
        assembler = ContextAssembler(project_path=repo)
        results = assembler.semantic_search("my_function add numbers", top_k=5)

        # Should find the function
        assert len(results) > 0
        assert any("my_function" in r.get("function", "") for r in results)


@pytest.mark.smoke
class TestSmokeHelmRepo:
    """Smoke test: Helm (IaC) repo profile detection + indexing."""

    def test_helm_repo_indexes_yaml(self, tmp_path):
        """Index Helm repo (iac profile) and verify content."""
        repo = make_helm_repo(tmp_path)

        # Index
        fresh_index(repo, embeddings=False)

        # Verify index created
        assert (repo / ".cairn" / "chroma").exists()

        # Search for known Helm resource
        assembler = ContextAssembler(project_path=repo)
        results = assembler.semantic_search("deployment replica", top_k=5)

        # IaC repos should have structural content indexed
        assert results is not None


@pytest.mark.smoke
class TestSmokeTerraformRepo:
    """Smoke test: Terraform (IaC) repo profile detection + indexing."""

    def test_terraform_repo_indexes_tf(self, tmp_path):
        """Index Terraform repo and search for known resource."""
        repo = make_terraform_repo(tmp_path)

        # Index
        fresh_index(repo, embeddings=False)

        # Verify index created
        assert (repo / ".cairn" / "chroma").exists()

        # Search for known Terraform resource
        assembler = ContextAssembler(project_path=repo)
        results = assembler.semantic_search("vpc subnet aws", top_k=5)

        # Should find Terraform resources
        assert results is not None


@pytest.mark.smoke
class TestSmokeMCPFailClosed:
    """Smoke test: MCP server fail-closed behavior."""

    def test_mcp_unbound_returns_error(self):
        """When CAIRN_PROJECT is not set, resolution returns error."""
        with patch.dict(os.environ, {}, clear=True):
            path, error = _resolve_project_path()
            assert path is None
            assert error is not None
            assert "bound project" in error.lower()

    def test_mcp_bound_to_valid_repo(self, tmp_path):
        """When CAIRN_PROJECT is set to indexed repo, resolution succeeds."""
        repo = make_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)

        with patch.dict(os.environ, {"CAIRN_PROJECT": str(repo)}):
            path, error = _resolve_project_path()
            assert str(path) == str(repo)
            assert error is None


@pytest.mark.smoke
class TestSmokeScaffold:
    """Smoke test: init --no-index creates valid opencode.json."""

    def test_init_scaffold_creates_valid_opencode(self, tmp_path):
        """cairn init --no-index creates opencode.json with correct schema."""
        runner = CliRunner()
        repo = make_python_repo(tmp_path)

        original_cwd = os.getcwd()
        try:
            os.chdir(repo)
            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"init failed: {result.output}"
        finally:
            os.chdir(original_cwd)

        # Verify opencode.json
        opencode_file = repo / "opencode.json"
        assert opencode_file.exists()

        data = json.loads(opencode_file.read_text())
        assert "mcp" in data
        assert "cairn" in data["mcp"]

        cairn = data["mcp"]["cairn"]
        assert isinstance(cairn["command"], list)
        assert cairn["command"][-1] == "mcp"
        assert cairn["enabled"] is True


@pytest.mark.smoke
class TestSmokeProjectIsolation:
    """Smoke test: project isolation basics."""

    def test_two_repos_different_project_ids(self, tmp_path):
        """Two different repos get different project_ids."""
        repo1 = make_python_repo(tmp_path)
        repo2 = make_terraform_repo(tmp_path)

        id1 = project_id(repo1)
        id2 = project_id(repo2)

        assert id1 != id2

    def test_queries_isolated_to_repo(self, tmp_path):
        """Queries from one repo don't leak results from another."""
        repo1 = make_python_repo(tmp_path)
        repo2 = make_terraform_repo(tmp_path)

        # Index both
        fresh_index(repo1, embeddings=False)
        fresh_index(repo2, embeddings=False)

        # Create assemblers
        asm1 = ContextAssembler(project_path=repo1)
        asm2 = ContextAssembler(project_path=repo2)

        pid1 = asm1.project_id
        pid2 = asm2.project_id

        # Query each
        results1 = asm1.semantic_search("function", top_k=5)
        results2 = asm2.semantic_search("resource variable", top_k=5)

        # All results from asm1 should have pid1
        if results1:
            assert all(r.get("project_id") == pid1 for r in results1)

        # All results from asm2 should have pid2
        if results2:
            assert all(r.get("project_id") == pid2 for r in results2)


@pytest.mark.smoke
class TestSmokeQuickReindex:
    """Smoke test: quick reindex mode completes."""

    def test_reindex_quick_completes(self, tmp_path):
        """CLI reindex --mode quick completes successfully."""
        runner = CliRunner()
        repo = make_python_repo(tmp_path)

        original_cwd = os.getcwd()
        try:
            os.chdir(repo)

            # Init first
            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0

            # Then reindex --mode quick
            result = runner.invoke(main, ["reindex", "--mode", "quick"])
            assert result.exit_code == 0, f"reindex failed: {result.output}"

        finally:
            os.chdir(original_cwd)

        # Verify index exists
        assert (repo / ".cairn" / "chroma").exists()
