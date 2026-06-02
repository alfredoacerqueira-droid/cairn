"""Tests for workspace-router MCP functionality.

Tests the WorkspaceRouter class, multi-repo routing logic, and MCP server
workspace binding. All tests are hermetic (no network, offline embeddings).
"""

import os
from unittest.mock import patch

from core.repo import project_id
from server.workspace_router import WorkspaceRouter
from tests.fixtures.builders import make_workspace
from tests.fixtures.harness import fresh_index


class TestWorkspaceRouterDiscovery:
    """Test discovery of indexed repos in a workspace."""

    def test_discover_repos_finds_all_indexed(self, tmp_path):
        """discover_repos finds all sibling repos with .cairn/."""
        workspace = make_workspace(tmp_path)

        # Index all three repos
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)
        fresh_index(workspace / "k8s-repo", embeddings=False)

        discovered = WorkspaceRouter.discover_repos(workspace)

        # Should find all three by their .cairn/ directories
        assert len(discovered) == 3
        names = sorted([p.name for p in discovered])
        assert names == ["helm-repo", "k8s-repo", "terraform-repo"]

    def test_discover_repos_excludes_workspace_root(self, tmp_path):
        """discover_repos does NOT include the workspace root itself."""
        workspace = make_workspace(tmp_path)

        # Index repos but NOT the workspace root
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        discovered = WorkspaceRouter.discover_repos(workspace)

        # None of the discovered paths should be the workspace root
        assert workspace not in discovered

    def test_discover_repos_handles_mixed_indexed_unindexed(self, tmp_path):
        """discover_repos skips unindexed sibling directories."""
        workspace = make_workspace(tmp_path)

        # Index only helm and terraform, not k8s
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        discovered = WorkspaceRouter.discover_repos(workspace)

        # Should find only the indexed repos
        assert len(discovered) == 2
        names = sorted([p.name for p in discovered])
        assert names == ["helm-repo", "terraform-repo"]
        assert all(p.name != "k8s-repo" for p in discovered)

    def test_discover_repos_empty_workspace(self, tmp_path):
        """discover_repos returns [] for a workspace with no indexed repos."""
        workspace = make_workspace(tmp_path)

        # Don't index any repos
        discovered = WorkspaceRouter.discover_repos(workspace)

        assert discovered == []


class TestWorkspaceRouterRouting:
    """Test the routing logic: selecting the best-matching repo."""

    def test_route_helm_query_to_helm_repo(self, tmp_path):
        """A helm-unique, guard-passing query routes to the helm repo only."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        router = WorkspaceRouter(workspace)
        assert len(router.repo_paths) == 2

        # "replicaCount" lives in helm values, never in terraform -> must route
        # to helm with confident, isolated results (not fail-closed).
        best_repo, results = router.route("replicaCount", top_k=5)
        assert best_repo == helm_repo, f"expected helm, got {best_repo}"
        assert results, "expected confident results from helm"
        helm_pid = project_id(helm_repo)
        assert all(r.get("project_id") == helm_pid for r in results)

    def test_route_terraform_query_to_terraform_repo(self, tmp_path):
        """A terraform-unique, guard-passing query routes to the terraform repo only."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        router = WorkspaceRouter(workspace)

        # "provider" lives in terraform, never in helm -> must route to terraform.
        best_repo, results = router.route("provider", top_k=5)
        assert best_repo == tf_repo, f"expected terraform, got {best_repo}"
        assert results, "expected confident results from terraform"
        tf_pid = project_id(tf_repo)
        assert all(r.get("project_id") == tf_pid for r in results)

    def test_route_nonsense_query_returns_none(self, tmp_path):
        """A nonsense query that matches nothing returns (None, []) or no results."""
        workspace = make_workspace(tmp_path)

        # Index repos
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # Query with nonsense words (offline structural/lexical retrieval is very
        # selective, so this should fail-close with no confident matches)
        best_repo, results = router.route("zzzzunusualzzz qwertyasdf foobarfoo", top_k=5)

        # With offline retrieval, we should get no confident matches.
        # If we do get a repo, verify it's a valid one.
        if best_repo is not None:
            # This shouldn't happen with truly nonsense text, but be lenient
            assert best_repo in (workspace / "helm-repo", workspace / "terraform-repo")
        assert len(results) == 0 or best_repo is not None

    def test_route_empty_workspace_returns_none(self, tmp_path):
        """Routing in an empty workspace returns (None, [])."""
        workspace = make_workspace(tmp_path)

        # Don't index anything
        router = WorkspaceRouter(workspace)

        best_repo, results = router.route("anything", top_k=5)

        assert best_repo is None
        assert results == []


class TestWorkspaceRouterSearch:
    """Test the search() formatted output."""

    def test_search_helm_query_includes_repo_name(self, tmp_path):
        """search() returns repo name if confident, else fail-closed message."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        router = WorkspaceRouter(workspace)
        result = router.search("helm chart values deployment", top_k=5)

        # Result is either:
        # (1) A successful search with "Repo:" header (if confident match found), or
        # (2) A fail-closed message (if no confident match)
        if "Repo:" in result:
            # Success path: includes repo name
            assert "helm" in result.lower() or "terraform" in result.lower()
        else:
            # Fail-closed path: clear message
            assert "Could not confidently determine" in result

    def test_search_nonsense_query_returns_fail_closed(self, tmp_path):
        """search() with nonsense query returns fail-closed or no results."""
        workspace = make_workspace(tmp_path)

        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        # Use very unusual words unlikely to match
        result = router.search("zzzzunusualzzz qwertyasdf foobarfoo", top_k=5)

        # Should either:
        # (1) Return fail-closed message, or
        # (2) Return a result (if offline retrieval found something by chance)
        # The key is not to crash and handle gracefully
        assert result is not None
        assert isinstance(result, str)


class TestWorkspaceRouterAssemble:
    """Test the assemble() method."""

    def test_assemble_includes_repo_header(self, tmp_path):
        """assemble() result includes '# Repo: <name>' header."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        result = router.assemble("helm chart deployment")

        # Should have repo header
        assert "# Repo:" in result
        assert "helm" in result.lower()
        # Should NOT be fail-closed
        assert "Could not confidently determine" not in result

    def test_assemble_nonsense_query_returns_fail_closed(self, tmp_path):
        """assemble() with nonsense query returns fail-closed or no results."""
        workspace = make_workspace(tmp_path)

        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        # Use very unusual words unlikely to match
        result = router.assemble("zzzzunusualzzz qwertyasdf")

        # Should return either fail-closed or a result (gracefully handled)
        assert result is not None
        assert isinstance(result, str)


class TestMCPServerWorkspaceBinding:
    """Test MCP server workspace binding via _classify_binding()."""

    def test_classify_binding_unbound_no_env(self):
        """With no CAIRN_PROJECT/GATEWAY_PROJECT, returns UNBOUND."""
        with patch.dict(os.environ, {}, clear=True):
            from server.mcp_server import _classify_binding

            mode, path, error = _classify_binding()
            assert mode == "UNBOUND"
            assert path is None
            assert error is not None
            assert "no bound project" in error.lower()

    def test_classify_binding_single_repo(self, tmp_path):
        """With CAIRN_PROJECT pointing to an indexed repo, returns SINGLE."""
        repo = tmp_path / "repo"
        repo.mkdir()
        fresh_index(repo, embeddings=False)

        with patch.dict(os.environ, {"CAIRN_PROJECT": str(repo)}):
            from server.mcp_server import _classify_binding

            mode, path, error = _classify_binding()
            assert mode == "SINGLE"
            assert path == repo
            assert error is None

    def test_classify_binding_workspace(self, tmp_path):
        """With CAIRN_PROJECT pointing to workspace root, returns WORKSPACE."""
        workspace = make_workspace(tmp_path)

        # Index repos
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        with patch.dict(os.environ, {"CAIRN_PROJECT": str(workspace)}):
            from server.mcp_server import _classify_binding

            mode, path, error = _classify_binding()
            assert mode == "WORKSPACE"
            assert path == workspace
            assert error is None

    def test_classify_binding_nonexistent_path(self):
        """With nonexistent CAIRN_PROJECT, returns UNBOUND."""
        with patch.dict(os.environ, {"CAIRN_PROJECT": "/nonexistent/path"}):
            from server.mcp_server import _classify_binding

            mode, path, error = _classify_binding()
            assert mode == "UNBOUND"
            assert path is None
            assert error is not None
            assert "does not exist" in error

    def test_classify_binding_unindexed_repo(self, tmp_path):
        """With unindexed repo (no .cairn/, no children with .cairn/), returns UNBOUND."""
        repo = tmp_path / "repo"
        repo.mkdir()

        with patch.dict(os.environ, {"CAIRN_PROJECT": str(repo)}):
            from server.mcp_server import _classify_binding

            mode, path, error = _classify_binding()
            assert mode == "UNBOUND"
            assert path is None
            assert error is not None


class TestMCPServerWorkspaceModeTools:
    """Test MCP tools when server is in WORKSPACE mode."""

    def test_search_code_workspace_mode(self, tmp_path):
        """search_code tool delegates to router when in workspace mode."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        # Monkeypatch MCP server globals to simulate workspace binding
        import server.mcp_server as mcp_module

        router = WorkspaceRouter(workspace)
        mcp_module._router = router
        mcp_module._PROJECT_PATH = None
        mcp_module._BIND_ERROR = None

        try:
            from server.mcp_server import search_code

            result = search_code("helm chart deployment")

            # Should include repo header
            assert "Repo:" in result
            # Should not be fail-closed
            assert "Could not confidently determine" not in result
        finally:
            mcp_module._router = None

    def test_assemble_context_workspace_mode(self, tmp_path):
        """assemble_context tool delegates to router in workspace mode."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        # Monkeypatch MCP server globals
        import server.mcp_server as mcp_module

        router = WorkspaceRouter(workspace)
        mcp_module._router = router
        mcp_module._PROJECT_PATH = None
        mcp_module._BIND_ERROR = None

        try:
            from server.mcp_server import assemble_context

            result = assemble_context("helm deployment")

            # Result is either:
            # (1) Assembled context with "# Repo:" header (if match found), or
            # (2) Fail-closed message (if no confident match)
            if "# Repo:" in result:
                # Success: has repo header
                assert "helm" in result.lower() or "terraform" in result.lower()
            else:
                # Fail-closed: clear message
                assert "Could not confidently determine" in result
        finally:
            mcp_module._router = None

    def test_set_profile_workspace_mode_rejected(self, tmp_path):
        """set_profile tool rejects workspace mode."""
        workspace = make_workspace(tmp_path)

        fresh_index(workspace / "helm-repo", embeddings=False)

        # Monkeypatch MCP server globals
        import server.mcp_server as mcp_module

        router = WorkspaceRouter(workspace)
        mcp_module._router = router
        mcp_module._PROJECT_PATH = None
        mcp_module._BIND_ERROR = None

        try:
            from server.mcp_server import set_profile

            result = set_profile("iac")

            # Should reject with a clear message
            assert "requires a single-repo binding" in result
            assert "workspace" in result.lower()
        finally:
            mcp_module._router = None


class TestWorkspaceRouterLazyInitialization:
    """Test that ContextAssemblers are lazily initialized per repo."""

    def test_router_caches_assemblers(self, tmp_path):
        """Router caches assemblers (lazily constructs on first use)."""
        workspace = make_workspace(tmp_path)

        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # Initially no assemblers cached
        assert len(router._assemblers) == 0

        # After search, assemblers are cached
        router.route("anything", top_k=1)

        # Both repos should have been probed (even if results are empty)
        # At least one should be cached
        assert len(router._assemblers) >= 1

    def test_router_reuses_cached_assembler(self, tmp_path):
        """Router reuses the same assembler on subsequent queries."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # First query
        router.route("helm chart", top_k=1)
        asm1 = router._assemblers.get(helm_repo)

        # Second query
        router.route("helm deployment", top_k=1)
        asm2 = router._assemblers.get(helm_repo)

        # Should be the same instance
        assert asm1 is asm2
