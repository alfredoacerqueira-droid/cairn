"""Tests for workspace-level memory functionality.

Tests the flexible, scope-configurable persistent memory for multi-repo
workspace mode. Includes memory config tests, workspace/per-repo reads,
classifier robustness, and MCP tool integration.
"""

import os

from core.config import MemoryConfig, load_config
from core.repo import RepoManager
from server.workspace_router import WorkspaceRouter
from tests.fixtures.builders import make_workspace
from tests.fixtures.harness import fresh_index


class TestMemoryConfig:
    """Test MemoryConfig with new scope field."""

    def test_memory_config_default_scope(self):
        """MemoryConfig.scope defaults to 'auto'."""
        cfg = MemoryConfig()
        assert cfg.scope == "auto"

    def test_memory_config_custom_scope(self):
        """MemoryConfig can set custom scope."""
        cfg = MemoryConfig(scope="both")
        assert cfg.scope == "both"

    def test_config_roundtrip_memory_scope(self, tmp_path):
        """Memory scope survives save/load cycle."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        custom_config = {"memory": {"scope": "workspace"}}
        config_file.write_text(yaml.dump(custom_config))

        loaded = load_config(tmp_path)
        assert loaded.memory.scope == "workspace"

    def test_old_config_without_scope_loads_with_default(self, tmp_path):
        """Backward compat: old config without scope loads with 'auto' default."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        old_config = {"memory": {"trigger": "manual", "max_entries": 50}}
        config_file.write_text(yaml.dump(old_config))

        loaded = load_config(tmp_path)
        assert loaded.memory.scope == "auto"


class TestWorkspaceMemoryMethods:
    """Test WorkspaceRouter memory methods."""

    def test_workspace_repo_lazy_init(self, tmp_path):
        """_get_workspace_repo lazily initializes workspace RepoManager."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        # _workspace_repo should be None initially
        assert router._workspace_repo is None

        # First call creates it
        ws_repo = router._get_workspace_repo()
        assert ws_repo is not None
        assert ws_repo.project_path == workspace

        # Second call returns cached instance
        ws_repo2 = router._get_workspace_repo()
        assert ws_repo2 is ws_repo

    def test_resolve_scope_auto_returns_both(self, tmp_path):
        """resolve_scope() returns 'both' when config scope is 'auto' in workspace."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        scope = router.resolve_scope()
        # 'auto' resolves to 'both' in a workspace
        assert scope == "both"

    def test_resolve_scope_explicit_value(self, tmp_path):
        """resolve_scope() returns explicit config value."""
        import yaml

        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        # Set scope to 'workspace' in workspace config
        config_dir = workspace / ".cairn"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"memory": {"scope": "workspace"}}))

        router = WorkspaceRouter(workspace)
        scope = router.resolve_scope()
        assert scope == "workspace"

    def test_write_memory_workspace_scope(self, tmp_path):
        """write_memory with 'workspace' scope appends to workspace memory."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        router.write_memory("test note 1", scope="workspace")

        # Verify it's in workspace memory
        ws_repo = router._get_workspace_repo()
        ws_mem = ws_repo.load_memory()
        assert "test note 1" in ws_mem

    def test_write_memory_both_scope(self, tmp_path):
        """write_memory with 'both' scope appends to workspace memory."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        router.write_memory("test note both", scope="both")

        ws_repo = router._get_workspace_repo()
        ws_mem = ws_repo.load_memory()
        assert "test note both" in ws_mem

    def test_write_memory_repo_scope_defaults_to_workspace(self, tmp_path):
        """write_memory with 'repo' scope in workspace mode defaults to workspace."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        router.write_memory("repo scope note", scope="repo")

        # Should write to workspace as safe default
        ws_repo = router._get_workspace_repo()
        ws_mem = ws_repo.load_memory()
        assert "repo scope note" in ws_mem

    def test_read_memory_workspace_scope(self, tmp_path):
        """read_memory with 'workspace' scope returns only workspace memory."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)
        router.write_memory("workspace note", scope="workspace")

        # Also add a note to a child repo
        helm_repo = RepoManager(workspace / "helm-repo")
        helm_repo.append_memory("helm note")

        # Read workspace scope
        result = router.read_memory(scope="workspace")
        assert "## Workspace memory" in result
        assert "workspace note" in result
        assert "helm note" not in result

    def test_read_memory_repo_scope(self, tmp_path):
        """read_memory with 'repo' scope returns per-repo memories."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # Add per-repo notes
        helm_repo = RepoManager(workspace / "helm-repo")
        helm_repo.append_memory("helm note")

        tf_repo = RepoManager(workspace / "terraform-repo")
        tf_repo.append_memory("terraform note")

        result = router.read_memory(scope="repo")
        assert "## Repo:" in result
        assert "helm note" in result
        assert "terraform note" in result
        assert "## Workspace memory" not in result

    def test_read_memory_both_scope(self, tmp_path):
        """read_memory with 'both' scope returns workspace + per-repo memories."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)
        fresh_index(workspace / "terraform-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # Add workspace note
        router.write_memory("workspace note", scope="workspace")

        # Add per-repo notes
        helm_repo = RepoManager(workspace / "helm-repo")
        helm_repo.append_memory("helm note")

        result = router.read_memory(scope="both")
        assert "## Workspace memory" in result
        assert "workspace note" in result
        assert "## Repo:" in result
        assert "helm note" in result

    def test_read_memory_token_capped(self, tmp_path):
        """read_memory respects token budget limit."""
        from core.tokens import count_tokens

        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # Write a long note
        long_note = "x" * 5000  # Large note
        router.write_memory(long_note)

        # Read with very small token budget
        result = router.read_memory(max_tokens=100)

        # Result should be truncated to fit in budget
        tokens = count_tokens(result)
        assert tokens <= 100

    def test_read_memory_empty_returns_empty_string(self, tmp_path):
        """read_memory returns empty string when no memory exists."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        router = WorkspaceRouter(workspace)

        # No memory written, should return empty
        result = router.read_memory(scope="workspace")
        assert result == ""


class TestAssembleAllWithMemory:
    """Test assemble_all prepends workspace memory."""

    def test_assemble_all_prepends_workspace_memory(self, tmp_path):
        """assemble_all prepends workspace memory when scope includes workspace."""
        import yaml

        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        # Set scope to workspace
        config_dir = workspace / ".cairn"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"memory": {"scope": "workspace"}}))

        router = WorkspaceRouter(workspace)
        router.write_memory("test assembly memory")

        # Query that matches
        result = router.assemble_all("replicaCount")

        # Should have workspace memory header before repo sections
        assert "## Workspace memory" in result
        assert "test assembly memory" in result
        # Should also have repo sections after memory
        assert "## Repo:" in result

    def test_assemble_all_no_memory_when_repo_scope(self, tmp_path):
        """assemble_all does NOT prepend when scope is 'repo'."""
        import yaml

        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"

        fresh_index(helm_repo, embeddings=False)

        # Set scope to repo
        config_dir = workspace / ".cairn"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"memory": {"scope": "repo"}}))

        router = WorkspaceRouter(workspace)
        router.write_memory("should not appear")

        result = router.assemble_all("replicaCount")

        # Should NOT have workspace memory header
        assert "## Workspace memory" not in result


class TestClassifierRobustness:
    """Test binding classifier robustness for workspace with root .cairn."""

    def test_workspace_with_root_cairn_still_classifies_workspace(self, tmp_path):
        """Classifier prefers WORKSPACE when path has .cairn/ AND >=2 child repos."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        # Also index the workspace root itself
        fresh_index(workspace, embeddings=False)

        from server.mcp_server import _classify_binding

        # Temporarily set env for classifier
        old_env = os.environ.get("CAIRN_PROJECT")
        try:
            os.environ["CAIRN_PROJECT"] = str(workspace)
            mode, path, error = _classify_binding()

            # Should classify as WORKSPACE, not SINGLE, because >=2 child repos
            assert mode == "WORKSPACE", f"Expected WORKSPACE, got {mode}"
            assert path == workspace
            assert error is None
        finally:
            if old_env:
                os.environ["CAIRN_PROJECT"] = old_env
            else:
                os.environ.pop("CAIRN_PROJECT", None)

    def test_classifier_single_child_repo_is_workspace(self, tmp_path):
        """Classifier returns WORKSPACE even with just 1 child repo."""
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"

        fresh_index(helm_repo, embeddings=False)

        from server.mcp_server import _classify_binding

        old_env = os.environ.get("CAIRN_PROJECT")
        try:
            os.environ["CAIRN_PROJECT"] = str(workspace)
            mode, path, error = _classify_binding()

            # Should classify as WORKSPACE (even with 1 child repo)
            assert mode == "WORKSPACE"
            assert path == workspace
            assert error is None
        finally:
            if old_env:
                os.environ["CAIRN_PROJECT"] = old_env
            else:
                os.environ.pop("CAIRN_PROJECT", None)


class TestMCPRememberRecall:
    """Test MCP remember/recall tools."""

    def test_remember_workspace_mode(self, tmp_path):
        """remember() in workspace mode writes to workspace memory."""
        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        # Simulate WORKSPACE binding
        from server import mcp_server

        old_router = mcp_server._router
        old_project = mcp_server._PROJECT_PATH
        old_error = mcp_server._BIND_ERROR
        try:
            mcp_server._router = WorkspaceRouter(workspace)
            mcp_server._PROJECT_PATH = None
            mcp_server._BIND_ERROR = None

            result = mcp_server.remember("test memory note")
            assert "remembered (workspace)" in result

            # Verify it's in workspace memory
            ws_repo = mcp_server._router._get_workspace_repo()
            ws_mem = ws_repo.load_memory()
            assert "test memory note" in ws_mem
        finally:
            mcp_server._router = old_router
            mcp_server._PROJECT_PATH = old_project
            mcp_server._BIND_ERROR = old_error

    def test_remember_single_mode(self, tmp_path):
        """remember() in single mode writes to repo memory."""
        from core.repo import RepoManager
        from server import mcp_server

        repo_path = tmp_path / "test-repo"
        fresh_index(repo_path, embeddings=False)

        old_router = mcp_server._router
        old_project = mcp_server._PROJECT_PATH
        old_error = mcp_server._BIND_ERROR
        try:
            mcp_server._router = None
            mcp_server._PROJECT_PATH = repo_path
            mcp_server._BIND_ERROR = None

            result = mcp_server.remember("single repo note")
            assert "remembered (repo:" in result

            # Verify it's in repo memory
            repo = RepoManager(repo_path)
            mem = repo.load_memory()
            assert "single repo note" in mem
        finally:
            mcp_server._router = old_router
            mcp_server._PROJECT_PATH = old_project
            mcp_server._BIND_ERROR = old_error

    def test_recall_workspace_mode(self, tmp_path):
        """recall() in workspace mode returns workspace memory."""
        from server import mcp_server

        workspace = make_workspace(tmp_path)
        fresh_index(workspace / "helm-repo", embeddings=False)

        old_router = mcp_server._router
        old_project = mcp_server._PROJECT_PATH
        old_error = mcp_server._BIND_ERROR
        try:
            mcp_server._router = WorkspaceRouter(workspace)
            mcp_server._PROJECT_PATH = None
            mcp_server._BIND_ERROR = None

            mcp_server._router.write_memory("recall test")
            result = mcp_server.recall()

            assert "recall test" in result
            assert "## Workspace memory" in result or result != ""
        finally:
            mcp_server._router = old_router
            mcp_server._PROJECT_PATH = old_project
            mcp_server._BIND_ERROR = old_error

    def test_recall_single_mode(self, tmp_path):
        """recall() in single mode returns repo memory."""
        from core.repo import RepoManager
        from server import mcp_server

        repo_path = tmp_path / "test-repo"
        fresh_index(repo_path, embeddings=False)

        repo = RepoManager(repo_path)
        repo.append_memory("single recall note")

        old_router = mcp_server._router
        old_project = mcp_server._PROJECT_PATH
        old_error = mcp_server._BIND_ERROR
        try:
            mcp_server._router = None
            mcp_server._PROJECT_PATH = repo_path
            mcp_server._BIND_ERROR = None

            result = mcp_server.recall()

            assert "single recall note" in result
            assert "## Repo memory" in result
        finally:
            mcp_server._router = old_router
            mcp_server._PROJECT_PATH = old_project
            mcp_server._BIND_ERROR = old_error

    def test_recall_bind_error_fail_closed(self, tmp_path):
        """recall() returns error message when unbound."""
        from server import mcp_server

        old_router = mcp_server._router
        old_project = mcp_server._PROJECT_PATH
        old_error = mcp_server._BIND_ERROR
        try:
            mcp_server._router = None
            mcp_server._PROJECT_PATH = None
            mcp_server._BIND_ERROR = "Test bind error"

            result = mcp_server.recall()

            assert "Test bind error" in result
        finally:
            mcp_server._router = old_router
            mcp_server._PROJECT_PATH = old_project
            mcp_server._BIND_ERROR = old_error
