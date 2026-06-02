"""Integration tests for cairn init MCP schema scaffolding and workspace detection."""

import json
from pathlib import Path

from click.testing import CliRunner

from cli.main import main
from tests.fixtures.builders import make_python_repo, make_workspace


class TestInitScaffoldSchema:
    """Test the opencode.json schema matches OpenCode 1.15+ requirements."""

    def test_opencode_json_schema_command_is_array(self, tmp_path):
        """Verify opencode.json command field is an array (OpenCode 1.15+)."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Use builder to create a git repo
            repo = make_python_repo(tmp_path)

            # Change to the repo and run init
            import os

            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"init failed: {result.output}"

            # Load opencode.json
            opencode_file = repo / "opencode.json"
            assert opencode_file.exists()
            opencode_data = json.loads(opencode_file.read_text())

            # Verify command is an array
            mcp_cfg = opencode_data["mcp"]["cairn"]
            assert isinstance(
                mcp_cfg["command"], list
            ), f"command should be array, got {type(mcp_cfg['command'])}"
            assert len(mcp_cfg["command"]) >= 1

            # Verify last element is "mcp"
            assert mcp_cfg["command"][-1] == "mcp"

    def test_opencode_json_schema_enabled_field(self, tmp_path):
        """Verify opencode.json has enabled: true field."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_python_repo(tmp_path)
            import os

            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0

            opencode_file = repo / "opencode.json"
            opencode_data = json.loads(opencode_file.read_text())

            mcp_cfg = opencode_data["mcp"]["cairn"]
            assert "enabled" in mcp_cfg, "enabled field missing"
            assert mcp_cfg["enabled"] is True, f"enabled should be true, got {mcp_cfg['enabled']}"

    def test_opencode_json_no_top_level_args(self, tmp_path):
        """Verify opencode.json MCP entry has no top-level args field."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_python_repo(tmp_path)
            import os

            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0

            opencode_file = repo / "opencode.json"
            opencode_data = json.loads(opencode_file.read_text())

            mcp_cfg = opencode_data["mcp"]["cairn"]
            assert "args" not in mcp_cfg, "top-level args field should not exist"

    def test_opencode_json_absolute_cairn_project_path(self, tmp_path):
        """Verify CAIRN_PROJECT env var is an absolute path."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_python_repo(tmp_path)
            import os

            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0

            opencode_file = repo / "opencode.json"
            opencode_data = json.loads(opencode_file.read_text())

            mcp_cfg = opencode_data["mcp"]["cairn"]
            cairn_project = mcp_cfg["env"]["CAIRN_PROJECT"]

            # Should be absolute
            assert Path(
                cairn_project
            ).is_absolute(), f"CAIRN_PROJECT should be absolute, got {cairn_project}"

            # Should resolve to the repo root
            assert Path(cairn_project).resolve() == repo.resolve()


class TestWorkspaceDetectionWarning:
    """Test that init warns when project is in a workspace."""

    def test_workspace_detection_outputs_warning(self, tmp_path):
        """Test that init detects workspace and prints warning."""
        runner = CliRunner()

        # Create a workspace with multiple repos
        workspace = make_workspace(tmp_path)

        # Enter one of the sibling repos (e.g., helm-repo)
        helm_repo = workspace / "helm-repo"
        import os

        os.chdir(helm_repo)

        result = runner.invoke(main, ["init", "--no-index"])
        assert result.exit_code == 0, f"init failed: {result.output}"

        # Should contain workspace warning
        assert (
            "Workspace detected" in result.output
        ), f"Expected workspace warning in output:\n{result.output}"
        assert "sibling git repo" in result.output or "monorepo" in result.output

        # Should suggest placing opencode.json at workspace root
        assert "workspace root" in result.output.lower() or "parent" in result.output.lower()

    def test_workspace_detection_shows_copy_suggestion(self, tmp_path):
        """Test that init provides a copy suggestion for workspace root."""
        runner = CliRunner()

        workspace = make_workspace(tmp_path)
        terraform_repo = workspace / "terraform-repo"
        import os

        os.chdir(terraform_repo)

        result = runner.invoke(main, ["init", "--no-index"])
        assert result.exit_code == 0

        # Should contain a copy command suggestion
        assert (
            "cp " in result.output and "opencode.json" in result.output
        ), f"Expected copy suggestion in output:\n{result.output}"

    def test_no_workspace_warning_in_single_repo(self, tmp_path):
        """Test that init does NOT warn for a standalone repo."""
        runner = CliRunner()

        # Create a single repo (not in workspace)
        repo = make_python_repo(tmp_path)
        import os

        os.chdir(repo)

        result = runner.invoke(main, ["init", "--no-index"])
        assert result.exit_code == 0

        # Should NOT contain workspace warning
        assert (
            "Workspace detected" not in result.output
        ), f"Standalone repo should not trigger workspace warning:\n{result.output}"
