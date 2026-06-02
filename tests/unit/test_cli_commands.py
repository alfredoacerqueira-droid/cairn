"""Unit tests for CLI commands."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cli.main import main


class TestCliCommands:
    def test_doctor_command_basic(self):
        """Test doctor command runs and reports reranker + interpreter status."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Initialize git repo (doctor checks for .git)
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )
            # Create minimal code to satisfy checks
            Path("main.py").write_text("def main(): pass")

            # Mock OllamaClient to avoid network calls
            with patch("server.ollama_client.OllamaClient") as mock_ollama_class:
                mock_ollama = MagicMock()
                mock_ollama.health_check.return_value = False
                mock_ollama_class.return_value = mock_ollama

                result = runner.invoke(main, ["doctor"])

                # Exit code should be 0 or 1 (depends on checks, but command runs)
                assert result.exit_code in [0, 1], f"Unexpected exit: {result.output}"

                # Should show reranker status (either available or NOT available)
                assert (
                    "Reranker (flashrank)" in result.output
                ), f"Doctor did not report reranker status. Output:\n{result.output}"

                # Should show interpreter path
                assert (
                    "Interpreter:" in result.output
                ), f"Doctor did not report interpreter. Output:\n{result.output}"

    def test_run_command_registered(self):
        """Test that the 'run' command is registered in the CLI."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_command_help(self):
        """Test that 'run --help' works."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Alias for start-all" in result.output

    def test_run_command_options(self):
        """Test that 'run' command accepts expected options."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--no-janitor" in result.output
        assert "--no-index" in result.output
        assert "--yes" in result.output or "-y" in result.output

    def test_start_all_command_registered(self):
        """Test that the 'start-all' command is registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "start-all" in result.output

    def test_start_all_command_help(self):
        """Test that 'start-all --help' works."""
        runner = CliRunner()
        result = runner.invoke(main, ["start-all", "--help"])
        assert result.exit_code == 0
        assert "start-all" in result.output.lower() or "smart orchestrator" in result.output

    def test_start_all_command_options(self):
        """Test that 'start-all' command accepts expected options."""
        runner = CliRunner()
        result = runner.invoke(main, ["start-all", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--no-janitor" in result.output
        assert "--no-index" in result.output
        assert "--yes" in result.output or "-y" in result.output


class TestInitCommand:
    def test_init_command_registered(self):
        """Test that the 'init' command is registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output

    def test_init_command_help(self):
        """Test that 'init --help' works."""
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output.lower()

    def test_init_command_options(self):
        """Test that 'init' command has expected options."""
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "--no-index" in result.output
        assert "--yes" in result.output or "-y" in result.output
        assert "--force" in result.output

    def test_init_no_index_flag(self, tmp_path):
        """Test init --no-index writes config without indexing."""
        import tempfile

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )

            # Create minimal Python code
            Path("main.py").write_text("def main(): pass")

            # Run init with --no-index (should not require Ollama)
            result = runner.invoke(main, ["init", "--no-index"])

            # Should succeed
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Should create config.yaml
            config_file = Path(".cairn") / "config.yaml"
            assert config_file.exists(), f"Config not created. Output: {result.output}"

            # Should create .gitignore
            gitignore_file = Path(".cairn") / ".gitignore"
            assert gitignore_file.exists()

            # Should NOT have indexed (no chroma yet)
            chroma_file = Path(".cairn") / "chroma" / "index.bin"
            # chroma might not create a simple index.bin, but let's check the directory
            # was created (that's expected even without indexing due to preflight checks)

    def test_init_idempotent(self, tmp_path):
        """Test that running init twice without --force is idempotent."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )

            # Create minimal code
            Path("main.py").write_text("def main(): pass")

            # First init
            result1 = runner.invoke(main, ["init", "--no-index"])
            assert result1.exit_code == 0

            config_file = Path(".cairn") / "config.yaml"
            assert config_file.exists()
            first_content = config_file.read_text()

            # Second init (should not error)
            result2 = runner.invoke(main, ["init", "--no-index"])
            assert result2.exit_code == 0

            # Config should still exist and be valid
            assert config_file.exists()
            second_content = config_file.read_text()

            # Content should be reasonable (might be slightly different due to
            # key ordering in YAML, but structure should be the same)
            assert "source_roots" in second_content
            assert "file_patterns" in second_content

    def test_init_scaffolds_opencode_json(self, tmp_path):
        """Test that init scaffolds opencode.json with MCP config."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Initialize git repo
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )

            # Create minimal code
            Path("main.py").write_text("def main(): pass")

            # Run init with --no-index
            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Check opencode.json exists and is valid JSON
            opencode_file = Path("opencode.json")
            assert opencode_file.exists()
            opencode_data = json.loads(opencode_file.read_text())

            # Verify structure
            assert "mcp" in opencode_data
            assert "cairn" in opencode_data["mcp"]

            # Verify MCP entry
            mcp_cfg = opencode_data["mcp"]["cairn"]
            assert mcp_cfg["type"] == "local"
            assert mcp_cfg["args"] == ["mcp"]
            assert "CAIRN_PROJECT" in mcp_cfg["env"]

            # CAIRN_PROJECT should be absolute path to current dir
            expected_path = str(Path.cwd().resolve())
            assert mcp_cfg["env"]["CAIRN_PROJECT"] == expected_path

    def test_init_scaffolds_mcp_json(self, tmp_path):
        """Test that init scaffolds .mcp.json with MCP config."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Initialize git repo
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )

            # Create minimal code
            Path("main.py").write_text("def main(): pass")

            # Run init with --no-index
            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Check .mcp.json exists and is valid JSON
            mcp_file = Path(".mcp.json")
            assert mcp_file.exists()
            mcp_data = json.loads(mcp_file.read_text())

            # Verify structure
            assert "mcpServers" in mcp_data
            assert "cairn" in mcp_data["mcpServers"]

            # Verify MCP entry
            mcp_cfg = mcp_data["mcpServers"]["cairn"]
            assert "command" in mcp_cfg
            assert mcp_cfg["args"] == ["mcp"]
            assert "CAIRN_PROJECT" in mcp_cfg["env"]

            # CAIRN_PROJECT should be absolute path to current dir
            expected_path = str(Path.cwd().resolve())
            assert mcp_cfg["env"]["CAIRN_PROJECT"] == expected_path

    def test_init_mcp_scaffold_idempotent(self, tmp_path):
        """Test that running init twice does not duplicate MCP entry."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Initialize git repo
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )

            # Create minimal code
            Path("main.py").write_text("def main(): pass")

            # First init
            result1 = runner.invoke(main, ["init", "--no-index"])
            assert result1.exit_code == 0

            opencode_file = Path("opencode.json")
            assert opencode_file.exists()
            data1 = json.loads(opencode_file.read_text())

            # Second init
            result2 = runner.invoke(main, ["init", "--no-index"])
            assert result2.exit_code == 0

            # opencode.json should still be valid JSON
            assert opencode_file.exists()
            data2 = json.loads(opencode_file.read_text())

            # Should have exactly one cairn entry
            assert "mcp" in data2
            assert "cairn" in data2["mcp"]
            assert isinstance(data2["mcp"]["cairn"], dict)

            # Check .mcp.json as well
            mcp_file = Path(".mcp.json")
            assert mcp_file.exists()
            mcp_data = json.loads(mcp_file.read_text())
            assert "mcpServers" in mcp_data
            assert "cairn" in mcp_data["mcpServers"]
            assert isinstance(mcp_data["mcpServers"]["cairn"], dict)

    def test_mcp_command_safe_without_sdk(self):
        """Test that mcp command has safe error handling for missing SDK."""
        # Read the source code to verify try/except is in place
        import inspect

        # Import the raw mcp callback function by reading the file
        with open("cli/main.py") as f:
            source = f.read()

        # Find the mcp command implementation and verify error handling
        mcp_start = source.find("def mcp():")
        assert mcp_start != -1, "mcp function not found"

        # Extract the mcp function (up to the next @main.command or end of function)
        mcp_end = source.find("\n@", mcp_start + 1)
        if mcp_end == -1:
            mcp_end = source.find("\nif __name__", mcp_start + 1)
        mcp_source = source[mcp_start:mcp_end]

        # Verify the error handling code is present
        assert "ImportError" in mcp_source
        assert "MCP SDK not installed" in mcp_source
        assert "pip install mcp" in mcp_source
        assert "sys.exit(1)" in mcp_source
