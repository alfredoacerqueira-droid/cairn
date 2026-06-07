"""Integration tests for fastembed auto-selection during cairn init."""

import os

import pytest
import yaml
from click.testing import CliRunner

from cli.main import main
from tests.fixtures.builders import make_helm_repo, make_python_repo

try:
    import fastembed  # noqa: F401

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


pytestmark = pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not installed")


class TestInitFastembedDefault:
    def test_init_python_repo_selects_fastembed(self, tmp_path):
        """On a python repo with NO Ollama, init auto-selects fastembed embedder."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_python_repo(tmp_path)
            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"init failed: {result.output}"

            config_file = repo / ".cairn" / "config.yaml"
            assert config_file.exists()

            config_data = yaml.safe_load(config_file.read_text())
            assert config_data["embeddings_enabled"] is True
            assert config_data["local_llm"]["embedder"] == "fastembed"

    def test_init_helm_repo_does_not_force_fastembed(self, tmp_path):
        """On an iac (helm) repo, embeddings stay OFF and embedder is not forced."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_helm_repo(tmp_path)
            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0, f"init failed: {result.output}"

            config_file = repo / ".cairn" / "config.yaml"
            assert config_file.exists()

            config_data = yaml.safe_load(config_file.read_text())
            assert config_data["embeddings_enabled"] is False

    def test_init_output_confirms_fastembed(self, tmp_path):
        """Init output confirms offline semantic search via fastembed."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = make_python_repo(tmp_path)
            os.chdir(repo)

            result = runner.invoke(main, ["init", "--no-index"])
            assert result.exit_code == 0

            assert "fastembed" in result.output
            assert "offline" in result.output
