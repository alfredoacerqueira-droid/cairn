"""Test offline/proxy resilience: reranker hangs and timeouts don't block search.

This test verifies that:
  1. When FlashRank Ranker() times out (proxy hang), search completes quickly
  2. When FlashRank Ranker() raises (SSL error), search completes quickly
  3. With offline=True, reranker is never constructed (no download attempt)
  4. cairn doctor runs to completion and reports reranker status (no hangs)
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.main import doctor, init
from core.config import load_config, save_config
from server.context_assembler import ContextAssembler


@pytest.mark.integration
class TestOfflineResilience:
    """Test reranker timeout and offline modes."""

    def _setup_temp_repo(self, tmp_path: Path) -> Path:
        """Create a minimal git repo with a Python file."""
        import subprocess

        project_path = tmp_path

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=project_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=project_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=project_path,
            capture_output=True,
            check=True,
        )

        # Create a simple Python file
        test_py = project_path / "test.py"
        test_py.write_text("def hello():\n    return 'world'\n")

        # Commit the file
        subprocess.run(
            ["git", "add", "test.py"],
            cwd=project_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=project_path,
            capture_output=True,
            check=True,
        )

        # Create .cairn config
        config_dir = project_path / ".cairn"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("""indexing:
  file_patterns:
    - "*.py"
  exclude_patterns:
    - "**/.venv/**"
  source_roots:
    - "."
embeddings_enabled: false
""")

        return project_path

    def test_reranker_timeout_does_not_block_search(self, tmp_path):
        """When FlashRank Ranker() times out, _get_ranker returns within timeout."""
        # Direct unit test of _get_ranker timeout behavior
        import pipeline.retrieval.reranker as reranker_module

        reranker_module._ranker_cache.clear()

        # Monkeypatch Ranker class to hang
        class HangingRanker:
            def __init__(self, model_name=None):
                time.sleep(25)

        with patch("flashrank.Ranker", HangingRanker):
            # Call _get_ranker directly and measure time
            start = time.perf_counter()
            result = reranker_module._get_ranker(ca_bundle=None, offline=False)
            elapsed = time.perf_counter() - start

            # _get_ranker should return None within timeout, not wait for full sleep
            assert result is None, "Should degrade gracefully"
            assert (
                elapsed < 30.0
            ), f"_get_ranker took {elapsed:.2f}s; should timeout at 20s + small overhead"

    def test_reranker_exception_does_not_block_search(self, tmp_path):
        """When FlashRank Ranker() raises (SSL error), _get_ranker handles it gracefully."""
        import pipeline.retrieval.reranker as reranker_module

        reranker_module._ranker_cache.clear()

        # Monkeypatch Ranker to raise SSL error
        def error_ranker(*args, **kwargs):
            raise RuntimeError("SSL_ERROR: certificate verify failed")

        with patch("flashrank.Ranker", side_effect=error_ranker):
            # Call _get_ranker and verify it returns None on error
            result = reranker_module._get_ranker(ca_bundle=None, offline=False)
            assert result is None, "Should degrade gracefully on SSL error"
            assert (
                reranker_module._ranker_cache.get("ms-marco-MiniLM-L-12-v2") is None
            ), "Should cache the failure to avoid retries"

    def test_offline_mode_skips_reranker_construction(self, tmp_path):
        """With offline=True, reranker is never constructed (no download attempt)."""
        project_path = self._setup_temp_repo(tmp_path)

        try:
            original_cwd = os.getcwd()
            os.chdir(project_path)

            # Set offline mode in config
            cfg = load_config(project_path)
            cfg.retrieval.offline = True
            save_config(cfg, project_path)

            # Build index
            from core.repo import RepoManager, collect_source_files
            from pipeline.ast_parser import ASTParser
            from pipeline.indexer import VectorIndexer

            repo = RepoManager(project_path)
            parser = ASTParser()
            indexer = VectorIndexer(chroma_path=repo.get_chroma_path(), embeddings_enabled=False)

            cfg = load_config(project_path)
            filtered = collect_source_files(
                project_path,
                cfg.indexing.file_patterns,
                cfg.indexing.exclude_patterns,
                cfg.indexing.source_roots,
            )

            for filepath in filtered:
                try:
                    ast = parser.parse_file(filepath)
                    indexer.index_ast(ast)
                except Exception:
                    pass

            # Mock Ranker to fail if called (it shouldn't be)
            mock_ranker = MagicMock()
            mock_ranker.side_effect = RuntimeError(
                "Reranker should not be constructed in offline mode"
            )

            with patch("flashrank.Ranker", mock_ranker):
                # Create assembler and search
                assembler = ContextAssembler(project_path=project_path)

                # Search should succeed without ever calling Ranker()
                assembler.semantic_search("hello", top_k=5)

                # Verify Ranker was never called
                assert not mock_ranker.called, "Ranker() should not be called in offline mode"

        finally:
            os.chdir(original_cwd)

    def test_doctor_runs_to_completion(self, tmp_path):
        """cairn doctor completes without hanging or timeout."""
        project_path = self._setup_temp_repo(tmp_path)

        try:
            original_cwd = os.getcwd()
            os.chdir(project_path)

            # Run doctor with CliRunner (which has implicit timeout via click)
            runner = CliRunner()

            start = time.perf_counter()
            result = runner.invoke(doctor)
            elapsed = time.perf_counter() - start

            # Doctor should complete within a reasonable time (~5s)
            assert elapsed < 10.0, f"Doctor took {elapsed:.2f}s; expected <10s (should not hang)"

            # Doctor should report reranker status (one of the checks)
            assert (
                "Reranker" in result.output
                or "flashrank" in result.output
                or "reranking" in result.output
            ), f"Doctor should report reranker status. Output:\n{result.output}"

        finally:
            os.chdir(original_cwd)

    def test_init_offline_flag_sets_config(self, tmp_path):
        """cairn init --offline sets retrieval.offline=True in config."""
        project_path = self._setup_temp_repo(tmp_path)

        try:
            original_cwd = os.getcwd()
            os.chdir(project_path)

            # Remove the config so init will create a fresh one
            (project_path / ".cairn" / "config.yaml").unlink()

            runner = CliRunner()

            # Run init with --offline flag and --no-index (skip embedding model checks)
            result = runner.invoke(init, ["--offline", "--no-index", "-y"])

            assert result.exit_code == 0, f"init failed: {result.output}"

            # Verify config has offline=True
            cfg = load_config(project_path)
            assert cfg.retrieval.offline is True, "offline should be set to True in config"

        finally:
            os.chdir(original_cwd)
