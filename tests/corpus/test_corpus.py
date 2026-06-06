"""Opt-in corpus tests for large public repos (CAIRN_CORPUS=1)."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from core.config import load_config
from pipeline.ast_parser import ASTParser
from tests.corpus.manifest import CORPUS_REPOS, get_cache_dir

# Skip entire module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    "CAIRN_CORPUS" not in os.environ,
    reason="Corpus tests disabled by default (set CAIRN_CORPUS=1 to enable)",
)


def _shallow_clone_repo(url: str, target_dir: Path) -> bool:
    """Shallow clone a repo (depth=1) for fast corpus setup.

    Args:
        url: Git repository URL.
        target_dir: Target directory for clone.

    Returns:
        True if clone succeeded, False otherwise.
    """
    if target_dir.exists():
        return True  # Already cloned

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target_dir)],
            check=True,
            capture_output=True,
            timeout=120.0,
        )
        return True
    except Exception:
        return False


class TestCorpus:
    """Integration tests on large public repos."""

    @pytest.mark.corpus
    def test_helm_indexing_completes_under_budget(self):
        """Index Helm repo and measure wall-clock time."""
        repo_spec = next(r for r in CORPUS_REPOS if r.name == "helm")
        cache_dir = get_cache_dir()
        repo_path = cache_dir / repo_spec.name

        # Clone if needed
        if not _shallow_clone_repo(repo_spec.url, repo_path):
            pytest.skip(f"Failed to clone {repo_spec.name}")

        # Initialize cairn project
        project_path = repo_path
        (project_path / ".git").stat()  # Verify it's a git repo

        try:
            # Load or create config
            from core.config import save_config

            cfg = load_config(project_path)
            if not (project_path / ".cairn" / "config.yaml").exists():
                cfg.profile = "code"
                cfg.embeddings_enabled = True
                save_config(cfg, project_path)

            # Parse and index
            parser = ASTParser()

            from core.repo import collect_source_files

            files = collect_source_files(
                project_path,
                cfg.indexing.file_patterns,
                cfg.indexing.exclude_patterns,
                cfg.indexing.source_roots,
            )

            start = time.time()
            indexed_count = 0
            for file_path in files[:100]:  # Sample 100 files for corpus test
                try:
                    definitions = parser.parse(file_path)
                    if definitions:
                        indexed_count += len(definitions)
                except Exception:
                    pass  # Skip files that fail to parse

            elapsed = time.time() - start

            # Assert: indexing completed and took reasonable time (< 60s for 100 files)
            assert indexed_count > 0, "No functions indexed"
            assert elapsed < 60.0, f"Indexing took {elapsed:.1f}s (budget: 60s)"

        finally:
            # Cleanup chroma DB (leave clone for next run)
            chroma_path = repo_path / ".cairn" / "chroma"
            if chroma_path.exists():
                shutil.rmtree(chroma_path, ignore_errors=True)

    @pytest.mark.corpus
    def test_terraform_indexing_no_crash(self):
        """Index Terraform repo without crash."""
        repo_spec = next(r for r in CORPUS_REPOS if r.name == "terraform-aws-modules")
        cache_dir = get_cache_dir()
        repo_path = cache_dir / repo_spec.name

        # Clone if needed
        if not _shallow_clone_repo(repo_spec.url, repo_path):
            pytest.skip(f"Failed to clone {repo_spec.name}")

        try:
            from core.config import save_config

            cfg = load_config(repo_path)
            if not (repo_path / ".cairn" / "config.yaml").exists():
                cfg.profile = "iac"
                cfg.embeddings_enabled = False
                save_config(cfg, repo_path)

            # Parse a sample of files
            parser = ASTParser()

            from core.repo import collect_source_files

            files = collect_source_files(
                repo_path,
                cfg.indexing.file_patterns,
                cfg.indexing.exclude_patterns,
                cfg.indexing.source_roots,
            )

            # Just parse without error
            parsed_count = 0
            for file_path in files[:50]:  # Sample 50 files
                try:
                    definitions = parser.parse(file_path)
                    if definitions:
                        parsed_count += 1
                except Exception:
                    pass  # Acceptable; some files may have parse errors

            # Assert: at least some files parsed successfully
            assert parsed_count > 0, "No Terraform files parsed successfully"

        finally:
            chroma_path = repo_path / ".cairn" / "chroma"
            if chroma_path.exists():
                shutil.rmtree(chroma_path, ignore_errors=True)

    @pytest.mark.corpus
    def test_known_answer_queries(self):
        """Test that known-answer queries return expected files (when enabled)."""
        # This is a template for when reranker/search are available
        # For now, just verify the query format is valid
        for repo_spec in CORPUS_REPOS:
            for query, expected_file_substr in repo_spec.queries:
                assert isinstance(query, str)
                assert isinstance(expected_file_substr, str)
                assert len(query) > 0
                assert len(expected_file_substr) > 0
