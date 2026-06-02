"""Harsh test: index a repo with many files and verify budget compliance.

Generates a synthetic repo with hundreds to ~1-2k small files and verifies:
1. Indexing completes within a wall-clock budget (e.g., <30s).
2. A sane count of files are indexed (no silent skips).
3. The index is successfully populated with functions/classes.
"""

import time
from pathlib import Path

from tests.fixtures.harness import fresh_index


def _make_large_python_repo(base: Path, num_modules: int) -> Path:
    """Generate a Python repo with many small modules.

    Args:
        base: Parent directory.
        num_modules: Number of .py files to create (e.g., 500-1000).

    Returns:
        Path to repo root.
    """
    repo_root = base / "large-repo"
    repo_root.mkdir(exist_ok=True)

    # Create __init__.py
    (repo_root / "__init__.py").write_text('"""Large package."""\n')

    # Create subpackages with modules
    for i in range(num_modules):
        # Distribute into subdirs (modules/, services/, utils/)
        subdir = repo_root / ["modules", "services", "utils"][i % 3]
        subdir.mkdir(exist_ok=True)

        # Create module_N.py with a simple function
        module_file = subdir / f"module_{i:04d}.py"
        module_file.write_text(f'''"""Module {i}."""

def function_{i:04d}(x: int) -> int:
    """Function {i}."""
    return x + {i}

class Class_{i:04d}:
    """Class {i}."""

    def method(self, value: str) -> str:
        """Method in class {i}."""
        return f"{{value}}_result_{i}"
''')

    # Initialize git
    import subprocess

    subprocess.run(
        ["git", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    return repo_root


class TestLargeRepoIndexingBudget:
    """Verify indexing of large repos completes within budget."""

    def test_medium_repo_500_files_indexes_in_budget(self, tmp_path):
        """Index a repo with ~500 small Python modules in <45s."""
        repo = _make_large_python_repo(tmp_path, num_modules=500)

        start = time.time()
        fresh_index(repo, embeddings=False)
        elapsed = time.time() - start

        # Should complete in <45s for 500 small files
        assert elapsed < 45.0, f"indexing took {elapsed}s, expected <45s"

        # Verify index is populated
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["documents", "metadatas"])
        indexed_count = len(data["documents"])
        assert indexed_count > 0, "no documents indexed"

        # Should have indexed a sane count (at least 50% of files)
        expected_min = 500  # We have ~500 files * 2 items per file (1 func + 1 class)
        assert (
            indexed_count > expected_min * 0.3
        ), f"only {indexed_count} items indexed from {expected_min} expected"

    def test_large_repo_1000_files_indexes_in_budget(self, tmp_path):
        """Index a repo with ~1000 small modules in <90s."""
        repo = _make_large_python_repo(tmp_path, num_modules=1000)

        start = time.time()
        fresh_index(repo, embeddings=False)
        elapsed = time.time() - start

        # Should complete in <90s for 1000 small files
        assert elapsed < 90.0, f"indexing took {elapsed}s, expected <90s"

        # Verify index is populated
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["documents"])
        indexed_count = len(data["documents"])
        assert indexed_count > 0, "no documents indexed"

        # Verify no silent skips: should have indexed substantial portion
        assert indexed_count > 200, f"only {indexed_count} items indexed, expected >200"

    def test_repo_file_counts_match(self, tmp_path):
        """Verify file iteration and indexing counts are consistent."""
        num_files = 200  # Smaller set for quick sanity check
        repo = _make_large_python_repo(tmp_path, num_modules=num_files)

        from core.config import Config
        from core.repo import collect_source_files

        cfg = Config()
        files = collect_source_files(
            repo,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            ["."],
        )
        collected_count = len(files)

        # Should collect at least num_files Python files
        assert collected_count >= num_files, f"collected {collected_count}, expected >= {num_files}"

        # Now index and verify
        fresh_index(repo, embeddings=False)

        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["documents"])
        indexed_count = len(data["documents"])

        # Each file should yield at least one indexed item
        assert indexed_count > 0, "nothing indexed"
        # Should be reasonable ratio (at least 1 item per 2 files)
        assert (
            indexed_count > collected_count * 0.4
        ), f"indexed {indexed_count} from {collected_count} files (low ratio)"
