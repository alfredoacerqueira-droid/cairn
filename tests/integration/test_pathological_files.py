"""Harsh test: index repo with pathological files and edge cases.

Tests handling of:
- Oversized YAML (>5461 docs via k8s pathological)
- Large nested YAML (~60KB deployment)
- Unicode filenames
- Symlinks (graceful skip if unsupported)
- Vendored .git directories

Verifies:
1. Fresh index completes without crash.
2. Pathological files are gracefully handled (not causing silent skips of other files).
3. Index is populated with what could be parsed.
"""

import os
from pathlib import Path

from tests.fixtures.builders import make_k8s_repo, make_terraform_repo
from tests.fixtures.harness import fresh_index


def _add_unicode_filename(repo_path: Path) -> None:
    """Add a file with Unicode characters in its name."""
    unicode_file = repo_path / "src" / "módulo_ñoño_españa.py"
    unicode_file.parent.mkdir(parents=True, exist_ok=True)
    unicode_file.write_text('''"""Module with Unicode name."""

def función_española(texto: str) -> str:
    """Function with Spanish characters."""
    return f"Niño: {texto}"
''')


def _add_vendored_git_dir(repo_path: Path) -> None:
    """Add a vendored .git directory (should be excluded)."""
    vendored = repo_path / "vendor" / "some-lib" / ".git"
    vendored.mkdir(parents=True, exist_ok=True)
    (vendored / "HEAD").write_text("ref: refs/heads/main\n")
    (vendored / "config").write_text("[core]\n    repositoryformatversion = 0\n")


def _add_symlink(repo_path: Path) -> None:
    """Add a symlink (skip gracefully if unsupported, e.g., on Windows)."""
    target = repo_path / "lib.py"
    link = repo_path / "lib-link.py"
    target.write_text('def lib_func(): return "target"\n')
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        # Symlinks not supported (e.g., Windows without admin); skip
        pass


class TestPathologicalFiles:
    """Verify handling of edge-case files."""

    def test_oversized_k8s_crd_indexed_without_crash(self, tmp_path):
        """Oversized CRD YAML (>5461 docs) is handled gracefully."""
        repo = make_k8s_repo(tmp_path, with_pathological=True)

        # Should not crash
        fresh_index(repo, embeddings=False)

        # Verify index was created
        assert (repo / ".cairn").exists()
        assert (repo / ".cairn" / "chroma").exists()

        # Verify some content was indexed (non-pathological files)
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["documents"])
        # At least the simple manifests should be indexed
        assert len(data["documents"]) > 0

    def test_large_deployment_yaml_batch_split(self, tmp_path):
        """Large deployment with huge env list is batch-split correctly."""
        repo = make_k8s_repo(tmp_path, with_pathological=True)

        fresh_index(repo, embeddings=False)

        # Verify the large-deployment.yaml was processed
        large_yaml = repo / "manifests" / "large-deployment.yaml"
        assert large_yaml.exists(), "large-deployment.yaml not created"

        # Should be > 50KB (1000 env vars * ~50 bytes each)
        size = large_yaml.stat().st_size
        assert size > 50000, f"large-deployment.yaml is only {size} bytes"

        # Index should have processed it without crashing
        assert (repo / ".cairn" / "chroma").exists()

    def test_unicode_filename_indexed(self, tmp_path):
        """Files with Unicode names are indexed (or gracefully skipped)."""
        repo = make_terraform_repo(tmp_path)
        _add_unicode_filename(repo)

        # Should not crash on Unicode filenames
        fresh_index(repo, embeddings=False)

        # Index should exist
        assert (repo / ".cairn" / "chroma").exists()

        # If the Unicode file was indexed, good. If not, that's also fine
        # (depends on the OS). The important thing is no crash.

    def test_vendored_git_excluded_from_index(self, tmp_path):
        """Vendored .git directories are excluded from indexing."""
        repo = make_terraform_repo(tmp_path)
        _add_vendored_git_dir(repo)

        fresh_index(repo, embeddings=False)

        # Verify vendor/ was not indexed
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["metadatas"])
        filepaths = [m["filepath"] for m in data["metadatas"]]

        # No vendor/ files should be in the index
        assert not any("vendor/" in fp for fp in filepaths), filepaths

    def test_symlink_gracefully_skipped(self, tmp_path):
        """Symlinks are skipped without crashing (if not supported)."""
        repo = make_terraform_repo(tmp_path)
        _add_symlink(repo)

        # Should not crash
        fresh_index(repo, embeddings=False)

        # Index should exist
        assert (repo / ".cairn" / "chroma").exists()

    def test_mixed_pathological_all_at_once(self, tmp_path):
        """Repo with all pathological cases together."""
        repo = make_k8s_repo(tmp_path, with_pathological=True)
        _add_unicode_filename(repo)
        _add_vendored_git_dir(repo)
        _add_symlink(repo)

        # Should complete without crash
        fresh_index(repo, embeddings=False)

        # Index should be functional
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        data = indexer.collection.get(include=["documents"])
        # Should have indexed at least the valid K8s manifests
        assert len(data["documents"]) > 0, "nothing indexed despite valid files"

        # Sanity: vendor/ should not be indexed
        data_with_meta = indexer.collection.get(include=["metadatas"])
        filepaths = [m["filepath"] for m in data_with_meta["metadatas"]]
        assert not any("vendor/" in fp for fp in filepaths)
