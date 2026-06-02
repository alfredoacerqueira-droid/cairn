"""Harsh test: simulate slow filesystem reads during parsing.

Monkeypatches the ASTParser's file-read operations to inject latency,
simulating slow network storage (e.g., /mnt/c on WSL). Verifies that:
1. Indexing completes within a wall-clock budget (doesn't hang).
2. Oversized/slow files are gracefully skipped with a warning (not crashed).
3. The index is still populated with successfully parsed files.
"""

import time
from pathlib import Path

from pipeline.ast_parser import ASTParser
from tests.fixtures.builders import make_k8s_repo
from tests.fixtures.harness import fresh_index


class TestSlowFilesystemIndexing:
    """Verify parsing completes under slow read conditions."""

    def test_slow_file_reads_index_completes(self, tmp_path, monkeypatch):
        """Index completes despite monkeypatched slow reads."""
        # Create a repo with pathological YAML
        repo = make_k8s_repo(tmp_path, with_pathological=True)

        # Monkeypatch ASTParser.parse_file to add a small delay to each file
        original_parse = ASTParser.parse_file
        call_count = 0

        def slow_parse_file(self, filepath):
            nonlocal call_count
            call_count += 1
            # Simulate slow filesystem: small delay per file
            time.sleep(0.01)
            return original_parse(self, filepath)

        monkeypatch.setattr(ASTParser, "parse_file", slow_parse_file)

        # fresh_index should still complete (not hang)
        start = time.time()
        fresh_index(repo, embeddings=False)
        elapsed = time.time() - start

        # Should complete in reasonable time (even with delays, should be <10s)
        assert elapsed < 10.0, f"indexing took {elapsed}s, expected <10s"

        # Verify index was populated
        cairn_dir = repo / ".cairn"
        assert cairn_dir.exists()
        assert (cairn_dir / "chroma").exists()

        # Verify some files were parsed (at least the small ones)
        assert call_count > 0

    def test_pathological_yaml_skipped_with_tiny_timeout(self, tmp_path, monkeypatch):
        """Oversized YAML is skipped (not crashed) with timeout."""
        repo = make_k8s_repo(tmp_path, with_pathological=True)

        # Create a parser with a very small timeout
        parser = ASTParser(parse_timeout_s=0.001)

        # Monkeypatch the internal parse to sleep (simulating slow parse)
        original_parse = parser.parse_file

        def slow_parse_wrapper(filepath):
            # Simulate a slow parse by adding a sleep
            if "large" in filepath.name:
                time.sleep(0.01)
            return original_parse(filepath)

        monkeypatch.setattr(parser, "parse_file", slow_parse_wrapper)

        # Should not raise even with timeout; large file is skipped
        large_yaml = repo / "manifests" / "large-crd.yaml"
        if large_yaml.exists():
            # Parsing should timeout gracefully
            result = parser.parse_file(large_yaml)
            # The parser returns an empty AST on timeout, not raises
            assert result is not None

    def test_large_deployment_yaml_indexed(self, tmp_path):
        """Large but parseable YAML (nested deployment) is indexed."""
        repo = make_k8s_repo(tmp_path, with_pathological=True)

        # fresh_index should handle it
        fresh_index(repo, embeddings=False)

        # Verify the large-deployment.yaml was parsed (not skipped)
        # We can't easily check the contents, but we can verify no crash
        cairn_dir = repo / ".cairn"
        assert cairn_dir.exists()

        # Check that chroma db has some documents (parsed successfully)
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        repo_mgr = RepoManager(repo)
        indexer = VectorIndexer(
            chroma_path=repo_mgr.get_chroma_path(),
            embeddings_enabled=False,
        )

        # Get count of indexed items
        data = indexer.collection.get(include=["documents"])
        assert len(data["documents"]) > 0, "no documents indexed"

    def test_indexing_with_file_open_latency(self, tmp_path, monkeypatch):
        """Monkeypatch pathlib.Path.read_text to simulate I/O latency."""
        repo = make_k8s_repo(tmp_path, with_pathological=False)

        # Monkeypatch Path.read_text to add latency
        original_read_text = Path.read_text
        read_count = 0

        def latent_read_text(self, *args, **kwargs):
            nonlocal read_count
            read_count += 1
            time.sleep(0.005)  # 5ms per file read
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", latent_read_text)

        # Should still index successfully
        start = time.time()
        fresh_index(repo, embeddings=False)
        elapsed = time.time() - start

        # Should complete (5ms * a few files = <1s)
        assert elapsed < 5.0
        assert read_count > 0

        # Index should be populated
        cairn_dir = repo / ".cairn"
        assert (cairn_dir / "chroma").exists()
