"""Tests for content-hash incremental indexing and IndexManifest."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from core.manifest import IndexManifest, sha256_of_file


class MockOllamaClient:
    KEYWORDS = {
        "auth": 0.8,
        "login": 0.9,
        "logout": 0.9,
        "user": 0.7,
        "pass": 0.6,
        "authentic": 0.95,
        "token": 0.85,
        "database": 0.8,
        "date": 0.3,
        "format": 0.2,
        "util": 0.1,
        "connect": 0.8,
    }

    def __init__(self, embed_model: str = "nomic-embed-text"):
        self.embed_model = embed_model

    def embed(self, text: str, model=None) -> list[float]:
        _ = model or self.embed_model
        text_lower = text.lower()
        vector = []
        keys = list(self.KEYWORDS.keys())
        for kw in keys:
            vector.append(self.KEYWORDS[kw] if kw in text_lower else 0.0)
        while len(vector) < 16:
            vector.append(0.0)
        return vector

    def embed_batch(self, texts: list[str], model=None) -> list[list[float]]:
        _ = model or self.embed_model
        return [self.embed(text, model) for text in texts]


class FakeRepoManager:
    def __init__(self, tmp_path: Path):
        self.data_dir = tmp_path / ".cairn"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.project_path = tmp_path

    def get_manifest_path(self) -> Path:
        return self.data_dir / "index_manifest.json"

    def get_chroma_path(self) -> Path:
        p = self.data_dir / "chroma"
        p.mkdir(parents=True, exist_ok=True)
        return p


class TestIndexManifest:
    @pytest.fixture
    def tmp_manifest(self, tmp_path):
        repo = FakeRepoManager(tmp_path)
        manifest = IndexManifest(repo.get_manifest_path(), "test-pid-001")
        return manifest, repo

    def test_save_and_load(self, tmp_manifest):
        manifest, repo = tmp_manifest
        manifest.set_entry("src/auth.py", "abc123", 5)
        manifest.save()
        loaded = IndexManifest.load(repo, "test-pid-001")
        assert loaded.files == manifest.files
        assert loaded.project_id == "test-pid-001"
        assert loaded.version == 1

    def test_has_same_hash_matches(self, tmp_manifest):
        manifest, _ = tmp_manifest
        manifest.set_entry("src/auth.py", "abc123", 5)
        assert manifest.has_same_hash("src/auth.py", "abc123") is True

    def test_has_same_hash_mismatch(self, tmp_manifest):
        manifest, _ = tmp_manifest
        manifest.set_entry("src/auth.py", "abc123", 5)
        assert manifest.has_same_hash("src/auth.py", "xyz789") is False

    def test_has_same_hash_missing_entry(self, tmp_manifest):
        manifest, _ = tmp_manifest
        assert manifest.has_same_hash("src/nonexistent.py", "abc123") is False

    def test_remove_entry(self, tmp_manifest):
        manifest, _ = tmp_manifest
        manifest.set_entry("src/auth.py", "abc123", 5)
        manifest.set_entry("src/utils.py", "def456", 3)
        manifest.remove_entry("src/auth.py")
        assert "src/auth.py" not in manifest.files
        assert "src/utils.py" in manifest.files

    def test_remove_nonexistent_entry_does_nothing(self, tmp_manifest):
        manifest, _ = tmp_manifest
        manifest.remove_entry("nonexistent.py")
        assert manifest.files == {}

    def test_load_missing_file_returns_empty(self, tmp_manifest):
        _, repo = tmp_manifest
        manifest = IndexManifest.load(repo, "test-pid-001")
        assert manifest.files == {}

    def test_load_corrupt_json_returns_empty(self, tmp_manifest):
        _, repo = tmp_manifest
        repo.get_manifest_path().write_text("this is not json")
        manifest = IndexManifest.load(repo, "test-pid-001")
        assert manifest.files == {}

    def test_load_wrong_project_id_returns_empty(self, tmp_manifest):
        _, repo = tmp_manifest
        manifest = IndexManifest(repo.get_manifest_path(), "other-pid")
        manifest.set_entry("src/auth.py", "abc123", 5)
        manifest.save()
        loaded = IndexManifest.load(repo, "test-pid-001")
        assert loaded.files == {}

    def test_save_then_reload_preserves_all_entries(self, tmp_manifest):
        manifest, repo = tmp_manifest
        for i in range(5):
            manifest.set_entry(
                f"src/file{i}.py",
                hashlib.sha256(f"content{i}".encode()).hexdigest(),
                i + 1,
            )
        manifest.save()
        loaded = IndexManifest.load(repo, "test-pid-001")
        assert len(loaded.files) == 5
        for i in range(5):
            assert f"src/file{i}.py" in loaded.files
            assert loaded.files[f"src/file{i}.py"]["blocks"] == i + 1

    def test_set_entry_overwrites(self, tmp_manifest):
        manifest, _ = tmp_manifest
        manifest.set_entry("src/auth.py", "abc123", 5)
        manifest.set_entry("src/auth.py", "xyz789", 10)
        assert manifest.files["src/auth.py"]["sha256"] == "xyz789"
        assert manifest.files["src/auth.py"]["blocks"] == 10

    def test_load_empty_file_returns_empty(self, tmp_manifest):
        _, repo = tmp_manifest
        repo.get_manifest_path().write_text("")
        manifest = IndexManifest.load(repo, "test-pid-001")
        assert manifest.files == {}

    def test_load_file_with_json_false(self, tmp_manifest):
        _, repo = tmp_manifest
        repo.get_manifest_path().write_text("false")
        manifest = IndexManifest.load(repo, "test-pid-001")
        assert manifest.files == {}

    def test_load_file_with_non_dict(self, tmp_manifest):
        _, repo = tmp_manifest
        repo.get_manifest_path().write_text(json.dumps([1, 2, 3]))
        manifest = IndexManifest.load(repo, "test-pid-001")
        assert manifest.files == {}

    def test_indexed_at_is_set(self, tmp_manifest):
        manifest, _ = tmp_manifest
        before = int(time.time())
        manifest.set_entry("src/auth.py", "abc123", 5)
        after = int(time.time())
        ts = manifest.files["src/auth.py"]["indexed_at"]
        assert before <= ts <= after


class TestSha256OfFile:
    def test_known_content(self, tmp_path):
        filepath = tmp_path / "test.py"
        filepath.write_text("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_of_file(filepath) == expected

    def test_empty_file(self, tmp_path):
        filepath = tmp_path / "empty.py"
        filepath.write_text("")
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_of_file(filepath) == expected

    def test_different_contents_different_hashes(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("foo")
        f2.write_text("bar")
        assert sha256_of_file(f1) != sha256_of_file(f2)


class TestIncrementalIndexing:
    @pytest.fixture
    def setup_indexer(self, tmp_path):
        from core.repo import project_id
        from pipeline.ast_parser import ASTParser
        from pipeline.indexer import VectorIndexer

        repo = FakeRepoManager(tmp_path)
        idx = VectorIndexer(
            chroma_path=repo.get_chroma_path(),
            ollama_client=MockOllamaClient(),
            project_root=tmp_path,
        )
        parser = ASTParser()
        pid = project_id(str(tmp_path.resolve()))
        return tmp_path, repo, idx, parser, pid

    def _write_source(self, setup_indexer, filename: str, code: str) -> Path:
        tmp_path, _, _, _, _ = setup_indexer
        filepath = tmp_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code)
        return filepath

    def test_unchanged_file_skipped(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        code219 = "def login():\n    pass\ndef logout():\n    pass"
        filepath = self._write_source(setup_indexer, "src/auth.py", code219)
        ast = parser.parse_file(filepath)
        idx.index_ast(ast)

        manifest = IndexManifest(repo.get_manifest_path(), pid)
        relpath = "src/auth.py"
        fhash = sha256_of_file(filepath)
        blocks = len(ast.functions) + sum(len(c.methods) for c in ast.classes)
        manifest.set_entry(relpath, fhash, blocks)
        manifest.save()

        assert manifest.has_same_hash(relpath, sha256_of_file(filepath))

    def test_modified_file_reindexed(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")

        initial_ast = parser.parse_file(filepath)
        idx.index_ast(initial_ast)

        manifest = IndexManifest(repo.get_manifest_path(), pid)
        relpath = "src/auth.py"
        n_blocks = len(initial_ast.functions) + sum(len(c.methods) for c in initial_ast.classes)
        manifest.set_entry(relpath, sha256_of_file(filepath), n_blocks)
        manifest.save()

        filepath.write_text("def login():\n    print('modified')\ndef register():\n    pass")

        assert not manifest.has_same_hash(relpath, sha256_of_file(filepath))
        idx.remove_file(str(filepath))
        new_ast = parser.parse_file(filepath)
        idx.index_ast(new_ast)

        new_blocks = len(new_ast.functions) + sum(len(c.methods) for c in new_ast.classes)
        manifest.set_entry(relpath, sha256_of_file(filepath), new_blocks)
        manifest.save()

        loaded = IndexManifest.load(repo, pid)
        assert loaded.files[relpath]["blocks"] == new_blocks

    def test_deleted_file_removed_from_manifest_and_index(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")
        ast = parser.parse_file(filepath)
        idx.index_ast(ast)

        manifest = IndexManifest(repo.get_manifest_path(), pid)
        relpath = "src/auth.py"
        manifest.set_entry(relpath, sha256_of_file(filepath), 1)
        manifest.save()

        os.remove(str(filepath))

        assert relpath in manifest.files
        idx.remove_file(str(filepath))
        manifest.remove_entry(relpath)
        manifest.save()

        loaded = IndexManifest.load(repo, pid)
        assert relpath not in loaded.files
        assert idx.count() == 0

    def test_missing_manifest_treats_all_as_changed(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")
        ast = parser.parse_file(filepath)
        idx.index_ast(ast)
        assert idx.count() == 1

    def test_full_mode_rebuilds_manifest(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")
        ast = parser.parse_file(filepath)
        idx.index_ast(ast)

        manifest = IndexManifest(repo.get_manifest_path(), pid)
        manifest.set_entry("src/auth.py", "old-fake-hash", 1)
        manifest.save()

        idx.clear()
        ast2 = parser.parse_file(filepath)
        idx.index_ast(ast2)

        new_manifest = IndexManifest(repo.get_manifest_path(), pid)
        fhash2 = sha256_of_file(filepath)
        blocks2 = len(ast2.functions) + sum(len(c.methods) for c in ast2.classes)
        new_manifest.set_entry("src/auth.py", fhash2, blocks2)
        new_manifest.save()

        loaded = IndexManifest.load(repo, pid)
        assert loaded.files["src/auth.py"]["sha256"] == fhash2
        assert loaded.files["src/auth.py"]["sha256"] != "old-fake-hash"

    def test_project_id_isolation(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")

        manifest_a = IndexManifest(repo.get_manifest_path(), "project-a")
        manifest_a.set_entry("src/auth.py", sha256_of_file(filepath), 1)
        manifest_a.save()

        loaded = IndexManifest.load(repo, pid)
        assert loaded.files == {}

        manifest_correct = IndexManifest(repo.get_manifest_path(), pid)
        manifest_correct.set_entry("src/auth.py", sha256_of_file(filepath), 1)
        manifest_correct.save()

        loaded2 = IndexManifest.load(repo, pid)
        assert len(loaded2.files) == 1

    def test_remove_file_handles_batched_blocks(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        code334 = (
            "def f1():\n    pass\n\ndef f2():\n    pass\n"
            "def f3():\n    pass\n\ndef f4():\n    pass\n"
        )
        filepath = self._write_source(setup_indexer, "src/many.py", code334)
        ast = parser.parse_file(filepath)
        idx.index_ast(ast)
        assert idx.count() == 4
        idx.remove_file(str(filepath))
        assert idx.count() == 0

    def test_manifest_save_is_atomic(self, tmp_path):
        repo = FakeRepoManager(tmp_path)
        manifest = IndexManifest(repo.get_manifest_path(), "test-pid-001")
        manifest.set_entry("src/auth.py", "abc123", 5)
        manifest.save()
        assert repo.get_manifest_path().exists()
        tmp_file_path = repo.get_manifest_path().with_suffix(".json.tmp")
        assert not tmp_file_path.exists()

    def test_full_reindex_no_source_files_writes_empty_manifest(self, setup_indexer):
        """Regression: FULL reindex of a repo with zero indexable source files
        writes an empty manifest consistent with the cleared index; a subsequent
        QUICK reindex does not index anything stale and does not crash."""
        tmp_path, repo, idx, parser, pid = setup_indexer

        # --- Full mode with no matching source files ---
        idx.clear()
        manifest = IndexManifest(repo.get_manifest_path(), pid)
        # filtered_files is empty — loop body never runs
        manifest.save()

        loaded = IndexManifest.load(repo, pid)
        assert loaded.files == {}
        assert idx.count() == 0
        assert loaded.project_id == pid

        # --- Subsequent quick reindex on the same empty repo ---
        manifest2 = IndexManifest.load(repo, pid)
        assert manifest2.files == {}
        # No filtered_files to remove stale entries for, no files to index
        # Must not crash

    def test_manifest_entry_removal_by_overwrite(self, setup_indexer):
        tmp_path, repo, idx, parser, pid = setup_indexer
        filepath = self._write_source(setup_indexer, "src/auth.py", "def login():\n    pass")

        manifest = IndexManifest(repo.get_manifest_path(), pid)
        manifest.set_entry("src/auth.py", sha256_of_file(filepath), 1)
        manifest.set_entry("src/utils.py", "fakehash", 2)
        manifest.save()

        manifest.remove_entry("src/utils.py")
        assert "src/utils.py" not in manifest.files
        assert "src/auth.py" in manifest.files
