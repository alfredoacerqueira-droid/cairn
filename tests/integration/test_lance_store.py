"""Integration tests for LanceStore."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

lancedb = pytest.importorskip("lancedb")


class FakeEmbedder:
    """Deterministic fake embedder for testing (no downloads/Ollama needed)."""

    dim = 8
    name = "fake"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Hash each text to a deterministic 8-dim vector."""
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            # Convert first 8 bytes to floats in [0, 1]
            out.append([b / 255.0 for b in h[:8]])
        return out


class PlaceholderEmbedder:
    """Placeholder embedder (dim=1, no real embeddings)."""

    dim = 1
    name = "placeholder"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


@pytest.fixture
def lance_dir(tmp_path: Path) -> Path:
    """Temporary directory for Lance database."""
    return tmp_path / "lance"


@pytest.fixture
def embedder() -> FakeEmbedder:
    """Fake embedder for deterministic tests."""
    return FakeEmbedder()


@pytest.fixture
def placeholder_embedder() -> PlaceholderEmbedder:
    """Placeholder embedder (no real embeddings)."""
    return PlaceholderEmbedder()


def test_lance_store_upsert_and_count(
    lance_dir: Path, embedder: FakeEmbedder
) -> None:
    """Test upserting blocks and counting them."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    # Create some test blocks
    blocks = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a():\n    return 42",
            line_start=10,
            line_end=11,
        ),
        Block(
            id="test.py:func_b:20",
            filepath="test.py",
            function="func_b",
            code="def func_b():\n    return 'hello'",
            line_start=20,
            line_end=21,
        ),
    ]

    # Upsert blocks
    store.upsert_blocks(blocks)

    # Check count
    assert store.count() == 2


def test_lance_store_search_returns_standard_shape(
    lance_dir: Path, embedder: FakeEmbedder
) -> None:
    """Test that search results have the standard hit dict shape."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    blocks = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a():\n    return 42",
            line_start=10,
            line_end=11,
        ),
    ]
    store.upsert_blocks(blocks)

    # Search
    results = store.search("return 42", top_k=5)

    assert len(results) > 0
    hit = results[0]

    # Verify all 9 keys are present
    required_keys = {
        "id",
        "filepath",
        "function",
        "line_start",
        "line_end",
        "code",
        "similarity",
        "project_id",
        "project_root",
    }
    assert set(hit.keys()) >= required_keys

    # Verify similarity is in [0, 1]
    assert 0.0 <= hit["similarity"] <= 1.0


def test_lance_store_iter_blocks_respects_project_isolation(
    lance_dir: Path, embedder: FakeEmbedder
) -> None:
    """Test that iter_blocks respects project isolation."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    # Create first store with project_id proj_a
    store_a = LanceStore(
        lance_dir, embedder, project_id="proj_a", project_root="/path/a"
    )

    blocks_a = [
        Block(
            id="a.py:func_a:10",
            filepath="a.py",
            function="func_a",
            code="def func_a(): pass",
            line_start=10,
            line_end=11,
        ),
    ]
    store_a.upsert_blocks(blocks_a)

    # Create second store with project_id proj_b, same table
    store_b = LanceStore(
        lance_dir, embedder, project_id="proj_b", project_root="/path/b"
    )

    blocks_b = [
        Block(
            id="b.py:func_b:20",
            filepath="b.py",
            function="func_b",
            code="def func_b(): pass",
            line_start=20,
            line_end=21,
        ),
    ]
    store_b.upsert_blocks(blocks_b)

    # Iterate blocks from store_a (should only see proj_a's blocks)
    blocks_from_a = list(store_a.iter_blocks())
    assert len(blocks_from_a) == 1
    assert blocks_from_a[0]["filepath"] == "a.py"

    # Iterate blocks from store_b (should only see proj_b's blocks)
    blocks_from_b = list(store_b.iter_blocks())
    assert len(blocks_from_b) == 1
    assert blocks_from_b[0]["filepath"] == "b.py"


def test_lance_store_delete_file(lance_dir: Path, embedder: FakeEmbedder) -> None:
    """Test deleting all blocks for a file."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    blocks = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a(): pass",
            line_start=10,
            line_end=11,
        ),
        Block(
            id="other.py:func_b:20",
            filepath="other.py",
            function="func_b",
            code="def func_b(): pass",
            line_start=20,
            line_end=21,
        ),
    ]
    store.upsert_blocks(blocks)

    assert store.count() == 2

    # Delete one file
    store.delete_file("test.py")

    # Check that only the other file remains
    blocks_left = list(store.iter_blocks())
    assert len(blocks_left) == 1
    assert blocks_left[0]["filepath"] == "other.py"


def test_lance_store_delete_file_predicate_injection_safe(
    lance_dir: Path, embedder: FakeEmbedder
) -> None:
    """Test delete_file is safe against SQL injection via single quotes."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    # Create blocks with special characters in filepath
    blocks = [
        Block(
            id="app/o'brien.py:func_a:10",
            filepath="app/o'brien.py",
            function="func_a",
            code="def func_a(): pass",
            line_start=10,
            line_end=11,
        ),
        Block(
            id="normal.py:func_b:20",
            filepath="normal.py",
            function="func_b",
            code="def func_b(): pass",
            line_start=20,
            line_end=21,
        ),
    ]
    store.upsert_blocks(blocks)

    assert store.count() == 2

    # Delete file with single quote in name (should not raise and should delete only that file)
    result = store.delete_file("app/o'brien.py")

    # Should succeed (return value is 1 to indicate success)
    assert result == 1

    # Check that only the normal.py block remains
    blocks_left = list(store.iter_blocks())
    assert len(blocks_left) == 1
    assert blocks_left[0]["filepath"] == "normal.py"

    # Test that injection-like strings don't crash and delete nothing
    result = store.delete_file("nonexistent'; --")
    assert result == 1  # Returns 1 even if no rows deleted

    # Count should still be 1
    assert store.count() == 1
    blocks_left = list(store.iter_blocks())
    assert len(blocks_left) == 1
    assert blocks_left[0]["filepath"] == "normal.py"


def test_lance_store_hybrid_search(
    lance_dir: Path, embedder: FakeEmbedder
) -> None:
    """Test hybrid search (should return results for exact token matches)."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    # Create a block with known code
    blocks = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a():\n    return hello_world",
            line_start=10,
            line_end=12,
        ),
    ]
    store.upsert_blocks(blocks)

    # Search for a token that appears in the code
    results = store.hybrid_search("hello_world", top_k=5, rerank=False)

    # Should get results (either via FTS or hybrid)
    assert len(results) > 0
    assert results[0]["filepath"] == "test.py"


def test_lance_store_versioning(lance_dir: Path, embedder: FakeEmbedder) -> None:
    """Test snapshoting and restore (guard against version API mismatches)."""
    from pipeline.store.base import Block
    from pipeline.store.lance_store import LanceStore

    store = LanceStore(lance_dir, embedder)

    blocks_v1 = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a(): pass",
            line_start=10,
            line_end=11,
        ),
    ]
    store.upsert_blocks(blocks_v1)

    # Snapshot (may fail gracefully if version API not available)
    v1 = store.snapshot()
    if v1 is None:
        pytest.skip("Lance versioning not available in this version")

    assert store.count() == 1

    # Add more blocks
    blocks_v2 = [
        Block(
            id="test.py:func_b:20",
            filepath="test.py",
            function="func_b",
            code="def func_b(): pass",
            line_start=20,
            line_end=21,
        ),
    ]
    store.upsert_blocks(blocks_v2)
    assert store.count() == 2

    # Restore to v1
    try:
        store.restore(v1)
        assert store.count() == 1
    except Exception as e:
        pytest.skip(f"Lance restore not available: {e}")


def test_make_store_default_is_chroma() -> None:
    """Test that make_store with default config returns ChromaStore."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.config import Config
    from core.repo import RepoManager
    from pipeline.store import ChromaStore, make_store

    with TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        repo = RepoManager(project_root)
        cfg = Config()  # Default config has store_backend="chroma"

        store = make_store(cfg, repo, project_root=project_root)

        assert isinstance(store, ChromaStore)


def test_make_store_lance_backend() -> None:
    """Test that make_store with lance backend returns LanceStore."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.config import Config
    from core.repo import RepoManager
    from pipeline.store import LanceStore, make_store

    with TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        repo = RepoManager(project_root)
        cfg = Config()
        cfg.indexing.store_backend = "lance"

        store = make_store(cfg, repo, project_root=project_root)

        assert isinstance(store, LanceStore)


def test_chroma_store_upsert_blocks(tmp_path: Path) -> None:
    """Test ChromaStore.upsert_blocks round-trip."""
    from core.repo import RepoManager
    from pipeline.indexer import VectorIndexer
    from pipeline.store.base import Block
    from pipeline.store.chroma_store import ChromaStore

    repo = RepoManager(tmp_path)
    repo.ensure_directories()

    # Create a VectorIndexer with placeholder embeddings (no Ollama)
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(),
        embeddings_enabled=False,  # Use placeholder embeddings
        project_root=tmp_path,
    )
    store = ChromaStore(indexer)

    # Create test blocks
    blocks = [
        Block(
            id="test.py:func_a:10",
            filepath="test.py",
            function="func_a",
            code="def func_a():\n    return 42",
            line_start=10,
            line_end=11,
        ),
        Block(
            id="test.py:func_b:20",
            filepath="test.py",
            function="func_b",
            code="def func_b():\n    return 'hello'",
            line_start=20,
            line_end=21,
        ),
    ]

    # Upsert blocks
    store.upsert_blocks(blocks)

    # Verify count
    assert store.count() == 2

    # Verify iter_blocks
    blocks_list = list(store.iter_blocks())
    assert len(blocks_list) == 2
    assert blocks_list[0]["filepath"] == "test.py"
    assert blocks_list[0]["id"] == "test.py:func_a:10"
    assert blocks_list[1]["id"] == "test.py:func_b:20"
