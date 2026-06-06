"""Test ChromaDB batch size limits and chunked upsert handling.

Verifies that:
1. Oversized batches (>5461 items) are automatically chunked
2. No records are dropped when batch size is exceeded
3. Large YAML/CRD files parse and index correctly without ValueError
"""

from pathlib import Path

import pytest

from pipeline.ast_parser import FileAST, FunctionDef
from pipeline.indexer import VectorIndexer
from tests.fixtures.builders import make_k8s_repo
from tests.fixtures.harness import fresh_index


class MockOllamaClient:
    """Mock Ollama client for testing without embeddings."""

    embed_model = "nomic-embed-text"

    def embed(self, text: str, model: str = None) -> list[float]:
        return [0.5] * 16

    def embed_batch(self, texts: list[str], model: str = None) -> list[list[float]]:
        return [[0.5] * 16 for _ in texts]


class TestBatchSizeLimits:
    """Test oversized batch handling in VectorIndexer."""

    def test_synthetic_6000_functions_no_drop(self, tmp_path):
        """Test that 6000+ functions are indexed without being dropped.

        Constructs a synthetic AST with 6000 functions (well above ChromaDB's
        5461 limit) and verifies all records appear in the index.
        """
        # Create a temporary chroma path
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=MockOllamaClient(),
            embeddings_enabled=False,
        )

        # Build a synthetic AST with 6000 functions
        num_functions = 6000
        ast_result = FileAST("synthetic.py")

        for i in range(num_functions):
            func = FunctionDef(
                name=f"function_{i:05d}",
                line_start=i * 2 + 1,
                line_end=i * 2 + 2,
                code=f"def function_{i:05d}():\n    pass",
            )
            ast_result.functions.append(func)

        # Index the oversized batch
        indexer.index_ast(ast_result)

        # Verify all records were indexed
        count = indexer.count()
        assert count == num_functions, (
            f"Expected {num_functions} indexed records, got {count} "
            f"(batch splitting failed or records were dropped)"
        )

    def test_large_crd_yaml_repo_indexes_without_error(self, tmp_path):
        """Test that large CRD YAML (500+ documents) indexes without ValueError.

        Uses the pathological K8s repo fixture with large YAML files and verifies:
        1. No ValueError: Batch size of N is greater than max batch size of 5461
        2. Index is populated (count > 0)
        3. Re-indexing is idempotent
        """
        # Build a K8s repo with large, pathological YAML files
        repo_root = make_k8s_repo(tmp_path, with_pathological=True)

        # Index the repo (this would fail before batch splitting)
        try:
            fresh_index(repo_root, embeddings=False)
        except ValueError as e:
            if "Batch size" in str(e) and "max batch size" in str(e):
                pytest.fail(f"Batch splitting failed: got ValueError on large CRD: {e}")
            raise

        # Verify the index is populated
        repo_root_resolved = Path(repo_root).resolve()
        chroma_path = repo_root_resolved / ".cairn" / "chroma"
        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=MockOllamaClient(),
            embeddings_enabled=False,
        )
        count = indexer.count()
        assert count > 0, "Large YAML repo indexed but resulted in empty index"

        # Verify idempotency: re-index and confirm count is stable
        fresh_index(repo_root, embeddings=False)
        count_after_reindex = indexer.count()
        assert count_after_reindex == count, (
            f"Reindex changed count from {count} to {count_after_reindex} "
            f"(idempotency violated)"
        )

    def test_mixed_functions_and_methods_large_batch(self, tmp_path):
        """Test batch splitting with both top-level functions and class methods.

        Ensures that the batching logic handles both functions and methods
        correctly without dropping either type.
        """
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=MockOllamaClient(),
            embeddings_enabled=False,
        )

        # Build AST with 3000 functions and 3000 methods across 100 classes
        num_functions = 3000
        num_classes = 100
        methods_per_class = 30

        ast_result = FileAST("mixed.py")

        # Add standalone functions
        for i in range(num_functions):
            func = FunctionDef(
                name=f"func_{i:05d}",
                line_start=i * 2 + 1,
                line_end=i * 2 + 2,
                code=f"def func_{i:05d}():\n    pass",
            )
            ast_result.functions.append(func)

        # Add classes with methods
        from pipeline.ast_parser import ClassDef

        for c in range(num_classes):
            cls = ClassDef(
                name=f"Class{c:03d}",
                line_start=num_functions * 2 + c * 100 + 1,
                line_end=num_functions * 2 + c * 100 + 50,
                code=f"class Class{c:03d}:\n    pass",
            )
            for m in range(methods_per_class):
                method = FunctionDef(
                    name=f"method_{m:02d}",
                    line_start=num_functions * 2 + c * 100 + m * 2 + 1,
                    line_end=num_functions * 2 + c * 100 + m * 2 + 2,
                    code=f"    def method_{m:02d}(self):\n        pass",
                )
                cls.methods.append(method)
            ast_result.classes.append(cls)

        # Index the large AST
        indexer.index_ast(ast_result)

        # Verify all records indexed
        expected_count = num_functions + (num_classes * methods_per_class)
        count = indexer.count()
        assert count == expected_count, (
            f"Expected {expected_count} records (functions + methods), " f"got {count}"
        )
