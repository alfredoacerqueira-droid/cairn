"""Test embedding dimension mismatch handling on reindex.

Verifies that when the embedding dimension changes (e.g. user toggles
embeddings on/off or switches embed models), ChromaDB collections are
automatically rebuilt so that reindex succeeds instead of failing with
"Collection expecting embedding with dimension of X, got Y".
"""

from pipeline.ast_parser import FileAST, FunctionDef
from pipeline.indexer import VectorIndexer


class _Dim8MockOllama:
    """Mock Ollama client returning 8-dimensional embeddings."""

    embed_model = "test-dim8"

    def embed(self, text: str, model: str = None) -> list[float]:
        return [0.5] * 8

    def embed_batch(self, texts: list[str], model: str = None) -> list[list[float]]:
        return [[0.5] * 8 for _ in texts]


class _Dim4MockOllama:
    """Mock Ollama client returning 4-dimensional embeddings."""

    embed_model = "test-dim4"

    def embed(self, text: str, model: str = None) -> list[float]:
        return [0.5] * 4

    def embed_batch(self, texts: list[str], model: str = None) -> list[list[float]]:
        return [[0.5] * 4 for _ in texts]


class TestDimensionMismatch:
    """Test automatic collection rebuild on embedding dimension change."""

    def test_index_ast_proactive_check_rebuilds_collection(self, tmp_path):
        """index_ast detects dimension change before upsert and rebuilds."""
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        # Round 1: index with dim-8 embeddings
        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )

        ast1 = FileAST("a.py")
        ast1.functions.append(FunctionDef("fn_a", 1, 2, "def fn_a(): pass"))
        indexer.index_ast(ast1)
        assert indexer.count() == 1

        # Round 2: index with dim-4 embeddings to the SAME collection.
        # The proactive dimension check must detect the mismatch and rebuild
        # the collection so the upsert succeeds.
        indexer2 = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim4MockOllama(),
            embeddings_enabled=True,
        )
        assert indexer2.collection.name == indexer.collection.name

        ast2 = FileAST("b.py")
        ast2.functions.append(FunctionDef("fn_b", 1, 2, "def fn_b(): pass"))
        indexer2.index_ast(ast2)

        assert indexer2.count() == 1

    def test_index_function_fallback_rebuilds_collection(self, tmp_path):
        """index_function catches dimension error and rebuilds on retry."""
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        # Round 1: index with dim-8
        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )
        indexer.index_function("a.py", "fn_a", "pass", 1, 1)
        assert indexer.count() == 1

        # Round 2: index with dim-4 via index_function (no proactive check,
        # relies entirely on the exception-catch fallback).
        indexer2 = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim4MockOllama(),
            embeddings_enabled=True,
        )
        indexer2.index_function("b.py", "fn_b", "pass", 1, 1)
        assert indexer2.count() == 1

    def test_same_dimension_no_rebuild(self, tmp_path):
        """Same-dimension reindex is unaffected (backward compatibility)."""
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )

        ast1 = FileAST("a.py")
        ast1.functions.append(FunctionDef("fn_a", 1, 2, "def fn_a(): pass"))
        indexer.index_ast(ast1)
        assert indexer.count() == 1

        # Same dim, same collection — records accumulate, no rebuild.
        indexer2 = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )

        ast2 = FileAST("b.py")
        ast2.functions.append(FunctionDef("fn_b", 1, 2, "def fn_b(): pass"))
        indexer2.index_ast(ast2)
        assert indexer2.count() == 2

    def test_placeholder_to_dim8_transition(self, tmp_path):
        """Placeholder (dim=1) → real embeddings (dim=8) triggers rebuild."""
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        # Round 1: embeddings disabled → placeholder vectors (dim=1)
        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            embeddings_enabled=False,
        )

        ast1 = FileAST("a.py")
        ast1.functions.append(FunctionDef("fn_a", 1, 2, "def fn_a(): pass"))
        ast1.functions.append(FunctionDef("fn_b", 3, 4, "def fn_b(): pass"))
        indexer.index_ast(ast1)
        assert indexer.count() == 2

        # Round 2: embeddings enabled with dim-8 → must rebuild
        indexer2 = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )

        ast2 = FileAST("b.py")
        ast2.functions.append(FunctionDef("fn_c", 1, 2, "def fn_c(): pass"))
        indexer2.index_ast(ast2)
        assert indexer2.count() == 1

    def test_embedding_to_placeholder_transition(self, tmp_path):
        """Real embeddings (dim=8) → placeholder (dim=1) triggers rebuild."""
        chroma_path = tmp_path / ".cairn" / "chroma"
        chroma_path.mkdir(parents=True)

        # Round 1: dim-8 embeddings
        indexer = VectorIndexer(
            chroma_path=str(chroma_path),
            ollama_client=_Dim8MockOllama(),
            embeddings_enabled=True,
        )

        ast1 = FileAST("a.py")
        ast1.functions.append(FunctionDef("fn_a", 1, 2, "def fn_a(): pass"))
        indexer.index_ast(ast1)
        assert indexer.count() == 1

        # Round 2: embeddings disabled → placeholder dim=1
        indexer2 = VectorIndexer(
            chroma_path=str(chroma_path),
            embeddings_enabled=False,
        )

        ast2 = FileAST("b.py")
        ast2.functions.append(FunctionDef("fn_b", 1, 2, "def fn_b(): pass"))
        indexer2.index_ast(ast2)
        assert indexer2.count() == 1
