"""Unit tests for pipeline.store.embedders and blocks_from_ast."""

from core.config import Config
from pipeline.ast_parser import ASTParser
from pipeline.store import (
    EmbeddingFn,
    FastEmbedEmbedder,
    OllamaEmbedder,
    PlaceholderEmbedder,
    blocks_from_ast,
    make_embedder,
)


class TestPlaceholderEmbedder:
    """Tests for PlaceholderEmbedder."""

    def test_dim(self):
        """Placeholder embedder has dim == 1."""
        embedder = PlaceholderEmbedder()
        assert embedder.dim == 1

    def test_name(self):
        """Placeholder embedder has readable name."""
        embedder = PlaceholderEmbedder()
        assert embedder.name == "placeholder"

    def test_call(self):
        """Placeholder embedder returns [0.0] vectors."""
        embedder = PlaceholderEmbedder()
        result = embedder(["hello", "world"])
        assert result == [[0.0], [0.0]]

    def test_embedding_fn_protocol(self):
        """PlaceholderEmbedder conforms to EmbeddingFn protocol."""
        embedder = PlaceholderEmbedder()
        assert isinstance(embedder, EmbeddingFn)


class TestFastEmbedEmbedder:
    """Tests for FastEmbedEmbedder."""

    def test_init_default_model(self):
        """FastEmbedEmbedder initializes with default model."""
        embedder = FastEmbedEmbedder()
        assert embedder._model_name == "BAAI/bge-small-en-v1.5"
        assert embedder._dim == 384  # Known dim for bge-small

    def test_init_custom_model(self):
        """FastEmbedEmbedder initializes with custom model."""
        embedder = FastEmbedEmbedder("some-other-model")
        assert embedder._model_name == "some-other-model"

    def test_name(self):
        """FastEmbedEmbedder name includes model."""
        embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
        assert embedder.name == "fastembed:BAAI/bge-small-en-v1.5"

    def test_dim_bge_small(self):
        """FastEmbedEmbedder returns known dim for bge-small."""
        embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
        assert embedder.dim == 384

    def test_embedding_fn_protocol(self):
        """FastEmbedEmbedder conforms to EmbeddingFn protocol."""
        embedder = FastEmbedEmbedder()
        assert isinstance(embedder, EmbeddingFn)

    def test_import_error_on_call(self):
        """FastEmbedEmbedder raises ImportError if fastembed not installed.

        Note: this test assumes fastembed is NOT installed (true in minimal envs).
        If fastembed IS installed, the test will actually call the embedder
        and should still pass. This test is a smoke test to ensure the error
        path is correct if the library is missing.
        """
        embedder = FastEmbedEmbedder()
        try:
            # Try to call it; if fastembed is installed, this succeeds
            # and the test verifies the call works. If not, ImportError is raised.
            result = embedder(["test"])
            assert len(result) == 1
            assert isinstance(result[0], list)
        except ImportError as e:
            assert "fastembed" in str(e)


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder."""

    def test_init(self):
        """OllamaEmbedder initializes with client and optional model."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, "nomic-embed-text")
        assert embedder._client is mock_client
        assert embedder._model == "nomic-embed-text"

    def test_name(self):
        """OllamaEmbedder name includes model."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, "nomic-embed-text")
        assert embedder.name == "ollama:nomic-embed-text"

    def test_name_default(self):
        """OllamaEmbedder name says 'default' if no model."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, None)
        assert embedder.name == "ollama:default"

    def test_dim(self):
        """OllamaEmbedder returns known dim for Ollama models."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, "nomic-embed-text")
        assert embedder.dim == 768

    def test_dim_fallback(self):
        """OllamaEmbedder defaults to 768 for unknown models."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, "unknown-model")
        assert embedder.dim == 768

    def test_embedding_fn_protocol(self):
        """OllamaEmbedder conforms to EmbeddingFn protocol."""
        mock_client = object()
        embedder = OllamaEmbedder(mock_client, "test")
        assert isinstance(embedder, EmbeddingFn)


class TestMakeEmbedder:
    """Tests for make_embedder factory."""

    def test_default_config_returns_placeholder(self):
        """Default Config (embeddings via ollama, but disabled) -> PlaceholderEmbedder."""
        cfg = Config()
        # Default: local_llm.enabled = False, embedder = "ollama"
        embedder = make_embedder(cfg)
        assert isinstance(embedder, PlaceholderEmbedder)

    def test_embedder_none_returns_placeholder(self):
        """Config with embedder='none' -> PlaceholderEmbedder."""
        cfg = Config()
        cfg.local_llm.embedder = "none"
        embedder = make_embedder(cfg)
        assert isinstance(embedder, PlaceholderEmbedder)

    def test_embedder_fastembed_returns_fastembed(self):
        """Config with embedder='fastembed' -> FastEmbedEmbedder."""
        cfg = Config()
        cfg.local_llm.embedder = "fastembed"
        embedder = make_embedder(cfg)
        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder.name.startswith("fastembed:")
        assert embedder.dim == 384

    def test_embedder_fastembed_custom_model(self):
        """Config with fastembed and custom model."""
        cfg = Config()
        cfg.local_llm.embedder = "fastembed"
        cfg.local_llm.fastembed_model = "some-custom-model"
        embedder = make_embedder(cfg)
        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder.name == "fastembed:some-custom-model"

    def test_local_llm_enabled_returns_ollama(self):
        """Config with local_llm.enabled=True -> OllamaEmbedder."""
        cfg = Config()
        cfg.local_llm.enabled = True
        cfg.local_llm.embed_model = "nomic-embed-text"
        embedder = make_embedder(cfg)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.name.startswith("ollama:")
        assert embedder.dim > 1

    def test_local_llm_enabled_no_embed_model(self):
        """Config with local_llm.enabled=True but no explicit embed_model."""
        cfg = Config()
        cfg.local_llm.enabled = True
        cfg.local_llm.embed_model = None
        embedder = make_embedder(cfg)
        assert isinstance(embedder, OllamaEmbedder)
        # Client is constructed but we don't call embed_batch (would require Ollama)


class TestBlocksFromAST:
    """Tests for blocks_from_ast."""

    def test_blocks_from_simple_function(self):
        """blocks_from_ast extracts a single top-level function."""
        parser = ASTParser()
        ast = parser.parse_string(
            """def greet(name):
    return f"Hello, {name}!"
""",
            "test.py",
        )
        blocks = blocks_from_ast(ast)

        assert len(blocks) == 1
        assert blocks[0].filepath == "test.py"
        assert blocks[0].function == "greet"
        assert blocks[0].id == "test.py:greet:1"
        assert blocks[0].line_start == 1
        assert "Hello" in blocks[0].code

    def test_blocks_from_class_with_method(self):
        """blocks_from_ast extracts class and its method."""
        parser = ASTParser()
        ast = parser.parse_string(
            """class Calculator:
    def add(self, a, b):
        return a + b
""",
            "calc.py",
        )
        blocks = blocks_from_ast(ast)

        # Should have 1 method (classes themselves are not indexed)
        methods = [b for b in blocks if "." in b.function]
        assert len(methods) == 1
        assert methods[0].function == "Calculator.add"
        assert methods[0].id == f"calc.py:Calculator.add:{methods[0].line_start}"

    def test_blocks_function_and_class(self):
        """blocks_from_ast extracts both top-level functions and class methods."""
        parser = ASTParser()
        ast = parser.parse_string(
            """def outer():
    return 1

class MyClass:
    def inner(self):
        return 2
""",
            "module.py",
        )
        blocks = blocks_from_ast(ast)

        # Should have 1 function + 1 method = 2 blocks
        assert len(blocks) == 2
        funcs = [b for b in blocks if "." not in b.function]
        methods = [b for b in blocks if "." in b.function]
        assert len(funcs) == 1
        assert len(methods) == 1
        assert funcs[0].function == "outer"
        assert methods[0].function == "MyClass.inner"

    def test_block_id_format(self):
        """Block IDs follow the format filepath:function:line_start."""
        parser = ASTParser()
        ast = parser.parse_string(
            """def test_func():
    pass
""",
            "myfile.py",
        )
        blocks = blocks_from_ast(ast)
        assert len(blocks) == 1
        block = blocks[0]
        # ID should be "myfile.py:test_func:<line_start>"
        assert block.id.startswith("myfile.py:test_func:")
        parts = block.id.split(":")
        assert len(parts) == 3
        assert parts[0] == "myfile.py"
        assert parts[1] == "test_func"
        assert int(parts[2]) > 0

    def test_block_fields(self):
        """Block has all required fields."""
        parser = ASTParser()
        ast = parser.parse_string(
            """def sample():
    return 42
""",
            "test.py",
        )
        blocks = blocks_from_ast(ast)
        assert len(blocks) == 1
        block = blocks[0]

        assert hasattr(block, "id")
        assert hasattr(block, "filepath")
        assert hasattr(block, "function")
        assert hasattr(block, "code")
        assert hasattr(block, "line_start")
        assert hasattr(block, "line_end")

        assert block.filepath == "test.py"
        assert "sample" in block.function
        assert "42" in block.code
        assert block.line_start > 0
        assert block.line_end >= block.line_start
