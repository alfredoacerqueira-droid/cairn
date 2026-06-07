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

    def test_device_cpu_builds_and_embeds(self):
        """device='cpu' builds and embeds, returns dim-384 vector."""
        embedder = FastEmbedEmbedder(device="cpu")
        try:
            result = embedder(["test"])
            assert len(result) == 1
            assert len(result[0]) == 384
        except ImportError as e:
            assert "fastembed" in str(e)

    def test_device_auto_cuda_providers(self, monkeypatch):
        """device='auto' with CUDA available -> passes GPU providers."""
        recorded_kwargs = {}

        class FakeTE:
            def __init__(self, model_name=None, providers=None, threads=None):
                recorded_kwargs["providers"] = providers
                recorded_kwargs["threads"] = threads

            def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr("fastembed.TextEmbedding", FakeTE)
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        embedder = FastEmbedEmbedder(device="auto")
        embedder._ensure()

        assert recorded_kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_gpu_init_fails_falls_back_to_cpu(self, monkeypatch):
        """GPU provider init raises -> falls back to CPU, no exception."""

        class FakeTE:
            def __init__(self, model_name=None, providers=None, threads=None):
                if providers and any(p in str(providers) for p in ["CUDA", "ROCM", "CoreML"]):
                    raise RuntimeError("GPU not available")
                self._providers = providers
                self._threads = threads

            def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr("fastembed.TextEmbedding", FakeTE)
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        embedder = FastEmbedEmbedder(device="auto")
        # Should not raise
        embedder._ensure()
        assert embedder._model is not None
        # Should have fallen back to CPU (no GPU providers in the constructed FakeTE)
        assert embedder._model._providers is None or "CUDAExecutionProvider" not in str(
            embedder._model._providers
        )

    def test_device_cuda_no_provider_falls_back(self, monkeypatch, caplog):
        """device='cuda' but no CUDA provider -> falls back to CPU, logs warning."""
        recorded_kwargs = {}

        class FakeTE:
            def __init__(self, model_name=None, providers=None, threads=None):
                recorded_kwargs["providers"] = providers
                recorded_kwargs["threads"] = threads

            def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr("fastembed.TextEmbedding", FakeTE)
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CPUExecutionProvider"],
        )

        import logging

        caplog.set_level(logging.WARNING)

        embedder = FastEmbedEmbedder(device="cuda")
        embedder._ensure()

        assert embedder._model is not None
        assert recorded_kwargs["providers"] is None  # CPU construction, no providers
        assert "embed_device=cuda" in caplog.text

    def test_threads_passed_to_cpu_path(self, monkeypatch):
        """threads=4 -> TextEmbedding receives threads=4 on CPU path."""
        recorded_threads = []

        class FakeTE:
            def __init__(self, model_name=None, providers=None, threads=None):
                recorded_threads.append(threads)

            def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr("fastembed.TextEmbedding", FakeTE)
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CPUExecutionProvider"],
        )

        embedder = FastEmbedEmbedder(threads=4)
        embedder._ensure()
        assert recorded_threads[0] == 4

    def test_threads_zero_passes_none(self, monkeypatch):
        """threads=0 -> TextEmbedding receives threads=None (fastembed default)."""
        recorded_threads = []

        class FakeTE:
            def __init__(self, model_name=None, providers=None, threads=None):
                recorded_threads.append(threads)

            def embed(self, texts):
                return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr("fastembed.TextEmbedding", FakeTE)
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["CPUExecutionProvider"],
        )

        embedder = FastEmbedEmbedder(threads=0)
        embedder._ensure()
        assert recorded_threads[0] is None

    def test_make_embedder_passes_device_and_threads(self):
        """make_embedder for fastembed passes device and threads from config."""
        cfg = Config()
        cfg.local_llm.embedder = "fastembed"
        cfg.local_llm.embed_device = "cpu"
        cfg.local_llm.embed_threads = 8

        embedder = make_embedder(cfg)
        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder._device == "cpu"
        assert embedder._threads == 8


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
        """blocks_from_ast extracts class definition and its method."""
        parser = ASTParser()
        ast = parser.parse_string(
            """class Calculator:
    def add(self, a, b):
        return a + b
""",
            "calc.py",
        )
        blocks = blocks_from_ast(ast)

        assert len(blocks) == 2

        class_blocks = [b for b in blocks if b.function == "Calculator"]
        assert len(class_blocks) == 1
        assert class_blocks[0].id == f"calc.py:Calculator:{class_blocks[0].line_start}"

        method_blocks = [b for b in blocks if "." in b.function]
        assert len(method_blocks) == 1
        assert method_blocks[0].function == "Calculator.add"
        assert method_blocks[0].id == f"calc.py:Calculator.add:{method_blocks[0].line_start}"

    def test_blocks_function_and_class(self):
        """blocks_from_ast extracts top-level functions, class definitions, and class methods."""
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

        # Should have 1 function + 1 class definition + 1 method = 3 blocks
        assert len(blocks) == 3
        funcs = [b for b in blocks if "." not in b.function]
        methods = [b for b in blocks if "." in b.function]
        assert len(funcs) == 2
        assert len(methods) == 1
        assert funcs[0].function == "outer"
        assert funcs[1].function == "MyClass"
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
