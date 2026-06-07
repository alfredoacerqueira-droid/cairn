"""Embedding function adapters for Cairn vector indices.

Provides PlaceholderEmbedder (embeddings OFF), FastEmbedEmbedder (local ONNX),
OllamaEmbedder (Ollama/OpenAI-compatible), and a factory function to select
the right embedder based on config.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _active_providers(model) -> list[str]:
    """Return the ACTIVE onnxruntime providers from a constructed fastembed TextEmbedding.

    Uses the real InferenceSession (model.model.model.get_providers()) rather than
    the compiled-in list, because onnxruntime-gpu can report CUDAExecutionProvider as
    available while silently falling back to CPU when CUDA libs are missing.
    Returns [] on any failure (never raises).
    """
    try:
        return model.model.model.get_providers()
    except Exception:
        return []


# Known Ollama embedding model dimensions (hint for vector-index sizing).
_OLLAMA_DIMS = {
    "nomic-embed-text": 768,
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "mxbai-embed-large": 1024,
}


class PlaceholderEmbedder:
    """Embeddings OFF. dim==1, returns a [0.0] vector per text.

    Used when embeddings are disabled (iac/shell profiles, or "none" mode).
    """

    @property
    def dim(self) -> int:
        """Embedding dimensionality (1 = no real embeddings)."""
        return 1

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "placeholder"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Return placeholder vectors (one [0.0] per text)."""
        return [[0.0] for _ in texts]


class FastEmbedEmbedder:
    """In-process ONNX embedder (fastembed).

    Mode 2: No Ollama/LLM needed. Embeddings run locally in-process.
    Lazy-loads the model on first use.
    """

    _GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider")

    def __init__(
        self, model: str = "BAAI/bge-small-en-v1.5", device: str = "auto", threads: int = 0
    ):
        """Initialize FastEmbedEmbedder.

        Args:
            model: HuggingFace model ID (default: "BAAI/bge-small-en-v1.5").
            device: "auto", "cpu", or "cuda" — fastembed execution device.
            threads: CPU intra-op threads; 0 = all cores.
        """
        self._model_name = model
        self._device = device
        self._threads = threads
        self._model = None  # Lazy-loaded
        self._active_providers: list[str] = []
        self._gpu_active: bool = False
        # Hint: only updated on first __call__ if 0
        self._dim = 384 if "bge-small" in model else 0

    @property
    def dim(self) -> int:
        """Embedding dimensionality (probed lazily if unknown)."""
        if self._dim == 0:
            self._ensure()
            # Probe model output dim on first call
            if self._model is not None:
                try:
                    sample = list(self._model.embed(["test"]))
                    if sample:
                        self._dim = len(sample[0])
                except Exception:
                    self._dim = 384  # Fallback
        return self._dim

    @property
    def name(self) -> str:
        """Human-readable name."""
        return f"fastembed:{self._model_name}"

    @property
    def gpu_active(self) -> bool:
        """Whether the loaded model is actually running on a GPU provider."""
        self._ensure()
        return self._gpu_active

    def _ensure(self):
        """Lazy-load the fastembed model, preferring GPU if available."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed is not installed. Install with: pip install fastembed")

        threads = self._threads if self._threads and self._threads > 0 else None

        try:
            import onnxruntime as ort

            avail = set(ort.get_available_providers())
        except Exception:
            avail = set()
        gpu = [p for p in self._GPU_PROVIDERS if p in avail]
        want_gpu = self._device == "cuda" or (self._device == "auto" and bool(gpu))

        if want_gpu and gpu:
            providers = [gpu[0], "CPUExecutionProvider"]
            try:
                self._model = TextEmbedding(
                    model_name=self._model_name, providers=providers, threads=threads
                )
                self._active_providers = _active_providers(self._model)
                self._gpu_active = any(p in self._active_providers for p in self._GPU_PROVIDERS)
                if self._gpu_active:
                    logger.debug("fastembed GPU active: %s", self._active_providers)
                else:
                    logger.warning(
                        "fastembed requested %s but onnxruntime fell back to CPU "
                        "(active=%s); GPU runtime libs likely missing "
                        "(need CUDA 12.x + cuDNN 9.x). Using CPU.",
                        gpu[0],
                        self._active_providers,
                    )
                return
            except Exception as e:
                logger.warning("fastembed GPU init failed (%s); falling back to CPU", e)
        elif self._device == "cuda" and not gpu:
            logger.warning(
                "embed_device=cuda but no GPU onnxruntime provider installed; "
                "install fastembed-gpu / onnxruntime-gpu — using CPU"
            )

        self._model = TextEmbedding(model_name=self._model_name, threads=threads)
        self._active_providers = _active_providers(self._model)
        self._gpu_active = False
        logger.debug("fastembed using CPU (threads=%s)", threads)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using fastembed (local ONNX).

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (numpy arrays converted to lists).
        """
        self._ensure()
        # fastembed.embed() returns a generator of numpy arrays
        embeddings = list(self._model.embed(texts))
        # Convert numpy arrays to lists
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]


class OllamaEmbedder:
    """Wraps an OllamaClient or OpenAICompatibleClient.

    Mode 1: Embeddings via Ollama or OpenAI-compatible server.
    """

    def __init__(self, client: object, model: str | None = None):
        """Initialize OllamaEmbedder.

        Args:
            client: OllamaClient or OpenAICompatibleClient instance.
            model: Optional embedding model name override.
        """
        self._client = client
        self._model = model

    @property
    def dim(self) -> int:
        """Embedding dimensionality (static hint based on model name)."""
        return _OLLAMA_DIMS.get(self._model or "", 768)

    @property
    def name(self) -> str:
        """Human-readable name."""
        return f"ollama:{self._model or 'default'}"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama/OpenAI-compatible server.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        return self._client.embed_batch(texts, model=self._model)


def make_embedder(cfg: object) -> object:
    """Pick the embedder from config.

    Config hierarchy:
      1. embedder == "none"      -> PlaceholderEmbedder
      2. embedder == "fastembed" -> FastEmbedEmbedder
      3. local_llm.enabled       -> OllamaEmbedder
      4. else                    -> PlaceholderEmbedder

    Args:
        cfg: Config object (from core.config) with .local_llm attribute.

    Returns:
        An EmbeddingFn instance.
    """
    # Get embedder setting from local_llm config
    embedder = getattr(cfg.local_llm, "embedder", "ollama")

    # Mode: embeddings OFF
    if embedder == "none":
        return PlaceholderEmbedder()

    # Mode 2: fastembed (in-process ONNX, no Ollama needed)
    if embedder == "fastembed":
        model = getattr(cfg.local_llm, "fastembed_model", "BAAI/bge-small-en-v1.5")
        device = getattr(cfg.local_llm, "embed_device", "auto")
        threads = getattr(cfg.local_llm, "embed_threads", 0)
        return FastEmbedEmbedder(model, device=device, threads=threads)

    # Mode 1: Ollama/OpenAI-compatible (if enabled)
    enabled = getattr(cfg.local_llm, "enabled", False)
    if enabled:
        # Lazy import to avoid hard dependency
        from server.ollama_client import make_llm_client

        client = make_llm_client(cfg.local_llm)
        embed_model = getattr(cfg.local_llm, "embed_model", None)
        return OllamaEmbedder(client, embed_model)

    # Fallback: embeddings OFF
    return PlaceholderEmbedder()
