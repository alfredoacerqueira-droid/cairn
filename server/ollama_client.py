"""Ollama API client for embeddings and text generation — env+config driven."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import httpx


def _default_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _default_embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _default_generate_model() -> str:
    # Local worker model for the cheap jobs (memory summarization, optional
    # validation/LLM-rerank). Configurable via OLLAMA_GENERATE_MODEL.
    # Default qwen2.5-coder:1.5b — MEASURED warm: 0.6s/gen vs 3b's 1.2s, at half
    # the VRAM, which reduces model-swap eviction on a 6GB GPU (the real
    # bottleneck: embedder+worker thrash, not generation speed). gemma4:e2b was
    # ~39s here (7.2GB → CPU offload). On big-VRAM/Apple Silicon, 3b/7b are fine.
    return os.environ.get("OLLAMA_GENERATE_MODEL", "qwen2.5-coder:1.5b")


def _worker_num_ctx() -> int:
    # Cap the worker model's context so the KV cache can't spill into system RAM
    # (KV scales linearly with context). The worker's jobs — summarize a diff,
    # score a block — are short, so 4096 is ample. Override via OLLAMA_NUM_CTX.
    try:
        return int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
    except ValueError:
        return 4096


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        embed_model: str | None = None,
        generate_model: str | None = None,
    ):
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.embed_model = embed_model or _default_embed_model()
        self.generate_model = generate_model or _default_generate_model()

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate embedding vector for text."""
        m = model or self.embed_model
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": m, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(
        self, texts: list[str], model: str | None = None, workers: int = 4
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in parallel."""
        m = model or self.embed_model
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.embed, text, m) for text in texts]
            return [f.result() for f in futures]

    def generate(self, prompt: str, model: str | None = None, stream: bool = False) -> str:
        """Generate text completion."""
        m = model or self.generate_model
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": m,
                "prompt": prompt,
                "stream": stream,
                "options": {"num_ctx": _worker_num_ctx()},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["response"]

    async def aembed(self, text: str, model: str | None = None) -> list[float]:
        """Async embedding generation."""
        m = model or self.embed_model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": m, "prompt": text},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def agenerate(self, prompt: str, model: str | None = None) -> str:
        """Async text generation."""
        m = model or self.generate_model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": m,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": _worker_num_ctx()},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["response"]

    def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def pull_model(self, model: str) -> bool:
        """Pull (download) a model from Ollama registry.

        Args:
            model: Model name (e.g. "nomic-embed-text", "qwen2.5-coder:3b")

        Returns:
            True if the model was successfully pulled, False otherwise.
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600.0,
            )
            return response.status_code == 200
        except Exception:
            return False


class OpenAICompatibleClient:
    """OpenAI-compatible local LLM client (LM Studio, llama.cpp, Jan, etc.).

    Mirrors the OllamaClient interface but uses OpenAI-style REST endpoints.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        embed_model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        # For OpenAI-compatible servers, we typically use the same model for both
        # embedding and generation, unless explicitly specified
        self.embed_model = embed_model or model or "model"
        self.generate_model = model or "model"

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate embedding vector for text."""
        m = model or self.embed_model
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": m, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_batch(
        self, texts: list[str], model: str | None = None, workers: int = 4
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in parallel."""
        m = model or self.embed_model
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.embed, text, m) for text in texts]
            return [f.result() for f in futures]

    def generate(self, prompt: str, model: str | None = None, stream: bool = False) -> str:
        """Generate text completion."""
        m = model or self.generate_model
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def aembed(self, text: str, model: str | None = None) -> list[float]:
        """Async embedding generation."""
        m = model or self.embed_model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": m, "input": text},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def agenerate(self, prompt: str, model: str | None = None) -> str:
        """Async text generation."""
        m = model or self.generate_model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": m,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        """Check if the OpenAI-compatible server is running."""
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        response = httpx.get(f"{self.base_url}/v1/models", timeout=10.0)
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def pull_model(self, model: str) -> bool:
        """Pull (download) a model — not supported by OpenAI-compatible servers.

        Returns False to indicate this is not available.
        """
        return False


def make_llm_client(
    local_llm: object | None = None,
) -> OllamaClient | OpenAICompatibleClient:
    """Factory function to create an LLM client based on config.

    Args:
        local_llm: LocalLLMConfig object (from core.config), or None to use defaults.

    Returns:
        OllamaClient if disabled or None, OpenAICompatibleClient if enabled with
        backend="openai_compatible", or OllamaClient otherwise.
    """
    # If local_llm is None or not enabled, return a basic OllamaClient.
    # (It won't be used when embeddings are gated via embeddings_enabled=False.)
    if local_llm is None:
        return OllamaClient()

    # Get attributes safely, handling both dict-like and object-like access
    enabled = getattr(local_llm, "enabled", None) or (
        local_llm.get("enabled") if isinstance(local_llm, dict) else False
    )

    if not enabled:
        return OllamaClient()

    backend = getattr(local_llm, "backend", None) or (
        local_llm.get("backend") if isinstance(local_llm, dict) else "ollama"
    )
    base_url = getattr(local_llm, "base_url", None) or (
        local_llm.get("base_url") if isinstance(local_llm, dict) else None
    )
    model = getattr(local_llm, "model", None) or (
        local_llm.get("model") if isinstance(local_llm, dict) else None
    )
    embed_model = getattr(local_llm, "embed_model", None) or (
        local_llm.get("embed_model") if isinstance(local_llm, dict) else None
    )

    if backend == "openai_compatible":
        if not base_url:
            raise ValueError(
                "OpenAI-compatible backend requires base_url to be set in local_llm config"
            )
        return OpenAICompatibleClient(base_url=base_url, model=model, embed_model=embed_model)
    else:
        # Ollama backend (default)
        return OllamaClient(base_url=base_url, embed_model=embed_model, generate_model=model)
