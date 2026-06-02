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
