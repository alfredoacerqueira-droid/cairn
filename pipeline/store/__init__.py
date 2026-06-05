"""Storage abstraction layer for Cairn vector indices.

This package provides the IndexStore protocol and embedder implementations
that ChromaStore and LanceStore will implement.
"""

from pathlib import Path

from core.repo import RepoManager, project_id
from pipeline.store.base import Block, EmbeddingFn, IndexStore, blocks_from_ast
from pipeline.store.chroma_store import ChromaStore
from pipeline.store.embedders import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    PlaceholderEmbedder,
    make_embedder,
)
from pipeline.store.lance_store import LanceStore


def make_store(cfg: object, repo: RepoManager, *, project_root: str | Path) -> IndexStore:
    """Build an IndexStore from config.

    Selects the backend by cfg.indexing.store_backend:
    - "chroma" (default): wraps a VectorIndexer in ChromaStore
    - "lance": builds a LanceStore backed by Lance

    Args:
        cfg: Config object (from core.config) with .indexing.store_backend.
        repo: RepoManager instance for data_dir access.
        project_root: Project root path (for VectorIndexer + LanceStore).

    Returns:
        An IndexStore implementation (ChromaStore or LanceStore).
    """
    from pipeline.indexer import VectorIndexer
    from server.ollama_client import make_llm_client

    # Determine embeddings flag: enabled if both config and local LLM are enabled
    emb_enabled = cfg.embeddings_enabled and cfg.local_llm.enabled

    # Get the backend choice
    backend = getattr(cfg.indexing, "store_backend", "chroma")

    # Build embedder (same for both backends)
    embedder = make_embedder(cfg)

    # Resolve project_id
    pid = project_id(project_root)

    if backend == "lance":
        # Build LanceStore
        return LanceStore(
            repo.get_lance_path(),
            embedder,
            project_id=pid,
            project_root=str(project_root),
        )
    else:
        # Default: build ChromaStore (wraps VectorIndexer)
        llm_client = make_llm_client(cfg.local_llm)
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(),
            ollama_client=llm_client,
            embedding_model=getattr(cfg.indexing, "embedding_model", None),
            embeddings_enabled=emb_enabled,
            project_root=project_root,
        )
        return ChromaStore(indexer)


__all__ = [
    "Block",
    "ChromaStore",
    "EmbeddingFn",
    "IndexStore",
    "LanceStore",
    "blocks_from_ast",
    "FastEmbedEmbedder",
    "OllamaEmbedder",
    "PlaceholderEmbedder",
    "make_embedder",
    "make_store",
]
