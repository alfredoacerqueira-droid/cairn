"""Storage abstraction layer for Cairn vector indices.

This package provides the IndexStore protocol and embedder implementations
that ChromaStore and LanceStore will implement.
"""

from pathlib import Path

from core.config import embeddings_available
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

    # Determine embeddings flag: enabled if any real embedder is available (no
    # longer requires local_llm.enabled — fastembed works without Ollama).
    emb_available, emb_name = embeddings_available(cfg)
    emb_enabled = emb_available

    # Get the backend choice
    backend = getattr(cfg.indexing, "store_backend", "chroma")

    # Build embedder only when needed:
    # - fastembed: in-process, no ollama client required
    # - lance backend: needs an embedder for all embedding types
    # - chroma + ollama: VectorIndexer uses ollama_client directly
    embedder = None
    if emb_enabled and (emb_name == "fastembed" or backend == "lance"):
        embedder = make_embedder(cfg)

    # Resolve project_id
    pid = project_id(project_root)

    # Get index_location from config to ensure consistency
    index_location = getattr(cfg.indexing, "index_location", "auto")

    if backend == "lance":
        # Build LanceStore
        return LanceStore(
            repo.get_lance_path(index_location),
            embedder,
            project_id=pid,
            project_root=str(project_root),
        )
    else:
        # Default: build ChromaStore (wraps VectorIndexer)
        llm_client = make_llm_client(cfg.local_llm)
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(index_location),
            ollama_client=llm_client,
            embedding_model=getattr(cfg.indexing, "embedding_model", None),
            embeddings_enabled=emb_enabled,
            project_root=project_root,
            embedder=embedder,
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
