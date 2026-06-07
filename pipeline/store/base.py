"""Storage abstraction layer for Cairn vector indices.

This module defines the IndexStore protocol and Block dataclass, which form
the backend-agnostic interface that ChromaStore and LanceStore implement.
The IndexStore hit shape (dict keys: id, filepath, function, line_start,
line_end, code, similarity, project_id, project_root) is stable across
all backends and must not change — retrieval legs, confidence guards, and
MCP formatters depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class Block:
    """One indexable code unit (function or method)."""

    id: str  # "<filepath>:<function>:<line_start>"
    filepath: str
    function: str
    code: str
    line_start: int
    line_end: int


@runtime_checkable
class EmbeddingFn(Protocol):
    """An embedding function. dim==1 means 'no real embeddings' (placeholder).

    Implementations must be callable and have dim/name properties.
    """

    @property
    def dim(self) -> int:
        """Embedding vector dimensionality. 1 = placeholder (no embeddings)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name (e.g. 'ollama:nomic-embed-text')."""
        ...

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per text).
        """
        ...


@runtime_checkable
class IndexStore(Protocol):
    """Backend-agnostic project index (ChromaDB, LanceDB, etc.).

    Hits returned by search/hybrid_search are plain dicts with keys:
    id, filepath, function, line_start, line_end, code, similarity,
    project_id, project_root (hybrid_search additionally sets raw_cosine
    and rerank_score). This shape is stable and must not change.
    """

    project_id: str | None
    project_root: str | None

    def upsert_blocks(self, blocks: list[Block], *, batch_size: int = 5000) -> None:
        """Insert or update blocks in the index.

        Args:
            blocks: List of Block objects to upsert.
            batch_size: Max blocks per batch (default 5000 for ChromaDB compat).
        """
        ...

    def delete_file(self, filepath: str) -> int:
        """Delete all blocks for a file.

        Args:
            filepath: Path to file.

        Returns:
            Number of blocks deleted.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        filepath_prefix: str | None = None,
        metrics: object | None = None,
    ) -> list[dict]:
        """Semantic search for relevant blocks.

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            filepath_prefix: Optional filter (startswith).
            metrics: Optional metrics object to record search latency.

        Returns:
            List of hit dicts with keys: id, filepath, function, line_start,
            line_end, code, similarity, project_id, project_root.
        """
        ...

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        rerank: bool = False,
        filepath_prefix: str | None = None,
    ) -> list[dict]:
        """Hybrid search (lexical + semantic, optionally reranked).

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            rerank: Whether to apply cross-encoder reranking.
            filepath_prefix: Optional filter (startswith).

        Returns:
            List of hit dicts. When rerank=True, additionally sets
            raw_cosine and rerank_score on each hit.
        """
        ...

    def iter_blocks(self, page: int = 2000) -> Iterator[dict]:
        """Iterate all blocks in the index (pagination).

        Args:
            page: Page size (number of blocks per iteration).

        Yields:
            Dicts representing blocks (with same keys as search hits).
        """
        ...

    def count(self) -> int:
        """Get total number of indexed blocks."""
        ...

    def clear(self) -> None:
        """Delete all blocks from the index."""
        ...

    def snapshot(self, tag: str | None = None) -> str | None:
        """Take a versioned snapshot of the index (no-op on backends without versioning).

        Args:
            tag: Optional human-readable tag for the snapshot.

        Returns:
            Version ID or None if backend doesn't support snapshots.
        """
        ...

    def restore(self, version: str | int) -> None:
        """Restore index to a previous snapshot (no-op on backends without versioning).

        Args:
            version: Version ID returned by snapshot().
        """
        ...

    def list_versions(self) -> list[dict]:
        """List all available snapshots (empty on backends without versioning).

        Returns:
            List of dicts with version metadata.
        """
        ...

    def checkpoint(self) -> None:
        """Checkpoint the index state (e.g., flush to disk).

        No-op on backends without explicit checkpoint semantics.
        """
        ...


def blocks_from_ast(ast_result) -> list[Block]:
    """Flatten a FileAST (top-level functions, class definitions, class methods) into Blocks.

    Mirrors pipeline/indexer.py::index_ast iteration logic, producing
    id=f'{filepath}:{name}:{line_start}'. Methods are named 'ClassName.method_name'.

    Args:
        ast_result: FileAST result from ASTParser.parse_file() or .parse_string().
                    Must have: .filepath, .functions[], .classes[].
                    Each function/method must have: .name, .code, .line_start, .line_end.
                    Each class must have: .name, .code, .line_start, .line_end, .methods[].

    Returns:
        List of Block objects ready for upsert.
    """
    blocks: list[Block] = []

    # Top-level functions
    for func in ast_result.functions:
        block_id = f"{ast_result.filepath}:{func.name}:{func.line_start}"
        blocks.append(
            Block(
                id=block_id,
                filepath=ast_result.filepath,
                function=func.name,
                code=func.code,
                line_start=func.line_start,
                line_end=func.line_end,
            )
        )

    # Class definitions + their methods
    for cls in ast_result.classes:
        blocks.append(
            Block(
                id=f"{ast_result.filepath}:{cls.name}:{cls.line_start}",
                filepath=ast_result.filepath,
                function=cls.name,
                code=cls.code,
                line_start=cls.line_start,
                line_end=cls.line_end,
            )
        )
        for method in cls.methods:
            method_name = f"{cls.name}.{method.name}"
            block_id = f"{ast_result.filepath}:{method_name}:{method.line_start}"
            blocks.append(
                Block(
                    id=block_id,
                    filepath=ast_result.filepath,
                    function=method_name,
                    code=method.code,
                    line_start=method.line_start,
                    line_end=method.line_end,
                )
            )

    return blocks
