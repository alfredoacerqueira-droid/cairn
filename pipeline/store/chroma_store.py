"""ChromaStore — IndexStore wrapper around VectorIndexer for the read path.

This module provides ChromaStore, which wraps the existing VectorIndexer
(composition, not inheritance) and implements the IndexStore protocol.
Used by ContextAssembler to abstract away VectorIndexer; other code paths
(CLI reindex, sync_engine, etc.) continue using VectorIndexer directly.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from pipeline.indexer import VectorIndexer

logger = logging.getLogger(__name__)


class ChromaStore:
    """IndexStore-compatible wrapper around VectorIndexer for semantic search + pagination.

    Implements the IndexStore protocol by delegating to an inner VectorIndexer instance.
    The read path (search, iter_blocks) is fully implemented. The write path
    (upsert_blocks) is a no-op (migrated in a later step).
    """

    def __init__(self, indexer: VectorIndexer):
        """Initialize ChromaStore with a VectorIndexer instance.

        Args:
            indexer: A VectorIndexer instance to wrap.
        """
        self._indexer = indexer

    @property
    def indexer(self) -> VectorIndexer:
        """Return the underlying VectorIndexer."""
        return self._indexer

    @property
    def collection(self):
        """Return the underlying ChromaDB collection (for legacy access)."""
        return self._indexer.collection

    @property
    def project_id(self) -> str | None:
        """Return the project ID, if set."""
        return getattr(self._indexer, "project_id", None)

    @property
    def project_root(self) -> str | None:
        """Return the project root path, if set."""
        return getattr(self._indexer, "project_root", None)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filepath_prefix: Optional[str] = None,
        metrics: Optional[object] = None,
    ) -> list[dict]:
        """Semantic search via ChromaDB.

        Delegates to VectorIndexer.search() with the same signature.
        Returns a list of hit dicts with keys: id, filepath, function,
        line_start, line_end, code, similarity, project_id, project_root.

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            filepath_prefix: Optional filter (startswith).
            metrics: Optional metrics object to record search latency.

        Returns:
            List of hit dicts.
        """
        return self._indexer.search(
            query, top_k=top_k, filepath_prefix=filepath_prefix, metrics=metrics
        )

    def iter_blocks(self, page: int = 2000) -> Iterator[dict]:
        """Iterate all blocks in ChromaDB (pagination).

        Reproduces the exact paged loop from ContextAssembler._load_function_texts,
        yielding dicts with the same keys (id, filepath, function, line_start,
        line_end, code, and project metadata). This is critical because the
        downstream BM25/AST legs depend on these exact keys.

        Args:
            page: Page size (number of blocks per iteration).

        Yields:
            Dicts representing blocks with keys matching the index store schema.
        """
        offset = 0
        while True:
            try:
                data = self.collection.get(
                    include=["metadatas", "documents"], limit=page, offset=offset
                )
            except Exception:
                # Same error resilience as the original code
                break

            batch_ids = list(data.get("ids") or [])
            if not batch_ids:
                break

            batch_metadatas = list(data.get("metadatas") or [])
            batch_documents = list(data.get("documents") or [])

            # Yield rows with the same structure as the original loop expects
            for i, doc_id in enumerate(batch_ids):
                meta = batch_metadatas[i] if i < len(batch_metadatas) else {}
                doc = batch_documents[i] if i < len(batch_documents) else ""

                # Skip provably foreign records (same logic as _load_function_texts)
                rec_pid = meta.get("project_id")
                if (
                    self.project_id is not None
                    and rec_pid is not None
                    and rec_pid != self.project_id
                ):
                    continue

                yield {
                    "id": doc_id,
                    "text": doc,
                    "filepath": meta.get("filepath", ""),
                    "function": meta.get("function", ""),
                    "line_start": meta.get("line_start", 0),
                    "line_end": meta.get("line_end", 0),
                    "project_id": meta.get("project_id"),
                    "project_root": meta.get("project_root"),
                }

            offset += len(batch_ids)
            if len(batch_ids) < page:
                break

    def count(self) -> int:
        """Get total number of indexed blocks."""
        if hasattr(self._indexer, "count"):
            return self._indexer.count()
        return self.collection.count()

    def clear(self) -> None:
        """Delete all blocks from the index."""
        if hasattr(self._indexer, "clear"):
            self._indexer.clear()
        else:
            # Fallback: delete all IDs in the collection
            try:
                all_ids = self.collection.get(include=[])
                if all_ids.get("ids"):
                    self.collection.delete(ids=all_ids["ids"])
            except Exception:
                logger.warning("Failed to clear ChromaDB collection")

    def delete_file(self, filepath: str) -> int:
        """Delete all blocks for a file.

        Args:
            filepath: Path to file.

        Returns:
            Number of blocks deleted.
        """
        self._indexer.remove_file(filepath)
        # VectorIndexer.remove_file doesn't return a count, so we return 0
        # (the write path isn't being migrated in this step anyway)
        return 0

    def upsert_blocks(self, blocks: list[Any], *, batch_size: int = 5000) -> None:
        """Upsert blocks into the index.

        Embeds each block's code using the inner VectorIndexer's embedding
        mechanism and upserts into the ChromaDB collection in batches.

        Args:
            blocks: List of Block objects to upsert.
            batch_size: Max blocks per batch (default 5000).
        """
        import time

        if not blocks:
            return

        # Extract codes for embedding
        codes = [block.code for block in blocks]

        # Embed in batches using VectorIndexer's embedding function
        all_embeddings = []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            for code in batch_codes:
                embedding = self._indexer.embed_function(code)
                all_embeddings.append(embedding)

        # Build metadata and document lists
        ids = []
        metadatas = []
        documents = []
        embeddings = []

        for block, embedding in zip(blocks, all_embeddings):
            doc_id = block.id
            ids.append(doc_id)
            embeddings.append(embedding)

            metadata = {
                "filepath": block.filepath,
                "function": block.function,
                "line_start": block.line_start,
                "line_end": block.line_end,
                "indexed_at": time.time(),
            }
            # Add project provenance if available
            if self._indexer.project_id is not None:
                metadata["project_id"] = self._indexer.project_id
                metadata["project_root"] = self._indexer.project_root
            metadatas.append(metadata)
            documents.append(block.code)

        # Determine batch size for ChromaDB
        try:
            client_max = self._indexer.client.get_max_batch_size()
        except Exception:
            client_max = None
        max_batch = (
            min(5000, client_max) if client_max else 5000
        )  # Use 5000 as default

        # Upsert in batches
        num_items = len(ids)
        for i in range(0, num_items, max_batch):
            end_idx = min(i + max_batch, num_items)
            self.collection.upsert(  # type: ignore[arg-type]
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],  # type: ignore[arg-type]
                metadatas=metadatas[i:end_idx],  # type: ignore[arg-type]
                documents=documents[i:end_idx],
            )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        rerank: bool = False,
        filepath_prefix: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search (not implemented — ChromaDB uses Python HybridRetriever)."""
        raise NotImplementedError(
            "ChromaStore uses the Python HybridRetriever, not native hybrid"
        )

    # Versioning hooks: no-ops (ChromaDB doesn't support snapshots)
    def snapshot(self, tag: Optional[str] = None) -> None:
        """Take a versioned snapshot (no-op on backends without versioning)."""
        return None

    def restore(self, version: str | int) -> None:
        """Restore index to a previous snapshot (no-op on backends without versioning)."""
        return None

    def list_versions(self) -> list[dict]:
        """List all available snapshots (empty on backends without versioning)."""
        return []

    def checkpoint(self) -> None:
        """Checkpoint the index state (no-op on backends without explicit checkpoint)."""
        return None
