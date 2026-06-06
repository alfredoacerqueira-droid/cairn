"""LanceDB-backed IndexStore implementation for semantic code search."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class LanceStore:
    """IndexStore-compatible wrapper around LanceDB for semantic search.

    Implements the IndexStore protocol using Lance (Apache Arrow-based
    vector database) as the backend. Supports vector search, full-text search,
    and hybrid search with optional reranking.
    """

    def __init__(
        self,
        lance_path: str | Path,
        embedder: object,
        *,
        project_id: str | None = None,
        project_root: str | None = None,
        table_name: str = "blocks",
    ):
        """Initialize LanceStore with a Lance database path and embedder.

        Args:
            lance_path: Path to the Lance database directory.
            embedder: EmbeddingFn instance (dim property and __call__ method).
            project_id: Optional project identifier for multi-repo isolation.
            project_root: Optional project root path (for provenance metadata).
            table_name: Name of the table to use in the Lance database.
        """
        self.lance_path = Path(lance_path)
        self._embedder = embedder
        self.project_id = project_id
        self.project_root = project_root
        self.table_name = table_name

        # Connect to Lance database (creates if doesn't exist)
        import lancedb

        self.db = lancedb.connect(str(self.lance_path))
        self.table = None
        # Try to open existing table, or None if it doesn't exist yet
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Table doesn't exist yet; will be created on first upsert
            pass

    def _get_or_create_table(self, sample_row: dict) -> Any:
        """Create table on first upsert if it doesn't exist yet.

        Lance infers the schema automatically from the sample row.

        Args:
            sample_row: A sample row to infer the schema from.

        Returns:
            The opened/created table.
        """
        if self.table is not None:
            return self.table

        # Create table from sample row (Lance infers schema automatically)
        table = self.db.create_table(
            self.table_name, data=[sample_row], mode="overwrite"
        )
        self.table = table
        return table

    def upsert_blocks(self, blocks: list[Any], *, batch_size: int = 5000) -> None:
        """Upsert blocks into the Lance table.

        Args:
            blocks: List of Block objects to upsert.
            batch_size: Max blocks per batch (default 5000).
        """
        if not blocks:
            return

        # Extract code texts for embedding
        codes = [block.code for block in blocks]

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_embeddings = self._embedder(batch_codes)
            all_embeddings.extend(batch_embeddings)

        # Build rows with embeddings and metadata
        rows = []
        for block, embedding in zip(blocks, all_embeddings):
            row = {
                "id": block.id,
                "filepath": block.filepath,
                "function": block.function,
                "code": block.code,
                "line_start": int(block.line_start),
                "line_end": int(block.line_end),
                "vector": embedding,
                "indexed_at": time.time(),
            }
            if self.project_id is not None:
                row["project_id"] = self.project_id
            if self.project_root is not None:
                row["project_root"] = self.project_root
            rows.append(row)

        # Ensure table exists (will be created if needed)
        if self.table is None:
            self._get_or_create_table(rows[0])

        # Upsert rows: merge on id, update all matched, insert new
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            self.table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(batch_rows)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filepath_prefix: str | None = None,
        metrics: object | None = None,
    ) -> list[dict]:
        """Semantic search for relevant blocks.

        If embedder.dim == 1 (placeholder), returns empty list (no semantic search).
        Otherwise, embeds query and runs vector search on the table.

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            filepath_prefix: Optional filter (startswith).
            metrics: Optional metrics object to record search latency.

        Returns:
            List of hit dicts with standard keys.
        """
        if self._embedder.dim == 1:
            # Placeholder embedder: semantic search disabled
            return []

        if self.table is None:
            return []

        import time as time_module

        start = time_module.perf_counter()

        # Embed the query
        query_embedding = self._embedder([query])[0]

        # Build filter for filepath prefix if provided
        where = None
        if filepath_prefix:
            # Lance supports LIKE for string matching
            where = f"filepath LIKE '{filepath_prefix}%'"

        # Vector search
        results = self.table.search(query_embedding).limit(top_k)
        if where:
            results = results.where(where)

        raw_results = results.to_list()

        # Convert to standard hit dict format
        formatted = []
        for row in raw_results:
            # Normalize cosine distance to similarity [0, 1]
            # Lance returns cosine distance (0=identical, 2=opposite)
            # Raw cosine similarity = 1 - distance
            distance = row.get("_distance", 0.0)
            similarity = 1.0 - (distance / 2.0) if distance else 1.0

            hit = {
                "id": row.get("id", ""),
                "filepath": row.get("filepath", ""),
                "function": row.get("function", ""),
                "line_start": int(row.get("line_start", 0)),
                "line_end": int(row.get("line_end", 0)),
                "code": row.get("code", ""),
                "similarity": max(0.0, min(1.0, similarity)),
                "project_id": row.get("project_id"),
                "project_root": row.get("project_root"),
            }
            formatted.append(hit)

        elapsed_ms = (time_module.perf_counter() - start) * 1000
        if metrics:
            metrics.record_search(query, len(formatted), elapsed_ms)

        return formatted

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        rerank: bool = False,
        filepath_prefix: str | None = None,
    ) -> list[dict]:
        """Hybrid search (lexical + semantic, optionally reranked).

        Attempts to use LanceDB native hybrid search if available.
        Falls back to FTS-only if hybrid is not supported.

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            rerank: Whether to apply cross-encoder reranking.
            filepath_prefix: Optional filter (startswith).

        Returns:
            List of hit dicts with raw_cosine and rerank_score when reranked.
        """
        if self.table is None:
            return []

        # Ensure FTS index exists on code column
        try:
            self.table.create_fts_index("code", replace=True)
        except Exception as e:
            logger.warning("Failed to create FTS index: %s", e)

        # Try hybrid search first
        where = None
        if filepath_prefix:
            where = f"filepath LIKE '{filepath_prefix}%'"

        try:
            results = self.table.search(query, query_type="hybrid").limit(top_k)
            if where:
                results = results.where(where)
            raw_results = results.to_list()
        except Exception as e:
            logger.warning("Hybrid search not supported, falling back to FTS: %s", e)
            # Fall back to FTS-only
            try:
                results = self.table.search(query, query_type="fts").limit(top_k)
                if where:
                    results = results.where(where)
                raw_results = results.to_list()
            except Exception as e2:
                logger.error("FTS search failed: %s", e2)
                return []

        # Convert to standard hit dict format
        formatted = []
        for row in raw_results:
            # Extract cosine distance if present (for hybrid results)
            distance = row.get("_distance", None)
            similarity = None
            if distance is not None:
                similarity = 1.0 - (distance / 2.0) if distance else 1.0
                similarity = max(0.0, min(1.0, similarity))

            hit = {
                "id": row.get("id", ""),
                "filepath": row.get("filepath", ""),
                "function": row.get("function", ""),
                "line_start": int(row.get("line_start", 0)),
                "line_end": int(row.get("line_end", 0)),
                "code": row.get("code", ""),
                "similarity": similarity or 0.5,  # Default if no distance
                "project_id": row.get("project_id"),
                "project_root": row.get("project_root"),
                "raw_cosine": similarity,
                "rerank_score": None,
            }

            # Apply reranking if requested and reranker available
            if rerank:
                try:
                    from flashrank import Reranker as FlashRankReranker

                    reranker = FlashRankReranker(model_name="ms-marco-MiniLM-L-12-v2")
                    rerank_result = reranker.rerank(
                        query=query, documents=[hit["code"]], top_k=1
                    )
                    if rerank_result:
                        hit["rerank_score"] = rerank_result[0].score
                except Exception as e:
                    logger.debug("Reranking failed: %s", e)
                    hit["rerank_score"] = None

            formatted.append(hit)

        return formatted

    def iter_blocks(self, page: int = 2000) -> Iterator[dict]:
        """Iterate all blocks in the Lance table (pagination).

        Args:
            page: Page size (number of blocks per iteration).

        Yields:
            Dicts representing blocks with keys: id, text, filepath, function,
            line_start, line_end, project_id, project_root.
        """
        # Reopen table in case it was updated by another store instance
        try:
            table = self.db.open_table(self.table_name)
        except Exception:
            return

        # Scan the entire table using search() with no query
        # (search with no vector does a full scan)
        offset = 0
        while True:
            try:
                results = table.search().limit(page).offset(offset).to_list()
            except Exception:
                break

            if not results:
                break

            for row in results:
                # Skip foreign projects if isolation is enabled
                rec_pid = row.get("project_id")
                if (
                    self.project_id is not None
                    and rec_pid is not None
                    and rec_pid != self.project_id
                ):
                    continue

                yield {
                    "id": row.get("id", ""),
                    "text": row.get("code", ""),
                    "filepath": row.get("filepath", ""),
                    "function": row.get("function", ""),
                    "line_start": int(row.get("line_start", 0)),
                    "line_end": int(row.get("line_end", 0)),
                    "project_id": row.get("project_id"),
                    "project_root": row.get("project_root"),
                }

            offset += len(results)
            if len(results) < page:
                break

    def delete_file(self, filepath: str) -> int:
        """Delete all blocks for a file.

        Args:
            filepath: Path to file.

        Returns:
            Number of blocks deleted (0 if table doesn't exist).
        """
        if self.table is None:
            return 0

        # Escape single quotes in filepath for SQL
        escaped_filepath = filepath.replace("'", "''")

        try:
            # Delete rows where filepath matches
            self.table.delete(f"filepath = '{escaped_filepath}'")
            return 1  # Lance doesn't return count, so return 1 to indicate success
        except Exception as e:
            logger.warning("Failed to delete file %s: %s", filepath, e)
            return 0

    def count(self) -> int:
        """Get total number of indexed blocks."""
        if self.table is None:
            return 0
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all blocks from the index."""
        if self.table is None:
            return

        try:
            self.db.drop_table(self.table_name)
            self.table = None
        except Exception as e:
            logger.warning("Failed to clear Lance table: %s", e)

    def snapshot(self, tag: str | None = None) -> str | None:
        """Take a versioned snapshot of the index.

        Lance natively supports versioning via table.version.

        Args:
            tag: Optional human-readable tag (not used by Lance).

        Returns:
            Current table version (int) or None if versioning unavailable.
        """
        if self.table is None:
            return None

        try:
            return str(self.table.version)
        except Exception:
            return None

    def restore(self, version: str | int) -> None:
        """Restore index to a previous snapshot.

        Args:
            version: Version ID returned by snapshot().
        """
        if self.table is None:
            return

        try:
            self.table.restore(int(version))
        except Exception as e:
            logger.warning("Failed to restore Lance table to version %s: %s", version, e)

    def list_versions(self) -> list[dict]:
        """List all available snapshots.

        Returns:
            List of version metadata dicts.
        """
        if self.table is None:
            return []

        try:
            versions = self.table.list_versions()
            return [
                {"version": v.version, "timestamp": v.timestamp} for v in versions
            ]
        except Exception:
            return []

    def checkpoint(self) -> None:
        """Checkpoint the index state (no-op; Lance auto-checkpoints on write)."""
        return None
