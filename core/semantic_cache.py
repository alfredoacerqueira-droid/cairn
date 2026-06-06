"""Local semantic prompt/response cache with embedding-based lookup.

A simple, persistent, embedded prompt/response cache for rapid retrieval
when prompts are paraphrased but semantically similar. Uses cosine similarity
to match queries against cached entries. SET by a local LLM, GET by the cloud
LLM (or CLI). Short TTL, no external service required.

Storage: JSON files under cache_dir (e.g., .cairn/cache/semantic/), one per entry,
named by MD5 of normalized query for O(1) exact-match probe + O(N) semantic scan.
Each entry: {"query": str, "vector": list[float] | null, "value": str, "expires_at": float}.

When embedder.dim == 1 (PlaceholderEmbedder), operates in EXACT-MATCH mode only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity in [0.0, 1.0].
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _normalize_query(query: str) -> str:
    """Normalize a query for exact-match lookup.

    Strips whitespace, collapses internal whitespace, and lowercases.
    Used to generate the filename key for O(1) exact-match probe.

    Args:
        query: Raw query string.

    Returns:
        Normalized query.
    """
    return " ".join(query.strip().lower().split())


class SemanticCache:
    """Local embedded prompt/response cache with similarity lookup.

    Provides get/set operations with:
    - Exact-match mode when embedder.dim == 1 (no real embeddings).
    - Semantic similarity mode when embeddings are available (dim > 1).
    - TTL-based expiry and LRU eviction by mtime.
    - Resilience to corrupt/unreadable files.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        embedder: object,
        *,
        ttl_seconds: int = 1800,
        sim_threshold: float = 0.92,
        max_entries: int = 200,
        time_fn: Optional[Callable[[], float]] = None,
    ):
        """Initialize SemanticCache.

        Args:
            cache_dir: Directory to store cache files (created if missing).
                Example: Path(".cairn/cache/semantic").
            embedder: An EmbeddingFn object with .dim and __call__(texts).
                If embedder.dim == 1 (PlaceholderEmbedder), only exact-match works.
            ttl_seconds: Time-to-live for each entry (default 30 minutes).
            sim_threshold: Minimum cosine similarity (0..1) for semantic match.
                Only used if embedder.dim > 1.
            max_entries: Maximum entries before LRU eviction by mtime.
            time_fn: Optional callable for getting current time (default time.time).
                Used for testing/injection.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries
        self.time_fn = time_fn or time.time

    def _entry_path(self, normalized_query: str) -> Path:
        """Get the cache file path for a normalized query (MD5 key).

        Args:
            normalized_query: Normalized query string.

        Returns:
            Path to the cache entry file.
        """
        key = hashlib.md5(normalized_query.encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def set(self, query: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        """Store a query-value pair in the cache.

        Args:
            query: The prompt/query string.
            value: The response/cached value.
            ttl_seconds: Optional TTL override (default: self.ttl_seconds).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        normalized = _normalize_query(query)
        expires_at = self.time_fn() + ttl

        # Compute embedding if embeddings are available
        vector = None
        if self.embedder.dim > 1:
            try:
                embeddings = self.embedder([query])
                if embeddings:
                    vector = embeddings[0]
            except Exception as e:
                logger.warning("Failed to embed query for cache: %s", e)
                # Continue without vector (fall back to exact-match only)

        entry = {
            "query": query,
            "vector": vector,
            "value": value,
            "expires_at": expires_at,
        }

        entry_path = self._entry_path(normalized)

        # Atomic write: write to tmp, then replace
        tmp_file = entry_path.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(entry, f)
            tmp_file.replace(entry_path)
        except Exception as e:
            logger.warning("Failed to write cache entry %s: %s", entry_path, e)
            try:
                tmp_file.unlink()
            except Exception:
                pass
            return

        # Evict oldest by mtime if over capacity
        self._evict_if_needed()

    def get(self, query: str) -> Optional[str]:
        """Retrieve a cached value by query (exact or semantic match).

        First tries EXACT match (MD5 filename probe). If miss, tries SEMANTIC
        match by scanning non-expired entries and comparing embeddings
        (only if embedder.dim > 1).

        Args:
            query: The prompt/query string.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        normalized = _normalize_query(query)

        # Try exact match first (O(1) filename probe)
        entry_path = self._entry_path(normalized)
        if entry_path.exists():
            entry = self._read_entry(entry_path)
            if entry is not None:
                # Check expiry
                if entry.get("expires_at", 0) > self.time_fn():
                    return entry.get("value")
                else:
                    # Expired: delete the file
                    try:
                        entry_path.unlink()
                    except Exception:
                        pass
            return None

        # Exact match missed. Try semantic match if embeddings available.
        if self.embedder.dim == 1:
            # No embeddings; semantic lookup disabled
            return None

        # Embed the query
        try:
            query_embedding = self.embedder([query])
            if not query_embedding:
                return None
            query_vec = query_embedding[0]
        except Exception as e:
            logger.warning("Failed to embed query for semantic lookup: %s", e)
            return None

        # Scan all non-expired entries with vectors
        best_match = None
        best_similarity = 0.0

        try:
            for entry_file in self.cache_dir.glob("*.json"):
                entry = self._read_entry(entry_file)
                if entry is None:
                    continue

                # Skip expired entries
                if entry.get("expires_at", 0) <= self.time_fn():
                    try:
                        entry_file.unlink()
                    except Exception:
                        pass
                    continue

                # Skip entries without vectors
                entry_vec = entry.get("vector")
                if not entry_vec:
                    continue

                # Compute similarity
                similarity = _cosine_similarity(query_vec, entry_vec)
                if similarity >= self.sim_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry.get("value")
        except Exception as e:
            logger.warning("Error scanning cache for semantic match: %s", e)

        return best_match

    def _read_entry(self, entry_path: Path) -> Optional[dict]:
        """Read and validate a cache entry file.

        Treats corrupt/unreadable files as a miss and optionally deletes them.

        Args:
            entry_path: Path to the cache entry JSON file.

        Returns:
            Parsed entry dict, or None if unreadable/invalid.
        """
        try:
            with open(entry_path) as f:
                data = json.load(f)
            # Basic validation
            if "query" in data and "value" in data and "expires_at" in data:
                return data
            else:
                logger.warning("Cache entry missing required fields: %s", entry_path)
                return None
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Corrupt cache entry %s, deleting: %s", entry_path, e)
            try:
                entry_path.unlink()
            except Exception:
                pass
            return None

    def _evict_if_needed(self):
        """Evict oldest entry (by mtime) if OVER capacity."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            # Evict only if strictly over max_entries (not equal)
            if len(cache_files) > self.max_entries:
                oldest = min(cache_files, key=lambda p: p.stat().st_mtime)
                oldest.unlink()
                logger.debug("Evicted oldest cache entry: %s", oldest.name)
        except Exception as e:
            logger.warning("Error during cache eviction: %s", e)

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
        except Exception as e:
            logger.warning("Error clearing cache: %s", e)
