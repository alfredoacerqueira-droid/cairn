"""Session cache for embeddings and search results."""

import hashlib
import time
from typing import Any


class SessionCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, max_entries: int = 100, ttl_seconds: int = 300):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, *parts: str) -> str:
        """Generate a cache key from parts."""
        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, *key_parts: str) -> Any | None:
        """Get a value from cache if it exists and is not expired."""
        key = self._key(*key_parts)
        if key not in self._store:
            self._misses += 1
            return None

        value, timestamp = self._store[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    def set(self, value: Any, *key_parts: str):
        """Store a value in cache."""
        key = self._key(*key_parts)

        # Evict oldest if at capacity
        if len(self._store) >= self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]

        self._store[key] = (value, time.time())

    def invalidate(self, *key_parts: str):
        """Remove a specific key from cache."""
        key = self._key(*key_parts)
        self._store.pop(key, None)

    def clear(self):
        """Clear all cached entries."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }
