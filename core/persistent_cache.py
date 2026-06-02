"""Persistent file-based cache for cross-process warmth.

Cache entries stored as JSON files in .cairn/cache/ so repeated queries
across separate CLI/MCP processes hit warm cache (much faster than cold
embedding/search). Same interface as SessionCache (get/set/clear) but persisted.

Per-key files + atomic writes (write tmp, os.replace) to handle concurrent
access. Defensive on corrupt/partial JSON (treat as cache miss).
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PersistentCache:
    """File-based cache with TTL and max_entries eviction (LRU by mtime)."""

    def __init__(
        self,
        cache_dir: Path,
        max_entries: int = 100,
        ttl_seconds: int = 300,
        time_fn: Callable[[], float] | None = None,
    ):
        """Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache files (created if missing).
            max_entries: Max entries before LRU eviction (by mtime).
            ttl_seconds: Time-to-live for each entry.
            time_fn: Optional callable for getting current time (default: time.time).
                Used for testing/injection.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.time_fn = time_fn or time.time
        self._hits = 0
        self._misses = 0

    def _key(self, *parts: str) -> str:
        """Generate cache key from parts."""
        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def _cache_file(self, key: str) -> Path:
        """Path to cache file for a key."""
        return self.cache_dir / f"{key}.json"

    def get(self, *key_parts: str) -> Any | None:
        """Get value from cache if exists and not expired."""
        key = self._key(*key_parts)
        cache_file = self._cache_file(key)

        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Corrupt cache file %s, treating as miss: %s", cache_file, e)
            self._misses += 1
            # Optionally delete corrupt file (but be defensive)
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None

        # Check expiry
        timestamp = data.get("ts", 0)
        if self.time_fn() - timestamp > self.ttl_seconds:
            self._misses += 1
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None

        self._hits += 1
        return data.get("value")

    def set(self, value: Any, *key_parts: str):
        """Store value in cache (atomic write with tmp file)."""
        key = self._key(*key_parts)
        cache_file = self._cache_file(key)

        data = {
            "value": value,
            "ts": self.time_fn(),
        }

        # Atomic write: write to tmp, then replace
        tmp_file = cache_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            tmp_file.replace(cache_file)
        except Exception as e:
            logger.warning("Failed to write cache file %s: %s", cache_file, e)
            try:
                tmp_file.unlink()
            except Exception:
                pass
            return

        # Evict oldest by mtime if over capacity
        self._evict_if_needed()

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

    def invalidate(self, *key_parts: str):
        """Remove a specific key from cache."""
        key = self._key(*key_parts)
        cache_file = self._cache_file(key)
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Error invalidating cache key %s: %s", key, e)

    def clear(self):
        """Clear all cache entries."""
        try:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            self._hits = 0
            self._misses = 0
        except Exception as e:
            logger.warning("Error clearing cache: %s", e)

    def stats(self) -> dict:
        """Return cache statistics."""
        try:
            entries = len(list(self.cache_dir.glob("*.json")))
        except Exception:
            entries = 0

        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }
