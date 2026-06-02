"""Tests for persistent file-based cache."""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from core.persistent_cache import PersistentCache


class TestPersistentCache:
    def test_set_get_roundtrip(self):
        """Test that set/get works across separate cache instances."""
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # First instance: set a value
            cache1 = PersistentCache(cache_dir, max_entries=10, ttl_seconds=300)
            cache1.set({"result": "data"}, "query", "v1")

            # Second instance: get the same value
            cache2 = PersistentCache(cache_dir, max_entries=10, ttl_seconds=300)
            value = cache2.get("query", "v1")

            assert value == {"result": "data"}

    def test_ttl_expiry(self):
        """Test that expired entries are treated as misses."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=1)
            cache.set("value1", "key1")

            # Should hit immediately
            assert cache.get("key1") == "value1"

            # Wait for expiry
            time.sleep(1.1)

            # Should miss
            assert cache.get("key1") is None

    def test_max_entries_eviction(self):
        """Test LRU eviction by mtime."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=3, ttl_seconds=300)

            # Add 3 entries
            cache.set("a", "key1")
            time.sleep(0.01)
            cache.set("b", "key2")
            time.sleep(0.01)
            cache.set("c", "key3")

            # All should exist
            assert cache.get("key1") == "a"
            assert cache.get("key2") == "b"
            assert cache.get("key3") == "c"

            # Add a 4th (should evict oldest, key1)
            time.sleep(0.01)
            cache.set("d", "key4")

            # key1 should be gone
            assert cache.get("key1") is None
            # Others should remain
            assert cache.get("key2") == "b"
            assert cache.get("key3") == "c"
            assert cache.get("key4") == "d"

    def test_corrupt_json_is_miss(self):
        """Test that corrupt JSON files are treated as cache miss."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=300)

            # Write a corrupt JSON file directly
            key = cache._key("bad_key")
            cache_file = cache._cache_file(key)
            cache_file.write_text("{ invalid json ")

            # Get should return None (miss) and not crash
            assert cache.get("bad_key") is None
            # Corrupt file should be deleted
            assert not cache_file.exists()

    def test_stats(self):
        """Test cache statistics."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=300)

            cache.set("a", "key1")
            assert cache.get("key1") == "a"  # hit
            cache.get("missing")  # miss
            cache.get("missing2")  # miss

            stats = cache.stats()
            assert stats["entries"] == 1
            assert stats["hits"] == 1
            assert stats["misses"] == 2
            assert stats["hit_rate"] == round(1 / 3, 3)

    def test_clear(self):
        """Test clearing all cache entries."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=300)

            cache.set("a", "key1")
            cache.set("b", "key2")
            assert cache.get("key1") == "a"

            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None
            assert cache.stats()["entries"] == 0

    def test_invalidate(self):
        """Test invalidating a specific key."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=300)

            cache.set("a", "key1")
            cache.set("b", "key2")

            cache.invalidate("key1")

            assert cache.get("key1") is None
            assert cache.get("key2") == "b"

    def test_serializable_types(self):
        """Test that common JSON-serializable types work."""
        with TemporaryDirectory() as tmpdir:
            cache = PersistentCache(Path(tmpdir), max_entries=10, ttl_seconds=300)

            # Test various types
            cache.set([1, 2, 3], "list")
            cache.set({"key": "value"}, "dict")
            cache.set("string", "str")
            cache.set(42, "int")
            cache.set(3.14, "float")
            cache.set(True, "bool")
            cache.set(None, "none")

            assert cache.get("list") == [1, 2, 3]
            assert cache.get("dict") == {"key": "value"}
            assert cache.get("str") == "string"
            assert cache.get("int") == 42
            assert cache.get("float") == 3.14
            assert cache.get("bool") is True
            assert cache.get("none") is None
