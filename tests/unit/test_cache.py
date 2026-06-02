"""Unit tests for core/cache.py — SessionCache."""

import time

from core.cache import SessionCache


class TestSessionCache:
    def test_set_and_get(self):
        cache = SessionCache(max_entries=10, ttl_seconds=60)
        cache.set("hello", "greeting", "en")
        result = cache.get("greeting", "en")
        assert result == "hello"

    def test_get_missing(self):
        cache = SessionCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiry(self):
        cache = SessionCache(max_entries=10, ttl_seconds=0.01)
        cache.set("value", "key")
        time.sleep(0.02)
        result = cache.get("key")
        assert result is None

    def test_max_entries_eviction(self):
        cache = SessionCache(max_entries=3, ttl_seconds=60)
        for i in range(5):
            cache.set(f"val_{i}", f"key_{i}")
        assert cache.stats()["entries"] <= 3

    def test_invalidate(self):
        cache = SessionCache()
        cache.set("value", "k")
        assert cache.get("k") == "value"
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_clear(self):
        cache = SessionCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_stats(self):
        cache = SessionCache()
        cache.set("val", "k")
        cache.get("k")
        cache.get("missing")

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "hit_rate" in stats

    def test_variable_arity_keys(self):
        cache = SessionCache()
        cache.set(42, "user", "123", "score")
        assert cache.get("user", "123", "score") == 42

    def test_commit_aware_keys(self):
        cache = SessionCache()
        cache.set("results", "search", "query", "top_5", "abc1234")
        assert cache.get("search", "query", "top_5", "abc1234") == "results"
        assert cache.get("search", "query", "top_5", "def5678") is None
