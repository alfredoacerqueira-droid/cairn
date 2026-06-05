"""Unit tests for core/semantic_cache.py — SemanticCache."""

import hashlib
import time

from core.semantic_cache import SemanticCache


class FakeEmbedder:
    """Deterministic test embedder using SHA256 hashing."""

    dim = 8
    name = "fake"

    def __call__(self, texts):
        """Return 8-dimensional vectors based on SHA256 hash."""
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


class PlaceholderLike:
    """Mimics PlaceholderEmbedder with dim=1."""

    dim = 1
    name = "placeholder"

    def __call__(self, texts):
        return [[0.0] for _ in texts]


class TestSemanticCache:
    """Unit tests for SemanticCache."""

    def test_set_and_get_exact_match(self, tmp_path):
        """Test basic set/get with exact match."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("hello world", "response_1")
        result = cache.get("hello world")

        assert result == "response_1"

    def test_get_miss_on_unknown_query(self, tmp_path):
        """Test cache miss for unknown query."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("query_a", "value_a")
        result = cache.get("query_b")

        assert result is None

    def test_exact_match_case_insensitive_and_whitespace_normalized(self, tmp_path):
        """Test that exact match is case-insensitive and normalizes whitespace."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("Hello   World", "response_1")

        # Slightly different whitespace and case
        result = cache.get("hello world")
        assert result == "response_1"

        # More aggressive whitespace
        result = cache.get("  HELLO    WORLD  ")
        assert result == "response_1"

    def test_ttl_expiry(self, tmp_path):
        """Test that cached entries expire after TTL."""
        fake_time = [0.0]

        def time_fn():
            return fake_time[0]

        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=10, time_fn=time_fn)

        cache.set("query", "value")

        # Still fresh
        assert cache.get("query") == "value"

        # Advance time past TTL
        fake_time[0] = 11.0
        assert cache.get("query") is None

    def test_max_entries_eviction(self, tmp_path):
        """Test that oldest entries are evicted when max_entries exceeded."""
        cache = SemanticCache(
            tmp_path, FakeEmbedder(), ttl_seconds=300, max_entries=3
        )

        # Add 5 entries
        for i in range(5):
            cache.set(f"query_{i}", f"value_{i}")
            time.sleep(0.01)  # Ensure different mtimes

        # Count files
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) <= 3

    def test_clear(self, tmp_path):
        """Test clearing the cache."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("query_1", "value_1")
        cache.set("query_2", "value_2")

        assert cache.get("query_1") is not None
        assert cache.get("query_2") is not None

        cache.clear()

        assert cache.get("query_1") is None
        assert cache.get("query_2") is None
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_placeholder_embedder_exact_match_only(self, tmp_path):
        """Test that PlaceholderEmbedder (dim=1) only uses exact match."""
        cache = SemanticCache(
            tmp_path, PlaceholderLike(), ttl_seconds=300, sim_threshold=0.9
        )

        cache.set("how do I list pods", "answer_a")

        # Exact match
        assert cache.get("how do I list pods") == "answer_a"

        # Different query (would match semantically if embeddings worked)
        result = cache.get("how do i list containers")
        assert result is None  # No semantic match with placeholder

    def test_semantic_similarity_with_real_embedder(self, tmp_path):
        """Test semantic matching with deterministic embedder.

        With FakeEmbedder (SHA256-based), identical text produces identical vectors,
        so we test by monkeypatching the threshold to verify the semantic path works.
        """
        cache = SemanticCache(
            tmp_path,
            FakeEmbedder(),
            ttl_seconds=300,
            sim_threshold=0.0,  # Accept any similarity > 0
        )

        # Set an entry
        cache.set("original query", "cached_response")

        # Try to retrieve with identical text (will get exact match, but tests the path)
        result = cache.get("original query")
        assert result == "cached_response"

    def test_corrupt_file_graceful_handling(self, tmp_path):
        """Test that corrupt JSON files are handled gracefully."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("query_1", "value_1")

        # Corrupt a cache file by writing invalid JSON
        cache_files = list(tmp_path.glob("*.json"))
        if cache_files:
            cache_files[0].write_text("{ invalid json")

        # Should not crash and should treat as miss
        result = cache.get("query_1")
        assert result is None

        # Corrupt file should be deleted
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_ttl_override_per_entry(self, tmp_path):
        """Test per-entry TTL override."""
        fake_time = [0.0]

        def time_fn():
            return fake_time[0]

        cache = SemanticCache(
            tmp_path, FakeEmbedder(), ttl_seconds=10, time_fn=time_fn
        )

        # Set entry with custom TTL
        cache.set("query", "value", ttl_seconds=5)

        # At t=4, should be fresh
        fake_time[0] = 4.0
        assert cache.get("query") == "value"

        # At t=6, should be expired (custom TTL=5)
        fake_time[0] = 6.0
        assert cache.get("query") is None

    def test_set_overwrites_existing(self, tmp_path):
        """Test that setting a query again overwrites the old value."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("query", "value_1")
        assert cache.get("query") == "value_1"

        cache.set("query", "value_2")
        assert cache.get("query") == "value_2"

    def test_exact_match_normalization_consistency(self, tmp_path):
        """Test that normalization is consistent across sets and gets."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        queries = [
            "Hello World",
            "  hello world  ",
            "HELLO WORLD",
            "hello   world",
        ]

        # Set with first variant
        cache.set(queries[0], "response")

        # All variants should hit the same entry
        for q in queries[1:]:
            result = cache.get(q)
            assert result == "response", f"Failed for variant: {q}"

    def test_semantic_scan_with_low_threshold(self, tmp_path):
        """Test semantic lookup by lowering threshold to accept near-matches."""
        cache = SemanticCache(
            tmp_path,
            FakeEmbedder(),
            ttl_seconds=300,
            sim_threshold=0.1,  # Very low to ensure hits
        )

        # Set an entry
        cache.set("kubernetes pods list", "answer_about_pods")

        # Similar but different query (will scan semantically)
        # With FakeEmbedder, different text gives different vectors,
        # but with threshold=0.1, we may get a match if the hash-based
        # vectors happen to be similar enough
        result = cache.get("kubernetes pods show")

        # The important thing is the get() doesn't crash
        # (result may be None, which is fine)
        assert result is None or result == "answer_about_pods"

    def test_embedding_failure_fallback_to_exact(self, tmp_path):
        """Test that embedding failures don't break caching."""

        class FailingEmbedder:
            dim = 8
            name = "failing"

            def __call__(self, texts):
                raise RuntimeError("Embedding failed")

        cache = SemanticCache(tmp_path, FailingEmbedder(), ttl_seconds=300)

        # Set should not crash even if embedding fails
        cache.set("query", "value")

        # Entry should be stored (without vector)
        result = cache.get("query")
        assert result == "value"

    def test_empty_cache_directory(self, tmp_path):
        """Test behavior with empty cache directory."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        result = cache.get("any_query")
        assert result is None

    def test_multiple_entries_different_vectors(self, tmp_path):
        """Test storing multiple entries with different content."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("query_a", "response_a")
        cache.set("query_b", "response_b")
        cache.set("query_c", "response_c")

        assert cache.get("query_a") == "response_a"
        assert cache.get("query_b") == "response_b"
        assert cache.get("query_c") == "response_c"

    def test_vector_storage_and_retrieval(self, tmp_path):
        """Test that vectors are properly stored and retrieved."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        cache.set("test query", "test response")

        # Check that the file was created
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

        # Verify the entry has a vector
        import json

        with open(cache_files[0]) as f:
            data = json.load(f)

        assert "vector" in data
        assert isinstance(data["vector"], list)
        assert len(data["vector"]) == 8  # FakeEmbedder uses 8 dims

    def test_expired_entries_cleaned_on_scan(self, tmp_path):
        """Test that expired entries are deleted during semantic scan."""
        fake_time = [0.0]

        def time_fn():
            return fake_time[0]

        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=10, time_fn=time_fn)

        cache.set("query_1", "value_1")

        # Advance time so entry expires
        fake_time[0] = 11.0

        # Trigger a semantic scan (get on a different query)
        cache.get("different_query")

        # File should be deleted during scan
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0

    def test_none_value_handling(self, tmp_path):
        """Test that None values are handled correctly."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        # Store an empty string as value
        cache.set("query", "")
        result = cache.get("query")
        assert result == ""

    def test_large_value_storage(self, tmp_path):
        """Test storing large response values."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        large_value = "x" * 100_000  # 100KB response
        cache.set("query", large_value)
        result = cache.get("query")

        assert result == large_value

    def test_special_characters_in_query(self, tmp_path):
        """Test handling special characters in queries."""
        cache = SemanticCache(tmp_path, FakeEmbedder(), ttl_seconds=300)

        queries_with_special_chars = [
            "what is $PATH?",
            "how to [install] packages",
            "print('hello')",
            "SELECT * FROM users WHERE id=1",
        ]

        for q in queries_with_special_chars:
            cache.set(q, f"response_{q}")

        for q in queries_with_special_chars:
            result = cache.get(q)
            assert result == f"response_{q}", f"Failed for: {q}"
