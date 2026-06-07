"""Unit tests for pipeline/retrieval/reranker.py — FlashRank cross-encoder."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from pipeline.retrieval.reranker import Reranker


class TestReranker:
    """Test the Reranker cross-encoder wrapper."""

    def test_rerank_orders_by_score(self):
        """Reranker reorders candidates by cross-encoder score (descending)."""
        reranker = Reranker()

        # Mock the FlashRank Ranker
        mock_ranker = MagicMock()
        mock_result1 = {"id": 0, "score": 0.3}
        mock_result2 = {"id": 1, "score": 0.9}
        mock_result3 = {"id": 2, "score": 0.6}
        mock_ranker.rerank.return_value = [
            mock_result2,
            mock_result3,
            mock_result1,
        ]

        candidates = [
            {"id": "func1", "text": "def foo(): pass", "similarity": 0.5},
            {"id": "func2", "text": "def bar(): pass", "similarity": 0.7},
            {"id": "func3", "text": "def baz(): pass", "similarity": 0.6},
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            result = reranker.rerank("query", candidates)

        # Should be reordered by score: func2 (0.9), func3 (0.6), func1 (0.3)
        assert len(result) == 3
        assert result[0]["id"] == "func2"
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["id"] == "func3"
        assert result[1]["rerank_score"] == 0.6
        assert result[2]["id"] == "func1"
        assert result[2]["rerank_score"] == 0.3

    def test_rerank_respects_top_k(self):
        """Reranker returns only top_k results when specified."""
        reranker = Reranker()

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 1, "score": 0.9},
            {"id": 0, "score": 0.8},
            {"id": 2, "score": 0.7},
        ]

        candidates = [
            {"id": "func1", "text": "def foo(): pass"},
            {"id": "func2", "text": "def bar(): pass"},
            {"id": "func3", "text": "def baz(): pass"},
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            result = reranker.rerank("query", candidates, top_k=2)

        # Should return only top 2
        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["rerank_score"] == 0.8

    def test_rerank_truncates_long_code(self):
        """Reranker truncates long code snippets to bound latency."""
        reranker = Reranker()

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [{"id": 0, "score": 0.5}]

        # Very long code (> 2000 chars)
        long_code = "def foo():\n" + "    x = 1\n" * 300

        candidates = [{"id": "long_func", "text": long_code}]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            reranker.rerank("query", candidates)

        # Check that passages passed to ranker were truncated
        call_args = mock_ranker.rerank.call_args
        assert call_args is not None
        # Try to extract passages from RerankRequest (first arg)
        if call_args[0]:
            rerank_request = call_args[0][0]
            passages = rerank_request.passages
        else:
            passages = call_args[1].get("passages", [])
        assert len(passages[0]["text"]) <= 2003  # 2000 + "..."

    def test_rerank_graceful_noop_no_ranker(self):
        """Reranker returns candidates unchanged if ranker unavailable."""
        reranker = Reranker()

        candidates = [
            {"id": "func1", "text": "def foo(): pass", "similarity": 0.5},
            {"id": "func2", "text": "def bar(): pass", "similarity": 0.7},
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=None):
            result = reranker.rerank("query", candidates)

        # Should return candidates unchanged with fallback scores
        assert len(result) == 2
        assert result[0]["id"] == "func1"
        assert result[0]["rerank_score"] == 0.5  # from similarity fallback
        assert result[1]["id"] == "func2"
        assert result[1]["rerank_score"] == 0.7

    def test_rerank_graceful_noop_exception(self):
        """Reranker returns candidates unchanged if ranking fails."""
        reranker = Reranker()

        mock_ranker = MagicMock()
        mock_ranker.rerank.side_effect = RuntimeError("Ranking failed")

        candidates = [
            {"id": "func1", "text": "def foo(): pass"},
            {"id": "func2", "text": "def bar(): pass"},
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            result = reranker.rerank("query", candidates)

        # Should return candidates unchanged with fallback
        assert len(result) == 2
        assert result[0]["id"] == "func1"
        assert result[0]["rerank_score"] == 0.0  # similarity is absent

    def test_rerank_empty_candidates(self):
        """Reranker handles empty candidate list."""
        reranker = Reranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_attaches_score_to_each_result(self):
        """Each reranked result has rerank_score attached."""
        reranker = Reranker()

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 0, "score": 0.9},
            {"id": 1, "score": 0.6},
        ]

        candidates = [
            {"id": "func1", "text": "code1"},
            {"id": "func2", "text": "code2"},
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            result = reranker.rerank("query", candidates)

        # Every result should have rerank_score
        for r in result:
            assert "rerank_score" in r
            assert isinstance(r["rerank_score"], float)

    def test_rerank_preserves_other_fields(self):
        """Reranking preserves non-text/id fields from original candidates."""
        reranker = Reranker()

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [{"id": 0, "score": 0.8}]

        candidates = [
            {
                "id": "func1",
                "text": "code",
                "filepath": "module.py",
                "line_start": 42,
                "similarity": 0.7,
                "custom_field": "preserved",
            }
        ]

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            result = reranker.rerank("query", candidates)

        assert len(result) == 1
        assert result[0]["filepath"] == "module.py"
        assert result[0]["line_start"] == 42
        assert result[0]["similarity"] == 0.7
        assert result[0]["custom_field"] == "preserved"
        assert result[0]["rerank_score"] == 0.8

    def test_model_name_plumbs_through(self):
        """Reranker(model_name=...) plumbs through to _get_ranker."""
        # Test with a different model name
        reranker_l6 = Reranker(model_name="ms-marco-MiniLM-L-6-v2")
        assert reranker_l6.model_name == "ms-marco-MiniLM-L-6-v2"

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [{"id": 0, "score": 0.5}]
        candidates = [{"id": "func1", "text": "code", "similarity": 0.5}]

        with patch(
            "pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker
        ) as mock_get:
            reranker_l6.rerank("query", candidates)
            mock_get.assert_called_once()
            _, kwargs = mock_get.call_args
            assert kwargs["model_name"] == "ms-marco-MiniLM-L-6-v2"

        # Default model name is L-12
        reranker_default = Reranker()
        assert reranker_default.model_name == "ms-marco-MiniLM-L-12-v2"

    def test_truncate_chars_configurable(self):
        """Reranker(truncate_chars=...) honors custom truncation limit."""
        reranker = Reranker(truncate_chars=500)

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [{"id": 0, "score": 0.5}]

        long_code = "def foo():\n" + "    x = 1\n" * 100  # ~1300 chars

        with patch("pipeline.retrieval.reranker._get_ranker", return_value=mock_ranker):
            reranker.rerank("query", [{"id": "long_func", "text": long_code}])

        # Check that text was truncated to 500 chars + "..."
        call_args = mock_ranker.rerank.call_args
        if call_args and call_args[0]:
            passages = call_args[0][0].passages
        else:
            passages = call_args[1].get("passages", [])
        assert len(passages[0]["text"]) <= 503  # 500 + "..."

        # Also test that the default truncate_chars is 2000
        reranker_default = Reranker()
        assert reranker_default.truncate_chars == 2000

    def test_singleton_cache_per_model(self):
        """Two different model names don't collide in the singleton cache."""
        with patch("flashrank.Ranker") as mock_ranker_cls:
            mock_l12 = MagicMock()
            mock_l6 = MagicMock()
            mock_ranker_cls.side_effect = [mock_l12, mock_l6]

            # First call with L-12
            from pipeline.retrieval.reranker import _get_ranker, _ranker_cache

            _ranker_cache.clear()
            result1 = _get_ranker(model_name="ms-marco-MiniLM-L-12-v2")
            assert result1 is mock_l12
            assert "ms-marco-MiniLM-L-12-v2" in _ranker_cache
            assert _ranker_cache["ms-marco-MiniLM-L-12-v2"] is mock_l12

            # Second call with L-6 should load a new instance
            result2 = _get_ranker(model_name="ms-marco-MiniLM-L-6-v2")
            assert result2 is mock_l6
            assert "ms-marco-MiniLM-L-6-v2" in _ranker_cache
            assert _ranker_cache["ms-marco-MiniLM-L-6-v2"] is mock_l6

            # The L-12 entry is still intact
            assert _ranker_cache["ms-marco-MiniLM-L-12-v2"] is mock_l12

            # Calling L-12 again returns the cached instance (no new Ranker call)
            result3 = _get_ranker(model_name="ms-marco-MiniLM-L-12-v2")
            assert result3 is mock_l12
            assert mock_ranker_cls.call_count == 2  # Only two constructor calls total
