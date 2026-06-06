"""Tests for context assembler compression."""

from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from server.context_assembler import ContextAssembler


class TestAssemblerCompression:
    @patch("server.context_assembler.load_config")
    @patch("server.context_assembler.RepoManager")
    @patch("server.context_assembler.VectorIndexer")
    def test_assemble_context_compressed_when_enabled(
        self,
        mock_vector_indexer,
        mock_repo_manager,
        mock_load_config,
    ):
        """Test that assemble_context compresses output when enabled."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock config with compression enabled
            mock_cfg = MagicMock()
            mock_cfg.cache.enabled = False
            mock_cfg.compression.enabled = True
            mock_cfg.compression.level = "minimal"
            mock_cfg.retrieval.rerank_enabled = False
            mock_cfg.retrieval.min_confidence = 0.0  # Disable confidence guard
            mock_load_config.return_value = mock_cfg

            # Mock repo manager
            mock_repo_mgr = MagicMock()
            mock_repo_mgr.get_chroma_path.return_value = project_path / "chroma"
            mock_repo_mgr.load_repo_map.return_value = {"test.py": {"functions": []}}
            mock_repo_mgr.load_memory.return_value = ""
            mock_repo_manager.return_value = mock_repo_mgr

            # Create assembler
            assembler = ContextAssembler(project_path=project_path)

            # Mock semantic_search to return a result
            assembler.semantic_search = MagicMock(
                return_value=[
                    {
                        "filepath": "test.py",
                        "function": "test_fn",
                        "line_start": 1,
                        "line_end": 10,
                        "code": "def test_fn():\n    pass",
                        "similarity": 0.9,
                        "raw_cosine": 0.9,
                        "rerank_score": 0.0,
                    }
                ]
            )

            result = assembler.assemble_context("test query")

            # Result should:
            # 1. Be compressed (have the compression marker)
            assert "[already-compressed]" in result

            # 2. Still contain content (not empty after compression)
            assert "test.py" in result or "Codebase Context" in result

            # 3. Be different from uncompressed (should have fewer chars due to
            #    comments removed)
            # Note: we can't directly compare to uncompressed, but we can verify
            # the marker exists

    @patch("server.context_assembler.load_config")
    @patch("server.context_assembler.RepoManager")
    @patch("server.context_assembler.VectorIndexer")
    def test_assemble_context_not_compressed_when_disabled(
        self,
        mock_vector_indexer,
        mock_repo_manager,
        mock_load_config,
    ):
        """Test that assemble_context skips compression when disabled."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock config with compression disabled
            mock_cfg = MagicMock()
            mock_cfg.cache.enabled = False
            mock_cfg.compression.enabled = False
            mock_cfg.compression.level = "minimal"
            mock_cfg.retrieval.rerank_enabled = False
            mock_cfg.retrieval.min_confidence = 0.0
            mock_load_config.return_value = mock_cfg

            # Mock repo manager
            mock_repo_mgr = MagicMock()
            mock_repo_mgr.get_chroma_path.return_value = project_path / "chroma"
            mock_repo_mgr.load_repo_map.return_value = {"test.py": {"functions": []}}
            mock_repo_mgr.load_memory.return_value = ""
            mock_repo_manager.return_value = mock_repo_mgr

            assembler = ContextAssembler(project_path=project_path)

            assembler.semantic_search = MagicMock(
                return_value=[
                    {
                        "filepath": "test.py",
                        "function": "test_fn",
                        "line_start": 1,
                        "line_end": 10,
                        "code": "def test_fn():\n    pass",
                        "similarity": 0.9,
                        "raw_cosine": 0.9,
                        "rerank_score": 0.0,
                    }
                ]
            )

            result = assembler.assemble_context("test query")

            # Result should NOT have compression marker
            assert "[already-compressed-by-gateway]" not in result

            # Should still have content
            assert "Codebase Context" in result

    @patch("server.context_assembler.load_config")
    @patch("server.context_assembler.RepoManager")
    @patch("server.context_assembler.VectorIndexer")
    def test_confidence_guard_not_compressed(
        self,
        mock_vector_indexer,
        mock_repo_manager,
        mock_load_config,
    ):
        """Test that confidence-guard rejection is not compressed."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock config with low confidence guard
            mock_cfg = MagicMock()
            mock_cfg.cache.enabled = False
            mock_cfg.compression.enabled = True
            mock_cfg.compression.level = "minimal"
            mock_cfg.retrieval.rerank_enabled = False
            mock_cfg.retrieval.min_confidence = 0.9  # High threshold
            mock_load_config.return_value = mock_cfg

            # Mock repo manager
            mock_repo_mgr = MagicMock()
            mock_repo_mgr.get_chroma_path.return_value = project_path / "chroma"
            mock_repo_manager.return_value = mock_repo_mgr

            assembler = ContextAssembler(project_path=project_path)

            # Mock semantic_search to return low-confidence result
            assembler.semantic_search = MagicMock(return_value=[])

            result = assembler.assemble_context("test query")

            # Result should be the rejection message, not compressed
            assert result == "*No confident matches found for this query.*"
            assert "[already-compressed-by-gateway]" not in result

    @patch("server.context_assembler.load_config")
    @patch("server.context_assembler.RepoManager")
    @patch("server.context_assembler.VectorIndexer")
    def test_assemble_context_persistent_cache_hit(
        self,
        mock_vector_indexer,
        mock_repo_manager,
        mock_load_config,
    ):
        """Test that persistent cache is used for cross-process warmth."""
        with TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock config with persistent cache enabled
            mock_cfg = MagicMock()
            mock_cfg.cache.enabled = True
            mock_cfg.cache.max_entries = 100
            mock_cfg.cache.ttl_seconds = 300
            mock_cfg.compression.enabled = True
            mock_cfg.compression.level = "minimal"
            mock_cfg.retrieval.rerank_enabled = False
            mock_cfg.retrieval.min_confidence = 0.0
            mock_load_config.return_value = mock_cfg

            # Mock repo manager
            mock_repo_mgr = MagicMock()
            mock_repo_mgr.get_chroma_path.return_value = project_path / "chroma"
            mock_repo_mgr.load_repo_map.return_value = {}
            mock_repo_mgr.load_memory.return_value = ""
            mock_repo_manager.return_value = mock_repo_mgr

            # First instance: populate cache
            assembler1 = ContextAssembler(project_path=project_path)
            assembler1.semantic_search = MagicMock(
                return_value=[
                    {
                        "filepath": "test.py",
                        "function": "fn",
                        "line_start": 1,
                        "line_end": 10,
                        "code": "def fn(): pass",
                        "similarity": 0.9,
                        "raw_cosine": 0.9,
                        "rerank_score": 0.0,
                    }
                ]
            )
            result1 = assembler1.assemble_context("query1")
            assert "[already-compressed]" in result1

            # Second instance: should hit persistent cache (no search needed)
            assembler2 = ContextAssembler(project_path=project_path)
            assembler2.semantic_search = MagicMock()  # Should not be called

            result2 = assembler2.assemble_context("query1")

            # Results should match
            assert result2 == result1
            # semantic_search should NOT have been called (hit persistent cache)
            assembler2.semantic_search.assert_not_called()
