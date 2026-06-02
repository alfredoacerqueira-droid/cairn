"""Unit tests for server/sync_engine.py — event-driven background sync."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from server.sync_engine import run_sync, should_sync
from throttle.vram import VRAMPriority


class TestRunSyncSingleFlight:
    """Test single-flight behavior: only one sync at a time."""

    def test_single_flight_blocks_concurrent_calls(self):
        """Two concurrent run_sync calls: second returns skipped."""
        vram = VRAMPriority()
        project_path = Path(".")

        results = []
        barrier = threading.Barrier(2)  # Synchronize thread starts

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = ([], [])
            with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                mock_freshness = MagicMock()
                mock_freshness_class.return_value = mock_freshness
                mock_freshness.get_current_commit.return_value = "abc123"

                def call_sync():
                    barrier.wait()  # Ensure both threads start nearly simultaneously
                    result = run_sync(project_path, vram)
                    results.append(result)

                # Run both in threads
                t1 = threading.Thread(target=call_sync)
                t2 = threading.Thread(target=call_sync)

                t1.start()
                t2.start()
                t1.join(timeout=5)
                t2.join(timeout=5)

                # Verify one succeeded and one was skipped
                assert len(results) == 2

                # One should be skipped (sync_in_progress=True)
                skipped_count = sum(1 for r in results if r.get("skipped") is True)
                assert skipped_count == 1

    def test_single_flight_reset_after_completion(self):
        """After a sync completes, the next one should run."""
        vram = VRAMPriority()
        project_path = Path(".")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = ([], [])
            with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                mock_freshness = MagicMock()
                mock_freshness_class.return_value = mock_freshness
                mock_freshness.get_current_commit.return_value = "abc123"

                # First sync should succeed
                result1 = run_sync(project_path, vram)
                assert result1.get("skipped") is not True

                # Second sync should also succeed (guard is reset)
                result2 = run_sync(project_path, vram)
                assert result2.get("skipped") is not True


class TestRunSyncGatewayBusy:
    """Test VRAM backoff: janitor backs off when gateway is active."""

    def test_returns_skipped_when_gateway_busy(self):
        """When vram.request('janitor') returns False, sync is skipped."""
        vram = VRAMPriority()
        project_path = Path(".")

        # Request VRAM as gateway (never releases)
        vram.request("gateway")

        with patch("server.sync_engine.VectorIndexer") as mock_indexer_class:
            with patch("server.sync_engine.changed_files_since_index") as mock_changed:
                # These should NOT be called
                mock_changed.return_value = (["file.py"], [])

                result = run_sync(project_path, vram)

                # Should be skipped because gateway is busy
                assert result.get("skipped") == "gateway_busy"

                # VectorIndexer should NOT have been instantiated
                mock_indexer_class.assert_not_called()

        vram.release("gateway")

    def test_vram_always_released_even_on_error(self):
        """VRAM is released in finally block even if sync fails."""
        vram = VRAMPriority()
        project_path = Path(".")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            # Simulate an error during DB sync
            mock_changed.side_effect = RuntimeError("Simulated error")

            result = run_sync(project_path, vram)

            # Should still complete (error is caught)
            assert result is not None

            # Janitor count should be 0 (VRAM released)
            assert vram.janitor_active is False


class TestShouldSyncThrottle:
    """Test should_sync throttle gate."""

    def test_returns_false_within_interval(self):
        """Within interval, should_sync returns False."""
        project_path = Path(".")
        now = time.time()
        last_check = now - 10.0  # 10 seconds ago
        interval = 30.0  # 30-second interval

        # Should return False because not enough time has passed
        result = should_sync(project_path, last_check, interval)
        assert result is False

    def test_returns_false_after_interval_if_not_behind(self):
        """After interval, but index is not behind: returns False."""
        project_path = Path(".")
        now = time.time()
        last_check = now - 60.0  # 60 seconds ago
        interval = 30.0  # 30-second interval

        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
            mock_freshness = MagicMock()
            mock_freshness_class.return_value = mock_freshness
            mock_freshness.check_freshness.return_value = {
                "commits_behind": 0,
            }

            result = should_sync(project_path, last_check, interval)
            assert result is False

    def test_returns_true_after_interval_if_behind(self):
        """After interval and index is behind: returns True."""
        project_path = Path(".")
        now = time.time()
        last_check = now - 60.0  # 60 seconds ago
        interval = 30.0  # 30-second interval

        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
            mock_freshness = MagicMock()
            mock_freshness_class.return_value = mock_freshness
            mock_freshness.check_freshness.return_value = {
                "commits_behind": 5,
            }

            result = should_sync(project_path, last_check, interval)
            assert result is True

    def test_catches_freshness_errors_and_returns_false(self):
        """If freshness check fails, returns False gracefully."""
        project_path = Path(".")
        now = time.time()
        last_check = now - 60.0

        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
            mock_freshness_class.side_effect = RuntimeError("Git error")

            result = should_sync(project_path, last_check, interval=30.0)
            assert result is False


class TestRunSyncDBPath:
    """Test DB sync logic: modified and deleted files."""

    def test_reindexes_modified_files(self):
        """Changed files are re-indexed via ASTParser + VectorIndexer."""
        vram = VRAMPriority()
        project_path = Path("/tmp/test_project")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = (["server/api.py", "core/cache.py"], [])

            with patch("server.sync_engine.RepoManager") as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_chroma_path.return_value = "/tmp/chroma"

                with patch("server.sync_engine.VectorIndexer") as mock_indexer_class:
                    mock_indexer = MagicMock()
                    mock_indexer_class.return_value = mock_indexer

                    with patch("server.sync_engine.ASTParser") as mock_parser_class:
                        mock_parser = MagicMock()
                        mock_parser_class.return_value = mock_parser

                        # Mock AST result
                        mock_ast = MagicMock()
                        mock_ast.functions = [MagicMock(), MagicMock()]
                        mock_ast.classes = [MagicMock(methods=[MagicMock(), MagicMock()])]
                        mock_parser.parse_file.return_value = mock_ast

                        # Mock freshness
                        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                            mock_freshness = MagicMock()
                            mock_freshness_class.return_value = mock_freshness
                            mock_freshness.get_current_commit.return_value = "xyz789"

                            # Mock Path.exists() to return True for test files
                            with patch("pathlib.Path.exists", return_value=True):
                                result = run_sync(project_path, vram)

                                # Verify index_ast was called for each modified file
                                assert mock_indexer.index_ast.call_count == 2

                                # Verify mark_indexed was called
                                mock_freshness.mark_indexed.assert_called_once_with("xyz789")

                                # Verify results
                                # Each file has 2 functions + 1 class with 2 methods
                                # = 2 + 2 = 4 items per file, times 2 files = 8
                                assert result["indexed"] == 8
                                assert result["removed"] == 0

    def test_removes_deleted_files(self):
        """Deleted files are removed from the index."""
        vram = VRAMPriority()
        project_path = Path("/tmp/test_project")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = ([], ["old_module.py", "deprecated.py"])

            with patch("server.sync_engine.RepoManager") as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_chroma_path.return_value = "/tmp/chroma"

                with patch("server.sync_engine.VectorIndexer") as mock_indexer_class:
                    mock_indexer = MagicMock()
                    mock_indexer_class.return_value = mock_indexer

                    with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                        mock_freshness = MagicMock()
                        mock_freshness_class.return_value = mock_freshness
                        mock_freshness.get_current_commit.return_value = "xyz789"

                        result = run_sync(project_path, vram)

                        # Verify remove_file was called for each deleted file
                        assert mock_indexer.remove_file.call_count == 2
                        mock_indexer.remove_file.assert_any_call("old_module.py")
                        mock_indexer.remove_file.assert_any_call("deprecated.py")

                        # Verify results
                        assert result["indexed"] == 0
                        assert result["removed"] == 2

    def test_memory_sync_called_when_files_changed(self):
        """When files change, MemorySummarizer.summarize_and_record is called."""
        vram = VRAMPriority()
        project_path = Path("/tmp/test_project")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = (["file.py"], [])

            with patch("server.sync_engine.RepoManager") as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_chroma_path.return_value = "/tmp/chroma"

                with patch("server.sync_engine.VectorIndexer"):
                    with patch("server.sync_engine.ASTParser") as mock_parser_class:
                        mock_parser = MagicMock()
                        mock_parser_class.return_value = mock_parser
                        mock_ast = MagicMock()
                        mock_ast.functions = []
                        mock_ast.classes = []
                        mock_parser.parse_file.return_value = mock_ast

                        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                            mock_freshness = MagicMock()
                            mock_freshness_class.return_value = mock_freshness
                            mock_freshness.get_current_commit.return_value = "abc"

                            with patch("server.sync_engine.MemorySummarizer") as mock_memory_class:
                                mock_memory = MagicMock()
                                mock_memory_class.return_value = mock_memory

                                result = run_sync(project_path, vram)

                                # Verify MemorySummarizer was instantiated and called
                                mock_memory_class.assert_called_once()
                                mock_memory.summarize_and_record.assert_called_once()

                                # Verify result
                                assert result["memory_updated"] is True

    def test_memory_sync_skipped_when_no_changes(self):
        """When no files changed, MemorySummarizer is not called."""
        vram = VRAMPriority()
        project_path = Path("/tmp/test_project")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = ([], [])

            with patch("server.sync_engine.MemorySummarizer") as mock_memory_class:
                with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                    mock_freshness = MagicMock()
                    mock_freshness_class.return_value = mock_freshness

                    result = run_sync(project_path, vram)

                    # MemorySummarizer should NOT be called
                    mock_memory_class.assert_not_called()

                    # Verify result
                    assert result["memory_updated"] is False

    def test_exceptions_dont_stop_sync(self):
        """A failure in one step doesn't stop the others."""
        vram = VRAMPriority()
        project_path = Path("/tmp/test_project")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = (["file.py"], [])

            with patch("server.sync_engine.RepoManager") as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_chroma_path.return_value = "/tmp/chroma"

                with patch("server.sync_engine.VectorIndexer") as mock_indexer_class:
                    # Indexer raises an exception
                    mock_indexer_class.side_effect = RuntimeError("Indexer failed")

                    with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                        mock_freshness = MagicMock()
                        mock_freshness_class.return_value = mock_freshness

                        # Should not raise; should catch and log
                        result = run_sync(project_path, vram)

                        # Should still complete
                        assert result is not None
                        # VRAM should be released
                        assert vram.janitor_active is False
