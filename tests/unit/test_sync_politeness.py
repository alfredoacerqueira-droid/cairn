"""Unit tests for sync politeness contract.

Tests the resource-politeness guarantees:
1. When gateway is active (holds VRAM slot), run_sync backs off (skips)
2. When gateway is idle, run_sync proceeds
3. Single-flight ensures only one sync runs at a time
4. VRAM is always released, even on exceptions
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from server.sync_engine import run_sync
from throttle.vram import VRAMPriority


class TestSyncPoliteness:
    """Test the politeness contract: janitor backs off when gateway is active."""

    def test_politeness_backs_off_when_gateway_busy(self):
        """TASK 1a: When gateway holds VRAM, sync skips and does no heavy work."""
        vram = VRAMPriority()
        project_path = Path(".")

        # Simulate live request: gateway claims VRAM
        vram.request("gateway")

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = (["modified.py"], [])

            with patch("server.sync_engine.VectorIndexer") as mock_indexer_class:
                mock_indexer = MagicMock()
                mock_indexer_class.return_value = mock_indexer

                with patch("server.sync_engine.ASTParser") as mock_parser_class:
                    mock_parser = MagicMock()
                    mock_parser_class.return_value = mock_parser

                    result = run_sync(project_path, vram)

                    # Assertion 1: sync returned early with gateway_busy
                    assert result.get("skipped") == "gateway_busy"

                    # Assertion 2: no heavy indexing work happened
                    mock_indexer_class.assert_not_called()
                    mock_parser_class.assert_not_called()

        vram.release("gateway")

    def test_politeness_proceeds_when_gateway_idle(self):
        """TASK 1a: When gateway is idle, sync proceeds normally."""
        vram = VRAMPriority()
        project_path = Path(".")

        # Gateway is idle (no active request)
        assert not vram.gateway_active

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = (["modified.py"], [])

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

                        mock_ast = MagicMock()
                        mock_ast.functions = [MagicMock()]
                        mock_ast.classes = []
                        mock_parser.parse_file.return_value = mock_ast

                        with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                            mock_freshness = MagicMock()
                            mock_freshness_class.return_value = mock_freshness
                            mock_freshness.get_current_commit.return_value = "commit1"

                            with patch("pathlib.Path.exists", return_value=True):
                                result = run_sync(project_path, vram)

                                # Assertion 1: sync did NOT skip
                                assert result.get("skipped") != "gateway_busy"

                                # Assertion 2: heavy work happened (parser + indexer called)
                                mock_parser_class.assert_called_once()
                                mock_indexer_class.assert_called_once()
                                mock_indexer.index_ast.assert_called_once()

                                # Assertion 3: result shows work was done
                                assert result["indexed"] == 1

    def test_politeness_vram_always_released_on_exception(self):
        """TASK 1c: VRAM is released in finally block even on exception."""
        vram = VRAMPriority()
        project_path = Path(".")

        # Simulate gateway holding VRAM, then release it manually
        # and verify janitor can acquire it

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.side_effect = RuntimeError("Simulated parse error")

            result = run_sync(project_path, vram)

            # Should complete despite error
            assert result is not None

            # Assertion: janitor VRAM is released (count = 0)
            assert vram.janitor_active is False

            # Verify a subsequent sync can acquire VRAM
            with patch("server.sync_engine.changed_files_since_index") as mock_changed2:
                mock_changed2.return_value = ([], [])

                with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                    mock_freshness = MagicMock()
                    mock_freshness_class.return_value = mock_freshness
                    mock_freshness.get_current_commit.return_value = "commit1"

                    result2 = run_sync(project_path, vram)
                    # Should NOT skip gateway_busy (no gateway is active)
                    assert result2.get("skipped") != "gateway_busy"


class TestSyncSingleFlightUnderLoad:
    """Test single-flight: only one sync runs at a time, even without VRAM."""

    def test_single_flight_blocks_concurrent_syncs(self):
        """TASK 1b: Two concurrent calls; second one is skipped by single-flight."""
        vram = VRAMPriority()
        project_path = Path(".")

        results = []
        start_sync = threading.Event()
        sync_started = threading.Event()

        with patch("server.sync_engine.changed_files_since_index") as mock_changed:
            mock_changed.return_value = ([], [])

            with patch("server.sync_engine.DBFreshness") as mock_freshness_class:
                mock_freshness = MagicMock()
                mock_freshness_class.return_value = mock_freshness
                mock_freshness.get_current_commit.return_value = "commit1"

                # Monkey-patch run_sync's entry to synchronize threads
                original_changed = mock_changed

                def blocking_changed(*args):
                    # First thread signals, then both wait
                    sync_started.set()
                    start_sync.wait(timeout=2)
                    return original_changed.return_value

                mock_changed.side_effect = blocking_changed

                def call_sync():
                    result = run_sync(project_path, vram)
                    results.append(result)

                # Start two threads
                t1 = threading.Thread(target=call_sync)
                t2 = threading.Thread(target=call_sync)

                t1.start()

                # Wait for t1 to enter the changed_files call, then start t2
                sync_started.wait(timeout=2)
                t2.start()

                # Now let both threads proceed
                start_sync.set()

                t1.join(timeout=5)
                t2.join(timeout=5)

        # Both threads should complete
        assert len(results) == 2, f"Expected 2 results, got {len(results)}: {results}"

        # One should be skipped (single-flight), one should proceed
        skipped = [r for r in results if r.get("skipped") is True]
        not_skipped = [r for r in results if r.get("skipped") is not True]

        assert len(skipped) == 1, f"Expected 1 skipped, got {len(skipped)}: {results}"
        assert len(not_skipped) == 1, f"Expected 1 not skipped, got {len(not_skipped)}: {results}"


class TestSyncConcurrencyWithGateway:
    """Test concurrent gateway requests don't block due to janitor."""

    def test_gateway_request_not_blocked_by_pending_janitor(self):
        """When janitor is waiting for VRAM, gateway should still be responsive."""
        vram = VRAMPriority()

        # Start janitor requesting VRAM (it will wait because gateway is busy)
        janitor_started = threading.Event()
        janitor_got_vram = threading.Event()

        def janitor_waits():
            janitor_started.set()
            # Request VRAM; gateway is active so this returns False
            result = vram.request("janitor")
            if result:
                janitor_got_vram.set()
                vram.release("janitor")

        janitor_thread = threading.Thread(target=janitor_waits)
        janitor_thread.start()

        janitor_started.wait(timeout=1)

        # Gateway arrives and requests VRAM (should succeed immediately)
        start = time.time()
        gateway_result = vram.request("gateway")
        gateway_elapsed = time.time() - start

        # Assertions
        assert gateway_result is True, "Gateway should always get VRAM"
        assert gateway_elapsed < 0.1, "Gateway should not be blocked by janitor waits"

        vram.release("gateway")
        janitor_thread.join(timeout=1)
