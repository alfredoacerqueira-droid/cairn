"""Unit tests for pipeline/watcher.py — FileWatcher."""

import time
from pathlib import Path

from pipeline.watcher import FileWatcher


class TestFileWatcher:
    def test_init_defaults(self):
        watcher = FileWatcher(
            project_path=Path("/tmp"),
            on_change=lambda fp: None,
        )
        assert watcher.is_running is False

    def test_start_and_stop(self):
        watcher = FileWatcher(
            project_path=Path("/tmp"),
            on_change=lambda fp: None,
        )
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()
        assert watcher.is_running is False

    def test_triggers_on_file_change(self, tmp_path):
        results = []

        def handler(fp: str):
            results.append(fp)

        watcher = FileWatcher(
            project_path=tmp_path,
            on_change=handler,
            file_patterns=["*.py"],
            debounce_s=0.1,
        )
        watcher.start()

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        time.sleep(1.0)

        watcher.stop()
        assert len(results) > 0, "Expected at least one file change detected"
        assert str(test_file) in results

    def test_exclude_patterns(self, tmp_path):
        results = []

        def handler(fp: str):
            results.append(fp)

        exclude_dir = tmp_path / "excluded"
        exclude_dir.mkdir()

        watcher = FileWatcher(
            project_path=tmp_path,
            on_change=handler,
            file_patterns=["*.py"],
            exclude_patterns=["**/excluded/**"],
            debounce_s=0.1,
        )
        watcher.start()

        test_file = exclude_dir / "test.py"
        test_file.write_text("y = 2")

        time.sleep(1.0)

        watcher.stop()
        assert str(test_file) not in results

    def test_ignores_non_python_files(self, tmp_path):
        results = []

        def handler(fp: str):
            results.append(fp)

        watcher = FileWatcher(
            project_path=tmp_path,
            on_change=handler,
            file_patterns=["*.py"],
            debounce_s=0.1,
        )
        watcher.start()

        test_file = tmp_path / "test.txt"
        test_file.write_text("not python")

        time.sleep(1.0)

        watcher.stop()
        assert not results
