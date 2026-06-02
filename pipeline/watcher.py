"""File system watcher for automatic re-indexing."""

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def _matches_pattern(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any pattern."""
    from fnmatch import fnmatch

    filename = Path(filepath).name
    return any(fnmatch(filename, p) for p in patterns)


def _matches_exclude(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any exclude pattern."""
    from fnmatch import fnmatch

    return any(fnmatch(filepath, p) for p in patterns)


class CodeFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        on_change: Callable[[str], None],
        file_patterns: list[str],
        exclude_patterns: list[str],
    ):
        super().__init__()
        self.on_change = on_change
        self.file_patterns = file_patterns
        self.exclude_patterns = exclude_patterns
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._debounce_s = 0.5

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def _schedule(self, filepath: str):
        if not _matches_pattern(filepath, self.file_patterns):
            return
        if _matches_exclude(filepath, self.exclude_patterns):
            return

        with self._lock:
            self._pending[filepath] = time.time()

    def flush_pending(self):
        """Process pending file changes after debounce window."""
        with self._lock:
            now = time.time()
            to_process = {
                fp: ts for fp, ts in self._pending.items() if now - ts >= self._debounce_s
            }
            for fp, ts in to_process.items():
                del self._pending[fp]

        for fp in to_process:
            self.on_change(fp)


class FileWatcher:
    def __init__(
        self,
        project_path: str | Path,
        on_change: Callable[[str], None],
        file_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        debounce_s: float = 0.5,
    ):
        self.project_path = Path(project_path)
        self.on_change = on_change
        self.file_patterns = file_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or [
            ".git/**",
            "__pycache__/**",
            ".venv/**",
            "node_modules/**",
        ]

        self.observer = Observer()
        self.handler = CodeFileHandler(
            on_change=on_change,
            file_patterns=self.file_patterns,
            exclude_patterns=self.exclude_patterns,
        )
        self.handler._debounce_s = debounce_s

        self._running = False
        self._flusher_thread: Optional[threading.Thread] = None

    def start(self):
        """Start watching the project directory."""
        if self._running:
            return

        self.observer.schedule(self.handler, str(self.project_path), recursive=True)
        self.observer.start()

        self._running = True
        self._flusher_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher_thread.start()

    def stop(self):
        """Stop watching."""
        self._running = False
        self.observer.stop()
        self.observer.join(timeout=5.0)
        if self._flusher_thread:
            self._flusher_thread.join(timeout=5.0)

    def _flush_loop(self):
        """Periodically flush pending file changes."""
        while self._running:
            time.sleep(0.5)
            self.handler.flush_pending()

    @property
    def is_running(self) -> bool:
        return self._running
