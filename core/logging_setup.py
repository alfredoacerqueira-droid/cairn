"""Centralized logging configuration for Cairn (CLI, MCP, tests).

Configure the root logger with DEBUG or WARNING level depending on the
debug flag or CAIRN_DEBUG env var. Ensures:
  - CLI: logs to STDERR (user-facing)
  - MCP: logs to STDERR or file, NEVER stdout (preserves JSON-RPC protocol)
  - Tests: logs to stderr via caplog fixture
"""

import logging
import os
import sys
from pathlib import Path


def configure_logging(debug: bool = False, log_file: str | None = None) -> None:
    """Configure root logger for Cairn.

    Args:
        debug: If True, set DEBUG level; otherwise WARNING.
        log_file: Optional file path for log output (in addition to stderr).
                  If not provided and debug=True, uses ~/.cache/cairn/cairn.log.
    """
    # Honor CAIRN_DEBUG env var if set
    if os.environ.get("CAIRN_DEBUG", "").lower() in ("1", "true"):
        debug = True

    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Set root level
    level = logging.DEBUG if debug else logging.WARNING

    root_logger.setLevel(level)

    # Format: compact, readable, with milliseconds
    fmt = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # STDERR handler (always present, logs go here)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # Optional file handler (debug mode only, if not explicitly disabled)
    if debug and log_file is None:
        # Default: ~/.cache/cairn/cairn.log
        cache_dir = Path.home() / ".cache" / "cairn"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(cache_dir / "cairn.log")

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.DEBUG)  # File gets all DEBUG+
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Non-fatal: if file logging fails, continue without it
            root_logger.warning(f"Failed to open log file {log_file}: {e}")
