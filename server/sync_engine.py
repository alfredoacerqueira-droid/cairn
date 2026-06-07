"""Event-driven background sync engine for keeping the vector index fresh.

This module decouples index/memory sync from the request path. A single-flight
sync ensures concurrent runs don't corrupt ChromaDB writes. Two triggers funnel
into run_sync():
  1. FileWatcher on code changes (debounced)
  2. Periodic background task (catches commits/branch switches)
"""

import logging
import threading
from pathlib import Path

from core.freshness import DBFreshness, changed_files_since_index
from core.repo import RepoManager
from pipeline.ast_parser import ASTParser
from pipeline.indexer import VectorIndexer
from pipeline.memory import MemorySummarizer
from throttle.vram import VRAMPriority

logger = logging.getLogger(__name__)

# Single-flight guard: ensures only one sync runs at a time
_sync_lock = threading.Lock()
_sync_in_progress = False


def run_sync(project_path: Path, vram: VRAMPriority) -> dict:
    """Run a single background sync cycle.

    Does NOT block: if another sync is running, returns immediately with
    {"skipped": True}. Also respects VRAM priority: if gateway is active,
    returns {"skipped": "gateway_busy"}.

    Sync steps (each wrapped in try/except so one failure doesn't kill others):
      1. Acquire single-flight guard
      2. Request VRAM (backs off if gateway busy)
      3. DB sync: reindex modified files, remove deleted ones
      4. Memory sync: summarize git-diff if there were changes
      5. Cache housekeeping (logged only; entries expire naturally)
      6. Release guards

    Args:
        project_path: Root directory of the project.
        vram: Shared VRAMPriority instance from api.py.

    Returns:
        A summary dict with keys: indexed, removed, memory_updated, cache_cleared,
        or "skipped" if sync was skipped (either already running or gateway busy).
    """
    global _sync_in_progress

    # Single-flight: if a sync is already running, return immediately
    with _sync_lock:
        if _sync_in_progress:
            return {"skipped": True}
        _sync_in_progress = True

    try:
        # Resource-polite: request VRAM, back off if gateway busy
        if not vram.request("janitor"):
            return {"skipped": "gateway_busy"}

        try:
            result = {
                "indexed": 0,
                "removed": 0,
                "memory_updated": False,
                "cache_cleared": False,
            }

            # Get current project path as absolute Path
            project_path = Path(project_path).resolve()

            # Step 1: DB sync (modified/deleted files)
            try:
                modified, deleted = changed_files_since_index(project_path)

                if modified or deleted:
                    repo = RepoManager(project_path)
                    indexer = VectorIndexer(
                        chroma_path=repo.get_chroma_path(), project_root=project_path
                    )
                    parser = ASTParser()

                    # Reindex modified/added files
                    for filepath_str in modified:
                        filepath = project_path / filepath_str
                        if filepath.exists():
                            try:
                                ast = parser.parse_file(filepath)
                                indexer.index_ast(ast)
                                result["indexed"] += len(ast.functions)
                                for cls in ast.classes:
                                    result["indexed"] += len(cls.methods)
                            except Exception as e:
                                logger.debug(
                                    "Failed to sync %s: %s",
                                    filepath,
                                    e,
                                )

                    # Remove deleted files from index
                    for filepath_str in deleted:
                        try:
                            indexer.remove_file(filepath_str)
                            result["removed"] += 1
                        except Exception as e:
                            logger.debug(
                                "Failed to remove %s: %s",
                                filepath_str,
                                e,
                            )

                    # Mark as re-indexed
                    freshness = DBFreshness(project_path)
                    freshness.mark_indexed(freshness.get_current_commit())

                    if result["indexed"] > 0 or result["removed"] > 0:
                        logger.info(
                            "DB sync: indexed %d, removed %d",
                            result["indexed"],
                            result["removed"],
                        )

            except Exception as e:
                logger.error("DB sync failed: %s", e)

            # Step 2: Memory sync (if there were changes)
            try:
                if modified or deleted:
                    summarizer = MemorySummarizer(repo_path=project_path)
                    summarizer.summarize_and_record()
                    result["memory_updated"] = True
                    logger.debug("Memory sync completed")
            except Exception as e:
                logger.debug("Memory sync skipped: %s", e)

            # Step 3: Cache housekeeping (logged only)
            # SessionCache is commit-keyed so stale entries are naturally bypassed.
            # Do NOT clear a live cache handle — we don't own it here; the assembler
            # holds its own instance. Commit-keying handles automatic invalidation
            # on branch switch/commit.
            logger.debug("Cache is naturally invalidated via commit-keying")

            return result

        finally:
            # Always release VRAM
            vram.release("janitor")

    finally:
        # Always release single-flight guard
        with _sync_lock:
            _sync_in_progress = False


def should_sync(
    project_path: Path,
    last_check: float,
    interval: float = 60.0,
) -> bool:
    """Cheap throttle gate for sync triggers.

    Returns False if not enough time has passed since last_check.
    Otherwise checks if the index is behind and returns True only if
    commits_behind > 0 OR the caller forces it.

    Args:
        project_path: Root directory of the project.
        last_check: Time (from time.time()) of the last check.
        interval: Minimum seconds between checks.

    Returns:
        True if sync should run; False if throttled or not behind.
    """
    import time

    now = time.time()
    if now - last_check < interval:
        return False

    try:
        project_path = Path(project_path).resolve()
        freshness = DBFreshness(project_path)
        info = freshness.check_freshness()

        # Only sync if behind on commits
        return info.get("commits_behind", 0) > 0

    except Exception:
        # If we can't check freshness, skip the sync
        return False
