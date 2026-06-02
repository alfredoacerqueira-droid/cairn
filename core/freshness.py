"""Stale DB detection and handling."""

import subprocess
from pathlib import Path
from typing import Optional


def changed_files_since_index(project_path: Path) -> tuple[list[str], list[str]]:
    """Return (modified_or_added, deleted) source files since last indexed commit.

    Uses git diff --name-status to track changes between the persisted
    last_indexed_commit and current HEAD.

    Args:
        project_path: Root directory of the project.

    Returns:
        A tuple of (modified_added, deleted) file paths (relative to project root).
        If no last commit recorded or git is unavailable, returns ([], []).
        Returns absolute paths (strings).
    """
    project_path = Path(project_path)

    # Load the last indexed commit
    freshness = DBFreshness(project_path)
    last_commit = freshness._last_indexed_commit

    if not last_commit:
        # No prior index, can't compute delta
        return [], []

    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", f"{last_commit}..HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Git error (e.g., invalid commit), return empty
            return [], []

        modified_added = []
        deleted = []

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            status, filepath = parts[0], parts[1]
            if status == "D":
                deleted.append(filepath)
            else:
                # All other statuses (M, A, R, C, etc.) are modifications/additions
                modified_added.append(filepath)

        return modified_added, deleted

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        # Git error or not a git repo
        return [], []


class DBFreshness:
    def __init__(
        self,
        project_path: Optional[Path] = None,
        quick_threshold: int = 1000,
        full_threshold: int = 10000,
    ):
        if project_path is None:
            project_path = Path.cwd()
        self.project_path = Path(project_path)
        self._last_indexed_commit: str | None = None
        self.quick_threshold = quick_threshold
        self.full_threshold = full_threshold
        self._load_last_indexed_commit()

    def _get_last_indexed_path(self) -> Path:
        """Get the path to the last_indexed.txt file."""
        return self.project_path / ".cairn" / "last_indexed.txt"

    def _load_last_indexed_commit(self):
        """Load the last indexed commit from persisted storage if available."""
        path = self._get_last_indexed_path()
        if path.exists():
            try:
                content = path.read_text().strip()
                if content:
                    self._last_indexed_commit = content
            except Exception:
                pass

    def _git_command(self, *args) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, ["git", *args], result.stdout, result.stderr
            )
        return result.stdout.strip()

    def get_current_commit(self) -> str:
        try:
            return self._git_command("rev-parse", "HEAD")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    def count_commits_behind(self, from_commit: str) -> int:
        try:
            result = self._git_command("rev-list", "--count", f"{from_commit}..HEAD")
            return int(result)
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return -1

    def get_changed_files(self, from_commit: str) -> list[str]:
        try:
            output = self._git_command("diff", "--name-only", from_commit, "HEAD")
            return [f for f in output.split("\n") if f.strip()]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []

    def check_freshness(self) -> dict:
        current = self.get_current_commit()
        last = self._last_indexed_commit
        behind = 0

        if last:
            behind = self.count_commits_behind(last)

        return {
            "current_commit": current,
            "last_indexed_commit": last,
            "commits_behind": behind,
        }

    def needs_quick_reindex(self) -> bool:
        info = self.check_freshness()
        return self.quick_threshold <= info["commits_behind"] < self.full_threshold

    def needs_full_reindex(self) -> bool:
        info = self.check_freshness()
        return info["commits_behind"] >= self.full_threshold

    def mark_indexed(self, commit_hash: str):
        """Mark the given commit as indexed and persist it."""
        self._last_indexed_commit = commit_hash
        path = self._get_last_indexed_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit_hash)
