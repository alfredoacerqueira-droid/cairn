"""Git diff summarization using local Ollama model."""

import subprocess
from pathlib import Path
from typing import Optional

from server.ollama_client import OllamaClient


class MemorySummarizer:
    def __init__(
        self,
        repo_path: str | Path,
        ollama_client: Optional[OllamaClient] = None,
        model: str = "qwen2.5-coder:3b",
        memory_file: Optional[str] = None,
        max_entries: int = 50,
    ):
        self.repo_path = Path(repo_path)
        self.ollama = ollama_client or OllamaClient()
        self.model = model
        self.memory_file = memory_file or str(self.repo_path / ".cairn" / "memory.md")
        self.max_entries = max_entries
        self._compaction_threshold = 200

    def get_recent_diff(self) -> str:
        """Get git diff for the last commit."""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return ""
            return result.stdout.strip()
        except Exception:
            return ""

    def get_uncommitted_diff(self) -> str:
        """Get git diff for uncommitted changes (both staged and unstaged)."""
        try:
            # Staged changes
            staged = subprocess.run(
                ["git", "diff", "--staged"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()

            # Unstaged changes
            unstaged = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()

            return f"{staged}\n{unstaged}".strip()
        except Exception:
            return ""

    def summarize_diff(self, diff: str) -> str:
        """Use local Qwen model to summarize a git diff."""
        if not diff or len(diff.strip()) < 10:
            return "No significant changes."

        prompt = f"Summarize this git diff in one short sentence:\n\n{diff[:2000]}"

        try:
            response = self.ollama.generate(prompt=prompt, model=self.model)
            return response.strip()
        except Exception:
            return "Failed to generate summary."

    def summarize_and_record(self, diff: Optional[str] = None):
        """Summarize a diff and append to MEMORY.md."""
        if diff is None:
            diff = self.get_recent_diff()

        if not diff:
            return

        summary = self.summarize_diff(diff)
        self.append_to_memory(summary)

    def append_to_memory(self, entry: str):
        """Append a summary entry to MEMORY.md with timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = entry.replace("\n", " ").strip()

        Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.memory_file, "a") as f:
            f.write(f"\n[{timestamp}] {entry}")

        self._rotate()  # Auto-rotate: keep only last N entries
        self._maybe_compact()

    def _rotate(self):
        """Keep only the last max_entries in memory.md to prevent unbounded growth."""
        try:
            with open(self.memory_file) as f:
                content = f.read()
        except FileNotFoundError:
            return

        # Split by timestamp markers
        import re

        entries = re.split(r"\n(?=\[\d{4}-\d{2}-\d{2}\s)", content)
        entries = [e.strip() for e in entries if e.strip()]

        if len(entries) <= self.max_entries:
            return

        # Keep only last N
        entries = entries[-self.max_entries :]
        with open(self.memory_file, "w") as f:
            f.write("\n\n".join(entries))

    def _maybe_compact(self):
        """Compact MEMORY.md if it exceeds the threshold."""
        try:
            with open(self.memory_file) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            return

        if len(lines) < self._compaction_threshold:
            return

        # Take first chunk of entries
        old_lines = lines[: self._compaction_threshold]
        recent_lines = lines[self._compaction_threshold :]

        # Generate a compacted summary
        combined = "\n".join(old_lines)
        prompt = f"Summarize these git diff summaries into one paragraph:\n\n{combined}"

        try:
            compact_summary = self.ollama.generate(prompt=prompt, model=self.model)
        except Exception:
            compact_summary = "Multiple changes over time."

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        with open(self.memory_file, "w") as f:
            f.write(f"[{timestamp}] [COMPACTED] {compact_summary.strip()}")
            if recent_lines:
                f.write("\n" + "\n".join(recent_lines))

    def load_recent(self, last_n: int = 10) -> str:
        """Load the last N memory entries."""
        try:
            with open(self.memory_file) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            return ""

        return "\n".join(lines[-last_n:])

    def clear(self):
        """Clear the memory file."""
        Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "w") as f:
            f.write("")
