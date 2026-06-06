"""Git diff summarization using local Ollama model or deterministic fallback."""

import re
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
        llm_enabled: bool = True,
    ):
        self.repo_path = Path(repo_path)
        self.ollama = ollama_client or OllamaClient()
        self.model = model
        self.memory_file = memory_file or str(self.repo_path / ".cairn" / "memory.md")
        self.max_entries = max_entries
        self._compaction_threshold = 200
        self.llm_enabled = llm_enabled

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

    def _extract_files_from_diff(self, diff: str) -> list[str]:
        """Extract changed filenames from a diff string."""
        files = []
        for line in diff.split("\n"):
            if line.startswith("diff --git"):
                # Extract file path from "diff --git a/path b/path"
                parts = line.split(" b/")
                if len(parts) > 1:
                    files.append(parts[1].strip())
            elif line.startswith("+++") or line.startswith("---"):
                # Fallback: extract from +++ --- lines
                path = line.lstrip("+- /").strip()
                if path and path != "/dev/null":
                    files.append(path)
        return list(set(files))  # Deduplicate

    def _deterministic_summary(self, diff: str) -> str:
        """Generate a deterministic summary of changes without LLM.

        Produces a bullet list of changed files and a change type heuristic.
        """
        if not diff or len(diff.strip()) < 10:
            return "No significant changes."

        files = self._extract_files_from_diff(diff)
        if not files:
            return "Modified codebase (diff details unavailable)."

        # Heuristic: guess change type from file patterns and diff size
        has_tests = any("test" in f.lower() for f in files)
        has_docs = any(f.endswith((".md", ".rst", ".txt")) for f in files)
        has_config = any(f.endswith((".yaml", ".yml", ".json", ".toml")) for f in files)

        type_hints = []
        if has_tests:
            type_hints.append("tests")
        if has_docs:
            type_hints.append("documentation")
        if has_config:
            type_hints.append("config")

        type_str = " + ".join(type_hints) if type_hints else "code"
        file_list = "; ".join(files[:5])
        if len(files) > 5:
            file_list += f"; +{len(files) - 5} more"

        return f"Modified {type_str}: {file_list}"

    def summarize_diff(self, diff: str) -> str:
        """Summarize a git diff using LLM if enabled, else deterministic fallback."""
        if not diff or len(diff.strip()) < 10:
            return "No significant changes."

        # Use deterministic summary if LLM is disabled
        if not self.llm_enabled:
            return self._deterministic_summary(diff)

        prompt = f"Summarize this git diff in one short sentence:\n\n{diff[:2000]}"

        try:
            response = self.ollama.generate(prompt=prompt, model=self.model)
            return response.strip()
        except Exception:
            # Fallback to deterministic if LLM call fails
            return self._deterministic_summary(diff)

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

        # Compaction: use LLM if enabled, else use deterministic method
        if self.llm_enabled:
            combined = "\n".join(old_lines)
            prompt = f"Summarize these git diff summaries into one paragraph:\n\n{combined}"
            try:
                compact_summary = self.ollama.generate(prompt=prompt, model=self.model)
            except Exception:
                compact_summary = "Multiple changes over time."
        else:
            # Deterministic compaction: count entries
            compact_summary = f"Multiple historical changes ({len(old_lines)} entries compacted)."

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
