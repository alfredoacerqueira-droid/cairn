"""Unit tests for pipeline/memory.py — MemorySummarizer."""

import subprocess
from pathlib import Path

from pipeline.memory import MemorySummarizer


class TestMemorySummarizer:
    def test_init_creates_memory_file(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path)
        assert ms.memory_file is not None

    def test_append_to_memory_creates_file(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path, model="test")
        ms.append_to_memory("Tested memory summarization")
        mem_path = Path(ms.memory_file)
        assert mem_path.exists()
        content = mem_path.read_text()
        assert "Tested memory summarization" in content

    def test_load_recent_returns_content(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path)
        ms.append_to_memory("Entry 1")
        ms.append_to_memory("Entry 2")
        ms.append_to_memory("Entry 3")

        recent = ms.load_recent(last_n=2)
        assert "Entry" in recent

    def test_clear_empties_file(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path)
        ms.append_to_memory("Stuff")
        mem_path = Path(ms.memory_file)
        assert mem_path.exists()

        ms.clear()
        content = mem_path.read_text()
        assert content.strip() == ""

    def test_get_uncommitted_diff_no_git(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path)
        diff = ms.get_uncommitted_diff()
        assert isinstance(diff, str)

    def test_get_recent_diff_no_git(self, tmp_path):
        ms = MemorySummarizer(repo_path=tmp_path)
        diff = ms.get_recent_diff()
        assert isinstance(diff, str)

    def test_diff_excludes_cairn_directory(self, tmp_path):
        """Recent diff must exclude .cairn/ internal state files."""
        ms = MemorySummarizer(repo_path=tmp_path, llm_enabled=False)

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        # First commit with a source file
        (tmp_path / "src.py").write_text("def hello():\n    return 'hello'\n")
        (tmp_path / ".cairn").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".cairn" / "index_meta.json").write_text('{"built_at": "0"}')
        subprocess.run(["git", "add", "src.py"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "first commit"], cwd=tmp_path, capture_output=True)

        # Second commit changes BOTH a real source file AND .cairn/ state
        (tmp_path / "src.py").write_text("def hello():\n    return 'hello world'\n")
        (tmp_path / ".cairn" / "index_meta.json").write_text('{"built_at": "1"}')
        subprocess.run(
            ["git", "add", "src.py", ".cairn/index_meta.json"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "commit", "-m", "second commit"], cwd=tmp_path, capture_output=True)

        diff = ms.get_recent_diff()

        # Diff should contain the source file change
        assert "hello world" in diff

        # Diff must NOT contain .cairn/ paths
        assert ".cairn" not in diff

        # Extracted files must not include .cairn/
        files = ms._extract_files_from_diff(diff)
        assert "src.py" in files
        assert not any(f.startswith(".cairn") for f in files)

        # Deterministic summary must mention the source file, not .cairn/
        summary = ms._deterministic_summary(diff)
        assert "src.py" in summary
        assert ".cairn" not in summary

        # summarize_and_record with only .cairn/ changes should not create an entry
        (tmp_path / "src.py").write_text("def hello():\n    return 'hello world'\n")
        (tmp_path / ".cairn" / "last_indexed.txt").write_text("abc123")
        subprocess.run(["git", "add", ".cairn/last_indexed.txt"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "only cairn change"], cwd=tmp_path, capture_output=True
        )
        # Force re-read of the memory file after summary
        cairn_only_diff = ms.get_recent_diff()
        # When only .cairn/ changed, diff should be empty (no non-cairn changes)
        assert cairn_only_diff == ""

    def test_uncommitted_diff_excludes_cairn(self, tmp_path):
        """Uncommitted diff must also exclude .cairn/ internal state."""
        ms = MemorySummarizer(repo_path=tmp_path, llm_enabled=False)

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        (tmp_path / "src.py").write_text("def hello():\n    return 'hello'\n")
        subprocess.run(["git", "add", "src.py"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, capture_output=True)

        (tmp_path / "src.py").write_text("def hello():\n    return 'bye'\n")
        (tmp_path / ".cairn").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".cairn" / "some_state.json").write_text("{}")

        diff = ms.get_uncommitted_diff()
        assert "bye" in diff
        assert ".cairn" not in diff
