"""Unit tests for pipeline/memory.py — MemorySummarizer."""

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
