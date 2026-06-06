"""Unit tests for memory token budgeting in core/repo.py."""

from core.repo import RepoManager
from core.tokens import count_tokens


class FakeConfig:
    """Fake config for testing."""

    class BudgetConfig:
        tool_max_tokens = 1000

    budget = BudgetConfig()


class TestMemoryBudget:
    """Test memory loading with token budgets via MemoryDoc."""

    def test_load_memory_default_no_budget(self, tmp_path):
        """Test default behavior with no budget returns structured memory."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Append 3 entries to build a sectioned memory file
        repo.append_memory("Entry 1", kind="change")
        repo.append_memory("Entry 2", kind="change")
        repo.append_memory("Entry 3", kind="change")

        # Load with no budget (returns full structured document)
        result = repo.load_memory(last_n=2)
        # Result is now structured markdown with headers
        assert "## Recent Changes" in result
        assert "Entry 1" in result
        assert "Entry 2" in result
        assert "Entry 3" in result

    def test_load_memory_with_unlimited_budget(self, tmp_path):
        """Test that unlimited budget returns full structured memory."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Entry 1", kind="change")
        repo.append_memory("Entry 2", kind="change")
        repo.append_memory("Entry 3", kind="change")

        # Load with a very high budget
        result = repo.load_memory(last_n=10, max_tokens=10000)
        assert "## Recent Changes" in result
        assert "Entry 1" in result
        assert "Entry 2" in result
        assert "Entry 3" in result

    def test_load_memory_with_tight_budget(self, tmp_path):
        """Test memory trimming when budget is exceeded."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create entries with varying lengths
        entry1 = (
            "This is a relatively long entry that contains quite a bit of text "
            "to ensure it takes up some tokens."
        )
        entry2 = "Another entry with some content here to vary the token count."
        entry3 = "Final entry that should be kept since it's the newest."

        repo.append_memory(entry1, kind="change")
        repo.append_memory(entry2, kind="change")
        repo.append_memory(entry3, kind="change")

        # Load with a budget that forces trimming (only newest entries survive)
        result = repo.load_memory(last_n=10, max_tokens=50)

        # Verify the result fits the budget
        assert count_tokens(result) <= 50

        # Newest entry should be present
        assert "Final entry" in result

    def test_load_memory_keep_newest_entries(self, tmp_path):
        """Test that oldest entries are dropped first, keeping newest."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create memory with old and new entries
        old_entry = "OLD " * 50  # Create a longer entry
        new_entry = "NEW " * 5  # Shorter entry

        repo.append_memory(old_entry, kind="change")
        repo.append_memory(new_entry, kind="change")

        # Load with higher budget to ensure we get the entries
        result = repo.load_memory(last_n=10, max_tokens=100)

        # Should fit within budget
        assert count_tokens(result) <= 100

        # Should contain the newer entry
        assert "NEW" in result

    def test_load_memory_single_entry_exceeds_budget(self, tmp_path):
        """Test hard truncation when a single entry exceeds the budget."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create a very long single entry
        long_entry = "word " * 500  # Many tokens

        repo.append_memory(long_entry, kind="change")

        # Load with budget that even one entry exceeds
        result = repo.load_memory(last_n=10, max_tokens=100)

        # Result should be truncated to fit the budget
        assert count_tokens(result) <= 100
        assert len(result) > 0  # Should not be empty

    def test_load_memory_empty_file(self, tmp_path):
        """Test loading memory from non-existent file returns empty structured doc."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        result = repo.load_memory(last_n=10, max_tokens=100)
        # Empty memory file loads as structured doc with headers but no entries
        assert "## Open Tasks" in result
        assert "## Recent Changes" in result
        # No actual entries
        assert result.count("- ") == 0

    def test_load_memory_multiple_entries_trim_gradually(self, tmp_path):
        """Test that entries are trimmed from oldest to newest until budget fits."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create 4 entries
        entries = [
            "First entry is quite long with multiple words",
            "Second entry also has some text here",
            "Third entry with content",
            "Fourth newest entry short",
        ]
        for entry in entries:
            repo.append_memory(entry, kind="change")

        # Load with a moderate budget
        result = repo.load_memory(last_n=10, max_tokens=100)

        # Should fit the budget
        assert count_tokens(result) <= 100

        # Newest entry must be present
        assert "Fourth newest entry" in result

    def test_load_memory_respects_last_n(self, tmp_path):
        """Test that last_n is kept for signature compatibility (token budget is the real trim)."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        entries = [
            "Entry 1",
            "Entry 2",
            "Entry 3",
            "Entry 4",
        ]
        for entry in entries:
            repo.append_memory(entry, kind="change")

        # Load with high budget - all entries should be present (MemoryDoc returns full structure)
        result = repo.load_memory(last_n=2, max_tokens=1000)

        # With structured memory, all entries are present in Recent Changes
        assert "Entry 1" in result
        assert "Entry 2" in result
        assert "Entry 3" in result
        assert "Entry 4" in result

    def test_load_memory_malformed_entries_graceful(self, tmp_path):
        """Test handling of old flat-log format (migration)."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Write old flat-log format
        memory_path = repo.get_memory_path()
        memory_path.write_text(
            "[2024-01-01 10:00] Proper entry 1\n"
            "[2024-01-01 12:00] Proper entry 2"
        )

        # Load with budget - should migrate old format to structured
        result = repo.load_memory(last_n=10, max_tokens=200)

        # Should not crash, should fit budget, and should now have section headers
        assert count_tokens(result) <= 200
        assert "## Recent Changes" in result

    def test_load_memory_default_behavior_returns_structured(self, tmp_path):
        """Test that load_memory now returns structured sectioned memory."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        for i in range(1, 6):
            repo.append_memory(f"Entry {i}", kind="change")

        # Load with default (no budget)
        result_no_budget = repo.load_memory(last_n=3)

        # Load with explicit max_tokens=None
        result_explicit_none = repo.load_memory(last_n=3, max_tokens=None)

        # Both should be identical and structured
        assert result_no_budget == result_explicit_none
        assert "## Recent Changes" in result_no_budget
        # All entries are in the structured format
        assert "Entry 1" in result_no_budget
        assert "Entry 2" in result_no_budget
        assert "Entry 3" in result_no_budget
        assert "Entry 4" in result_no_budget
        assert "Entry 5" in result_no_budget


class TestMemoryKinds:
    """Test routing of different memory kinds to their sections."""

    def test_append_memory_task_to_section(self, tmp_path):
        """Test that kind='task' routes to Open Tasks section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Do X", kind="task")
        result = repo.load_memory()

        assert "## Open Tasks" in result
        assert "Do X" in result

    def test_append_memory_decision_to_section(self, tmp_path):
        """Test that kind='decision' routes to Decisions section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Use approach Y", kind="decision")
        result = repo.load_memory()

        assert "## Decisions" in result
        assert "Use approach Y" in result

    def test_append_memory_convention_to_section(self, tmp_path):
        """Test that kind='convention' routes to Conventions section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Always format with black", kind="convention")
        result = repo.load_memory()

        assert "## Conventions" in result
        assert "Always format with black" in result

    def test_append_memory_prompt_to_section(self, tmp_path):
        """Test that kind='prompt' routes to Recent User Prompts section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Stand up cluster", kind="prompt")
        result = repo.load_memory()

        assert "## Recent User Prompts" in result
        assert "Stand up cluster" in result

    def test_append_memory_change_default(self, tmp_path):
        """Test that kind='change' (default) routes to Recent Changes section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Modified foo.py")  # No kind specified, defaults to 'change'
        result = repo.load_memory()

        assert "## Recent Changes" in result
        assert "Modified foo.py" in result

    def test_append_memory_mixed_kinds(self, tmp_path):
        """Test that multiple kinds are routed to their respective sections."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        repo.append_memory("Task 1", kind="task")
        repo.append_memory("Decision 1", kind="decision")
        repo.append_memory("Convention 1", kind="convention")
        repo.append_memory("Change 1", kind="change")
        repo.append_memory("Prompt 1", kind="prompt")

        result = repo.load_memory()

        # All sections should be present
        assert "## Open Tasks" in result
        assert "Task 1" in result

        assert "## Decisions" in result
        assert "Decision 1" in result

        assert "## Conventions" in result
        assert "Convention 1" in result

        assert "## Recent Changes" in result
        assert "Change 1" in result

        assert "## Recent User Prompts" in result
        assert "Prompt 1" in result

    def test_old_flat_memory_migrates_to_changes(self, tmp_path):
        """Test that old flat-log memory is migrated into Recent Changes section."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Write old flat-log format
        memory_path = repo.get_memory_path()
        memory_path.write_text(
            "[2024-01-01 10:00] Old entry 1\n"
            "[2024-01-01 11:00] Old entry 2"
        )

        # Load should migrate to structured format
        result = repo.load_memory()

        assert "## Recent Changes" in result
        assert "Old entry 1" in result
        assert "Old entry 2" in result

    def test_memory_bounded_by_caps(self, tmp_path):
        """Test that memory sections are bounded by configured caps."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Add many tasks (default cap is 20)
        for i in range(30):
            repo.append_memory(f"Task {i}", kind="task")

        result = repo.load_memory()

        # Should have at most 20 tasks
        task_count = result.count("Task ")
        assert task_count <= 20

    def test_local_llm_disabled_deterministic_generation(self, tmp_path):
        """Test that MemorySummarizer works with local_llm disabled."""
        from pipeline.memory import MemorySummarizer

        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create summarizer with LLM disabled
        summarizer = MemorySummarizer(
            repo_path=tmp_path,
            llm_enabled=False,
            memory_file=str(repo.get_memory_path()),
        )

        # Append without LLM - should use deterministic summary
        summarizer.append_to_memory("Modified foo.py and bar.py")

        result = repo.load_memory()
        # Should be in Recent Changes via MemoryDoc
        assert "## Recent Changes" in result
        assert "Modified foo.py and bar.py" in result
