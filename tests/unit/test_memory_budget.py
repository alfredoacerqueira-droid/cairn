"""Unit tests for memory token budgeting in core/repo.py."""

from core.repo import RepoManager
from core.tokens import count_tokens


class FakeConfig:
    """Fake config for testing."""

    class BudgetConfig:
        tool_max_tokens = 1000

    budget = BudgetConfig()


class TestMemoryBudget:
    """Test memory loading with token budgets."""

    def test_load_memory_default_no_budget(self, tmp_path):
        """Test default behavior with no budget (backward compatible)."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Write a memory file with 3 entries
        memory_path = repo.get_memory_path()
        memory_path.write_text(
            "[2024-01-01 10:00] Entry 1\n"
            "[2024-01-01 11:00] Entry 2\n"
            "[2024-01-01 12:00] Entry 3"
        )

        # Load last 2 entries with no budget (old behavior)
        result = repo.load_memory(last_n=2)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "Entry 2" in result
        assert "Entry 3" in result
        assert "Entry 1" not in result

    def test_load_memory_with_unlimited_budget(self, tmp_path):
        """Test that unlimited budget doesn't affect default behavior."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        memory_path = repo.get_memory_path()
        memory_path.write_text(
            "[2024-01-01 10:00] Entry 1\n"
            "[2024-01-01 11:00] Entry 2\n"
            "[2024-01-01 12:00] Entry 3"
        )

        # Load with a very high budget
        result = repo.load_memory(last_n=10, max_tokens=10000)
        assert "Entry 1" in result
        assert "Entry 2" in result
        assert "Entry 3" in result

    def test_load_memory_with_tight_budget(self, tmp_path):
        """Test memory trimming when budget is exceeded."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create a memory file with several entries
        entry1 = (
            "This is a relatively long entry that contains quite a bit of text "
            "to ensure it takes up some tokens."
        )
        entry2 = "Another entry with some content here to vary the token count."
        entry3 = "Final entry that should be kept since it's the newest."

        memory_path = repo.get_memory_path()
        memory_path.write_text(
            f"[2024-01-01 10:00] {entry1}\n"
            f"[2024-01-01 11:00] {entry2}\n"
            f"[2024-01-01 12:00] {entry3}"
        )

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

        memory_path = repo.get_memory_path()
        memory_path.write_text(
            f"[2024-01-01 10:00] {old_entry}\n"
            f"[2024-01-01 12:00] {new_entry}"
        )

        # Load with tight budget
        result = repo.load_memory(last_n=10, max_tokens=30)

        # Should fit within budget
        assert count_tokens(result) <= 30

        # Should contain the newer entry
        assert "NEW" in result

    def test_load_memory_single_entry_exceeds_budget(self, tmp_path):
        """Test hard truncation when a single entry exceeds the budget."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create a very long single entry
        long_entry = "word " * 500  # Many tokens

        memory_path = repo.get_memory_path()
        memory_path.write_text(f"[2024-01-01 10:00] {long_entry}")

        # Load with budget that even one entry exceeds
        result = repo.load_memory(last_n=10, max_tokens=100)

        # Result should be truncated to fit the budget
        assert count_tokens(result) <= 100
        assert len(result) > 0  # Should not be empty

    def test_load_memory_empty_file(self, tmp_path):
        """Test loading memory from non-existent file."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        result = repo.load_memory(last_n=10, max_tokens=100)
        assert result == ""

    def test_load_memory_multiple_entries_trim_gradually(self, tmp_path):
        """Test that entries are trimmed from oldest to newest until budget fits."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Create 4 entries
        memory_path = repo.get_memory_path()
        entries = [
            "[2024-01-01 10:00] First entry is quite long with multiple words",
            "[2024-01-01 11:00] Second entry also has some text here",
            "[2024-01-01 12:00] Third entry with content",
            "[2024-01-01 13:00] Fourth newest entry short",
        ]
        memory_path.write_text("\n".join(entries))

        # Load with a moderate budget
        result = repo.load_memory(last_n=10, max_tokens=100)

        # Should fit the budget
        assert count_tokens(result) <= 100

        # Newest entry must be present
        assert "Fourth newest entry" in result

    def test_load_memory_respects_last_n(self, tmp_path):
        """Test that last_n is respected before budget trimming."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        memory_path = repo.get_memory_path()
        entries = [
            "[2024-01-01 10:00] Entry 1",
            "[2024-01-01 11:00] Entry 2",
            "[2024-01-01 12:00] Entry 3",
            "[2024-01-01 13:00] Entry 4",
        ]
        memory_path.write_text("\n".join(entries))

        # Load last 2 entries with high budget
        result = repo.load_memory(last_n=2, max_tokens=1000)

        assert "Entry 3" in result
        assert "Entry 4" in result
        assert "Entry 1" not in result
        assert "Entry 2" not in result

    def test_load_memory_malformed_entries_graceful(self, tmp_path):
        """Test handling of malformed timestamp entries."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        # Write mixed content (some with timestamps, some without)
        memory_path = repo.get_memory_path()
        memory_path.write_text(
            "Some loose text\n"
            "[2024-01-01 10:00] Proper entry 1\n"
            "More loose text\n"
            "[2024-01-01 12:00] Proper entry 2"
        )

        # Load with budget
        result = repo.load_memory(last_n=10, max_tokens=200)

        # Should not crash and should fit budget
        assert count_tokens(result) <= 200

    def test_load_memory_default_behavior_unchanged(self, tmp_path):
        """Test that default behavior (max_tokens=None) is identical to old behavior."""
        repo = RepoManager(tmp_path)
        repo.ensure_directories()

        memory_path = repo.get_memory_path()
        test_content = (
            "[2024-01-01 10:00] Entry 1\n"
            "[2024-01-01 11:00] Entry 2\n"
            "[2024-01-01 12:00] Entry 3\n"
            "[2024-01-01 13:00] Entry 4\n"
            "[2024-01-01 14:00] Entry 5"
        )
        memory_path.write_text(test_content)

        # Load with default (no budget)
        result_no_budget = repo.load_memory(last_n=3)

        # Load with explicit max_tokens=None
        result_explicit_none = repo.load_memory(last_n=3, max_tokens=None)

        # Both should be identical
        assert result_no_budget == result_explicit_none
        assert "Entry 3" in result_no_budget
        assert "Entry 4" in result_no_budget
        assert "Entry 5" in result_no_budget
        assert "Entry 1" not in result_no_budget
        assert "Entry 2" not in result_no_budget
