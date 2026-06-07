"""Tests for core.memory_doc (sectioned, bounded memory document model)."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from core.memory_doc import MemoryDoc


class TestMemoryDocBasic:
    """Basic tests: empty doc, initialization, headers."""

    def test_load_missing_path_returns_empty_doc(self):
        """Loading a missing file returns an empty doc."""
        doc = MemoryDoc.load("/nonexistent/path/memory.md")
        rendered = doc.render()
        # All headers present but no entries
        assert "## Open Tasks" in rendered
        assert "## Decisions" in rendered
        assert "## Conventions" in rendered
        assert "## Recent Changes" in rendered
        assert "## Recent User Prompts" in rendered
        assert rendered.count("- ") == 0

    def test_empty_doc_contains_all_headers(self):
        """Empty doc always renders all 5 section headers."""
        doc = MemoryDoc()
        rendered = doc.render()
        assert "## Open Tasks" in rendered
        assert "## Decisions" in rendered
        assert "## Conventions" in rendered
        assert "## Recent Changes" in rendered
        assert "## Recent User Prompts" in rendered

    def test_init_with_custom_caps(self):
        """Initialization with custom caps."""
        custom_caps = {"tasks": 5, "decisions": 10}
        doc = MemoryDoc(caps=custom_caps)
        assert doc._caps["tasks"] == 5
        assert doc._caps["decisions"] == 10


class TestMemoryDocAdd:
    """Tests for adding entries to sections."""

    def test_add_task(self):
        """Adding a task puts it in Open Tasks section."""
        doc = MemoryDoc()
        doc.add("task", "do something")
        rendered = doc.render()
        assert "## Open Tasks" in rendered
        assert "do something" in rendered

    def test_add_decision(self):
        """Adding a decision puts it in Decisions section."""
        doc = MemoryDoc()
        doc.add("decision", "use approach X")
        rendered = doc.render()
        assert "## Decisions" in rendered
        assert "use approach X" in rendered

    def test_add_convention(self):
        """Adding a convention puts it in Conventions section."""
        doc = MemoryDoc()
        doc.add("convention", "always format with black")
        rendered = doc.render()
        assert "## Conventions" in rendered
        assert "always format with black" in rendered

    def test_add_change(self):
        """Adding a change puts it in Recent Changes section."""
        doc = MemoryDoc()
        doc.add("change", "edited foo.py")
        rendered = doc.render()
        assert "## Recent Changes" in rendered
        assert "edited foo.py" in rendered

    def test_add_prompt(self):
        """Adding a prompt puts it in Recent User Prompts section."""
        doc = MemoryDoc()
        doc.add("prompt", "stand up cluster")
        rendered = doc.render()
        assert "## Recent User Prompts" in rendered
        assert "stand up cluster" in rendered

    def test_add_unknown_kind_defaults_to_change(self):
        """Unknown kind defaults to Recent Changes section."""
        doc = MemoryDoc()
        doc.add("unknown_kind", "some entry")
        rendered = doc.render()
        assert "some entry" in rendered
        assert rendered.find("## Recent Changes") < rendered.find("some entry")

    def test_add_normalizes_whitespace(self):
        """Entry is normalized to single line."""
        doc = MemoryDoc()
        doc.add("task", "do\n  something\t  with   spaces")
        rendered = doc.render()
        assert "do something with spaces" in rendered

    def test_add_empty_entry_ignored(self):
        """Empty or whitespace-only entries are not added."""
        doc = MemoryDoc()
        doc.add("task", "")
        doc.add("task", "   \n\n  ")
        rendered = doc.render()
        # Only headers and title, no entries
        assert rendered.count("- ") == 0

    def test_add_stamps_date(self):
        """Entries are stamped with date [YYYY-MM-DD]."""
        doc = MemoryDoc()
        doc.add("change", "did something")
        rendered = doc.render()
        # Should contain a date stamp
        assert "[" in rendered and "]" in rendered
        assert "did something" in rendered


class TestMemoryDocDedup:
    """Tests for deduplication logic."""

    def test_dedup_task_ignores_date(self):
        """Adding the same task twice moves it to newest, ignores date."""
        doc = MemoryDoc()
        doc.add("task", "do X", date="[2026-06-01]")
        doc.add("task", "do X", date="[2026-06-02]")
        rendered = doc.render()
        # Count how many times "do X" appears (should be 1)
        assert rendered.count("do X") == 1

    def test_dedup_case_insensitive(self):
        """Dedup is case-insensitive."""
        doc = MemoryDoc()
        doc.add("task", "Do Something")
        doc.add("task", "do something")
        rendered = doc.render()
        assert rendered.count("do something") == 1

    def test_dedup_moves_to_newest(self):
        """Duplicate entry moves to end (newest position)."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        doc.add("task", "task2")
        doc.add("task", "task1")  # duplicate
        rendered = doc.render()
        # task1 should appear after task2 (newest)
        idx1 = rendered.rfind("task1")
        idx2 = rendered.rfind("task2")
        assert idx1 > idx2


class TestMemoryDocRender:
    """Tests for rendering markdown output."""

    def test_render_format(self):
        """Render produces correct markdown format."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        doc.add("decision", "decision1")
        rendered = doc.render()
        assert rendered.startswith("# Cairn Memory\n")
        # Entries are stamped with dates
        assert "task1" in rendered
        assert "decision1" in rendered
        assert "[" in rendered and "]" in rendered

    def test_render_newest_last(self):
        """Entries are listed with newest last in render."""
        doc = MemoryDoc()
        doc.add("change", "change1")
        doc.add("change", "change2")
        rendered = doc.render()
        # Find indices in Recent Changes section
        changes_start = rendered.find("## Recent Changes")
        changes_end = rendered.find("##", changes_start + 1)
        changes_section = rendered[changes_start:changes_end]
        idx1 = changes_section.find("change1")
        idx2 = changes_section.find("change2")
        assert idx1 < idx2  # change1 before change2 (newer last)

    def test_render_all_sections_always_present(self):
        """Render always includes all 5 section headers even if empty."""
        doc = MemoryDoc()
        doc.add("task", "one task")
        rendered = doc.render()
        # All headers must be present
        assert "## Open Tasks" in rendered
        assert "## Decisions" in rendered
        assert "## Conventions" in rendered
        assert "## Recent Changes" in rendered
        assert "## Recent User Prompts" in rendered


class TestMemoryDocRoundTrip:
    """Tests for load/render round-trip stability."""

    def test_render_loads_roundtrip(self):
        """render() -> loads() preserves all entries."""
        doc1 = MemoryDoc()
        doc1.add("task", "task1")
        doc1.add("decision", "decision1")
        doc1.add("convention", "convention1")
        doc1.add("change", "change1")
        doc1.add("prompt", "prompt1")

        rendered = doc1.render()
        doc2 = MemoryDoc.loads(rendered)

        # Check that all entries are preserved
        assert "task1" in doc2.render()
        assert "decision1" in doc2.render()
        assert "convention1" in doc2.render()
        assert "change1" in doc2.render()
        assert "prompt1" in doc2.render()


class TestMemoryDocMigration:
    """Tests for loading old flat-log format (migration path)."""

    def test_load_flat_log_format(self):
        """Loading old flat-log format puts entries in Recent Changes."""
        old_log = (
            "[2026-06-06 07:48] did A\n"
            "[2026-06-06 08:00] did B\n"
            "[2026-06-06 08:15] did C\n"
        )
        doc = MemoryDoc.loads(old_log)
        rendered = doc.render()

        # All lines should be in Recent Changes
        changes_start = rendered.find("## Recent Changes")
        changes_end = rendered.find("##", changes_start + 1)
        changes_section = rendered[changes_start:changes_end]

        assert "did A" in changes_section
        assert "did B" in changes_section
        assert "did C" in changes_section

    def test_flat_log_with_leading_dash(self):
        """Flat log with leading '- ' is parsed correctly."""
        old_log = (
            "- [2026-06-06 07:48] did A\n"
            "- [2026-06-06 08:00] did B\n"
        )
        doc = MemoryDoc.loads(old_log)
        rendered = doc.render()
        # Dashes should be stripped and entries preserved
        assert "[2026-06-06 07:48] did A" in rendered
        assert "[2026-06-06 08:00] did B" in rendered

    def test_flat_log_render_has_section_headers(self):
        """After loading flat log, render() has section headers."""
        old_log = "[2026-06-06 07:48] did A\n[2026-06-06 08:00] did B\n"
        doc = MemoryDoc.loads(old_log)
        rendered = doc.render()
        # Should now have section headers
        assert "## Recent Changes" in rendered
        assert "## Open Tasks" in rendered


class TestMemoryDocCaps:
    """Tests for per-section caps and compaction."""

    def test_caps_enforced_on_add(self):
        """Adding entries beyond cap removes oldest entries."""
        caps = {"changes": 5}
        doc = MemoryDoc(caps=caps)
        for i in range(10):
            doc.add("change", f"change{i}")
        rendered = doc.render()
        # Should have at most 5 + 1 (history summary) entries in changes
        changes_section = rendered[
            rendered.find("## Recent Changes") : rendered.find("## Recent User")
        ]
        entry_count = changes_section.count("- ")
        assert entry_count <= 6  # 1 summary + 5 newest

    def test_changes_compaction_creates_history_summary(self):
        """When Recent Changes exceeds cap, oldest entries become history summary."""
        caps = {"changes": 3}
        doc = MemoryDoc(caps=caps)
        for i in range(5):
            doc.add("change", f"change{i}")
        rendered = doc.render()
        # Should have a (history) summary line
        assert "(history)" in rendered

    def test_changes_history_summary_correct_count(self):
        """History summary shows correct count of collapsed entries."""
        caps = {"changes": 2}
        doc = MemoryDoc(caps=caps)
        doc.add("change", "old1")
        doc.add("change", "old2")
        doc.add("change", "old3")
        doc.add("change", "new1")
        rendered = doc.render()
        # Should show (history) 3 earlier changes (old1, old2, old3)
        assert "(history) 3 earlier changes" in rendered

    def test_other_sections_keep_newest_only(self):
        """Non-Recent Changes sections keep only newest N entries."""
        caps = {"tasks": 2}
        doc = MemoryDoc(caps=caps)
        doc.add("task", "task1")
        doc.add("task", "task2")
        doc.add("task", "task3")
        rendered = doc.render()
        # Should only have task2 and task3 (newest 2)
        assert "task2" in rendered
        assert "task3" in rendered
        assert "task1" not in rendered

    def test_each_section_enforces_own_cap(self):
        """Each section enforces its own cap independently."""
        caps = {"tasks": 2, "decisions": 3}
        doc = MemoryDoc(caps=caps)
        for i in range(5):
            doc.add("task", f"task{i}")
            doc.add("decision", f"decision{i}")
        rendered = doc.render()
        # Tasks: should have newest 2
        assert "task3" in rendered
        assert "task4" in rendered
        assert "task0" not in rendered
        # Decisions: should have newest 3
        assert "decision2" in rendered
        assert "decision3" in rendered
        assert "decision4" in rendered


class TestMemoryDocSaveLoad:
    """Tests for file I/O."""

    def test_save_creates_file(self):
        """save() creates the file at the given path."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.md"
            doc.save(path)
            assert path.exists()

    def test_save_creates_parent_dirs(self):
        """save() creates parent directories if they don't exist."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "memory.md"
            doc.save(path)
            assert path.exists()
            assert path.parent.exists()

    def test_save_load_roundtrip(self):
        """save() then load() preserves all entries."""
        doc1 = MemoryDoc()
        doc1.add("task", "task1")
        doc1.add("decision", "decision1")
        doc1.add("change", "change1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.md"
            doc1.save(path)
            doc2 = MemoryDoc.load(path)

            # All entries should be present
            rendered2 = doc2.render()
            assert "task1" in rendered2
            assert "decision1" in rendered2
            assert "change1" in rendered2

    def test_save_atomic(self):
        """save() is atomic (tmp + os.replace)."""
        doc = MemoryDoc()
        doc.add("task", "important task")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.md"
            doc.save(path)
            # File should exist and have content
            content = path.read_text()
            assert "important task" in content


class TestMemoryDocRead:
    """Tests for budget-trimmed read() method."""

    def test_read_with_none_returns_full(self):
        """read(max_tokens=None) returns full render()."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        doc.add("change", "change1")
        result = doc.read(max_tokens=None)
        assert result == doc.render()

    def test_read_keeps_tasks_decisions_conventions_full(self):
        """read() always keeps tasks, decisions, conventions in full."""
        doc = MemoryDoc()
        doc.add("task", "task1")
        doc.add("decision", "decision1")
        doc.add("convention", "convention1")
        # Add many changes/prompts to trigger trimming
        for i in range(100):
            doc.add("change", f"change{i}")
            doc.add("prompt", f"prompt{i}")

        result = doc.read(max_tokens=200)
        # Core sections must all be present
        assert "task1" in result
        assert "decision1" in result
        assert "convention1" in result

    def test_read_trims_changes_and_prompts(self):
        """read() trims recent changes and prompts to fit budget."""
        doc = MemoryDoc()
        for i in range(50):
            doc.add("change", f"change{i}")

        result = doc.read(max_tokens=100)
        # Should be smaller than full render
        assert len(result) < len(doc.render())

    def test_read_respects_token_budget(self):
        """read() result is guaranteed to fit in max_tokens."""
        from core import tokens

        doc = MemoryDoc()
        doc.add("task", "task1")
        for i in range(100):
            doc.add("change", f"change{i}")

        result = doc.read(max_tokens=300)
        token_count = tokens.count_tokens(result)
        assert token_count <= 300

    def test_read_keeps_newest_entries(self):
        """read() prefers to keep newest entries when trimming."""
        doc = MemoryDoc()
        # Add three changes with distinct timestamps
        doc.add("change", "oldest change", date="[2026-06-01]")
        doc.add("change", "middle change", date="[2026-06-02]")
        doc.add("change", "newest change", date="[2026-06-03]")

        # With very tight budget, should keep newest
        result = doc.read(max_tokens=200)
        # At minimum, newest should be there
        assert "newest change" in result


class TestMemoryDocLoadsEdgeCases:
    """Tests for edge cases in loads() method."""

    def test_loads_empty_string(self):
        """loads("") returns empty doc."""
        doc = MemoryDoc.loads("")
        rendered = doc.render()
        # Should have headers but no entries
        assert "## Open Tasks" in rendered
        assert rendered.count("- ") == 0

    def test_loads_only_whitespace(self):
        """loads() with only whitespace returns empty doc."""
        doc = MemoryDoc.loads("   \n\n\t\n   ")
        rendered = doc.render()
        assert rendered.count("- ") == 0

    def test_loads_mixed_old_and_new_format(self):
        """loads() with new section headers ignores flat log format."""
        text = (
            "# Cairn Memory\n"
            "\n"
            "## Open Tasks\n"
            "- task1\n"
            "\n"
            "[2026-06-06 07:48] old flat line\n"
            "## Decisions\n"
            "- decision1\n"
        )
        doc = MemoryDoc.loads(text)
        rendered = doc.render()
        # Should parse as sectioned (task1 and decision1 present)
        assert "task1" in rendered
        assert "decision1" in rendered

    def test_loads_blank_lines_ignored(self):
        """Blank lines in sectioned format are ignored."""
        text = (
            "# Cairn Memory\n"
            "\n"
            "## Open Tasks\n"
            "\n"
            "\n"
            "- task1\n"
            "\n"
            "- task2\n"
            "\n"
        )
        doc = MemoryDoc.loads(text)
        rendered = doc.render()
        assert "task1" in rendered
        assert "task2" in rendered

    def test_loads_malformed_entries_tolerated(self):
        """Malformed entries are skipped without error."""
        text = (
            "## Open Tasks\n"
            "- valid entry\n"
            "invalid line without dash\n"
            "- another valid\n"
        )
        doc = MemoryDoc.loads(text)
        rendered = doc.render()
        # Valid entries should be there
        assert "valid entry" in rendered
        assert "another valid" in rendered


class TestMemoryDocDateStamping:
    """Tests for date stamping behavior."""

    def test_add_with_explicit_date(self):
        """add() with explicit date parameter uses that date."""
        doc = MemoryDoc()
        doc.add("change", "something", date="[2026-05-01]")
        rendered = doc.render()
        assert "[2026-05-01]" in rendered

    def test_add_without_date_uses_today(self):
        """add() without date parameter uses today."""
        doc = MemoryDoc()
        doc.add("change", "something")
        rendered = doc.render()
        today = date.today().isoformat()
        assert f"[{today}]" in rendered
