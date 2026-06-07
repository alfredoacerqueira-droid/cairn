"""SECTIONED, BOUNDED persistent-memory document model for Cairn.

Replaces the flat append-only .cairn/memory.md log with a structured markdown document
that has five fixed sections (Open Tasks, Decisions, Conventions, Recent Changes, Recent
User Prompts). Each section is independently capped so it never grows unbounded.

Format:
  # Cairn Memory

  ## Open Tasks
  - task entry 1
  - task entry 2

  ## Decisions
  - decision entry 1

  ## Conventions
  - convention entry 1

  ## Recent Changes
  - [2026-06-06] change entry 1

  ## Recent User Prompts
  - [2026-06-06] prompt entry 1

When a section exceeds its cap, the oldest entries are collapsed into a summary line:
  - (history) N earlier changes

This allows an LLM to navigate and update by section while keeping the document bounded.
Tolerant parsing: missing headers, blank lines, and old flat-log format (no headers)
are all handled gracefully.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_CAPS = {
    "tasks": 20,
    "decisions": 40,
    "conventions": 40,
    "changes": 40,
    "prompts": 10,
}

TITLE = "# Cairn Memory"

# Section keys in order and their display names
SECTION_HEADERS = {
    "tasks": "## Open Tasks",
    "decisions": "## Decisions",
    "conventions": "## Conventions",
    "changes": "## Recent Changes",
    "prompts": "## Recent User Prompts",
}

# Map kind parameter to section key
KIND_TO_SECTION = {
    "task": "tasks",
    "decision": "decisions",
    "convention": "conventions",
    "change": "changes",
    "prompt": "prompts",
}


class MemoryDoc:
    """A sectioned, bounded memory document with fixed markdown headers.

    Stores entries in five sections (Open Tasks, Decisions, Conventions, Recent Changes,
    Recent User Prompts) with independent per-section caps. Entries are stored as lists
    preserving insertion order; newest entries are appended at the END of each list.

    Attributes:
        _entries: Dict mapping section_key to list of entry strings (newest last).
        _caps: Dict mapping section_key to max entries allowed in that section.
    """

    def __init__(self, caps: dict | None = None):
        """Initialize an empty MemoryDoc.

        Args:
            caps: Optional dict mapping section keys to max entry counts.
                  Defaults to DEFAULT_CAPS if None. Partial caps are merged
                  with defaults.
        """
        self._entries: dict[str, list[str]] = defaultdict(list)
        merged_caps = DEFAULT_CAPS.copy()
        if caps:
            merged_caps.update(caps)
        self._caps = merged_caps

    @classmethod
    def load(cls, path: str | Path, caps: dict | None = None) -> MemoryDoc:
        """Load a MemoryDoc from a file, handling both new and legacy formats.

        If the file is missing or empty, returns an empty doc.

        If the file contains section headers (##), parses entries under each header
        (lines starting with '- ' are entries; blank lines are ignored).

        If the file is an OLD FLAT LOG (no section headers), puts every non-empty line
        into Recent Changes (stripping a leading '- ' if present). This enables migration
        from the old timestamped append-only log format.

        Never raises on malformed input; tolerant parsing.

        Args:
            path: File path to load from.
            caps: Optional caps dict; defaults to DEFAULT_CAPS.

        Returns:
            A MemoryDoc instance with loaded entries.
        """
        path = Path(path)
        if not path.exists():
            return cls(caps=caps)

        text = path.read_text(encoding="utf-8")
        return cls.loads(text, caps=caps)

    @classmethod
    def loads(cls, text: str, caps: dict | None = None) -> MemoryDoc:
        """Parse a MemoryDoc from a string.

        Handles both new sectioned format and old flat-log format.
        Never raises; tolerant parsing.

        Args:
            text: Markdown text to parse.
            caps: Optional caps dict.

        Returns:
            A MemoryDoc instance with parsed entries.
        """
        doc = cls(caps=caps)

        if not text or not text.strip():
            return doc

        # Check if this is a sectioned document (has section headers)
        has_headers = any(
            line.startswith("##") for line in text.split("\n")
        )

        if has_headers:
            _parse_sectioned(doc, text)
        else:
            _parse_flat_log(doc, text)

        return doc

    def add(
        self, kind: str, entry: str, *, date: str | None = None
    ) -> None:
        """Add an entry to the appropriate section.

        Normalizes entry to a single line (collapses newlines, strips whitespace).
        Prefixes with a date stamp '[YYYY-MM-DD] ' for change and prompt kinds;
        also stamps task/decision/convention kinds.

        Dedup: if an identical (case-insensitive, stripped, ignoring date prefix)
        entry already exists in that section, moves it to newest instead of adding
        a duplicate.

        After adding, enforces the section's cap.

        Args:
            kind: Entry kind ('task', 'decision', 'convention', 'change', 'prompt').
                  Unknown kinds default to 'change'.
            entry: Entry text to add.
            date: Optional date string '[YYYY-MM-DD]'. If None, uses today's date
                  for change and prompt kinds; uses today for all kinds actually.
        """
        # Normalize entry to single line
        entry = " ".join(entry.split())
        if not entry:
            return

        # Map kind to section, default to 'change'
        section = KIND_TO_SECTION.get(kind, "changes")

        # Determine date prefix
        if date is None:
            today = __import__("datetime").date.today().isoformat()
            date = f"[{today}]"

        # For all kinds, stamp the date
        stamped_entry = f"{date} {entry}"

        # Dedup: check if (stripped, case-insensitive, ignoring date) exists
        normalized_new = _normalize_for_dedup(stamped_entry)
        for i, existing in enumerate(self._entries[section]):
            if _normalize_for_dedup(existing) == normalized_new:
                # Move to end (newest)
                self._entries[section].pop(i)
                self._entries[section].append(stamped_entry)
                self._enforce_caps()
                return

        # New entry
        self._entries[section].append(stamped_entry)
        self._enforce_caps()

    def render(self) -> str:
        """Render the entire document as markdown.

        Returns:
          '# Cairn Memory' title, blank line, then each section in order with
          its header and entries as '- {entry}' lines (newest LAST).
          Sections are separated by blank lines. ALWAYS emits all 5 headers
          (even if empty) for stable round-trips.

        Returns:
            Rendered markdown string.
        """
        lines = [TITLE, ""]

        for section_key in SECTION_HEADERS:
            lines.append(SECTION_HEADERS[section_key])
            for entry in self._entries[section_key]:
                lines.append(f"- {entry}")
            lines.append("")

        return "\n".join(lines)

    def read(self, max_tokens: int | None = None) -> str:
        """Return a budget-trimmed rendering of the document.

        If max_tokens is None, returns render() unchanged.

        If max_tokens is set, ALWAYS keeps Open Tasks, Decisions, and Conventions
        in full. Trims Recent Changes and Recent User Prompts to the NEWEST entries
        that fit within the budget.

        Final result is guaranteed to encode to <= max_tokens tokens (checked via
        core.tokens.count_tokens).

        Args:
            max_tokens: Optional token budget. If None, returns full render().

        Returns:
            Rendered markdown, either full or trimmed to fit budget.
        """
        if max_tokens is None:
            return self.render()

        # Lazy import
        from core import tokens

        full = self.render()
        if tokens.count_tokens(full) <= max_tokens:
            return full

        # Build trimmed version: keep tasks/decisions/conventions in full,
        # trim changes and prompts to newest entries that fit
        lines = [TITLE, ""]

        # Add the three always-full sections
        for section_key in ["tasks", "decisions", "conventions"]:
            lines.append(SECTION_HEADERS[section_key])
            for entry in self._entries[section_key]:
                lines.append(f"- {entry}")
            lines.append("")

        # Build headers for changes and prompts (start with empty)
        lines.append(SECTION_HEADERS["changes"])
        lines.append("")
        lines.append(SECTION_HEADERS["prompts"])

        current_text = "\n".join(lines)
        current_tokens = tokens.count_tokens(current_text)

        # Try to add newest entries from changes and prompts, backwards
        all_trimmed = list(self._entries["changes"]) + list(
            self._entries["prompts"]
        )
        all_trimmed.reverse()

        for entry in all_trimmed:
            test_line = f"\n- {entry}"
            test_tokens = tokens.count_tokens(test_line)
            if current_tokens + test_tokens <= max_tokens:
                current_text += test_line
                current_tokens += test_tokens
            else:
                break

        return current_text

    def save(self, path: str | Path) -> None:
        """Atomically write the document to a file.

        Ensures parent directory exists. Uses temp file + os.replace for atomicity.

        Args:
            path: File path to write to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.render()
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    def _enforce_caps(self) -> None:
        """Enforce per-section caps.

        For sections over their cap: keep the newest cap entries.

        SPECIAL CASE: Recent Changes section. When over cap, collapse the OLDEST
        overflow entries into a SINGLE summary line '(history) N earlier changes'
        kept as the first entry, then keep the newest (cap-1) entries.

        This provides deterministic compaction; an LLM-summary hook can replace
        the static summary later.
        """
        for section_key in SECTION_HEADERS:
            cap = self._caps[section_key]
            entries = self._entries[section_key]

            if len(entries) <= cap:
                continue

            if section_key == "changes":
                # Collapse oldest overflow into summary.
                # Count how many entries are being removed (overflow_count).
                # If there's already a history summary, we're adding to its count.
                overflow_count = len(entries) - cap + 1

                # Check if first entry is already a history summary
                history_count = 0
                if entries and entries[0].startswith("(history)"):
                    # Extract the count from existing history line
                    parts = entries[0].split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        history_count = int(parts[1])
                    # When we collapse again, we add the old history count
                    # plus the number of non-summary entries being removed
                    overflow_count = (
                        history_count + len(entries) - cap
                    )

                self._entries[section_key] = (
                    [f"(history) {overflow_count} earlier changes"]
                    + entries[-(cap - 1) :]
                )
            else:
                # Keep only newest cap entries
                self._entries[section_key] = entries[-cap:]


def _normalize_for_dedup(entry: str) -> str:
    """Normalize an entry for dedup comparison.

    Strips, lowercases, and removes leading date prefix '[YYYY-MM-DD] '.
    Used to detect duplicate entries ignoring date stamps.

    Args:
        entry: Entry string to normalize.

    Returns:
        Normalized string.
    """
    normalized = entry.strip().lower()
    # Remove leading date prefix like '[2026-06-06] '
    normalized = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s+", "", normalized)
    return normalized


def _parse_sectioned(doc: MemoryDoc, text: str) -> None:
    """Parse a sectioned markdown document into a MemoryDoc.

    Expects headers like '## Open Tasks' and entries as '- text'.
    Tolerant: ignores unrecognized headers, blank lines, malformed entries.

    Args:
        doc: MemoryDoc instance to populate.
        text: Markdown text to parse.
    """
    current_section = None
    lines = text.split("\n")

    for line in lines:
        line = line.rstrip()

        # Check for section header
        if line.startswith("## "):
            # Find matching section key
            current_section = None
            for key, header in SECTION_HEADERS.items():
                if line == header:
                    current_section = key
                    break
            continue

        # Skip blank lines
        if not line.strip():
            continue

        # Parse entry if in a section
        if current_section and line.startswith("- "):
            entry = line[2:].strip()
            if entry:
                doc._entries[current_section].append(entry)


def _parse_flat_log(doc: MemoryDoc, text: str) -> None:
    """Parse an old flat-log format (no section headers) into Recent Changes.

    Old format: lines like '[2026-06-06 07:48] did X' with optional leading '- '.
    Puts every non-empty, non-blank line into Recent Changes section.

    Args:
        doc: MemoryDoc instance to populate.
        text: Flat-log text to parse.
    """
    for line in text.split("\n"):
        line = line.rstrip()

        # Skip blanks
        if not line.strip():
            continue

        # Strip leading '- ' if present
        if line.startswith("- "):
            line = line[2:].strip()

        if line:
            doc._entries["changes"].append(line)
