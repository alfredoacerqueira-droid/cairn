"""Declarative TOML filter system (RTK-style).

Allows easy customization of compression rules without code changes.
Each .toml file defines regex patterns for filtering output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


class TomlFilter:
    """Declarative filter defined in TOML."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.description = config.get("description", "")
        self.strip_lines_matching: list[str] = config.get("strip_lines_matching", [])
        self.keep_lines_matching: list[str] = config.get("keep_lines_matching", [])
        self.max_lines: Optional[int] = config.get("max_lines")
        self.on_empty: str = config.get("on_empty", "")

    def apply(self, text: str) -> str:
        """Apply filter to text."""
        lines = text.split("\n")

        # Strip lines matching patterns
        if self.strip_lines_matching:
            filtered = []
            for line in lines:
                if not any(re.search(p, line) for p in self.strip_lines_matching):
                    filtered.append(line)
            lines = filtered

        # Keep only lines matching patterns
        if self.keep_lines_matching:
            filtered = []
            for line in lines:
                if any(re.search(p, line) for p in self.keep_lines_matching):
                    filtered.append(line)
            lines = filtered

        # Apply max_lines
        if self.max_lines and len(lines) > self.max_lines:
            lines = lines[: self.max_lines]

        result = "\n".join(lines)

        # Apply on_empty
        if not result.strip() and self.on_empty:
            result = self.on_empty

        return result


class TomlFilterRegistry:
    """Registry of TOML filters."""

    def __init__(self):
        self.filters: dict[str, TomlFilter] = {}

    def load_from_directory(self, directory: Path) -> None:
        """Load all .toml files from directory."""
        if not directory.exists():
            return

        for toml_file in sorted(directory.glob("*.toml")):
            try:
                config = self._parse_file(toml_file)
                if isinstance(config, dict):
                    for filter_name, filter_config in config.items():
                        if isinstance(filter_config, dict):
                            self.filters[filter_name] = TomlFilter(filter_name, filter_config)
            except Exception as e:
                print(f"Warning: Failed to load {toml_file}: {e}")

    def _parse_file(self, path: Path) -> dict[str, Any]:
        """Parse a TOML file."""
        with open(path) as f:
            content = f.read()
        # Simple parse for our TOML-like format
        return self._parse_toml_simple(content)

    def _parse_toml_simple(self, content: str) -> dict[str, Any]:
        """Simple TOML parser for our filter format."""
        result: dict[str, Any] = {}
        current_section: Optional[str] = None

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header [filters.name]
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                parts = section.split(".", 1)
                if len(parts) == 2 and parts[0] == "filters":
                    current_section = parts[1]
                    result[current_section] = {}
                continue

            # Key-value pairs
            if current_section and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Parse lists
                if value.startswith("[") and value.endswith("]"):
                    items = [
                        i.strip().strip('"').strip("'") for i in value[1:-1].split(",") if i.strip()
                    ]
                    result[current_section][key] = items
                # Parse integers
                elif value.isdigit():
                    result[current_section][key] = int(value)
                else:
                    result[current_section][key] = value

        return result

    def get_filter(self, name: str) -> Optional[TomlFilter]:
        """Get filter by name."""
        return self.filters.get(name)

    def apply_filter(self, name: str, text: str) -> str:
        """Apply named filter to text."""
        filter_obj = self.get_filter(name)
        if filter_obj:
            return filter_obj.apply(text)
        return text
