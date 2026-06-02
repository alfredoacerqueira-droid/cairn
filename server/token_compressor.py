"""RTK-inspired token compression for code context.

Implements multi-level, language-aware compression strategies.
Inspired by: RTK (Rust Token Killer) - https://github.com/rtk-ai/rtk

Key features:
- 3 compression levels: None (0%), Minimal (20-40%), Aggressive (60-90%)
- Language-aware: Different strategies for Python, JS, Rust, Go
- 8-stage pipeline: Systematic, ordered compression
- Deterministic: No LLM involved, always the same output
- Fast: <10ms overhead (RTK-style performance)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FilterLevel(Enum):
    """Compression levels (RTK-style)."""

    NONE = "none"
    MINIMAL = "minimal"
    AGGRESSIVE = "aggressive"


class Language(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass
class CommentPatterns:
    """Language-specific comment patterns."""

    line: Optional[str]
    block_start: Optional[str]
    block_end: Optional[str]
    doc_line: Optional[str]
    doc_block_start: Optional[str]


class TokenCompressor:
    """RTK-style token compressor with multi-level filtering.

    Applies an 8-stage compression pipeline:
    1. Strip ANSI codes
    2. Remove comments (language-aware)
    3. Remove docstrings (aggressive only)
    4. Collapse whitespace
    5. Remove imports
    6. Truncate functions (aggressive only)
    7. Deduplicate patterns
    8. Apply max_lines limit
    """

    def __init__(self, level: FilterLevel = FilterLevel.MINIMAL, max_tokens: int = 2000):
        self.level = level
        self.max_tokens = max_tokens
        self.stats: dict = {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "reduction_pct": 0.0,
            "strategies_applied": [],
        }

    def compress(self, context: str, language: Language = Language.PYTHON) -> str:
        """Apply 8-stage compression pipeline."""
        self.stats["original_tokens"] = self._estimate_tokens(context)
        self.stats["strategies_applied"] = []

        if self.level == FilterLevel.NONE:
            self.stats["compressed_tokens"] = self.stats["original_tokens"]
            return context

        result = context
        result = self._stage1_strip_ansi(result)
        result = self._stage2_remove_comments(result, language)
        result = self._stage3_remove_docstrings(result, language)
        result = self._stage4_collapse_whitespace(result)
        result = self._stage5_remove_imports(result, language)

        if self.level == FilterLevel.AGGRESSIVE:
            result = self._stage6_truncate_functions(result, max_lines=15)
            result = self._stage7_deduplicate_patterns(result)

        result = self._stage8_apply_max_lines(result, max_lines=100)

        self.stats["compressed_tokens"] = self._estimate_tokens(result)
        self.stats["reduction_pct"] = self._calculate_reduction()

        return result

    # ── Stage 1: Strip ANSI codes ─────────────────────────────────────

    def _stage1_strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        result = ansi_escape.sub("", text)
        self.stats["strategies_applied"].append("strip_ansi")
        return result

    # ── Stage 2: Remove comments (language-aware) ──────────────────────

    def _stage2_remove_comments(self, text: str, lang: Language) -> str:
        """Remove comments based on language."""
        patterns = self._get_comment_patterns(lang)
        if not patterns or not patterns.line:
            return text

        lines = text.split("\n")
        filtered = []

        for line in lines:
            stripped = line.strip()
            important_markers = [
                "TODO",
                "FIXME",
                "XXX",
                "HACK",
                "NOTE",
                "BUG",
                "OPTIMIZE",
                "REVIEW",
                "WARNING",
            ]
            if (
                any(marker in stripped for marker in important_markers)
                and patterns.line in stripped
            ):
                filtered.append(line)
                continue

            # Find the last comment marker NOT inside a string literal
            comment_pos = -1
            pos = 0
            while True:
                pos = line.find(patterns.line, pos)
                if pos < 0:
                    break
                if not _is_inside_string(line, pos):
                    comment_pos = pos  # Keep searching for later genuine comment
                pos += len(patterns.line)

            if comment_pos >= 0:
                # Safe to strip from the genuine comment position
                code_part = line[:comment_pos].rstrip()
                if code_part and not line.lstrip().startswith(patterns.line):
                    filtered.append(code_part)
                elif not line.lstrip().startswith(patterns.line):
                    filtered.append(line)
                # else: whole line was a comment → drop it
            else:
                filtered.append(line)

        self.stats["strategies_applied"].append("remove_comments")
        return "\n".join(filtered)

    # ── Stage 3: Remove docstrings (aggressive only) ───────────────────

    def _stage3_remove_docstrings(self, text: str, lang: Language) -> str:
        """Remove docstrings (aggressive mode only)."""
        if self.level != FilterLevel.AGGRESSIVE:
            return text

        if lang == Language.PYTHON:
            text = re.sub(r'"""[\s\S]*?"""', '"""..."""', text)
            text = re.sub(r"'''[\s\S]*?'''", "'''...'''", text)
        elif lang in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.RUST):
            text = re.sub(r"/\*\*[\s\S]*?\*/", "/** ... */", text)
            text = re.sub(r"/\*![\s\S]*?\*/", "/*! ... */", text)

        self.stats["strategies_applied"].append("remove_docstrings")
        return text

    # ── Stage 4: Collapse whitespace ───────────────────────────────────

    def _stage4_collapse_whitespace(self, text: str) -> str:
        """Remove extra blank lines and trailing whitespace."""
        # Remove 3+ blank lines
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        self.stats["strategies_applied"].append("collapse_whitespace")
        return "\n".join(lines)

    # ── Stage 5: Remove imports ────────────────────────────────────────

    def _stage5_remove_imports(self, text: str, lang: Language) -> str:
        """Remove import statements (already in repo map)."""
        lines = text.split("\n")
        filtered = []

        import_prefixes: dict[Language, tuple] = {
            Language.PYTHON: ("import ", "from "),
            Language.JAVASCRIPT: ("import ", "export ", "require("),
            Language.TYPESCRIPT: ("import ", "export ", "require("),
            Language.RUST: ("use ", "extern crate "),
            Language.GO: ("import ",),
        }

        prefixes = import_prefixes.get(lang, ("import ",))
        import_section = False

        for line in lines:
            stripped = line.strip()

            # Detect start of import section
            is_exception = stripped.startswith("importlib") or stripped.startswith("import_string")
            if stripped.startswith(prefixes) and not is_exception:
                if not import_section:
                    filtered.append("# [imports removed - see repo map]")
                    import_section = True
                continue

            import_section = False
            filtered.append(line)

        self.stats["strategies_applied"].append("remove_imports")
        return "\n".join(filtered)

    # ── Stage 6: Truncate functions (aggressive only) ──────────────────

    def _stage6_truncate_functions(self, text: str, max_lines: int = 15) -> str:
        """Truncate long function bodies, keep signature + key lines."""
        if self.level != FilterLevel.AGGRESSIVE:
            return text

        lines = text.split("\n")
        result = []
        in_function = False
        function_lines: list[str] = []
        indent_level = 0

        # Regex for function/method declarations
        func_pattern = re.compile(r"^\s*(def |async def |function |fn |func |class |@\w+)")

        for line in lines:
            if line.strip().startswith("class ") or func_pattern.match(line):
                # Flush previous function
                if function_lines:
                    result.extend(self._truncate_block(function_lines, max_lines))
                function_lines = [line]
                in_function = True
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                stripped = line.strip()
                current_indent = len(line) - len(line.lstrip()) if stripped else indent_level + 4
                if stripped and current_indent <= indent_level:
                    result.extend(self._truncate_block(function_lines, max_lines))
                    function_lines = [line]
                    in_function = False
                else:
                    function_lines.append(line)
            else:
                result.append(line)

        if function_lines:
            result.extend(self._truncate_block(function_lines, max_lines))

        self.stats["strategies_applied"].append("truncate_functions")
        return "\n".join(result)

    def _truncate_block(self, lines: list[str], max_lines: int) -> list[str]:
        """Truncate a block of lines if it exceeds max_lines."""
        if len(lines) <= max_lines or not lines:
            return lines

        keep_first = min(5, len(lines))
        keep_last = min(5, len(lines) - keep_first)
        skipped = len(lines) - keep_first - keep_last

        if skipped <= 0:
            return lines

        return (
            lines[:keep_first]
            + [f"{' ' * 4}# ... ({skipped} lines omitted) ..."]
            + lines[-keep_last:]
        )

    # ── Stage 7: Deduplicate patterns ──────────────────────────────────

    def _stage7_deduplicate_patterns(self, text: str) -> str:
        """Collapse repeated code patterns in aggressive mode."""
        if self.level != FilterLevel.AGGRESSIVE:
            return text

        lines = text.split("\n")
        result = []
        seen_patterns: dict[str, int] = {}
        skip_count = 0

        for line in lines:
            if not line.strip():
                result.append(line)
                skip_count = 0
                continue

            pattern = re.sub(r"\d+", "N", line.strip())  # Normalize numbers
            pattern = re.sub(r"['\"][^'\"]*['\"]", '"..."', pattern)  # Normalize strings

            if pattern in seen_patterns:
                skip_count += 1
                if skip_count == 1:
                    result.append(f"    # [duplicate pattern ({seen_patterns[pattern]})]")
            else:
                seen_patterns[pattern] = seen_patterns.get(pattern, 0) + 1
                result.append(line)
                skip_count = 0

        self.stats["strategies_applied"].append("deduplicate_patterns")
        return "\n".join(result)

    # ── Stage 8: Apply max_lines limit ─────────────────────────────────

    def _stage8_apply_max_lines(self, text: str, max_lines: int = 100) -> str:
        """Apply maximum line limit."""
        lines = text.split("\n")
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines])
            text += f"\n\n# ... ({len(lines) - max_lines} more lines omitted) ..."

        self.stats["strategies_applied"].append("max_lines")
        return text

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_comment_patterns(self, lang: Language) -> Optional[CommentPatterns]:
        """Get comment patterns for a language."""
        patterns: dict[Language, CommentPatterns] = {
            Language.PYTHON: CommentPatterns(
                line="#", block_start='"""', block_end='"""', doc_line=None, doc_block_start='"""'
            ),
            Language.JAVASCRIPT: CommentPatterns(
                line="//", block_start="/*", block_end="*/", doc_line=None, doc_block_start="/**"
            ),
            Language.TYPESCRIPT: CommentPatterns(
                line="//", block_start="/*", block_end="*/", doc_line=None, doc_block_start="/**"
            ),
            Language.RUST: CommentPatterns(
                line="//", block_start="/*", block_end="*/", doc_line="///", doc_block_start="/**"
            ),
            Language.GO: CommentPatterns(
                line="//", block_start="/*", block_end="*/", doc_line=None, doc_block_start=None
            ),
        }
        return patterns.get(lang)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return max(1, len(text) // 4)

    def _calculate_reduction(self) -> float:
        """Calculate compression percentage."""
        original = self.stats["original_tokens"]
        compressed = self.stats["compressed_tokens"]
        if original <= 0:
            return 0.0
        return round(100.0 * (1 - compressed / original), 1)

    def get_stats(self) -> dict:
        """Return compression statistics."""
        return self.stats.copy()


# ── Module-level helpers ──────────────────────────────────────────────────────


def _is_inside_string(line: str, pos: int) -> bool:
    """Check if a position in a line is inside a quoted string literal.

    Counts quote characters before the position.  If an odd number of single
    or double quotes precedes the position, it is inside a quoted string.
    """
    prefix = line[:pos]
    in_single = False
    in_double = False
    escaped = False

    for ch in prefix:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double

    return in_single or in_double
