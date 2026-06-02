# ruff: noqa: E501  # Regex patterns are inherently long, not breakable

"""Deterministic AST parsing using tree-sitter (Python/HCL/YAML/C#/bash) + regex (others)."""

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

try:
    from tree_sitter_language_pack import get_parser as _get_parser

    get_parser: Callable[[str], Any] = _get_parser
except ImportError:
    get_parser = None  # type: ignore

PY_LANGUAGE = Language(tspython.language())
PY_PARSER = Parser(PY_LANGUAGE)

# ── Language patterns for regex-based extraction ──────────────────────────────

LANGUAGE_PATTERNS: dict[str, dict[str, str | None]] = {
    "python": {
        "function": r"^\s*(async\s+)?def\s+(\w+)\s*\(",
        "class": r"^\s*class\s+(\w+)\s*[(:]",
        "method": r"^\s+def\s+(\w+)\s*\(",
    },
    "javascript": {
        "function": r"(?:function\s+(\w+)\s*\(|(\w+)\s*=\s*(?:async\s+)?\(|(\w+)\s*=\s*function)",
        "class": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?\{",
        "method": r"(?:^\s+(\w+)\s*\(|async\s+(\w+)\s*\()",
    },
    "typescript": {
        "function": r"(?:function\s+(\w+)\s*\(|(\w+)\s*=\s*(?:async\s+)?\(|(\w+)\s*=\s*function)",
        "class": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?(?:implements\s+[\w,\s]+)?\{",
        "method": r"(?:^\s+(?:public\s+|private\s+|protected\s+)?(?:async\s+)?(\w+)\s*\()",
    },
    "rust": {
        "function": r"^\s*(?:pub\s+(?:async\s+)?)?fn\s+(\w+)\s*[<(]",
        "class": r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)",
        "method": r"^\s+fn\s+(\w+)\s*[<(]",
    },
    "go": {
        "function": r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        "class": r"^\s*type\s+(\w+)\s+struct\s*\{",
        "method": r"^\s*func\s+\(\w+\s+\*?\w+\)\s+(\w+)\s*\(",
    },
    "csharp": {
        "function": r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|async\s+|virtual\s+|override\s+)*(?:void|int|string|bool|var|Task|async|[\w<>[\],\s]+)\s+(\w+)\s*\(",
        "class": r"^\s*(?:public\s+|private\s+|internal\s+)?(?:static\s+)?class\s+(\w+)",
        "method": r"^\s+(?:public\s+|private\s+|protected\s+)*(?:void|int|string|bool|var|Task|[\w<>[\],\s]+)\s+(\w+)\s*\(",
    },
    "java": {
        "function": r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*(?:void|int|String|boolean|List|Map|[\w<>[\],\s]+)\s+(\w+)\s*\(",
        "class": r"^\s*(?:public\s+|private\s+)?(?:abstract\s+)?class\s+(\w+)",
        "method": r"^\s+(?:public\s+|private\s+|protected\s+)*(?:void|int|String|boolean|[\w<>[\],\s]+)\s+(\w+)\s*\(",
    },
    "ruby": {
        "function": r"^\s*def\s+(self\.)?(\w+)",
        "class": r"^\s*class\s+(\w+)\s*(?:<\s+\w+)?",
        "method": r"^\s+def\s+(\w+)",
    },
    "cpp": {
        "function": r"^\s*(?:virtual\s+)?(?:void|int|char|bool|float|double|auto|[\w:*&<>,\s]+)\s+(\w+)\s*\(",
        "class": r"^\s*class\s+(\w+)\s*(?::\s*public\s+\w+)?\s*\{",
        "method": r"^\s+(?:virtual\s+)?(?:void|int|char|bool|[\w:*&<>,\s]+)\s+(\w+)\s*\(",
    },
    "bash": {
        "function": r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{",
        "class": None,  # Bash has no classes
        "method": None,
    },
    "yaml": {
        "function": r"^\s*(\w[\w-]*)\s*:",  # Top-level keys as "functions"
        "class": None,
        "method": None,
    },
    "hcl": {  # Terraform
        "function": r'^\s*(resource|data|module|variable|output|provider|locals|terraform)\s+"(\w+)"\s+"(\w+)"',
        "class": None,
        "method": None,
    },
    "json": {
        "function": r'^\s*"(\w+)"\s*:',  # JSON keys as "functions"
        "class": None,
        "method": None,
    },
    "toml": {
        "function": r"^\s*\[(\w[\w.-]*)\]",  # TOML sections as "functions"
        "class": None,
        "method": None,
    },
}

# File extension → language mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".c": "cpp",
    ".h": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".json": "json",
    ".toml": "toml",
    ".cfg": "toml",
    ".ini": "toml",
}


# ── Data classes ──────────────────────────────────────────────────────────────


class FunctionDef:
    def __init__(self, name: str, line_start: int, line_end: int, code: str):
        self.name = name
        self.line_start = line_start
        self.line_end = line_end
        self.code = code

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code": self.code,
        }


class ClassDef:
    def __init__(self, name: str, line_start: int, line_end: int, code: str):
        self.name = name
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        self.methods: list[FunctionDef] = []

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code": self.code,
            "methods": [m.to_dict() for m in self.methods],
        }


class FileAST:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.functions: list[FunctionDef] = []
        self.classes: list[ClassDef] = []

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
        }


# ── Parser ────────────────────────────────────────────────────────────────────


class ASTParser:
    def __init__(
        self,
        max_file_kb: int = 0,
        parse_timeout_s: float = 10.0,
    ):
        """Initialize parser with optional size and timeout limits.

        Args:
            max_file_kb: Maximum file size in KB before skipping. 0 = unlimited.
            parse_timeout_s: Maximum parse time in seconds per file. 0 = no timeout.
        """
        self.max_file_kb = max_file_kb
        self.parse_timeout_s = parse_timeout_s

    def parse_file(self, filepath: str | Path) -> FileAST:
        filepath = Path(filepath)

        # SIZE LIMIT: check file size before reading
        if self.max_file_kb > 0:
            file_size_bytes = filepath.stat().st_size
            file_size_kb = file_size_bytes / 1024
            if file_size_kb > self.max_file_kb:
                logger.warning(
                    f"Skipping {filepath}: {file_size_kb:.1f}KB exceeds "
                    f"max_file_kb={self.max_file_kb}"
                )
                return FileAST(str(filepath))

        # Always decode as UTF-8 (the de-facto encoding for source). Without an
        # explicit encoding, read_text() uses the platform locale (cp1252 on native
        # Windows), which mojibakes any non-ASCII identifier/comment and yields a
        # different corruption per OS. errors="replace" keeps a single bad byte from
        # aborting the whole file's indexing.
        code = filepath.read_text(encoding="utf-8", errors="replace")
        return self.parse_string(code, str(filepath))

    def parse_string(self, code: str, filepath: str = "<string>", lang: str = "") -> FileAST:
        if not lang:
            lang = self._detect_language(filepath)

        # Default to Python for inline code (tests, REPL)
        if lang == "unknown":
            lang = "python"

        # PARSE TIMEOUT: wrap the parse operation with a timeout if configured
        if self.parse_timeout_s > 0:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._parse_impl, code, filepath, lang)
                    return future.result(timeout=self.parse_timeout_s)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Parse timeout ({self.parse_timeout_s}s) on {filepath}; skipping")
                return FileAST(filepath)
        else:
            return self._parse_impl(code, filepath, lang)

    def _parse_impl(self, code: str, filepath: str, lang: str) -> FileAST:
        """Internal parse implementation (language routing).

        This is extracted to be called either directly or via timeout wrapper.
        """
        # Route to appropriate parser
        if lang == "python":
            return self._tree_sitter_parse(code, filepath)
        elif lang in ("hcl", "yaml", "csharp", "bash"):
            try:
                return self._treesitter_parse_generic(code, filepath, lang)
            except Exception:
                # Fallback to regex on any tree-sitter error
                return self._regex_parse(code, filepath, lang)
        else:
            # Regex for all others
            return self._regex_parse(code, filepath, lang)

    def _detect_language(self, filepath: str) -> str:
        """Detect language from file extension."""
        path = Path(filepath)
        ext = path.suffix.lower()
        if not ext and path.name == "Dockerfile":
            return "bash"
        return EXTENSION_MAP.get(ext, "unknown")

    # ── Tree-sitter (Python) ─────────────────────────────────

    def _tree_sitter_parse(self, code: str, filepath: str) -> FileAST:
        tree = PY_PARSER.parse(code.encode())
        root = tree.root_node
        lines = code.split("\n")

        result = FileAST(filepath)
        self._collect_defs(root, lines, result)
        return result

    def _collect_defs(
        self,
        node: Node,
        lines: list[str],
        result: FileAST,
        inside_class: Optional[ClassDef] = None,
    ):
        if node.type == "function_definition":
            func = self._extract_function(node, lines)
            if inside_class:
                inside_class.methods.append(func)
            else:
                result.functions.append(func)
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    self._collect_defs(child, lines, result, inside_class)

        elif node.type == "class_definition":
            cls = self._extract_class(node, lines)
            result.classes.append(cls)
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    self._collect_defs(child, lines, result, inside_class=cls)

        elif node.type == "decorated_definition":
            for child in node.children:
                self._collect_defs(child, lines, result, inside_class)

        else:
            for child in node.children:
                self._collect_defs(child, lines, result, inside_class)

    def _extract_function(self, node: Node, lines: list[str]) -> FunctionDef:
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode() if (name_node and name_node.text) else "<anonymous>"
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        code_lines = lines[node.start_point[0] : node.end_point[0] + 1]
        code = "\n".join(code_lines).strip()
        return FunctionDef(name=name, line_start=line_start, line_end=line_end, code=code)

    def _extract_class(self, node: Node, lines: list[str]) -> ClassDef:
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode() if (name_node and name_node.text) else "<anonymous>"
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        code_lines = lines[node.start_point[0] : node.end_point[0] + 1]
        code = "\n".join(code_lines).strip()
        return ClassDef(name=name, line_start=line_start, line_end=line_end, code=code)

    # ── Tree-sitter generic (HCL/YAML/C#/bash) ───────────────

    def _treesitter_parse_generic(self, code: str, filepath: str, lang: str) -> FileAST:
        """Parse HCL/YAML/C#/bash using tree-sitter language pack."""
        if get_parser is None:
            raise ImportError("tree-sitter-language-pack not installed")

        # Map our language names to pack's parser names
        lang_map = {
            "hcl": "hcl",
            "yaml": "yaml",
            "csharp": "csharp",
            "bash": "bash",
        }
        parser_name = lang_map.get(lang, lang)
        parser = get_parser(parser_name)
        # Strip a leading UTF-8 BOM before parsing so offsets, lines, and stored
        # block code are all BOM-free and consistent (Windows/C# files often have one).
        code = code.lstrip("\ufeff")
        tree = parser.parse(code)  # String, not bytes!
        lines = code.split("\n")
        result = FileAST(filepath)

        # tree-sitter reports start_byte()/end_byte() as offsets into the UTF-8
        # encoding, NOT character offsets. Slicing the str directly is only correct
        # for pure-ASCII files; a UTF-8 BOM (3 bytes / 1 char, common in Windows &
        # C# files) or any multibyte char shifts every name. Slice the bytes.
        code_bytes = code.encode("utf-8")

        if lang == "hcl":
            self._extract_hcl_blocks(tree.root_node(), lines, code_bytes, result)
        elif lang == "yaml":
            self._extract_yaml_blocks(lines, code, result)
        elif lang == "csharp":
            self._extract_csharp_types(tree.root_node(), lines, code_bytes, result)
        elif lang == "bash":
            self._extract_bash_functions(tree.root_node(), lines, code_bytes, result)

        return result

    def _extract_hcl_blocks(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract Terraform blocks using tree-sitter-language-pack API."""
        kind = node.kind()
        if kind == "block":
            # Extract block type and labels
            block_type = None
            labels = []
            block_start = node.start_position().row
            block_end = node.end_position().row

            for i in range(node.child_count()):
                child = node.child(i)
                child_kind = child.kind()

                if child_kind == "identifier":
                    text = code_bytes[child.start_byte() : child.end_byte()].decode(
                        "utf-8", "replace"
                    )
                    if block_type is None:
                        block_type = text
                    else:
                        labels.append(text)
                elif child_kind == "string_lit":
                    # Extract content of string literal
                    text = code_bytes[child.start_byte() : child.end_byte()].decode(
                        "utf-8", "replace"
                    )
                    # Remove quotes if present
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    labels.append(text)

            if block_type:
                name_parts = [block_type] + labels
                name = ".".join(name_parts)
                block_code = "\n".join(lines[block_start : block_end + 1]).strip()
                if len(block_code) < 100000:
                    result.functions.append(
                        FunctionDef(
                            name=name,
                            line_start=block_start + 1,
                            line_end=block_end + 1,
                            code=block_code,
                        )
                    )

        # Recurse into children
        for i in range(node.child_count()):
            child = node.child(i)
            self._extract_hcl_blocks(child, lines, code_bytes, result)

    def _extract_yaml_blocks(self, lines: list[str], code: str, result: FileAST):
        """Extract YAML documents by kind and name."""
        docs = code.split("\n---\n")
        start_line = 0

        for doc_text in docs:
            doc_lines = doc_text.split("\n")
            doc_line_count = len(doc_lines)

            kind = None
            name = None

            for line in doc_lines:
                if line.strip().startswith("kind:"):
                    kind = line.split(":", 1)[1].strip()
                elif "metadata:" in line:
                    idx = doc_lines.index(line)
                    for j in range(idx + 1, min(idx + 5, len(doc_lines))):
                        if doc_lines[j].strip().startswith("name:"):
                            name = doc_lines[j].split(":", 1)[1].strip()
                            break

            block_name = f"{kind}.{name}" if kind and name else (kind or doc_lines[0][:30])
            if len(doc_text.strip()) > 0 and len(doc_text) < 100000:
                result.functions.append(
                    FunctionDef(
                        name=block_name,
                        line_start=start_line + 1,
                        line_end=start_line + doc_line_count,
                        code=doc_text.strip(),
                    )
                )

            start_line += doc_line_count + 1

    def _extract_csharp_types(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract C# classes, interfaces, structs, records."""
        kind = node.kind()
        if kind in (
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "record_declaration",
        ):
            # Find identifier child
            name = None
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "identifier":
                    name = code_bytes[child.start_byte() : child.end_byte()].decode(
                        "utf-8", "replace"
                    )
                    break

            if name:
                block_start = node.start_position().row
                block_end = node.end_position().row
                block_code = "\n".join(lines[block_start : block_end + 1]).strip()

                if len(block_code) < 100000:
                    cls = ClassDef(
                        name=name,
                        line_start=block_start + 1,
                        line_end=block_end + 1,
                        code=block_code,
                    )
                    result.classes.append(cls)

                    # Extract methods from this type
                    self._extract_csharp_methods(node, lines, code_bytes, cls)

        # Recurse to find nested types
        for i in range(node.child_count()):
            child = node.child(i)
            self._extract_csharp_types(child, lines, code_bytes, result)

    def _extract_csharp_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Recursively extract method_declaration nodes as methods."""
        kind = node.kind()
        if kind == "method_declaration":
            # Find identifier right before parameter_list (the actual method name)
            name = None
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "parameter_list":
                    # Look backwards for the last identifier
                    for j in range(i - 1, -1, -1):
                        prev_child = node.child(j)
                        if prev_child.kind() == "identifier":
                            name = code_bytes[
                                prev_child.start_byte() : prev_child.end_byte()
                            ].decode("utf-8", "replace")
                            break
                    break

            if name:
                method_start = node.start_position().row
                method_end = node.end_position().row
                method_code = "\n".join(lines[method_start : method_end + 1]).strip()

                if len(method_code) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=name,
                            line_start=method_start + 1,
                            line_end=method_end + 1,
                            code=method_code,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            child = node.child(i)
            self._extract_csharp_methods(child, lines, code_bytes, cls)

    def _extract_bash_functions(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract bash function_definition nodes."""
        kind = node.kind()
        if kind == "function_definition":
            # Get function name from first word child
            name = None
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "word":
                    name = code_bytes[child.start_byte() : child.end_byte()].decode(
                        "utf-8", "replace"
                    )
                    break

            if name:
                func_start = node.start_position().row
                func_end = node.end_position().row
                func_code = "\n".join(lines[func_start : func_end + 1]).strip()

                if len(func_code) < 100000:
                    result.functions.append(
                        FunctionDef(
                            name=name,
                            line_start=func_start + 1,
                            line_end=func_end + 1,
                            code=func_code,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            child = node.child(i)
            self._extract_bash_functions(child, lines, code_bytes, result)

    # ── Regex-based (all other languages) ────────────────────────

    def _regex_parse(self, code: str, filepath: str, lang: str) -> FileAST:
        """Extract functions/classes using language-specific regex patterns."""
        result = FileAST(filepath)
        patterns = LANGUAGE_PATTERNS.get(lang)

        if not patterns:
            return result  # Unknown language → empty result

        lines = code.split("\n")

        # Extract top-level functions
        func_pattern = patterns.get("function")
        if func_pattern:
            for i, line in enumerate(lines):
                match = re.search(func_pattern, line)
                if match:
                    # Get name from captured group
                    groups = [g for g in match.groups() if g]
                    name = (
                        groups[0]
                        if groups
                        else (match.group(0).split()[-1] if match.group(0).split() else "<unknown>")
                    )
                    name = name.strip()

                    # Extract function body
                    body_lines = self._extract_block(lines, i)
                    body = "\n".join(body_lines).strip()
                    if len(body) < 10000:  # Skip massive blobs
                        result.functions.append(
                            FunctionDef(
                                name=name,
                                line_start=i + 1,
                                line_end=i + len(body_lines),
                                code=body,
                            )
                        )

        # Extract classes
        class_pattern = patterns.get("class")
        if class_pattern:
            for i, line in enumerate(lines):
                match = re.search(class_pattern, line)
                if match:
                    name = next(
                        (g for g in match.groups() if g),
                        match.group(0).split()[-1] if match.group(0).split() else "<unknown>",
                    )
                    name = name.strip()

                    body_lines = self._extract_block(lines, i)
                    body = "\n".join(body_lines).strip()
                    if len(body) < 10000:
                        cls = ClassDef(
                            name=name,
                            line_start=i + 1,
                            line_end=i + len(body_lines),
                            code=body,
                        )
                        result.classes.append(cls)

        return result

    def _extract_block(self, lines: list[str], start: int) -> list[str]:
        """Extract indented/braced block."""
        if start >= len(lines):
            return [lines[start]] if start < len(lines) else []

        first_line = lines[start]

        # Brace-based languages
        if "{" in first_line:
            depth = 0
            block: list[str] = []
            for i in range(start, len(lines)):
                block.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                if depth == 0 and i > start:
                    return block
            return block

        # Indent-based languages
        base_indent = len(first_line) - len(first_line.lstrip())
        indent_block: list[str] = [first_line]
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                indent_block.append(line)
                continue
            indent = len(line) - len(line.lstrip())
            if indent > base_indent or line.strip().startswith(("#", "//")):
                indent_block.append(line)
            else:
                break
        return indent_block

    def diff_update(self, filepath: str | Path, old_ast: FileAST) -> FileAST:
        """Incremental update - re-parse only if file changed."""
        filepath = Path(filepath)
        if not filepath.exists():
            return FileAST(str(filepath))
        new_ast = self.parse_file(filepath)
        return new_ast
