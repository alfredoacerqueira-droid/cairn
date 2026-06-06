# ruff: noqa: E501  # Regex patterns are inherently long, not breakable

"""Deterministic AST parsing using tree-sitter for Python/Go/Rust/Java/JS/TS/HCL/YAML/C#/bash + regex fallback."""

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

# Cache for tree-sitter parsers from language pack
_PARSER_CACHE: dict[str, Any] = {}

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
    ".hcl": "hcl",  # Terragrunt (terragrunt.hcl, root.hcl, account.hcl, ...)
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
    # Hard ceiling for tree-sitter language-pack parsers (Go/Rust/Java/JS/TS).
    # These are ~linear with input size, but pathological inputs (deeply nested,
    # huge minified, etc.) can stall indexing on slow FSes (/mnt/c on WSL2).
    # 1500 KB is ~100K lines of source, well above typical files but prevents
    # indexing a multi-MB bundle from hanging the janitor.
    TREESITTER_ML_MAX_KB = 1500

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

        # INPUT-SIZE CEILING for language-pack parsers (Go/Rust/Java/JS/TS/TSX/C++/Ruby).
        # These are ~linear with input size but can stall on pathological files
        # (minified, deeply nested, huge). Language-pack parsers are not thread-safe,
        # so we cannot use thread-based timeout. Input-size ceiling is the hang guard.
        ml_languages = {"go", "rust", "java", "javascript", "typescript", "tsx", "cpp", "ruby"}
        if lang in ml_languages:
            code_bytes = code.encode("utf-8")
            code_kb = len(code_bytes) / 1024
            if code_kb > self.TREESITTER_ML_MAX_KB:
                logger.warning(
                    f"Skipping {filepath}: {code_kb:.1f}KB exceeds "
                    f"max for tree-sitter ML parsers ({self.TREESITTER_ML_MAX_KB}KB); "
                    f"falling back to regex"
                )
                return self._regex_parse(code, filepath, lang)
            # Skip thread-based timeout for language-pack (not thread-safe)
            return self._parse_impl(code, filepath, lang)

        # PARSE TIMEOUT: wrap the parse operation with a timeout if configured.
        # Python parser is thread-safe, so we can use thread-based timeout.
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
        elif lang in ("go", "rust", "java", "javascript", "typescript", "tsx", "cpp", "ruby"):
            # Real tree-sitter AST for these languages
            try:
                return self._treesitter_parse_ml(code, filepath, lang)
            except Exception as e:
                logger.warning(
                    f"Tree-sitter parse failed for {lang} in {filepath}: {e}; falling back to regex"
                )
                return self._regex_parse(code, filepath, lang)
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

    # ── Tree-sitter multi-language (Go/Rust/Java/JS/TS) ───────

    def _get_cached_parser(self, lang: str):
        """Get or cache a tree-sitter parser for the given language."""
        if lang not in _PARSER_CACHE:
            if get_parser is None:
                raise ImportError("tree-sitter-language-pack not installed")
            _PARSER_CACHE[lang] = get_parser(lang)
        return _PARSER_CACHE[lang]

    def _treesitter_parse_ml(self, code: str, filepath: str, lang: str) -> FileAST:
        """Parse Go/Rust/Java/JavaScript/TypeScript/C++/Ruby using tree-sitter language pack.

        Extracts top-level functions, classes/types, and methods (for class members).
        Method names are prefixed with their class/receiver type (e.g., 'Class.method').

        Note: language-pack parsers do not expose timeout_micros (thread-safe timeout).
        Hang safety relies on the input-size ceiling applied in parse_string().
        """
        parser = self._get_cached_parser(lang)
        code = code.lstrip("﻿")  # Strip BOM

        tree = parser.parse(code)
        if tree is None:
            # Parser returned None (unlikely with language-pack, but be safe).
            logger.debug(f"Parse returned None for {filepath} ({lang}); falling back to regex")
            return self._regex_parse(code, filepath, lang)

        lines = code.split("\n")
        code_bytes = code.encode("utf-8")
        result = FileAST(filepath)

        if lang == "go":
            self._extract_go_defs(tree.root_node(), lines, code_bytes, result)
        elif lang == "rust":
            self._extract_rust_defs(tree.root_node(), lines, code_bytes, result)
        elif lang == "java":
            self._extract_java_defs(tree.root_node(), lines, code_bytes, result)
        elif lang in ("javascript", "typescript", "tsx"):
            self._extract_js_defs(tree.root_node(), lines, code_bytes, result, lang)
        elif lang == "cpp":
            self._extract_cpp_defs(tree.root_node(), lines, code_bytes, result)
        elif lang == "ruby":
            self._extract_ruby_defs(tree.root_node(), lines, code_bytes, result)

        return result

    # ── Go extraction ────────────────────────────────────────────

    def _extract_go_defs(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract Go functions, methods, and types (structs/interfaces)."""
        kind = node.kind()

        if kind == "function_declaration":
            name = self._get_go_func_name(node, code_bytes)
            if name:
                self._add_function(node, lines, name, result)

        elif kind == "method_declaration":
            name, receiver = self._get_go_method_name(node, code_bytes)
            if name:
                # For Go methods, prefix with receiver type (e.g., "Receiver.method")
                full_name = f"{receiver}.{name}" if receiver else name
                self._add_function(node, lines, full_name, result)

        elif kind == "type_declaration":
            # Go type_declaration contains type_spec nodes
            # Extract each type_spec within
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "type_spec":
                    self._extract_go_type_spec(child, lines, code_bytes, result)

        # Recurse
        for i in range(node.child_count()):
            self._extract_go_defs(node.child(i), lines, code_bytes, result)

    def _get_go_func_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract function name from function_declaration."""
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "identifier":
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _get_go_method_name(self, node, code_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
        """Extract method name and receiver type from method_declaration.

        Returns (method_name, receiver_type_name) or (None, None) if extraction fails.
        """
        method_name = None
        receiver_type = None
        param_list_seen = False

        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "parameter_list":
                # First parameter_list is the receiver, second is the actual parameters
                if not param_list_seen:
                    receiver_type = self._get_go_receiver_type(child, code_bytes)
                    param_list_seen = True
            elif child.kind() == "field_identifier":
                # Method name comes after receiver
                method_name = code_bytes[child.start_byte() : child.end_byte()].decode(
                    "utf-8", "replace"
                )

        return method_name, receiver_type

    def _get_go_receiver_type(self, param_list_node, code_bytes: bytes) -> Optional[str]:
        """Extract receiver type name from parameter list (e.g., *Calculator from (c *Calculator))."""
        for i in range(param_list_node.child_count()):
            child = param_list_node.child(i)
            if child.kind() == "parameter_declaration":
                # Look for type identifier or pointer within this parameter
                for j in range(child.child_count()):
                    type_child = child.child(j)
                    kind = type_child.kind()
                    if kind == "type_identifier":
                        return code_bytes[type_child.start_byte() : type_child.end_byte()].decode(
                            "utf-8", "replace"
                        )
                    elif kind == "pointer_type":
                        # Extract type from *Type
                        for k in range(type_child.child_count()):
                            ptr_child = type_child.child(k)
                            if ptr_child.kind() == "type_identifier":
                                return code_bytes[
                                    ptr_child.start_byte() : ptr_child.end_byte()
                                ].decode("utf-8", "replace")
        return None

    def _extract_go_type_spec(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract a single Go type specification (from within type_declaration)."""
        # type_spec contains: type_identifier struct_type/interface_type
        type_name = None
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "type_identifier":
                type_name = code_bytes[child.start_byte() : child.end_byte()].decode(
                    "utf-8", "replace"
                )
                break

        if type_name:
            # Record as a class
            start_row = node.start_position().row
            end_row = node.end_position().row
            code_text = "\n".join(lines[start_row : end_row + 1]).strip()
            if len(code_text) < 100000:
                result.classes.append(
                    ClassDef(
                        name=type_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                )

    # ── Rust extraction ──────────────────────────────────────────

    def _extract_rust_defs(
        self, node, lines: list[str], code_bytes: bytes, result: FileAST, class_map: dict = None
    ):
        """Extract Rust functions, methods, and types (struct/enum/trait)."""
        if class_map is None:
            class_map = {}

        kind = node.kind()

        if kind == "function_item":
            name = self._get_identifier_name(node, code_bytes)
            if name:
                self._add_function(node, lines, name, result)

        elif kind == "impl_item":
            # Extract impl block and its methods
            self._extract_rust_impl(node, lines, code_bytes, result, class_map)

        elif kind in ("struct_item", "enum_item", "trait_item"):
            # Types/classes
            name = self._get_identifier_name(node, code_bytes)
            if name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    # Store in map for later impl blocks
                    class_map[name] = cls

        # Recurse
        for i in range(node.child_count()):
            self._extract_rust_defs(node.child(i), lines, code_bytes, result, class_map)

    def _extract_rust_impl(
        self, node, lines: list[str], code_bytes: bytes, result: FileAST, class_map: dict
    ):
        """Extract methods from Rust impl blocks."""
        # Find impl<T> Type or impl Type
        impl_type = self._get_rust_impl_type(node, code_bytes)

        # Find or create the class/type
        if impl_type and impl_type in class_map:
            cls = class_map[impl_type]
            # Add methods to existing class (recursively find function_items)
            self._collect_impl_methods(node, lines, code_bytes, cls)
        else:
            # No corresponding type definition, add as top-level functions
            self._collect_impl_methods_as_functions(node, lines, code_bytes, result, impl_type)

    def _collect_impl_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Recursively collect function_items from impl block and add as methods."""
        if node.kind() == "function_item":
            method_name = self._get_identifier_name(node, code_bytes)
            if method_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=method_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        for i in range(node.child_count()):
            self._collect_impl_methods(node.child(i), lines, code_bytes, cls)

    def _collect_impl_methods_as_functions(
        self, node, lines: list[str], code_bytes: bytes, result: FileAST, impl_type: Optional[str]
    ):
        """Recursively collect function_items from impl block and add as top-level functions."""
        if node.kind() == "function_item":
            method_name = self._get_identifier_name(node, code_bytes)
            if method_name and impl_type:
                full_name = f"{impl_type}.{method_name}"
                self._add_function(node, lines, full_name, result)

        for i in range(node.child_count()):
            self._collect_impl_methods_as_functions(
                node.child(i), lines, code_bytes, result, impl_type
            )

    def _get_rust_impl_type(self, impl_node, code_bytes: bytes) -> Optional[str]:
        """Extract type name from impl block (e.g., 'MyStruct' from 'impl MyStruct')."""
        for i in range(impl_node.child_count()):
            child = impl_node.child(i)
            kind = child.kind()
            if kind == "type_identifier":
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
            elif kind == "generic_type":
                # For generic types, try to extract the base type name
                for j in range(child.child_count()):
                    sub_child = child.child(j)
                    if sub_child.kind() == "type_identifier":
                        return code_bytes[sub_child.start_byte() : sub_child.end_byte()].decode(
                            "utf-8", "replace"
                        )
        return None

    # ── Java extraction ──────────────────────────────────────────

    def _extract_java_defs(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract Java classes, interfaces, enums, and methods."""
        kind = node.kind()

        if kind in ("class_declaration", "interface_declaration", "enum_declaration"):
            # Extract class and its methods
            class_name = self._get_identifier_name(node, code_bytes)
            if class_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=class_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    # Extract methods
                    self._extract_java_methods(node, lines, code_bytes, cls)

        # Recurse (for nested types)
        for i in range(node.child_count()):
            self._extract_java_defs(node.child(i), lines, code_bytes, result)

    def _extract_java_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Extract methods from Java class."""
        kind = node.kind()

        if kind in ("method_declaration", "constructor_declaration"):
            # For methods: skip modifiers and return type, get the identifier before formal_parameters
            method_name = self._get_java_method_name(node, code_bytes)
            if method_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=method_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            self._extract_java_methods(node.child(i), lines, code_bytes, cls)

    # ── JavaScript/TypeScript extraction ─────────────────────────

    def _extract_js_defs(
        self, node, lines: list[str], code_bytes: bytes, result: FileAST, lang: str
    ):
        """Extract JS/TS functions, classes, methods, arrow functions."""
        kind = node.kind()

        if kind == "function_declaration":
            name = self._get_identifier_name(node, code_bytes)
            if name:
                self._add_function(node, lines, name, result)

        elif kind == "class_declaration":
            class_name = self._get_identifier_name(node, code_bytes)
            if class_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=class_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    # Extract methods
                    self._extract_js_methods(node, lines, code_bytes, cls)

        elif kind == "lexical_declaration":
            # Handle const/let NAME = () => {} or const NAME = function() {}
            func_info = self._extract_js_arrow_or_func_expr(node, lines, code_bytes)
            if func_info:
                name, line_start, line_end, code_text = func_info
                result.functions.append(
                    FunctionDef(
                        name=name,
                        line_start=line_start,
                        line_end=line_end,
                        code=code_text,
                    )
                )

        elif lang in ("typescript", "tsx") and kind in (
            "interface_declaration",
            "type_alias_declaration",
        ):
            # TypeScript-specific types
            type_name = self._get_identifier_name(node, code_bytes)
            if type_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    result.classes.append(
                        ClassDef(
                            name=type_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            self._extract_js_defs(node.child(i), lines, code_bytes, result, lang)

    def _extract_js_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Extract methods from JS/TS class body."""
        kind = node.kind()

        if kind == "method_definition":
            method_name = self._get_identifier_name(node, code_bytes)
            if method_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=method_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            self._extract_js_methods(node.child(i), lines, code_bytes, cls)

    def _extract_js_arrow_or_func_expr(
        self, node, lines: list[str], code_bytes: bytes
    ) -> Optional[tuple[str, int, int, str]]:
        """Extract name and function from 'const NAME = () => {}' or 'const NAME = function() {}'.

        Returns (name, line_start, line_end, code) or None if not a function expression.
        """
        # lexical_declaration has structure: const/let/var NAME = arrow_function/function_expression
        var_declarator = None
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "variable_declarator":
                var_declarator = child
                break

        if not var_declarator:
            return None

        name = None
        func_expr = None

        for i in range(var_declarator.child_count()):
            child = var_declarator.child(i)
            if child.kind() == "identifier" and not name:
                name = code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
            elif child.kind() in ("arrow_function", "function_expression"):
                func_expr = child
                break

        if name and func_expr:
            start_row = var_declarator.start_position().row
            end_row = var_declarator.end_position().row
            code_text = "\n".join(lines[start_row : end_row + 1]).strip()
            if len(code_text) < 100000:
                return (name, start_row + 1, end_row + 1, code_text)

        return None

    # ── C++ extraction ──────────────────────────────────────────

    def _extract_cpp_defs(
        self, node, lines: list[str], code_bytes: bytes, result: FileAST, class_map: dict = None
    ):
        """Extract C++ functions, methods, classes, and structs."""
        if class_map is None:
            class_map = {}

        kind = node.kind()

        if kind == "function_definition":
            name = self._get_cpp_func_name(node, code_bytes)
            if name:
                self._add_function(node, lines, name, result)

        elif kind in ("class_specifier", "struct_specifier"):
            # Extract class/struct and its methods
            class_name = self._get_cpp_class_name(node, code_bytes)
            if class_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=class_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    class_map[class_name] = cls
                    # Extract methods from the class body
                    self._extract_cpp_methods(node, lines, code_bytes, cls)

        elif kind == "namespace_definition":
            # Recurse into namespace
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "declaration_list":
                    for j in range(child.child_count()):
                        self._extract_cpp_defs(child.child(j), lines, code_bytes, result, class_map)

        # Recurse
        for i in range(node.child_count()):
            self._extract_cpp_defs(node.child(i), lines, code_bytes, result, class_map)

    def _get_cpp_func_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract function name from C++ function_definition.

        C++ functions have a declarator that contains the name.
        """
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "function_declarator":
                # function_declarator has the actual declarator inside
                for j in range(child.child_count()):
                    decl_child = child.child(j)
                    if decl_child.kind() in ("identifier", "field_identifier"):
                        return code_bytes[decl_child.start_byte() : decl_child.end_byte()].decode(
                            "utf-8", "replace"
                        )
                    elif decl_child.kind() == "qualified_identifier":
                        # For qualified names like Foo::bar, extract the last part
                        last_id = None
                        for k in range(decl_child.child_count()):
                            qual_child = decl_child.child(k)
                            if qual_child.kind() in ("identifier", "field_identifier"):
                                last_id = code_bytes[
                                    qual_child.start_byte() : qual_child.end_byte()
                                ].decode("utf-8", "replace")
                        if last_id:
                            return last_id
            elif child.kind() == "pointer_declarator":
                # Handle pointer declarators
                return self._get_cpp_func_name_from_declarator(child, code_bytes)
        return None

    def _get_cpp_func_name_from_declarator(self, decl_node, code_bytes: bytes) -> Optional[str]:
        """Extract function name from a declarator (handles pointer_declarator, etc)."""
        for i in range(decl_node.child_count()):
            child = decl_node.child(i)
            if child.kind() == "function_declarator":
                for j in range(child.child_count()):
                    func_child = child.child(j)
                    if func_child.kind() in ("identifier", "field_identifier"):
                        return code_bytes[func_child.start_byte() : func_child.end_byte()].decode(
                            "utf-8", "replace"
                        )
            elif child.kind() in ("identifier", "field_identifier", "qualified_identifier"):
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _get_cpp_class_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract class/struct name from class_specifier or struct_specifier."""
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() in ("type_identifier", "identifier"):
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _extract_cpp_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Extract methods from C++ class/struct body."""
        kind = node.kind()

        if kind == "function_definition":
            method_name = self._get_cpp_func_name(node, code_bytes)
            if method_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=method_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        elif kind == "field_declaration_list":
            # Recurse into field_declaration_list to find functions
            for i in range(node.child_count()):
                self._extract_cpp_methods(node.child(i), lines, code_bytes, cls)
            return

        # Recurse
        for i in range(node.child_count()):
            self._extract_cpp_methods(node.child(i), lines, code_bytes, cls)

    # ── Ruby extraction ─────────────────────────────────────────

    def _extract_ruby_defs(self, node, lines: list[str], code_bytes: bytes, result: FileAST):
        """Extract Ruby methods, classes, and modules."""
        kind = node.kind()

        if kind == "method":
            # Top-level method definition (not inside a class)
            # Check if we're at the top level by looking at parent context
            name = self._get_ruby_method_name(node, code_bytes)
            if name:
                self._add_function(node, lines, name, result)

        elif kind == "class":
            # Extract class and its methods
            class_name = self._get_ruby_class_name(node, code_bytes)
            if class_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=class_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    # Extract methods from the class body
                    self._extract_ruby_methods(node, lines, code_bytes, cls)

        elif kind == "module":
            # Extract module and its methods
            module_name = self._get_ruby_module_name(node, code_bytes)
            if module_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls = ClassDef(
                        name=module_name,
                        line_start=start_row + 1,
                        line_end=end_row + 1,
                        code=code_text,
                    )
                    result.classes.append(cls)
                    # Extract methods from the module
                    self._extract_ruby_methods(node, lines, code_bytes, cls)

        # Recurse
        for i in range(node.child_count()):
            self._extract_ruby_defs(node.child(i), lines, code_bytes, result)

    def _get_ruby_method_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract method name from Ruby method node."""
        # Method node has children: def, identifier, method_parameters, body_statement, end
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "identifier":
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _get_ruby_class_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract class name from Ruby class node."""
        # Class node has: class, constant, body_statement, end
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "constant":
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _get_ruby_module_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract module name from Ruby module node."""
        # Module node has: module, constant, body_statement, end
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "constant":
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _extract_ruby_methods(self, node, lines: list[str], code_bytes: bytes, cls: ClassDef):
        """Extract methods from Ruby class/module body."""
        kind = node.kind()

        if kind == "method":
            method_name = self._get_ruby_method_name(node, code_bytes)
            if method_name:
                start_row = node.start_position().row
                end_row = node.end_position().row
                code_text = "\n".join(lines[start_row : end_row + 1]).strip()
                if len(code_text) < 100000:
                    cls.methods.append(
                        FunctionDef(
                            name=method_name,
                            line_start=start_row + 1,
                            line_end=end_row + 1,
                            code=code_text,
                        )
                    )

        # Recurse
        for i in range(node.child_count()):
            self._extract_ruby_methods(node.child(i), lines, code_bytes, cls)

    # ── Helpers ──────────────────────────────────────────────────

    def _get_identifier_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Get the identifier/name of a node (first identifier-like child)."""
        id_kinds = {"identifier", "type_identifier", "property_identifier", "field_identifier"}
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() in id_kinds:
                return code_bytes[child.start_byte() : child.end_byte()].decode("utf-8", "replace")
        return None

    def _get_java_method_name(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract method name from Java method/constructor declaration.

        Method structure: modifiers? type_identifier identifier formal_parameters ...
        We want the identifier that comes right before formal_parameters.
        """
        formal_params_idx = None
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "formal_parameters":
                formal_params_idx = i
                break

        if formal_params_idx is not None and formal_params_idx > 0:
            # Look backwards from formal_parameters for the first identifier
            for i in range(formal_params_idx - 1, -1, -1):
                child = node.child(i)
                if child.kind() == "identifier":
                    return code_bytes[child.start_byte() : child.end_byte()].decode(
                        "utf-8", "replace"
                    )

        return None

    def _add_function(self, node, lines: list[str], name: str, result: FileAST):
        """Helper to add a function to the result."""
        start_row = node.start_position().row
        end_row = node.end_position().row
        code_text = "\n".join(lines[start_row : end_row + 1]).strip()
        if len(code_text) < 100000:
            result.functions.append(
                FunctionDef(
                    name=name,
                    line_start=start_row + 1,
                    line_end=end_row + 1,
                    code=code_text,
                )
            )

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
        """Extract YAML blocks by kind+name (k8s resources) or top-level keys (plain YAML).

        For each YAML document:
          - If it has a 'kind:' field (k8s/ArgoCD resource), emit one block per resource
            with name '{kind}.{name}' or just '{kind}'.
          - Otherwise (plain YAML like values.yaml), parse with PyYAML and emit one block
            per top-level key. Each block spans from that key's line to the next top-level key.
        """
        docs = code.split("\n---\n")
        start_line = 0

        for doc_text in docs:
            doc_lines = doc_text.split("\n")
            doc_line_count = len(doc_lines)

            # Check for 'kind:' field (k8s/ArgoCD resource)
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

            if kind:
                # K8s/ArgoCD resource: use per-resource naming
                block_name = f"{kind}.{name}" if kind and name else kind
                if len(doc_text.strip()) > 0 and len(doc_text) < 100000:
                    result.functions.append(
                        FunctionDef(
                            name=block_name,
                            line_start=start_line + 1,
                            line_end=start_line + doc_line_count,
                            code=doc_text.strip(),
                        )
                    )
            else:
                # Plain YAML (no 'kind'): try to extract per top-level key
                try:
                    import yaml

                    parsed = yaml.safe_load(doc_text)
                    if isinstance(parsed, dict) and parsed:
                        # Extract blocks for each top-level key
                        self._extract_yaml_top_level_keys(doc_text, doc_lines, start_line, result)
                    else:
                        # Not a dict or empty: fall back to document-level block
                        if len(doc_text.strip()) > 0 and len(doc_text) < 100000:
                            result.functions.append(
                                FunctionDef(
                                    name=doc_lines[0][:30] if doc_lines else "yaml",
                                    line_start=start_line + 1,
                                    line_end=start_line + doc_line_count,
                                    code=doc_text.strip(),
                                )
                            )
                except Exception:
                    # YAML parse error: fall back to document-level block
                    if len(doc_text.strip()) > 0 and len(doc_text) < 100000:
                        result.functions.append(
                            FunctionDef(
                                name=doc_lines[0][:30] if doc_lines else "yaml",
                                line_start=start_line + 1,
                                line_end=start_line + doc_line_count,
                                code=doc_text.strip(),
                            )
                        )

            start_line += doc_line_count + 1

    def _extract_yaml_top_level_keys(
        self, doc_text: str, doc_lines: list[str], doc_start_line: int, result: FileAST
    ):
        """Extract YAML top-level keys as separate FunctionDef blocks.

        For a plain YAML dict (no 'kind'), emits one block per top-level key.
        Each block spans from the key's line to the next top-level key (or end).

        Args:
            doc_text: The YAML document text.
            doc_lines: The document split into lines.
            doc_start_line: The absolute line number where this document starts (in the file).
            result: The FileAST to append blocks to.
        """
        # Find all top-level keys (column 0, no leading whitespace)
        key_positions = []
        for line_idx, line in enumerate(doc_lines):
            if line and not line[0].isspace() and ":" in line:
                key_name = line.split(":")[0].strip()
                if key_name:
                    key_positions.append((line_idx, key_name))

        # For each top-level key, extract the block from that key to the next
        for i, (key_line_idx, key_name) in enumerate(key_positions):
            # Find the end line: either the next top-level key line or the end
            if i + 1 < len(key_positions):
                end_line_idx = key_positions[i + 1][0]
            else:
                end_line_idx = len(doc_lines)

            # Extract code for this key (from key line to just before next key)
            key_block_lines = doc_lines[key_line_idx:end_line_idx]
            key_block_text = "\n".join(key_block_lines).rstrip()

            if len(key_block_text) > 0 and len(key_block_text) < 100000:
                result.functions.append(
                    FunctionDef(
                        name=key_name,
                        line_start=doc_start_line + key_line_idx + 1,
                        line_end=doc_start_line + end_line_idx,
                        code=key_block_text,
                    )
                )

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
