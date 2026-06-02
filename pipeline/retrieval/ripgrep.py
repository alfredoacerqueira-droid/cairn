"""Ripgrep-based lexical retriever with BM25 fallback.

Uses ripgrep (rg) for fast exact-match search against live working tree.
Maps hits to enclosing functions via AST parsing. Falls back to in-memory
BM25 when ripgrep is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from pipeline.ast_parser import ASTParser
from pipeline.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)

# Common stopwords to skip (generic, low-signal terms)
STOPWORDS = {
    "the",
    "how",
    "does",
    "a",
    "to",
    "of",
    "in",
    "is",
    "are",
    "be",
    "and",
    "or",
    "not",
    "but",
    "if",
    "else",
    "for",
    "while",
    "do",
    "as",
    "by",
    "with",
    "from",
    "on",
    "at",
    "this",
    "that",
    "which",
    "when",
    "where",
    "what",
    "why",
    "who",
    "can",
    "should",
    "would",
    "could",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}


def _extract_search_terms(query: str) -> list[str]:
    """Extract identifiers/words from query, drop stopwords and short tokens.

    Returns deduplicated, lowercase terms of 3+ chars, excluding stopwords.
    """
    # Extract [A-Za-z_][A-Za-z0-9_]{2,} — identifiers with 3+ chars
    pattern = r"[A-Za-z_][A-Za-z0-9_]{2,}"
    matches = re.findall(pattern, query.lower())

    # Deduplicate and filter out stopwords
    terms = []
    seen = set()
    for term in matches:
        if term not in seen and term not in STOPWORDS:
            terms.append(term)
            seen.add(term)

    return terms


def _map_hit_to_function(
    filepath: str,
    hit_line: int,
    ast_cache: dict[str, Any],
    parser: ASTParser,
) -> Optional[tuple[str, str]]:
    """Map a hit line to the enclosing function.

    Args:
        filepath: Path to the file
        hit_line: Line number of the hit (1-indexed)
        ast_cache: Dict mapping filepath -> FileAST
        parser: ASTParser instance

    Returns:
        Tuple of (function_id, function_code) if found, else None.
        ID format: "filepath:function_name:line_start" or
        "filepath:ClassName.method_name:line_start" for methods.
    """
    # Parse file if not in cache
    if filepath not in ast_cache:
        try:
            ast_cache[filepath] = parser.parse_file(filepath)
        except Exception:
            logger.debug(f"Failed to parse {filepath}")
            return None

    ast = ast_cache[filepath]

    # Search for enclosing function
    # Check top-level functions first
    for func in ast.functions:
        if func.line_start <= hit_line <= func.line_end:
            func_id = f"{filepath}:{func.name}:{func.line_start}"
            return (func_id, func.code)

    # Check methods in classes
    for cls in ast.classes:
        for method in cls.methods:
            if method.line_start <= hit_line <= method.line_end:
                func_id = f"{filepath}:{cls.name}.{method.name}:{method.line_start}"
                return (func_id, method.code)

    return None


class RipgrepRetriever:
    """Lexical retriever using ripgrep with BM25 fallback."""

    def __init__(
        self,
        project_path: Path | str,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        source_roots: list[str] | None = None,
        fallback_items: list[dict[str, Any]] | None = None,
    ):
        """Initialize ripgrep retriever.

        Args:
            project_path: Root of the project to search
            file_patterns: File patterns to include (e.g., ["*.py"])
            exclude_patterns: Patterns to exclude
            source_roots: Subdirs within project to search (e.g., ["src"])
            fallback_items: List of dicts with 'id' and 'text' for BM25 fallback
        """
        self.project_path = Path(project_path)
        self.file_patterns = file_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or []
        self.source_roots = source_roots or ["."]
        self.fallback_items = fallback_items or []

        # Lazy-initialized BM25 fallback
        self._bm25: Optional[BM25Retriever] = None

    @staticmethod
    def available() -> bool:
        """Check if ripgrep is installed and available."""
        return shutil.which("rg") is not None

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search for relevant functions.

        Extracts search terms from query, uses ripgrep to find matches in live
        tree, maps hits to enclosing functions, scores by match count, returns
        top_k. Falls back to BM25 if ripgrep unavailable.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with keys: id, text, score, source
        """
        # Extract and validate search terms
        terms = _extract_search_terms(query)
        if not terms:
            return []

        # Try ripgrep first if available
        if self.available():
            return self._search_ripgrep(terms, top_k)

        # Fall back to BM25
        return self._search_bm25(query, top_k)

    def _search_ripgrep(self, terms: list[str], top_k: int) -> list[dict[str, Any]]:
        """Search via ripgrep.

        Args:
            terms: List of search terms (already extracted/cleaned)
            top_k: Number of results to return

        Returns:
            List of result dicts
        """
        if not terms:
            return []

        try:
            # Build search paths from source_roots
            search_paths = []
            for root in self.source_roots:
                root_path = self.project_path / root
                if root_path.exists():
                    search_paths.append(str(root_path))

            if not search_paths:
                logger.debug("No search paths found, falling back to BM25")
                return self._search_bm25("", top_k)

            # Build ripgrep command: rg --json -i -e term1 -e term2 ... paths
            cmd = ["rg", "--json", "-i"]  # -i for case-insensitive
            for term in terms:
                cmd.extend(["-e", term])
            cmd.extend(search_paths)

            # Run ripgrep with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10.0,
            )

            # Parse JSON lines output
            # Each match line has type="match" with data.path.text and data.line_number
            hit_map: dict[tuple[str, int], int] = defaultdict(int)
            # Map: (filepath, line_number) -> count

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") != "match":
                        continue

                    data = event.get("data", {})
                    filepath = data.get("path", {}).get("text", "")
                    line_number = data.get("line_number", 0)

                    if filepath and line_number:
                        # Normalize filepath to relative
                        try:
                            rel_path = str(Path(filepath).relative_to(self.project_path))
                        except ValueError:
                            rel_path = filepath

                        hit_map[(rel_path, line_number)] += 1
                except json.JSONDecodeError:
                    continue

            # Map hits to functions and aggregate scores
            ast_cache: dict[str, Any] = {}
            parser = ASTParser()

            function_scores: dict[str, tuple[str, int]] = {}
            # Map: function_id -> (code, score)

            for (filepath, line_number), count in hit_map.items():
                result_tuple = _map_hit_to_function(filepath, line_number, ast_cache, parser)
                if result_tuple:
                    func_id, func_code = result_tuple
                    if func_id in function_scores:
                        _, existing_score = function_scores[func_id]
                        function_scores[func_id] = (func_code, existing_score + count)
                    else:
                        function_scores[func_id] = (func_code, count)

            # Sort by score descending and return top_k
            ranked = sorted(
                function_scores.items(),
                key=lambda x: x[1][1],
                reverse=True,
            )

            results: list[dict[str, Any]] = []
            for func_id, (code, score) in ranked[:top_k]:
                results.append(
                    {
                        "id": func_id,
                        "text": code,
                        "score": float(score),
                        "source": "ripgrep",
                    }
                )

            return results

        except subprocess.TimeoutExpired:
            logger.warning("Ripgrep search timed out")
            return self._search_bm25("", top_k)
        except Exception as e:
            logger.warning(f"Ripgrep search error: {e}")
            return self._search_bm25("", top_k)

    def _search_bm25(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fall back to BM25 search.

        Args:
            query: Search query (used if we have fallback_items)
            top_k: Number of results

        Returns:
            List of result dicts with source="bm25"
        """
        if not self.fallback_items:
            return []

        # Lazily build BM25 index from fallback items
        if self._bm25 is None:
            self._bm25 = BM25Retriever()
            self._bm25.index(self.fallback_items)

        return self._bm25.search(query, top_k=top_k)
