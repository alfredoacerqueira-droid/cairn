"""Unit tests for ripgrep-based lexical retriever with BM25 fallback."""

from unittest.mock import patch

import pytest

from pipeline.ast_parser import ASTParser, ClassDef, FileAST, FunctionDef
from pipeline.retrieval.ripgrep import (
    RipgrepRetriever,
    _extract_search_terms,
    _map_hit_to_function,
)


class TestExtractSearchTerms:
    """Test term extraction from queries."""

    def test_extracts_identifiers(self):
        """Extract identifiers of 3+ chars."""
        terms = _extract_search_terms("find the create_nail function")
        assert "create_nail" in terms
        assert "function" in terms

    def test_drops_stopwords(self):
        """Stopwords are filtered out."""
        terms = _extract_search_terms("how does the function work")
        assert "the" not in terms
        assert "how" not in terms
        assert "does" not in terms
        assert "work" in terms

    def test_drops_short_tokens(self):
        """Tokens < 3 chars are dropped."""
        terms = _extract_search_terms("create an api")
        assert "create" in terms
        assert "an" not in terms  # 2 chars
        assert "api" in terms  # 3 chars, should be included
        # All extracted terms should be 3+ chars
        assert len([t for t in terms if len(t) >= 3]) == len(terms)

    def test_deduplicates(self):
        """Duplicate terms are removed."""
        terms = _extract_search_terms("create create function function")
        assert terms.count("create") == 1
        assert terms.count("function") == 1

    def test_case_insensitive(self):
        """Terms are lowercase."""
        terms = _extract_search_terms("CreateNail FUNCTION")
        assert all(t.islower() for t in terms)

    def test_empty_query_returns_empty(self):
        """Empty query returns no terms."""
        terms = _extract_search_terms("")
        assert terms == []

    def test_query_with_only_stopwords(self):
        """Query with only stopwords returns empty."""
        terms = _extract_search_terms("the how does a to")
        assert terms == []


class TestMapHitToFunction:
    """Test mapping hit lines to enclosing functions."""

    def test_hit_in_top_level_function(self):
        """Hit inside a top-level function is correctly mapped."""
        # Create a synthetic FileAST
        ast = FileAST("test.py")
        func = FunctionDef("my_func", line_start=10, line_end=20, code="def my_func():\n    pass")
        ast.functions.append(func)

        ast_cache = {}
        parser = ASTParser()

        # Mock parser.parse_file to return our synthetic AST
        with patch.object(parser, "parse_file", return_value=ast):
            result = _map_hit_to_function("test.py", 15, ast_cache, parser)

        assert result is not None
        func_id, code = result
        assert func_id == "test.py:my_func:10"
        assert code == "def my_func():\n    pass"

    def test_hit_in_method(self):
        """Hit inside a method is correctly mapped with ClassName.method format."""
        ast = FileAST("test.py")
        cls = ClassDef("MyClass", line_start=5, line_end=25, code="class MyClass:\n    pass")
        method = FunctionDef(
            "my_method", line_start=10, line_end=20, code="    def my_method(self):\n        pass"
        )
        cls.methods.append(method)
        ast.classes.append(cls)

        ast_cache = {}
        parser = ASTParser()

        with patch.object(parser, "parse_file", return_value=ast):
            result = _map_hit_to_function("test.py", 15, ast_cache, parser)

        assert result is not None
        func_id, code = result
        assert func_id == "test.py:MyClass.my_method:10"
        assert code == "    def my_method(self):\n        pass"

    def test_hit_not_in_any_function(self):
        """Hit outside all functions returns None."""
        ast = FileAST("test.py")
        func = FunctionDef("my_func", line_start=10, line_end=20, code="def my_func():\n    pass")
        ast.functions.append(func)

        ast_cache = {}
        parser = ASTParser()

        with patch.object(parser, "parse_file", return_value=ast):
            result = _map_hit_to_function("test.py", 30, ast_cache, parser)

        assert result is None

    def test_caches_parsed_files(self):
        """Parsed files are cached to avoid re-parsing."""
        ast = FileAST("test.py")
        func = FunctionDef("my_func", line_start=10, line_end=20, code="def my_func():\n    pass")
        ast.functions.append(func)

        ast_cache = {}
        parser = ASTParser()

        with patch.object(parser, "parse_file", return_value=ast) as mock_parse:
            _map_hit_to_function("test.py", 15, ast_cache, parser)
            _map_hit_to_function("test.py", 16, ast_cache, parser)

            # parse_file should only be called once
            assert mock_parse.call_count == 1

    def test_parse_error_returns_none(self):
        """Parse error gracefully returns None."""
        ast_cache = {}
        parser = ASTParser()

        with patch.object(parser, "parse_file", side_effect=Exception("Parse error")):
            result = _map_hit_to_function("test.py", 15, ast_cache, parser)

        assert result is None


class TestRipgrepRetrieverAvailable:
    """Test ripgrep availability detection."""

    def test_available_returns_bool(self):
        """available() returns a boolean."""
        result = RipgrepRetriever.available()
        assert isinstance(result, bool)

    def test_available_returns_false_when_not_found(self):
        """available() returns False when rg is not installed."""
        with patch("shutil.which", return_value=None):
            assert RipgrepRetriever.available() is False


class TestRipgrepRetrieverFallback:
    """Test fallback behavior when ripgrep is absent."""

    def test_search_falls_back_to_bm25_when_ripgrep_missing(self):
        """When ripgrep is unavailable, search delegates to BM25."""
        fallback_items = [
            {"id": "test.py:func1:10", "text": "def func1():\n    pass"},
            {"id": "test.py:func2:20", "text": "def func2():\n    order"},
        ]

        retriever = RipgrepRetriever(
            project_path="/tmp/test",
            fallback_items=fallback_items,
        )

        # Mock ripgrep as unavailable
        with patch.object(RipgrepRetriever, "available", return_value=False):
            results = retriever.search("order", top_k=5)

        # Should return BM25 results
        assert len(results) > 0
        assert all(r["source"] == "bm25" for r in results)
        assert all("id" in r and "text" in r and "score" in r for r in results)

    def test_search_with_no_fallback_items_returns_empty(self):
        """Without fallback items, search returns empty when ripgrep unavailable."""
        retriever = RipgrepRetriever(
            project_path="/tmp/test",
            fallback_items=None,
        )

        with patch.object(RipgrepRetriever, "available", return_value=False):
            results = retriever.search("query", top_k=5)

        assert results == []

    def test_search_returns_correct_shape(self):
        """Results have the correct dict shape."""
        fallback_items = [
            {"id": "test.py:func:10", "text": "def func():\n    pass"},
        ]

        retriever = RipgrepRetriever(
            project_path="/tmp/test",
            fallback_items=fallback_items,
        )

        with patch.object(RipgrepRetriever, "available", return_value=False):
            results = retriever.search("func", top_k=5)

        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "source" in r

    def test_search_empty_query_returns_empty(self):
        """Query that extracts no terms returns empty."""
        retriever = RipgrepRetriever(
            project_path="/tmp/test",
            fallback_items=[],
        )

        results = retriever.search("the how does", top_k=5)
        assert results == []


# Skip ripgrep-specific tests if ripgrep is not available (it's not on this machine)
@pytest.mark.skipif(not RipgrepRetriever.available(), reason="ripgrep not installed")
class TestRipgrepRetrieverWithRipgrep:
    """Tests that exercise ripgrep (only run if ripgrep is installed)."""

    def test_search_finds_unique_token(self, tmp_path):
        """Search finds a function containing a unique token."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "def unique_function_xyz():\n    pass\n\n" "def other_function():\n    pass\n"
        )

        retriever = RipgrepRetriever(
            project_path=tmp_path,
            source_roots=["."],
            file_patterns=["*.py"],
        )

        results = retriever.search("unique_function_xyz", top_k=5)

        assert len(results) > 0
        assert any("unique_function_xyz" in r["id"] for r in results)
        assert all(r["source"] == "ripgrep" for r in results)

    def test_search_respects_top_k(self, tmp_path):
        """Search respects top_k limit."""
        # Create multiple functions
        test_file = tmp_path / "test.py"
        content = ""
        for i in range(10):
            content += f"def func{i}():\n    pass\n\n"
        test_file.write_text(content)

        retriever = RipgrepRetriever(
            project_path=tmp_path,
            source_roots=["."],
            file_patterns=["*.py"],
        )

        results = retriever.search("func", top_k=3)

        assert len(results) <= 3

    def test_max_files_bounds_results_to_top_files(self, tmp_path):
        """With max_files=2, results come from at most 2 distinct files."""
        num_files = 10
        for i in range(num_files):
            f = tmp_path / f"mod{i}.py"
            f.write_text(f"def handle_common_thing_{i}():\n    # common_term_xyz\n    pass\n")

        retriever = RipgrepRetriever(
            project_path=tmp_path,
            source_roots=["."],
            file_patterns=["*.py"],
            max_files=2,
            max_count_per_file=50,
        )

        results = retriever.search("common_term_xyz", top_k=20)

        assert len(results) > 0
        distinct_files = set(r["id"].split(":", 1)[0] for r in results)
        assert len(distinct_files) <= 2

    def test_small_repo_search_unchanged_with_bounding(self, tmp_path):
        """Bounding does not change results for a small repo."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def unique_fn_abc():\n    pass\n")

        retriever = RipgrepRetriever(
            project_path=tmp_path,
            source_roots=["."],
            file_patterns=["*.py"],
            max_files=5,
            max_count_per_file=50,
        )

        results = retriever.search("unique_fn_abc", top_k=5)
        assert len(results) > 0
        assert any("unique_fn_abc" in r["id"] for r in results)
        assert all(r["source"] == "ripgrep" for r in results)

    def test_max_files_zero_disables_file_bounding(self, tmp_path):
        """max_files=0 disables file-level bounding."""
        num_files = 10
        for i in range(num_files):
            f = tmp_path / f"mod{i}.py"
            f.write_text(f"def fn_{i}():\n    # common_term_abc\n    pass\n")

        retriever = RipgrepRetriever(
            project_path=tmp_path,
            source_roots=["."],
            file_patterns=["*.py"],
            max_files=0,
            max_count_per_file=50,
        )

        results = retriever.search("common_term_abc", top_k=20)
        assert len(results) > 0
        distinct_files = set(r["id"].split(":", 1)[0] for r in results)
        assert len(distinct_files) >= 3
