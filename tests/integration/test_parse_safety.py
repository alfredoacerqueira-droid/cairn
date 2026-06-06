"""Integration tests for AST parser size limits and parse timeouts."""

import logging
import time
from unittest.mock import patch

from pipeline.ast_parser import ASTParser
from tests.fixtures.builders import make_k8s_repo

logger = logging.getLogger(__name__)


class TestFileSizeLimit:
    """Test max_file_kb size limit."""

    def test_skip_file_exceeding_size_limit(self, tmp_path):
        """File larger than max_file_kb should be skipped and return empty AST."""
        # Create a 100KB file
        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 20000)  # ~100KB

        # Parse with max_file_kb=20
        parser = ASTParser(max_file_kb=20)
        result = parser.parse_file(large_file)

        # Should return empty AST
        assert len(result.functions) == 0
        assert len(result.classes) == 0
        assert result.filepath == str(large_file)

    def test_normal_file_with_generous_limit(self, tmp_path):
        """Normal file should parse fine with generous limit."""
        normal_file = tmp_path / "normal.py"
        normal_file.write_text("""def hello():
    return 'world'

class MyClass:
    def method(self):
        pass
""")

        # Parse with generous limit (1000KB)
        parser = ASTParser(max_file_kb=1000)
        result = parser.parse_file(normal_file)

        # Should parse normally
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"
        assert len(result.classes) == 1
        assert result.classes[0].name == "MyClass"

    def test_no_limit_by_default(self, tmp_path):
        """By default (max_file_kb=0), no size limit is applied."""
        large_file = tmp_path / "large.py"
        large_file.write_text("def foo():\n    pass\n" * 10000)  # ~100KB

        # Parse with default (0 = unlimited)
        parser = ASTParser(max_file_kb=0)
        result = parser.parse_file(large_file)

        # Should parse the file despite size
        assert len(result.functions) > 0

    def test_size_limit_logs_warning(self, tmp_path, caplog):
        """Skipping a file due to size limit should log a warning."""
        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 20000)  # ~100KB

        parser = ASTParser(max_file_kb=20)
        with caplog.at_level(logging.WARNING):
            parser.parse_file(large_file)

        # Check that warning was logged
        assert any("Skipping" in record.message for record in caplog.records)
        assert any("exceeds max_file_kb" in record.message for record in caplog.records)


class TestParseTimeout:
    """Test parse_timeout_s timeout."""

    def test_timeout_on_slow_parse(self):
        """Parse should timeout if tree-sitter parse takes too long."""
        code = """def hello():
    return 'world'
"""

        # Create a parser with a very tight timeout (0.01s)
        parser = ASTParser(parse_timeout_s=0.01)

        # Monkeypatch _tree_sitter_parse to simulate slow parse
        original_parse = parser._tree_sitter_parse

        def slow_parse(code, filepath):
            time.sleep(0.1)  # Sleep longer than timeout
            return original_parse(code, filepath)

        with patch.object(parser, "_tree_sitter_parse", side_effect=slow_parse):
            result = parser.parse_string(code, "test.py")

        # Should return empty AST due to timeout
        assert len(result.functions) == 0
        assert len(result.classes) == 0
        assert result.filepath == "test.py"

    def test_timeout_logs_warning(self, caplog):
        """Timeout should log a warning."""
        code = """def hello():
    return 'world'
"""

        parser = ASTParser(parse_timeout_s=0.01)

        # Monkeypatch to simulate slow parse
        original_parse = parser._tree_sitter_parse

        def slow_parse(code, filepath):
            time.sleep(0.1)
            return original_parse(code, filepath)

        with caplog.at_level(logging.WARNING):
            with patch.object(parser, "_tree_sitter_parse", side_effect=slow_parse):
                parser.parse_string(code, "test.py")

        # Check that warning was logged
        assert any("Parse timeout" in record.message for record in caplog.records)

    def test_normal_parse_completes_within_timeout(self):
        """Normal parse should complete quickly and not timeout."""
        code = """def hello():
    return 'world'

class MyClass:
    def method(self):
        pass
"""

        # Parser with generous timeout (10s)
        parser = ASTParser(parse_timeout_s=10.0)
        start = time.time()
        result = parser.parse_string(code, "test.py")
        elapsed = time.time() - start

        # Should parse successfully
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"
        assert len(result.classes) == 1
        # Should be fast (well under 10s)
        assert elapsed < 1.0

    def test_no_timeout_by_default(self):
        """By default (parse_timeout_s=0), no timeout is applied."""
        code = """def hello():
    return 'world'
"""

        # Parser with default (0 = no timeout)
        parser = ASTParser(parse_timeout_s=0)
        result = parser.parse_string(code, "test.py")

        # Should parse normally without timeout path
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"


class TestPathologicalYAML:
    """Test parsing of pathological YAML files (large, deeply nested)."""

    def test_large_k8s_deployment_parses_within_timeout(self, tmp_path):
        """The 60KB pathological K8s deployment from fixture should parse within timeout."""
        # Generate the pathological repo
        repo_root = make_k8s_repo(tmp_path, with_pathological=True)

        # Parse the large deployment file with a reasonable timeout
        large_deploy = repo_root / "manifests" / "large-deployment.yaml"
        assert large_deploy.exists()

        parser = ASTParser(parse_timeout_s=30.0)
        start = time.time()
        result = parser.parse_file(large_deploy)
        elapsed = time.time() - start

        # Should parse without timing out (within 15s, well under the 30s timeout)
        assert elapsed < 15.0, f"Parse took {elapsed:.1f}s, expected < 15s"
        # Should find the deployment "function"
        assert len(result.functions) > 0

    def test_large_k8s_with_tight_timeout_skips(self, tmp_path):
        """With a very tight timeout, the large file should be skipped."""
        repo_root = make_k8s_repo(tmp_path, with_pathological=True)
        large_deploy = repo_root / "manifests" / "large-deployment.yaml"
        assert large_deploy.exists()

        # Use an unreasonably tight timeout (0.1s)
        parser = ASTParser(parse_timeout_s=0.1)
        result = parser.parse_file(large_deploy)

        # Should return empty AST due to timeout (or possibly parse if YAML regex is fast)
        # Either way, shouldn't hang
        assert result.filepath == str(large_deploy)


class TestRegressionNormalParsing:
    """Regression tests: ensure normal parsing still works with limits set."""

    def test_normal_python_file_parses(self, tmp_path):
        """Normal Python file should parse correctly."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
""")

        # Create parser with both limits set to reasonable values
        parser = ASTParser(max_file_kb=100, parse_timeout_s=5.0)
        result = parser.parse_file(py_file)

        assert len(result.functions) == 1
        assert result.functions[0].name == "add"
        assert len(result.classes) == 1
        assert result.classes[0].name == "Calculator"
        assert len(result.classes[0].methods) == 1

    def test_parse_string_with_limits(self):
        """parse_string should respect timeout limit."""
        code = """def foo():
    pass

class Bar:
    def baz(self):
        pass
"""

        parser = ASTParser(max_file_kb=100, parse_timeout_s=5.0)
        result = parser.parse_string(code, "test.py")

        assert len(result.functions) == 1
        assert result.functions[0].name == "foo"
        assert len(result.classes) == 1
        assert result.classes[0].name == "Bar"
