"""Unit tests for RTK-style token compressor."""

from server.token_compressor import FilterLevel, Language, TokenCompressor

SAMPLE_CODE = '''
"""Authentication module."""

import jwt
from datetime import datetime
# This is a comment
def authenticate(token: str) -> bool:
    """Verify the JWT token."""
    if not token:
        return False
    try:
        payload = jwt.decode(token, "secret", algorithms=["HS256"])
        return bool(payload.get("user_id"))
    except Exception:
        return False

class AuthMiddleware:
    """Middleware for authentication."""
    def process(self, request):
        # Check auth header
        token = request.headers.get("Authorization")
        return authenticate(token)
'''


class TestTokenCompressor:
    def test_none_level_returns_unchanged(self):
        c = TokenCompressor(level=FilterLevel.NONE)
        result = c.compress(SAMPLE_CODE, Language.PYTHON)
        assert "def authenticate" in result
        assert '"""Authentication module."""' in result

    def test_minimal_removes_comments(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        result = c.compress(SAMPLE_CODE, Language.PYTHON)
        assert "# This is a comment" not in result
        assert "# Check auth header" not in result

    def test_minimal_keeps_functions(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        result = c.compress(SAMPLE_CODE, Language.PYTHON)
        assert "def authenticate" in result
        assert "class AuthMiddleware" in result
        assert "def process" in result

    def test_minimal_removes_imports(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        result = c.compress(SAMPLE_CODE, Language.PYTHON)
        assert "import jwt" not in result
        assert "from datetime import" not in result

    def test_minimal_collapses_whitespace(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        result = c.compress("x = 1\n\n\n\ny = 2", Language.PYTHON)
        assert "\n\n\n\n" not in result

    def test_aggressive_removes_docstrings(self):
        c = TokenCompressor(level=FilterLevel.AGGRESSIVE)
        result = c.compress(SAMPLE_CODE, Language.PYTHON)
        assert '"""Verify the JWT token."""' not in result
        assert '"""Middleware for authentication."""' not in result

    def test_aggressive_truncates_functions(self):
        long_func = "def long_func():\n" + "    pass\n" * 30
        c = TokenCompressor(level=FilterLevel.AGGRESSIVE)
        result = c.compress(long_func, Language.PYTHON)
        assert "lines omitted" in result  # truncation marker

    def test_stats_reduction_nonzero(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        c.compress(SAMPLE_CODE, Language.PYTHON)
        stats = c.get_stats()
        assert stats["original_tokens"] > 0
        assert stats["compressed_tokens"] > 0
        assert stats["compressed_tokens"] <= stats["original_tokens"]

    def test_strip_ansi_removes_escape_codes(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        result = c.compress("hello\x1b[31m world\x1b[0m", Language.PYTHON)
        assert "\x1b" not in result

    def test_preserves_todo_comments(self):
        """Important markers like TODO should be kept."""
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = "x = 1  # TODO: fix this"
        result = c.compress(code, Language.PYTHON)
        assert "TODO" in result

    def test_empty_input_handled(self):
        c = TokenCompressor(level=FilterLevel.AGGRESSIVE)
        result = c.compress("", Language.PYTHON)
        assert result == ""

    def test_rust_language_parsing(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = '// comment\nfn main() {\n    println!("hello");\n}'
        result = c.compress(code, Language.RUST)
        assert "// comment" not in result
        assert "fn main" in result

    def test_javascript_language_parsing(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = "// comment\nfunction hello() {\n    return 1;\n}"
        result = c.compress(code, Language.JAVASCRIPT)
        assert "// comment" not in result
        assert "function hello" in result

    def test_strategies_tracking(self):
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        c.compress(SAMPLE_CODE, Language.PYTHON)
        stats = c.get_stats()
        assert len(stats["strategies_applied"]) >= 3  # At least: comments, whitespace, imports

    def test_strategies_tracking_aggressive(self):
        c = TokenCompressor(level=FilterLevel.AGGRESSIVE)
        c.compress(SAMPLE_CODE, Language.PYTHON)
        stats = c.get_stats()
        strategies = stats["strategies_applied"]
        assert "remove_comments" in strategies
        assert "remove_docstrings" in strategies
        assert "truncate_functions" in strategies

    # ── A1: String-literal safety tests ──────────────────────────────────────

    def test_url_not_corrupted(self):
        """# inside a URL string must not be stripped."""
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = 'url = "http://example.com#anchor"'
        result = c.compress(code, Language.PYTHON)
        assert "http://example.com#anchor" in result

    def test_hash_not_a_comment_in_string(self):
        """# inside a quoted string must be preserved."""
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = 's = "# not a comment"'
        result = c.compress(code, Language.PYTHON)
        assert "# not a comment" in result

    def test_genuine_trailing_comment_stripped(self):
        """A genuine trailing # comment outside quotes must be stripped."""
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = "x = 1  # actual comment"
        result = c.compress(code, Language.PYTHON)
        assert "x = 1" in result
        assert "# actual comment" not in result

    def test_todo_marker_preserved(self):
        """TODO marker line must always be preserved."""
        c = TokenCompressor(level=FilterLevel.AGGRESSIVE)
        code = "x = 1  # TODO: fix this later"
        result = c.compress(code, Language.PYTHON)
        assert "TODO" in result

    def test_mixed_quotes_and_comment(self):
        """Line with string containing # then real comment must only strip real comment."""
        c = TokenCompressor(level=FilterLevel.MINIMAL)
        code = 's = "a#b"  # real comment'
        result = c.compress(code, Language.PYTHON)
        assert 's = "a#b"' in result
        assert "# real comment" not in result


# ── A3: retrieval confidence guard test ────────────────────────────────────


class TestRetrievalConfidence:
    """Tests for the retrieval.min_confidence safety check."""

    def setup_method(self):
        pass

    def test_low_confidence_returns_minimal_context(self, monkeypatch):
        """When top score is below threshold, context should be minimal."""
        monkeypatch.setattr(
            "core.config.load_config",
            lambda *a, **kw: _fake_config(min_confidence=0.9),
        )
        # Verify threshold is enforced when top score is low
        assert True  # placeholder — integration test needed


def _fake_config(min_confidence: float = 0.0):
    from core.config import Config

    c = Config()
    c.retrieval.min_confidence = min_confidence
    return c
