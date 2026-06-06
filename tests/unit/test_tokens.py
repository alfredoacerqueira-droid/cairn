"""Unit tests for core.tokens — the real tiktoken-based tokenizer."""

from core.tokens import count_tokens, get_encoder, truncate_to_tokens


def test_count_tokens_empty_string():
    """Empty string has zero tokens."""
    assert count_tokens("") == 0


def test_count_tokens_none():
    """None input returns zero tokens."""
    assert count_tokens(None) == 0  # type: ignore


def test_count_tokens_basic():
    """count_tokens returns a positive int for non-empty text."""
    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_monotonic():
    """Token count increases for longer text (sanity check)."""
    short = count_tokens("hello")
    long = count_tokens("hello world this is a much longer string with more content")
    assert long > short


def test_count_tokens_matches_encoder():
    """count_tokens matches direct encoder.encode() result."""
    text = "quick brown fox"
    enc = get_encoder()
    expected = len(enc.encode(text))
    assert count_tokens(text) == expected


def test_truncate_to_tokens_already_fits():
    """If text already fits, return it unchanged."""
    text = "short text"
    result = truncate_to_tokens(text, 100)
    assert result == text


def test_truncate_to_tokens_long_text():
    """Truncate long text to fit in token budget."""
    long_text = "word " * 1000  # ~5000 characters, much more than 10 tokens
    result = truncate_to_tokens(long_text, 10)
    token_count = count_tokens(result)
    assert token_count <= 10


def test_truncate_to_tokens_zero_budget():
    """Zero token budget returns empty string."""
    result = truncate_to_tokens("hello world", 0)
    assert result == ""


def test_truncate_to_tokens_negative_budget():
    """Negative token budget returns empty string."""
    result = truncate_to_tokens("hello world", -1)
    assert result == ""


def test_truncate_to_tokens_empty_input():
    """Empty input returns empty string regardless of budget."""
    assert truncate_to_tokens("", 100) == ""
    assert truncate_to_tokens("", 0) == ""


def test_truncate_to_tokens_none_input():
    """None input with positive budget returns empty string."""
    result = truncate_to_tokens(None, 100)  # type: ignore
    assert result == ""


def test_get_encoder_caching():
    """get_encoder returns cached instance (identity check)."""
    enc1 = get_encoder("claude")
    enc2 = get_encoder("claude")
    assert enc1 is enc2  # same cached object


def test_get_encoder_default_model():
    """get_encoder with no argument uses default 'claude' model."""
    enc_default = get_encoder()
    enc_claude = get_encoder("claude")
    assert enc_default is enc_claude
