"""Real token counting wrapper using tiktoken cl100k_base encoding.

This module provides the single authoritative tokenizer for Cairn. Uses tiktoken's
cl100k_base encoding as a conservative proxy for Claude/Sonnet models (the official
Claude tokenizer is private). The cl100k_base encoding slightly under-counts code,
so callers should keep a safety margin when budgeting tokens.

Encoders are cached per model name, so repeated calls to get_encoder() do not rebuild
the encoding object.
"""

from __future__ import annotations

import functools

_DEFAULT_MODEL = "claude"


@functools.lru_cache(maxsize=4)
def get_encoder(model: str = "claude"):
    """Return a tiktoken Encoding for the given model.

    Uses cl100k_base as the proxy encoding for all Claude/Sonnet models.
    The encoding object is cached per model name, so repeated calls with the
    same model return the exact same cached instance.

    Args:
        model: Model name. Currently all models map to cl100k_base. The
            parameter is retained to allow refinement of the mapping later.

    Returns:
        A tiktoken.Encoding instance for token encoding/decoding.
    """
    import tiktoken

    # All model names map to cl100k_base for now (Claude proxy). Keep the param
    # so callers can pass a model and we can refine the mapping later.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str | None, model: str = "claude") -> int:
    """Return the exact token count of `text` under the proxy tokenizer.

    Empty strings and None return 0.

    Args:
        text: Text to count tokens in. None is treated as empty.
        model: Model name (currently all models use cl100k_base proxy).

    Returns:
        Non-negative integer token count.
    """
    if not text:
        return 0
    return len(get_encoder(model).encode(text))


def truncate_to_tokens(
    text: str | None, max_tokens: int, model: str = "claude"
) -> str:
    """Truncate `text` to fit within `max_tokens` tokens.

    Encodes `text`, truncates the token sequence to at most `max_tokens` tokens,
    and decodes back to text. If the text already fits, returns it unchanged.
    If max_tokens <= 0, returns empty string. None input returns "".

    Args:
        text: Text to truncate. None is treated as empty string.
        max_tokens: Maximum token budget. If <= 0, returns "".
        model: Model name (currently all models use cl100k_base proxy).

    Returns:
        Truncated text, guaranteed to encode to <= max_tokens tokens.
    """
    if not text or max_tokens <= 0:
        return ""
    enc = get_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])
