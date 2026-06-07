"""Unit tests for core.config.embeddings_available()."""

from core.config import Config, embeddings_available


def test_embeddings_available_off():
    """embeddings_enabled=False -> False."""
    cfg = Config()
    cfg.embeddings_enabled = False
    assert embeddings_available(cfg) == (False, None)


def test_embeddings_available_fastembed_no_ollama():
    """embedder='fastembed' returns True even with local_llm.enabled=False."""
    cfg = Config()
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = False
    cfg.local_llm.embedder = "fastembed"
    assert embeddings_available(cfg) == (True, "fastembed")


def test_embeddings_available_ollama_enabled():
    """embedder='ollama' + local_llm.enabled=True -> (True, 'ollama')."""
    cfg = Config()
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = True
    cfg.local_llm.embedder = "ollama"
    assert embeddings_available(cfg) == (True, "ollama")


def test_embeddings_available_embedder_none():
    """embedder='none' -> False regardless of other flags."""
    cfg = Config()
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = True
    cfg.local_llm.embedder = "none"
    assert embeddings_available(cfg) == (False, None)


def test_embeddings_available_ollama_disabled():
    """embedder='ollama' but local_llm.enabled=False -> False."""
    cfg = Config()
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = False
    cfg.local_llm.embedder = "ollama"
    assert embeddings_available(cfg) == (False, None)
