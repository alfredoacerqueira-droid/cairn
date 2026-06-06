"""Shared pytest fixtures and isolation guards.

Markers are registered in pyproject.toml ([tool.pytest.ini_options].markers).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _preserve_cwd_and_env():
    """Guard global process state so a leaky test can't pollute the suite.

    Some tests `os.chdir()` into a tmp dir (which pytest later deletes) without
    restoring the original working directory; subsequent tests that resolve a
    project from CWD then fail spuriously. We also snapshot the project-selecting
    env vars so a test that sets them cannot bleed into the next.
    """
    original_cwd = os.getcwd()
    saved_env = {k: os.environ.get(k) for k in ("CAIRN_PROJECT", "GATEWAY_PROJECT")}
    try:
        yield
    finally:
        try:
            os.chdir(original_cwd)
        except OSError:
            pass
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _reset_reranker_singleton():
    """Reset the FlashRank reranker singleton between tests.

    pipeline.retrieval.reranker caches a module-level `_ranker` / `_ranker_failed`.
    Tests that force the reranker to fail (offline-resilience) would otherwise
    leave it disabled for every test that runs afterwards, making the suite
    order-dependent (e.g. router routing then fail-closes spuriously).
    """
    import pipeline.retrieval.reranker as _rr

    before = (_rr._ranker, _rr._ranker_failed)
    _rr._ranker, _rr._ranker_failed = None, False
    try:
        yield
    finally:
        _rr._ranker, _rr._ranker_failed = before
