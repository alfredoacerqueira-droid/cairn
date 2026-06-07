"""Shared pytest fixtures for e2e tests (gated: opt-in via CAIRN_E2E=1).

E2E tests are slow and resource-intensive. They are skipped by default unless
explicitly enabled via the CAIRN_E2E env var or -m e2e pytest marker.
This mirrors the corpus test pattern for consistency.
"""

import os

import pytest  # noqa: F401, used in module-level pytestmark

# Skip entire e2e module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    "CAIRN_E2E" not in os.environ,
    reason="E2E tests disabled by default (set CAIRN_E2E=1 to enable, or use -m e2e)",
)
