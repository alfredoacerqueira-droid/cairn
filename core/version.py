"""Single source of truth for runtime version + index schema.

CAIRN_VERSION must match the ``version`` field in pyproject.toml (kept in sync
manually — pyproject is build metadata and can't be imported at build time).

INDEX_SCHEMA_VERSION is bumped whenever the on-disk index format or the parsing
that produced it changes in a way that makes an existing index stale. The number
is written into ``.cairn/index_meta.json`` at index build time; a mismatch
on load means the user should reindex.

History:
  1 — initial schema.
  2 — BOM-safe name extraction + embeddings-off placeholder vectors + root-module
      indexing. Indexes built before this are missing/garbled blocks → reindex.
"""

CAIRN_VERSION = "0.6.0"
INDEX_SCHEMA_VERSION = 2
