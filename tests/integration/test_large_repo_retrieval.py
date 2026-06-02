"""Regression: retrieval must not die on large repos (ChromaDB SQL-variable limit).

A real bug found by the hard-test campaign: on a repo with tens of thousands of
indexed blocks, ContextAssembler._load_function_texts() did one unbounded
collection.get(include=["metadatas", ...]), which makes Chroma bind one SQL
variable per row and raises SQLite's "too many SQL variables" error. The except
clause then silently returned nothing -> retrieval returned 0 results on every
big repo. The loader must page through instead.
"""

from __future__ import annotations

from server.context_assembler import ContextAssembler
from tests.fixtures.builders import make_python_repo
from tests.fixtures.harness import fresh_index


def test_load_function_texts_pages_and_survives_unbounded_get_failure(tmp_path, monkeypatch):
    repo = make_python_repo(tmp_path)
    fresh_index(repo, embeddings=False)

    asm = ContextAssembler(project_path=repo)
    collection = asm.vector_indexer.collection
    real_get = collection.get

    # Simulate the SQLite "too many SQL variables" failure: any get WITHOUT a
    # bounded `limit` (the old unbounded path) blows up; paginated gets succeed.
    def guarded_get(*args, **kwargs):
        if kwargs.get("limit") is None:
            raise RuntimeError("too many SQL variables")
        return real_get(*args, **kwargs)

    monkeypatch.setattr(collection, "get", guarded_get)

    bm25_items, ast_items = asm._load_function_texts()

    # If the loader still did a single unbounded get it would have caught the
    # error and returned []. Pagination means we still get the repo's blocks.
    assert bm25_items, "retrieval loader returned nothing — unbounded get regression"
    assert len(bm25_items) == collection.count()
