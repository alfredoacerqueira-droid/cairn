"""Cold-index harness for hermetic test setup.

Provides utilities to:
1. Delete an existing .cairn index
2. Run init + reindex programmatically (no CLI wrapper)
3. Return the indexed project path
4. Re-run indexing to verify determinism

All operations run offline (embeddings_enabled=False) with no Ollama/network.
"""

import shutil
from pathlib import Path

from core.config import Config, save_config
from core.freshness import DBFreshness
from core.repo import RepoManager, collect_source_files, detect_source_layout
from pipeline.ast_parser import ASTParser
from pipeline.indexer import VectorIndexer


class _ExplodingOllama:
    """Stand-in OllamaClient that fails if embeddings are attempted.

    Used in tests to ensure the iac profile (or any embeddings_enabled=False
    profile) never touches embeddings.
    """

    embed_model = "should-never-be-called"

    def embed(self, *a, **k):  # pragma: no cover
        raise AssertionError("embeddings disabled, but embed() was called")

    def embed_batch(self, *a, **k):  # pragma: no cover
        raise AssertionError("embeddings disabled, but embed_batch() was called")


def fresh_index(repo_path: Path, *, embeddings: bool = False) -> Path:
    """Run a cold index on the repo (delete .cairn, init, reindex).

    This mimics the CLI's init + reindex flow:
    1. Detect layout (source roots + file patterns)
    2. Delete existing .cairn if present
    3. Write fresh config.yaml
    4. Collect source files
    5. Parse and index via VectorIndexer
    6. Mark freshness

    All indexing runs with embeddings_enabled=False (no Ollama/network).

    Args:
        repo_path: Path to an initialized git repo with source files.
        embeddings: If True, enable embeddings (requires Ollama). Default False.

    Returns:
        The repo_path, now with a fresh .cairn/chroma index.
    """
    repo_path = Path(repo_path)

    # Step 1: Detect layout
    detected_roots, detected_patterns = detect_source_layout(repo_path)

    # Step 2: Delete existing .cairn
    cairn_dir = repo_path / ".cairn"
    if cairn_dir.exists():
        shutil.rmtree(cairn_dir)

    # Step 3: Write fresh config with detected values
    cfg = Config()
    cfg.indexing.source_roots = detected_roots
    cfg.indexing.file_patterns = detected_patterns
    cfg.embeddings_enabled = embeddings
    save_config(cfg, repo_path)

    # Step 4: Collect source files
    files = collect_source_files(
        repo_path,
        cfg.indexing.file_patterns,
        cfg.indexing.exclude_patterns,
        cfg.indexing.source_roots,
    )

    # Step 5: Index via VectorIndexer (offline, no embeddings)
    repo = RepoManager(repo_path)
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(),
        ollama_client=_ExplodingOllama() if not embeddings else None,
        embeddings_enabled=embeddings,
        project_root=repo_path,
    )

    parser = ASTParser()
    for filepath in files:
        try:
            ast = parser.parse_file(filepath)
            indexer.index_ast(ast)
        except Exception:
            # Silently skip unparseable files (like non-.py terraform, etc.)
            pass

    # Step 6: Mark freshness
    freshness = DBFreshness(
        repo_path,
        quick_threshold=cfg.stale_db.quick_reindex_threshold,
        full_threshold=cfg.stale_db.full_reindex_threshold,
    )
    freshness.mark_indexed(freshness.get_current_commit())
    repo.write_index_meta()

    return repo_path


def reindex_fresh(repo_path: Path) -> Path:
    """Wipe .cairn and re-run fresh_index.

    Useful for testing idempotency: index, capture results, re-index,
    verify results are identical.

    Args:
        repo_path: Path to the repo.

    Returns:
        The repo_path with a freshly re-indexed .cairn.
    """
    return fresh_index(repo_path, embeddings=False)
