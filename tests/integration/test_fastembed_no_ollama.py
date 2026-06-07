"""Integration test: fastembed semantic search works without Ollama.

Proves that fastembed (in-process ONNX) can power the full index+search pipeline
when config.local_llm.enabled=False. Skips cleanly if fastembed is not installed.
"""

import subprocess

import pytest

from core.config import Config, save_config
from core.profiles import get_profile
from pipeline.ast_parser import ASTParser
from server.context_assembler import ContextAssembler

fastembed = pytest.importorskip("fastembed")


def test_fastembed_index_and_search_no_ollama(tmp_path, monkeypatch):
    """Build a small Python repo, index with fastembed (NO Ollama), and search.

    Verifies that:
    - The indexer stores real 384-dim embeddings (bge-small-en-v1.5)
    - ContextAssembler.semantic_search() returns non-empty results with raw_cosine > 0
    - No Ollama client is needed at any point
    """
    repo_path = tmp_path / "fastembed-repo"
    repo_path.mkdir()

    (repo_path / "main.py").write_text(
        '"""Authentication module."""\n'
        "\n"
        "def authenticate_user(token):\n"
        '    """Validate the authentication token."""\n'
        "    if not token or len(token) < 8:\n"
        "        return False\n"
        "    return token == 'super-secret-token-12345'\n"
        "\n"
        "def hash_password(pw):\n"
        '    """Hash a password using sha256."""\n'
        "    import hashlib\n"
        "    return hashlib.sha256(pw.encode()).hexdigest()\n"
        "\n"
        "def compute_checksum(data):\n"
        '    """Compute a checksum for data integrity."""\n'
        "    return sum(ord(c) for c in data) % 65536\n"
    )

    cairn_dir = repo_path / ".cairn"
    cairn_dir.mkdir()

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    (repo_path / ".gitkeep").write_text("")
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    cfg = Config()
    cfg.embeddings_enabled = True
    cfg.local_llm.enabled = False
    cfg.local_llm.embedder = "fastembed"
    cfg.local_llm.fastembed_model = "BAAI/bge-small-en-v1.5"
    cfg.profile = "python"
    cfg.indexing.file_patterns = ["*.py"]
    cfg.indexing.source_roots = ["."]
    cfg.retrieval.rerank_enabled = False
    cfg.retrieval.mode = "embeddings"
    cfg.retrieval.min_confidence = 0.0
    save_config(cfg, repo_path)

    from core.config import clear_config_cache

    clear_config_cache(repo_path)

    assembler = ContextAssembler(project_path=repo_path)

    parser = ASTParser()
    ast = parser.parse_file(repo_path / "main.py")
    assembler.vector_indexer.index_ast(ast)

    assert assembler.vector_indexer.count() >= 1, "Expected at least one indexed block"

    existing = assembler.vector_indexer.collection.get(limit=1, include=["embeddings"])
    stored_emb = None
    try:
        stored_embs = existing["embeddings"]
    except Exception:
        stored_embs = []
    if len(stored_embs) > 0:
        stored_emb = stored_embs[0]
    assert stored_emb is not None, "Expected at least one stored embedding"
    stored_dim = len(stored_emb)
    assert stored_dim == 384, (
        f"Expected bge-small dim=384, got dim={stored_dim} — "
        "embedder may not be wired through VectorIndexer correctly"
    )

    profile = get_profile("python")
    assert profile.embedding_enabled, "python profile should have embedding enabled"

    results = assembler.semantic_search("authenticate user token", top_k=3, apply_guard=True)
    assert len(results) >= 1, (
        f"Expected >=1 results for 'authenticate user token', got {len(results)}. "
        "Semantic search with fastembed embeddings returned empty — "
        "the embeddings leg may not be running."
    )

    raw_cosines = [r.get("raw_cosine", 0.0) for r in results]
    assert any(c > 0.0 for c in raw_cosines), (
        f"Expected raw_cosine > 0.0 in at least one result, got {raw_cosines}. "
        "Embeddings leg did not contribute (no raw_cosine signal)."
    )
