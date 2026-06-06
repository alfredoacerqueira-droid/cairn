"""Regression test: native index_location collection namespace roundtrip.

Verifies that when index_location='native', a VectorIndexer built the way the
CLI 'reindex' command does (chroma_path from repo.get_chroma_path()) writes to
the SAME namespaced collection that ContextAssembler reads from.

Before the fix, CLI reindex wrote to the un-namespaced 'functions' collection
while ContextAssembler read from 'functions_<project_id>', causing zero search
results (split-brain bug). This test reproduces that scenario and proves the fix.
"""

from core.config import Config, save_config
from core.repo import RepoManager, project_id
from pipeline.ast_parser import ASTParser
from pipeline.indexer import VectorIndexer
from server.context_assembler import ContextAssembler


class TestNativeIndexRoundtrip:
    def test_index_and_search_same_collection_native(self, tmp_path, monkeypatch):
        repo_path = tmp_path / "project"
        repo_path.mkdir()
        (repo_path / "src.py").write_text(
            "def authenticate_user(token):\n"
            '    """Validate the authentication token."""\n'
            "    if not token or len(token) < 8:\n"
            "        return False\n"
            "    return token == 'super-secret-token-12345'\n"
        )

        cairn_dir = repo_path / ".cairn"
        cairn_dir.mkdir()

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        cfg = Config()
        cfg.indexing.index_location = "native"
        cfg.embeddings_enabled = False
        cfg.local_llm.enabled = False
        cfg.indexing.file_patterns = ["*.py"]
        cfg.indexing.source_roots = ["."]
        cfg.retrieval.rerank_enabled = False
        cfg.retrieval.mode = "hybrid"
        save_config(cfg, repo_path)

        pid = project_id(repo_path)
        expected_chroma = fake_home / ".cache" / "cairn" / pid / "chroma"

        repo = RepoManager(repo_path)
        chroma_path = repo.get_chroma_path()
        assert str(chroma_path) == str(expected_chroma), (
            f"chroma_path {chroma_path} != expected {expected_chroma}"
        )

        indexer = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=False,
            project_root=repo_path,
        )

        assert indexer.collection.name == f"functions_{pid}", (
            f"Indexer collection {indexer.collection.name} != functions_{pid}"
        )

        parser = ASTParser()
        ast = parser.parse_file(repo_path / "src.py")
        indexer.index_ast(ast)

        assert indexer.collection.count() >= 1, "Indexer collection should have >=1 entry"

        assembler = ContextAssembler(project_path=repo_path)

        assert assembler.vector_indexer.collection.name == f"functions_{pid}", (
            f"Assembler collection {assembler.vector_indexer.collection.name} != functions_{pid}"
        )

        assert assembler.vector_indexer.collection.name == indexer.collection.name, (
            f"Collection mismatch: indexer={indexer.collection.name}, "
            f"assembler={assembler.vector_indexer.collection.name}"
        )

        assert assembler.vector_indexer.collection.count() >= 1, (
            "Assembler collection should see the same data — got 0 entries "
            "(split-brain: read/write pointing at different collections)"
        )

        results = assembler.semantic_search("authenticate", top_k=5, apply_guard=False)
        assert len(results) >= 1, (
            f"Got {len(results)} results for 'authenticate', expected >=1. "
            f"Indexer wrote to '{indexer.collection.name}' ({indexer.collection.count()} entries), "
            f"Assembler reads from '{assembler.vector_indexer.collection.name}' "
            f"({assembler.vector_indexer.collection.count()} entries). "
            f"chroma_path={chroma_path}"
        )

    def test_collection_naming_parity_with_and_without_project_root(self, tmp_path, monkeypatch):
        repo_path = tmp_path / "project"
        repo_path.mkdir()
        (repo_path / ".cairn").mkdir()

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        cfg = Config()
        cfg.indexing.index_location = "native"
        cfg.embeddings_enabled = False
        cfg.local_llm.enabled = False
        save_config(cfg, repo_path)

        repo = RepoManager(repo_path)
        chroma_path = repo.get_chroma_path()

        indexer_with = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=False,
            project_root=repo_path,
        )

        indexer_without = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=False,
        )

        pid = project_id(repo_path)
        assert indexer_with.collection.name == f"functions_{pid}"
        assert indexer_without.collection.name == "functions"
        assert indexer_with.collection.name != indexer_without.collection.name, (
            "With and without project_root MUST produce different collection names "
            "under native location (project_root=repo → functions_<id>, "
            "no project_root → functions)."
        )
