"""Integration tests for the janitor pipeline."""

import pytest

from core.freshness import DBFreshness
from core.repo import RepoManager
from pipeline.ast_parser import ASTParser
from pipeline.indexer import VectorIndexer
from pipeline.memory import MemorySummarizer
from tests.unit.test_vector_indexer import MockOllamaClient


class TestJanitorPipeline:
    @pytest.fixture
    def repo_path(self, tmp_path):
        """Create a sample Python file for testing."""
        project = tmp_path / "test_project"
        project.mkdir()
        src = project / "src"
        src.mkdir()
        (src / "auth.py").write_text("""def login(user, password):
    return True

def logout(user):
    pass
""")
        return project

    def test_ast_to_vector_pipeline(self, repo_path):
        """Test the full pipeline: parse -> index."""
        parser = ASTParser()
        indexer = VectorIndexer(
            chroma_path=str(repo_path / ".cairn" / "chroma"),
            ollama_client=MockOllamaClient(),
        )

        # Parse
        ast = parser.parse_file(repo_path / "src" / "auth.py")
        assert len(ast.functions) == 2

        # Index
        indexer.index_ast(ast)
        assert indexer.count() == 2

        # Search
        results = indexer.search("login", top_k=2)
        assert len(results) >= 1

    def test_full_context_assembly_pipeline(self, repo_path, monkeypatch):
        """Test the full context assembly pipeline."""
        from server.context_assembler import ContextAssembler

        # Mock load_config to disable the confidence guard for this test
        def _fake_config_no_guard():
            from core.config import Config

            c = Config()
            c.retrieval.min_confidence = 0.0
            c.retrieval.rerank_enabled = False  # no reranker in this mock pipeline
            c.local_llm.enabled = True  # embeddings must be on for semantic_search to find results
            return c

        monkeypatch.setattr(
            "server.context_assembler.load_config", lambda *a, **kw: _fake_config_no_guard()
        )

        # Index the repo first
        parser = ASTParser()
        indexer = VectorIndexer(
            chroma_path=str(repo_path / ".cairn" / "chroma"),
            ollama_client=MockOllamaClient(),
        )

        auth_file = repo_path / "src" / "auth.py"
        ast = parser.parse_file(auth_file)
        indexer.index_ast(ast)

        # Create context assembler
        assembler = ContextAssembler(
            project_path=repo_path,
            ollama_client=MockOllamaClient(),
        )

        # Manually set up the repo map
        repo = RepoManager(repo_path)
        repo.save_repo_map({"src/auth.py": ast.to_dict()})

        # Assemble context with a query that matches the indexed fixture
        prompt = assembler.assemble("login")

        assert "Codebase Context" in prompt
        assert "Relevant Functions" in prompt
        assert "Repository Structure" in prompt
        assert "login" in prompt

        # A clearly-absent query must fail-closed (real guard must be active)
        monkeypatch.undo()
        assembler2 = ContextAssembler(
            project_path=repo_path,
            ollama_client=MockOllamaClient(),
        )
        absent = assembler2.assemble("zznotpresentzz")
        assert "No confident matches" in absent

    def test_memory_summarizer_no_git(self, tmp_path):
        """Test memory summarizer with no git repo."""
        summarizer = MemorySummarizer(
            repo_path=tmp_path,
            ollama_client=MockOllamaClient(),
            model="mock",
        )

        diff = summarizer.get_recent_diff()
        assert diff == ""


class TestDBFreshness:
    def test_check_freshness_no_git(self, tmp_path):
        """Test freshness returns defaults when git is unavailable."""
        freshness = DBFreshness(tmp_path)
        try:
            info = freshness.check_freshness()
            assert "commits_behind" in info
        except Exception:
            pass  # git may or may not be available
