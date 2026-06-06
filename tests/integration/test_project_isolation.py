"""Hard multi-repo project isolation tests.

Verifies that:
1. Each repo gets a unique project_id from its resolved absolute path.
2. ChromaDB collections are namespaced by project_id (functions_<id>).
3. Metadata on every indexed record includes project_id and project_root.
4. Queries from one project filter out results from another.
5. MCP server gracefully fails when CAIRN_PROJECT is not set or invalid.
"""

import os
from unittest.mock import patch

from core.repo import project_id
from pipeline.indexer import VectorIndexer
from server.context_assembler import ContextAssembler
from tests.fixtures.builders import make_workspace
from tests.fixtures.harness import fresh_index


class MockOllamaClient:
    """Mock Ollama client for deterministic embeddings."""

    def __init__(self, embed_model: str = "nomic-embed-text"):
        self.embed_model = embed_model

    def embed(self, text: str, model: str = None) -> list[float]:
        # Simple deterministic embedding: just return a fixed vector
        return [0.1] * 16

    def embed_batch(self, texts: list[str], model: str = None) -> list[list[float]]:
        return [[0.1] * 16 for _ in texts]


class TestProjectIDGeneration:
    """Test that project_id is stable and deterministic."""

    def test_project_id_same_for_same_path(self, tmp_path):
        """Same path always yields same project_id."""
        repo = tmp_path / "my-repo"
        repo.mkdir()
        id1 = project_id(repo)
        id2 = project_id(repo)
        assert id1 == id2
        assert len(id1) == 12
        assert all(c in "0123456789abcdef" for c in id1)

    def test_project_id_different_for_different_paths(self, tmp_path):
        """Different paths yield different project_ids."""
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()
        id1 = project_id(repo1)
        id2 = project_id(repo2)
        assert id1 != id2

    def test_project_id_resolves_relative_paths(self, tmp_path):
        """Relative paths resolve to their absolute form."""
        repo = tmp_path / "my-repo"
        repo.mkdir()
        # Change to tmp_path so we can use relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            id_relative = project_id("my-repo")
            id_absolute = project_id(repo)
            assert id_relative == id_absolute
        finally:
            os.chdir(original_cwd)


class TestVectorIndexerProjectIsolation:
    """Test that VectorIndexer respects project_root and namespaces collections."""

    def test_indexer_without_project_root_uses_legacy_collection(self, tmp_path):
        """When project_root is None, uses 'functions' collection (backward compat)."""
        indexer = VectorIndexer(
            chroma_path=tmp_path / "chroma",
            embeddings_enabled=False,
        )
        assert indexer.project_id is None
        assert indexer.project_root is None
        assert indexer.collection.name == "functions"

    def test_indexer_with_project_root_namespaces_collection(self, tmp_path):
        """When project_root is set, collection is named 'functions_<id>'."""
        repo_path = tmp_path / "myrepo"
        repo_path.mkdir()
        indexer = VectorIndexer(
            chroma_path=tmp_path / "chroma",
            embeddings_enabled=False,
            project_root=repo_path,
        )
        expected_id = project_id(repo_path)
        assert indexer.project_id == expected_id
        assert indexer.collection.name == f"functions_{expected_id}"

    def test_indexed_records_have_project_metadata(self, tmp_path):
        """Every indexed record has project_id and project_root metadata."""
        repo_path = tmp_path / "myrepo"
        repo_path.mkdir()
        indexer = VectorIndexer(
            chroma_path=tmp_path / "chroma",
            embeddings_enabled=False,
            project_root=repo_path,
        )

        indexer.index_function(
            filepath="src/auth.py",
            function_name="authenticate",
            code="def authenticate(): pass",
            line_start=10,
            line_end=12,
        )

        data = indexer.collection.get(include=["metadatas"])
        assert len(data["metadatas"]) == 1
        metadata = data["metadatas"][0]
        assert metadata["project_id"] == project_id(repo_path)
        assert metadata["project_root"] == str(repo_path.resolve())

    def test_search_filters_by_project_id(self, tmp_path):
        """Search applies project_id filter at ChromaDB level."""
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()

        # Create two indexers pointing to the SAME chroma_path (simulating collision)
        chroma_path = tmp_path / "shared-chroma"
        mock_ollama = MockOllamaClient()
        indexer1 = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=True,
            project_root=repo1,
            ollama_client=mock_ollama,
        )
        indexer2 = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=True,
            project_root=repo2,
            ollama_client=mock_ollama,
        )

        # Index into repo1
        indexer1.index_function(
            filepath="src/auth.py",
            function_name="authenticate",
            code="def authenticate(user): return user in auth_list",
            line_start=10,
            line_end=12,
        )

        # Index into repo2
        indexer2.index_function(
            filepath="src/deploy.tf",
            function_name="aws_vpc",
            code='resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }',
            line_start=1,
            line_end=3,
        )

        # Search in repo1 with query that matches repo2's content
        results = indexer1.search("aws vpc", top_k=10)

        # All results must be from repo1 (auth.py), never repo2 (deploy.tf)
        assert all(r["filepath"] == "src/auth.py" for r in results)
        assert all(r.get("project_id") == project_id(repo1) for r in results)

    def test_cross_project_record_dropped(self, tmp_path):
        """If a cross-project record slips through, search filters it out."""
        repo1 = tmp_path / "repo1"
        repo1.mkdir()

        chroma_path = tmp_path / "chroma"
        mock_ollama = MockOllamaClient()
        indexer = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=True,
            project_root=repo1,
            ollama_client=mock_ollama,
        )

        # Manually insert a record with a foreign project_id (simulates corruption)
        # Must match embedding dimension (16)
        indexer.collection.upsert(
            ids=["src/bad.py:foreign_func:1"],
            embeddings=[[0.1] * 16],
            metadatas=[
                {
                    "filepath": "src/bad.py",
                    "function": "foreign_func",
                    "line_start": 1,
                    "line_end": 2,
                    "project_id": "foreign_id_12",
                    "project_root": "/other/repo",
                }
            ],
            documents=["def foreign(): pass"],
        )

        # Now index a real record
        indexer.index_function(
            filepath="src/good.py",
            function_name="good_func",
            code="def good(): pass",
            line_start=5,
            line_end=6,
        )

        # Search should only return the real record, not the foreign one
        results = indexer.search("func", top_k=10)

        # All results must have the correct project_id
        assert all(r.get("project_id") == project_id(repo1) for r in results)
        # The foreign record should have been dropped (logged as warning)
        assert all(r["filepath"] != "src/bad.py" for r in results)


class TestContextAssemblerProjectIsolation:
    """Test that ContextAssembler filters results by project_id."""

    def test_context_assembler_has_project_id(self, tmp_path):
        """ContextAssembler stores its project_id on init."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".cairn").mkdir()

        assembler = ContextAssembler(project_path=repo)
        assert assembler.project_id == project_id(repo)

    def test_semantic_search_filters_cross_project(self, tmp_path):
        """ContextAssembler.semantic_search drops cross-project results."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Fresh index the repo
        fresh_index(repo, embeddings=False)

        assembler = ContextAssembler(project_path=repo)

        # Manually insert a cross-project record
        pid = project_id(repo)
        assembler.vector_indexer.collection.upsert(
            ids=["src/bad.py:cross_project:1"],
            embeddings=[[0.0]],
            metadatas=[
                {
                    "filepath": "src/bad.py",
                    "function": "cross_project",
                    "line_start": 1,
                    "line_end": 2,
                    "project_id": "foreign_id_12",  # Wrong project_id
                    "project_root": "/other",
                }
            ],
            documents=["def cross_project(): pass"],
        )

        # Add a real record for this project
        assembler.vector_indexer.index_function(
            filepath="src/good.py",
            function_name="good",
            code="def good(): pass",
            line_start=5,
            line_end=6,
        )

        # Search
        results = assembler.semantic_search("good", top_k=10)

        # All results must be from this project
        assert all(r.get("project_id") == pid for r in results)
        assert all(r["filepath"] != "src/bad.py" for r in results)


class TestMCPServerProjectBinding:
    """Test that MCP server handles missing CAIRN_PROJECT gracefully."""

    def test_mcp_unbound_returns_error(self, tmp_path):
        """When CAIRN_PROJECT is not set, tools return error message."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear CAIRN_PROJECT and GATEWAY_PROJECT
            from server.mcp_server import _resolve_project_path

            path, error = _resolve_project_path()
            assert path is None
            assert error is not None
            assert "Cairn MCP server has no bound project" in error

    def test_mcp_nonexistent_path_returns_error(self, tmp_path):
        """When CAIRN_PROJECT path doesn't exist, tools return error."""
        with patch.dict(os.environ, {"CAIRN_PROJECT": "/nonexistent/repo"}):
            from server.mcp_server import _resolve_project_path

            path, error = _resolve_project_path()
            assert path is None
            assert error is not None
            assert "does not exist" in error

    def test_mcp_unindexed_repo_returns_error(self, tmp_path):
        """When CAIRN_PROJECT has no .cairn/, tools return error."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with patch.dict(os.environ, {"CAIRN_PROJECT": str(repo)}):
            from server.mcp_server import _resolve_project_path

            path, error = _resolve_project_path()
            assert path is None
            assert error is not None
            assert ".cairn/" in error or "not indexed" in error


class TestMultiRepoEndToEnd:
    """End-to-end test: two repos in a workspace, isolated indexing + search."""

    def test_helm_terraform_isolation(self, tmp_path):
        """Index Helm + Terraform in same workspace, verify isolation."""
        # Build workspace with both repos
        workspace = make_workspace(tmp_path)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        # Index both separately (each gets its own .cairn)
        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        # Create assemblers pointing to each
        helm_asm = ContextAssembler(project_path=helm_repo)
        tf_asm = ContextAssembler(project_path=tf_repo)

        # Each project has a unique project_id
        assert helm_asm.project_id != tf_asm.project_id

        # Verify they're using different collections
        helm_pid = helm_asm.project_id
        tf_pid = tf_asm.project_id

        assert helm_asm.vector_indexer.collection.name == f"functions_{helm_pid}"
        assert tf_asm.vector_indexer.collection.name == f"functions_{tf_pid}"
        assert helm_asm.vector_indexer.collection.name != tf_asm.vector_indexer.collection.name

        # The real disaster scenario: an assembler bound to one repo must NEVER
        # return files from the other repo, no matter what we query. Query each
        # assembler with terms biased toward the OTHER repo and assert zero leakage.
        helm_hits = helm_asm.semantic_search("resource variable module terraform", top_k=10)
        assert all(
            h.get("project_id") == helm_pid for h in helm_hits
        ), f"helm assembler leaked foreign project: {[h.get('project_id') for h in helm_hits]}"
        assert all(
            tf_pid not in (h.get("project_id") or "") for h in helm_hits
        ), "helm assembler returned a terraform project_id"

        tf_hits = tf_asm.semantic_search("helm chart values deployment service", top_k=10)
        assert all(
            h.get("project_id") == tf_pid for h in tf_hits
        ), f"terraform assembler leaked foreign project: {[h.get('project_id') for h in tf_hits]}"
        assert all(
            helm_pid not in (h.get("project_id") or "") for h in tf_hits
        ), "terraform assembler returned a helm project_id"
