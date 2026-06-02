"""Prove cold vs. warm index determinism (no warm-up dependency).

This test verifies that indexing is NOT dependent on warm-up state:
1. Build a synthetic Terraform/Helm repo
2. Fresh-index it once (cold)
3. Capture retrieval results for fixed queries
4. Re-index N times and verify results remain identical
5. No embeddings, no Ollama, fully hermetic and offline

This catches regressions where:
- Random initialization affects indexing
- Cache state pollutes results
- Commit-dependent logic breaks determinism
"""

import pytest

from pipeline.retrieval.structural import StructuralRetriever
from tests.fixtures import fresh_index, make_terraform_repo


class TestColdVsWarmIndexDeterminism:
    """Prove retrieval results are identical across cold + warm index runs."""

    @pytest.mark.integration
    def test_terraform_index_determinism_cold_vs_warm(self, tmp_path):
        """Build Terraform repo, index cold, verify N re-indexes give same results.

        Strategy:
        1. Create a Terraform repo with structural patterns (resource types,
           variable names, etc.)
        2. Run fresh_index once (cold)
        3. Use StructuralRetriever (no embeddings) on three fixed queries
        4. Capture the ordered list of (id, score) hits
        5. Re-index fresh N=3 times
        6. Assert each re-index produces identical hit ordering
        """
        # Setup: build Terraform repo
        repo_path = make_terraform_repo(tmp_path)

        # Cold index: initial fresh_index
        fresh_index(repo_path, embeddings=False)

        # Retrieve index data and setup structural retriever
        repo_manager = __import__("core.repo", fromlist=["RepoManager"]).RepoManager(repo_path)
        indexer = __import__("pipeline.indexer", fromlist=["VectorIndexer"]).VectorIndexer(
            chroma_path=repo_manager.get_chroma_path(),
            embeddings_enabled=False,
        )

        # Get indexed data
        data = indexer.collection.get(include=["metadatas", "documents"])
        items = [{"id": i, "text": t} for i, t in zip(data["ids"], data["documents"])]

        # Setup structural retriever (no embeddings)
        structural = StructuralRetriever()
        structural.index(items)

        # Define test queries (structural, lexical patterns)
        queries = [
            "vpc cidr aws_vpc",  # Should match aws_vpc resource
            "subnet availability_zone",  # Should match aws_subnet + availability_zone var
            "internet gateway igw",  # Should match aws_internet_gateway
        ]

        # Cold results: capture ordered hits for each query
        cold_results = {}
        for query in queries:
            hits = structural.search(query, top_k=5)
            cold_results[query] = [(h["id"], h["score"]) for h in hits]

        # Verify we got some hits for each query
        for query in queries:
            assert cold_results[query], f"No hits for query '{query}' (cold index broken?)"

        # Warm tests: re-index N times and verify results are identical
        num_reindexes = 3
        for iteration in range(num_reindexes):
            # Re-index
            fresh_index(repo_path, embeddings=False)

            # Re-retrieve index data
            indexer = __import__("pipeline.indexer", fromlist=["VectorIndexer"]).VectorIndexer(
                chroma_path=repo_manager.get_chroma_path(),
                embeddings_enabled=False,
            )
            data = indexer.collection.get(include=["metadatas", "documents"])
            items = [{"id": i, "text": t} for i, t in zip(data["ids"], data["documents"])]

            # Re-run retriever
            structural = StructuralRetriever()
            structural.index(items)

            # Verify results match cold for each query
            for query in queries:
                warm_hits = structural.search(query, top_k=5)
                warm_results = [(h["id"], h["score"]) for h in warm_hits]

                assert warm_results == cold_results[query], (
                    f"Iteration {iteration + 1}: Query '{query}' results changed.\n"
                    f"  Cold:  {cold_results[query]}\n"
                    f"  Warm:  {warm_results}"
                )

    @pytest.mark.integration
    def test_helm_index_determinism_structure_and_metadata(self, tmp_path):
        """Verify Helm repo indexing captures consistent structure across re-indexes.

        This test ensures that even for YAML-based repos (where parser behavior
        might be sensitive to whitespace/ordering), the indexed results are
        deterministic.
        """
        from tests.fixtures import make_helm_repo

        repo_path = make_helm_repo(tmp_path)

        # Cold index
        fresh_index(repo_path, embeddings=False)

        repo_manager = __import__("core.repo", fromlist=["RepoManager"]).RepoManager(repo_path)
        indexer = __import__("pipeline.indexer", fromlist=["VectorIndexer"]).VectorIndexer(
            chroma_path=repo_manager.get_chroma_path(),
            embeddings_enabled=False,
        )

        cold_data = indexer.collection.get(include=["metadatas", "documents"])
        cold_ids = set(cold_data["ids"])
        cold_names = {m["function"] for m in cold_data["metadatas"]}

        # Warm re-indexes
        for iteration in range(2):
            fresh_index(repo_path, embeddings=False)

            indexer = __import__("pipeline.indexer", fromlist=["VectorIndexer"]).VectorIndexer(
                chroma_path=repo_manager.get_chroma_path(),
                embeddings_enabled=False,
            )
            warm_data = indexer.collection.get(include=["metadatas", "documents"])
            warm_ids = set(warm_data["ids"])
            warm_names = {m["function"] for m in warm_data["metadatas"]}

            # Verify IDs and function names are identical
            assert warm_ids == cold_ids, (
                f"Iteration {iteration + 1}: IDs changed.\n"
                f"  Cold:  {cold_ids}\n"
                f"  Warm:  {warm_ids}"
            )
            assert warm_names == cold_names, (
                f"Iteration {iteration + 1}: Function names changed.\n"
                f"  Cold:  {cold_names}\n"
                f"  Warm:  {warm_names}"
            )
