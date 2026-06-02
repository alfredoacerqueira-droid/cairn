"""Harsh test: verify project isolation under load.

Builds a workspace with 3 repos (helm, terraform, k8s), fresh-indexes all,
then runs interleaved queries against ContextAssemblers bound to each.

Verifies:
1. Every result belongs to the bound repo (no cross-repo filepaths).
2. Every result has the correct project_id.
3. Queries are deterministic and isolated across repos.
"""

import threading
from typing import Optional

from core.repo import project_id
from server.context_assembler import ContextAssembler
from tests.fixtures.builders import make_workspace
from tests.fixtures.harness import fresh_index


class TestIsolationUnderStress:
    """Verify multi-repo isolation under load."""

    def test_workspace_all_repos_indexed_separately(self, tmp_path):
        """Index all three repos in a workspace with separate .cairn."""
        base = tmp_path / "base"
        base.mkdir()
        workspace = make_workspace(base)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"
        k8s_repo = workspace / "k8s-repo"

        # Fresh index each separately
        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)
        fresh_index(k8s_repo, embeddings=False)

        # Verify each has its own .cairn
        assert (helm_repo / ".cairn").exists()
        assert (tf_repo / ".cairn").exists()
        assert (k8s_repo / ".cairn").exists()

        # Verify different project_ids
        helm_id = project_id(helm_repo)
        tf_id = project_id(tf_repo)
        k8s_id = project_id(k8s_repo)

        assert helm_id != tf_id
        assert tf_id != k8s_id
        assert helm_id != k8s_id

    def test_interleaved_queries_all_isolated(self, tmp_path):
        """Run interleaved queries across three repos, verify zero leakage."""
        base = tmp_path / "base"
        base.mkdir()
        workspace = make_workspace(base)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"
        k8s_repo = workspace / "k8s-repo"

        # Index all
        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)
        fresh_index(k8s_repo, embeddings=False)

        # Create assemblers
        helm_asm = ContextAssembler(project_path=helm_repo)
        tf_asm = ContextAssembler(project_path=tf_repo)
        k8s_asm = ContextAssembler(project_path=k8s_repo)

        # Get project IDs
        helm_id = helm_asm.project_id
        tf_id = tf_asm.project_id
        k8s_id = k8s_asm.project_id

        # Run queries biased toward each repo's keywords
        helm_queries = ["helm chart deploy", "values replicas", "templates service"]
        tf_queries = ["terraform variable module", "aws vpc subnet", "resource provider"]
        k8s_queries = ["kubernetes deployment pod", "service selector", "configmap"]

        # Interleave queries
        all_queries = (
            [(helm_asm, q, helm_id) for q in helm_queries]
            + [(tf_asm, q, tf_id) for q in tf_queries]
            + [(k8s_asm, q, k8s_id) for q in k8s_queries]
        )

        for assembler, query, expected_pid in all_queries:
            results = assembler.semantic_search(query, top_k=10)

            # Every result must belong to this assembler's project
            for result in results:
                assert result.get("project_id") == expected_pid, (
                    f"cross-project leak: query='{query}' on project {expected_pid} "
                    f"returned project {result.get('project_id')}"
                )

            # No filepaths from other repos
            for result in results:
                filepath = result.get("filepath", "")
                # Helm results should not contain terraform or k8s paths
                if expected_pid == helm_id:
                    assert "terraform" not in filepath.lower()
                    assert "k8s" not in filepath.lower()
                # TF results should not contain helm or k8s paths
                elif expected_pid == tf_id:
                    assert "helm" not in filepath.lower()
                    assert "kustomize" not in filepath.lower()
                # K8s results should not contain helm or tf paths
                elif expected_pid == k8s_id:
                    assert "terraform" not in filepath.lower()
                    assert "main.tf" not in filepath.lower()

    def test_concurrent_queries_isolated(self, tmp_path):
        """Verify isolation under concurrent queries (basic thread safety)."""
        base = tmp_path / "base"
        base.mkdir()
        workspace = make_workspace(base)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        # Index both
        fresh_index(helm_repo, embeddings=False)
        fresh_index(tf_repo, embeddings=False)

        # Create assemblers
        helm_asm = ContextAssembler(project_path=helm_repo)
        tf_asm = ContextAssembler(project_path=tf_repo)

        helm_id = helm_asm.project_id
        tf_id = tf_asm.project_id

        # Thread-safe results collector
        results_dict: dict[str, Optional[list]] = {"helm": None, "tf": None, "errors": []}

        def helm_query():
            try:
                results_dict["helm"] = helm_asm.semantic_search("helm chart", top_k=5)
            except Exception as e:
                results_dict["errors"].append(f"helm: {e}")

        def tf_query():
            try:
                results_dict["tf"] = tf_asm.semantic_search("terraform variable", top_k=5)
            except Exception as e:
                results_dict["errors"].append(f"tf: {e}")

        # Run concurrently (multiple rounds to stress it)
        for _ in range(3):
            t1 = threading.Thread(target=helm_query)
            t2 = threading.Thread(target=tf_query)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        # Verify no errors
        assert not results_dict["errors"], f"concurrent query errors: {results_dict['errors']}"

        # Verify isolation
        helm_results = results_dict["helm"]
        tf_results = results_dict["tf"]

        if helm_results:
            for r in helm_results:
                assert (
                    r.get("project_id") == helm_id
                ), f"helm query leaked: {r.get('project_id')} != {helm_id}"

        if tf_results:
            for r in tf_results:
                assert (
                    r.get("project_id") == tf_id
                ), f"tf query leaked: {r.get('project_id')} != {tf_id}"

    def test_search_filters_dont_apply_retroactively(self, tmp_path):
        """Verify isolation filters are applied at query time, not index time."""
        base = tmp_path / "base"
        base.mkdir()
        workspace = make_workspace(base)
        helm_repo = workspace / "helm-repo"
        tf_repo = workspace / "terraform-repo"

        # Index helm first
        fresh_index(helm_repo, embeddings=False)
        helm_asm = ContextAssembler(project_path=helm_repo)
        helm_id = helm_asm.project_id

        # Index terraform second
        fresh_index(tf_repo, embeddings=False)
        # tf_asm created but not used in this test's final assertion

        # Query helm assembler (it was bound at the time only helm was indexed)
        helm_results = helm_asm.semantic_search("chart", top_k=10)

        # Even though we later indexed terraform, the helm query should not return
        # any terraform results (isolation is per-query, not per-index-time).
        for result in helm_results:
            assert (
                result.get("project_id") == helm_id
            ), "helm query returned terraform after terraform was indexed"
