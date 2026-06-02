"""Run all benchmarks — retrieval eval, token reduction, and latency."""

from __future__ import annotations

from pathlib import Path

from benchmarks.datasets import load_demo_queries
from benchmarks.latency import measure_latency
from benchmarks.retrieval_eval import evaluate_retrieval
from benchmarks.token_reduction import file_tree_baseline, whole_repo_baseline


def main() -> None:
    """Run all benchmarks on the current repository."""
    project_path = Path.cwd()

    # Load queries
    try:
        queries = load_demo_queries()
        print(f"Loaded {len(queries)} benchmark queries")
    except Exception as e:
        print(f"SKIPPED (no queries): {e}")
        return

    # Build search function
    try:
        from server.context_assembler import ContextAssembler

        assembler = ContextAssembler(project_path=project_path)

        def search_fn(q: str, k: int) -> list[dict]:
            results = assembler.semantic_search(q, top_k=k)
            if not results:
                return []
            # Add 'id' field in the format expected by retrieval_eval
            for r in results:
                r["id"] = f"{r.get('filepath')}:{r.get('function')}:{r.get('line_start')}"
            return results

    except Exception as e:
        print(f"SKIPPED (no index/Ollama): {e}")
        return

    # Test search once
    try:
        test_results = search_fn("test query", 5)
        print(f"Search function working, found {len(test_results)} results")
    except Exception as e:
        print(f"SKIPPED (search failed): {e}")
        return

    # Retrieval evaluation
    try:
        print("\n=== Retrieval Evaluation ===")
        metrics = evaluate_retrieval(search_fn, queries, k_values=(1, 5, 10))
        for k, v in metrics.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"SKIPPED (retrieval eval): {e}")

    # Token reduction
    try:
        print("\n=== Token Reduction ===")
        if queries:
            q_text = queries[0][0]
            context = assembler.assemble_context(q_text)
            tokens = len(context) // 4  # Approximate token count
            print(f"Assembled context tokens: {tokens}")

            whole_repo = whole_repo_baseline(project_path)
            file_tree = file_tree_baseline(project_path)
            print(f"Baseline whole-repo tokens: {whole_repo}")
            print(f"Baseline file-tree-50 tokens: {file_tree}")

            reduction = round(100 * (1 - tokens / max(whole_repo, 1)), 1)
            print(f"Token reduction: {reduction}%")
    except Exception as e:
        print(f"SKIPPED (token reduction): {e}")

    # Latency
    try:
        print("\n=== Latency ===")
        if queries:
            q_text = queries[0][0]
            result = measure_latency(
                lambda: search_fn(q_text, 5),
                iters=20,
                warmup=3,
            )
            print(f"p50_ms: {result['p50_ms']}")
            print(f"p95_ms: {result['p95_ms']}")
            print(f"p99_ms: {result['p99_ms']}")
            print(f"mean_ms: {result['mean_ms']}")
    except Exception as e:
        print(f"SKIPPED (latency): {e}")


if __name__ == "__main__":
    main()
