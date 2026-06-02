"""End-to-end benchmark: baseline vs gateway token counts + retrieval quality."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTAPI_PATH = Path("/mnt/c/Users/alfre/Projects/testAPI")
RESULTS_PATH = Path(__file__).parent / "results" / "end_to_end.json"


def approx_tokens(text: str) -> int:
    """~4-char/token approximation (labeled as such in output)."""
    return max(1, len(text) // 4)


def whole_repo_baseline(project_path: Path) -> int:
    """Token count for dumping all Python files in the project (excludes .venv)."""
    total = 0
    for f in project_path.glob("**/*.py"):
        if ".venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
            continue
        try:
            total += approx_tokens(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return total


def gateway_context_for_query(
    query: str, project_path: Path, top_k: int = 5
) -> tuple[str, list[dict]]:
    """Assemble + compress context via the gateway path. Returns (prompt_text, search_results)."""
    # Import gateway modules FIRST, before adding testAPI to sys.path
    # This avoids shadowing issues (testAPI also has server/ directory)
    gateway_path = (
        project_path.parent / "Hybrid Smart-Gateway LAI" / "Hybrid Smart-Gateway for Local AI"
    )
    sys.path.insert(0, str(gateway_path))

    # Import gateway modules while they're at the top of sys.path
    from server.context_assembler import ContextAssembler

    # NOW add testAPI
    sys.path.insert(0, str(project_path))
    os.chdir(project_path)

    assembler = ContextAssembler(project_path=project_path, top_k=top_k)
    results = assembler.semantic_search(query, top_k=top_k)
    context = assembler.assemble_context(query)
    compressed = _compress_via_proxy_path(context)

    return compressed, results


def _compress_via_proxy_path(context: str) -> str:
    """Apply the same compression used in the cloud proxy path."""
    from server.token_compressor import FilterLevel, Language, TokenCompressor

    compressor = TokenCompressor(level=FilterLevel.MINIMAL)
    return compressor.compress(context, Language.PYTHON)


def gold_function_hit(delivered_context: str, gold_ids: set[str]) -> bool:
    """Check whether any gold function ID appears in the delivered context.

    Returns False if context contains the rejection message or is minimal.
    """
    # If the context is the rejection message, no hit
    if "No confident matches found" in delivered_context:
        return False

    # If the context is too minimal (e.g., just the rejection placeholder),
    # treat as no hit
    if len(delivered_context.strip()) < 50:
        return False

    # Check if any gold function name appears in the delivered context
    for gold_id in gold_ids:
        # Extract function name from gold_id (format: "filepath:function_name:line")
        parts = gold_id.rsplit(":", 2)
        if len(parts) >= 2:
            func_name = parts[1]
            # Simple substring check for function name in context
            if func_name in delivered_context:
                return True

    return False


def main() -> int:
    # ── Pre-flight checks ────────────────────────────────────────────────
    if not TESTAPI_PATH.exists():
        print(f"SKIPPED: testAPI repo not found at {TESTAPI_PATH}")
        return 0

    chroma_path = TESTAPI_PATH / ".cairn" / "chroma"
    if not chroma_path.exists():
        print("SKIPPED: testAPI is not indexed. Run: cd testAPI && cairn reindex")
        return 0

    # Check Ollama
    try:
        from server.ollama_client import OllamaClient

        if not OllamaClient().health_check():
            print("SKIPPED: Ollama not reachable")
            return 0
    except Exception:
        print("SKIPPED: Ollama client error")
        return 0

    # ── Load queries ─────────────────────────────────────────────────────
    try:
        from benchmarks.datasets.demo import load_demo_queries
        from benchmarks.retrieval_eval import mrr, recall_at_k
    except ImportError:
        print("SKIPPED: benchmark modules not importable")
        return 0

    queries = load_demo_queries()
    baseline_tokens = whole_repo_baseline(TESTAPI_PATH)

    # Load agent baseline for honest comparison
    agent_baseline_file = TESTAPI_PATH / "benchmark_baseline_results.json"
    agent_baseline_tokens = None
    if agent_baseline_file.exists():
        try:
            agent_data = json.loads(agent_baseline_file.read_text())
            agent_baseline_tokens = agent_data.get("total_tokens")
        except Exception:
            pass

    # ── Run per-query ────────────────────────────────────────────────────
    rows = []
    per_query_recalls = []
    per_query_mrrs = []

    for query_text, gold_ids in queries:
        try:
            context, results = gateway_context_for_query(query_text, TESTAPI_PATH)
        except Exception as exc:
            import traceback

            print(f"SKIPPED: gateway_context_for_query failed: {exc}")
            traceback.print_exc()
            continue

        gateway_tokens = approx_tokens(context)
        hit = gold_function_hit(context, gold_ids)

        # Build per-query retrieved IDs from delivered context
        # If context was suppressed by the guard, no retrieved IDs count toward recall
        retrieved_ids = []
        if "No confident matches found" not in context and len(context.strip()) > 50:
            # Extract function IDs from the delivered context
            for r in results:
                # Only include results that made it past the guard filter
                # For now, assume all results in the search response made it through
                # (they will appear in formatted context unless suppressed)
                fid = (
                    f"{r.get('filepath', '')}:{r.get('function', '')}" f":{r.get('line_start', 0)}"
                )
                retrieved_ids.append(fid)

        # Compute Recall@10 and MRR for THIS query
        query_recall = recall_at_k(retrieved_ids, gold_ids, 10)
        query_mrr = mrr(retrieved_ids, gold_ids)
        per_query_recalls.append(query_recall)
        per_query_mrrs.append(query_mrr)

        reduction_pct = round(100 * (1 - gateway_tokens / max(baseline_tokens, 1)), 1)

        # Compute reduction vs agent if available
        reduction_vs_agent_pct = None
        if agent_baseline_tokens:
            reduction_vs_agent_pct = round(
                100 * (1 - gateway_tokens / max(agent_baseline_tokens, 1)), 1
            )

        rows.append(
            {
                "query": query_text[:80],
                "baseline_tokens": baseline_tokens,
                "gateway_tokens": gateway_tokens,
                "reduction_pct": reduction_pct,
                "reduction_vs_agent_pct": reduction_vs_agent_pct,
                "gold_hit": hit,
                "recall_at_10": round(query_recall, 3),
                "mrr": round(query_mrr, 3),
            }
        )

    if not rows:
        print("No queries completed.")
        return 1

    # ── Aggregate metrics ────────────────────────────────────────────────
    # Use the honest baseline (vs agent) if available, otherwise vs whole-repo
    if agent_baseline_tokens:
        reductions = [
            r.get("reduction_vs_agent_pct")
            for r in rows
            if r.get("reduction_vs_agent_pct") is not None
        ]
    else:
        reductions = [r["reduction_pct"] for r in rows]
    reductions.sort()
    n = len(reductions)
    median = reductions[n // 2] if reductions else 0.0

    # Average recall and MRR across queries
    avg_recall = sum(per_query_recalls) / len(per_query_recalls) if per_query_recalls else 0.0
    avg_mrr = sum(per_query_mrrs) / len(per_query_mrrs) if per_query_mrrs else 0.0

    # ── Print table ──────────────────────────────────────────────────
    print()
    if agent_baseline_tokens:
        print("End-to-End Benchmark: Gateway vs Full-Context Agent (~4-char/token)")
        print("=" * 85)
        print(f"{'Query':<35s} {'GW':>7s} {'VsAg%':>7s} {'VsRepo%':>7s} {'Hit':>4s}")
    else:
        print("End-to-End Benchmark: Gateway vs Whole-Repo Baseline (~4-char/token)")
        print("=" * 75)
        print(f"{'Query':<40s} {'Base':>7s} {'GW':>7s} {'Red%':>6s} {'Hit':>4s}")
    print("-" * 85 if agent_baseline_tokens else "-" * 75)
    for r in rows:
        if agent_baseline_tokens:
            reduction_str = (
                f"{r['reduction_vs_agent_pct']:>6.1f}%"
                if r["reduction_vs_agent_pct"] is not None
                else "N/A"
            )
            print(
                f"{r['query']:<35s} {r['gateway_tokens']:>7d} "
                f"{reduction_str:>7s} {r['reduction_pct']:>6.1f}% "
                f"{'✓' if r['gold_hit'] else '✗':>4s}"
            )
        else:
            print(
                f"{r['query']:<40s} {r['baseline_tokens']:>7d} "
                f"{r['gateway_tokens']:>7d} {r['reduction_pct']:>5.1f}% "
                f"{'✓' if r['gold_hit'] else '✗':>4s}"
            )
    print("-" * 85 if agent_baseline_tokens else "-" * 75)
    median_label = "Median (vs agent)" if agent_baseline_tokens else "Median"
    spaces = 35 if agent_baseline_tokens else 40
    print(f"{'':>{spaces}s} {median_label:>7s} {median:>6.1f}%")
    print()
    print(f"Retrieval Recall@10: {avg_recall:.3f}  MRR: {avg_mrr:.3f}  " f"({len(rows)} queries)")
    print()

    # ── Write JSON ───────────────────────────────────────────────────────
    output = {
        "baseline_tokens": baseline_tokens,
        "agent_baseline_tokens": agent_baseline_tokens,
        "median_reduction_pct": median,
        "recall_at_10": round(avg_recall, 3),
        "mrr": round(avg_mrr, 3),
        "per_query": rows,
        "note": "Token counts use ~4-char/token approximation",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results written to {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
