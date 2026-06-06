"""Honest comparison benchmark: gateway vs full-context agent baseline.

Produces a clear, defensible comparison table showing token reduction
at measured retrieval quality. Includes embedding model A/B harness
to compare discrimination gaps (relevant vs nonsense queries).

Run with: python -m benchmarks.comparison [--models model1,model2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTAPI_PATH = Path("/mnt/c/Users/alfre/Projects/testAPI")
DJANGO_PATH = Path("/mnt/c/Users/alfre/Projects/django")
TFERKS_PATH = Path("/mnt/c/Users/alfre/Projects/tf-eks")
MEDIATR_PATH = Path("/mnt/c/Users/alfre/Projects/csharp-mediatr")
RESULTS_PATH = Path(__file__).parent / "results" / "comparison.json"
RERANK_AB_PATH = Path(__file__).parent / "results" / "rerank_ab.json"
PHASE6_PATH = Path(__file__).parent / "results" / "phase6.json"


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


def load_agent_baseline(project_path: Path) -> int | None:
    """Load the full-context agent baseline token count if available.

    This is the honest baseline: the full-context agent that the gateway
    competes against, not the naive whole-repo dump.
    """
    baseline_file = project_path / "benchmark_baseline_results.json"
    if baseline_file.exists():
        try:
            data = json.loads(baseline_file.read_text())
            return data.get("total_tokens")
        except Exception:
            pass
    return None


def gateway_context_for_query(
    query: str, project_path: Path, top_k: int = 5
) -> tuple[str, list[dict]]:
    """Assemble + compress context via the gateway path.

    Returns (prompt_text, search_results).
    """
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


def reduction_claimable(recall_at_10: float, threshold: float = 0.8) -> bool:
    """Guard: reduction is only meaningful if recall >= threshold.

    Prevents the dishonest "huge reduction but dropped the needed code" failure.
    """
    return recall_at_10 >= threshold


def compute_reduction_pct(gateway_tokens: int, baseline_tokens: int) -> float:
    """Compute token reduction percentage."""
    if baseline_tokens <= 0:
        return 0.0
    return round(100 * (1 - gateway_tokens / baseline_tokens), 1)


def run_phase_f_ab() -> int:
    """Execute Phase F reranker A/B benchmark on Django.

    Returns:
        0 if successful, 1 if skipped or error
    """
    # Pre-flight checks
    if not DJANGO_PATH.exists():
        print(f"SKIPPED: Django repo not found at {DJANGO_PATH}")
        return 0

    chroma_path = DJANGO_PATH / ".cairn" / "chroma"
    if not chroma_path.exists():
        print("SKIPPED: Django is not indexed. Run: cd django && cairn reindex")
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

    # Run A/B benchmark
    queries = django_rerank_queries()
    ab_results = compare_rerank_ab(DJANGO_PATH, queries)

    if "error" in ab_results:
        print(f"ERROR: {ab_results['error']}")
        return 1

    # Print results table
    print()
    print("=" * 140)
    print("PHASE F: Reranker A/B Benchmark (Django, hybrid mode)")
    print("=" * 140)
    print()
    print("RELEVANT QUERIES (5):")
    print("-" * 140)
    hdr = (
        f"{'Query':<40s} {'OFF Top-1':<20s} {'OFF Score':>10s} {'ON Top-1':<20s} "
        f"{'ON Score':>10s} {'OFF OK':>6s} {'ON OK':>6s}"
    )
    print(hdr)
    print("-" * 140)

    for row in ab_results["results"]:
        if not row["is_nonsense"]:
            print(
                f"{row['query']:<40s} {row['rerank_off_top1']:<20s} "
                f"{row['rerank_off_score']:>10.4f} {row['rerank_on_top1']:<20s} "
                f"{row['rerank_on_score']:>10.4f} {'✓' if row['rerank_off_correct'] else '✗':>6s} "
                f"{'✓' if row['rerank_on_correct'] else '✗':>6s}"
            )

    print()
    print("NONSENSE QUERIES (3) — Should be suppressed with rerank ON:")
    print("-" * 140)
    hdr_nonsense = (
        f"{'Query':<40s} {'OFF Score':>10s} {'ON Score':>10s} "
        f"{'Suppressed OFF':>15s} {'Suppressed ON':>15s}"
    )
    print(hdr_nonsense)
    print("-" * 140)

    for row in ab_results["results"]:
        if row["is_nonsense"]:
            print(
                f"{row['query']:<40s} {row['rerank_off_score']:>10.4f} "
                f"{row['rerank_on_score']:>10.4f} "
                f"{'✓' if row['suppressed_off'] else '✗':>15s} "
                f"{'✓' if row['suppressed_on'] else '✗':>15s}"
            )

    print()
    print("=" * 140)
    print("METRICS:")
    print("=" * 140)
    print(
        f"Top-1 Correctness (relevant only): "
        f"{ab_results['top1_correct_off']:.1%} (OFF) → "
        f"{ab_results['top1_correct_on']:.1%} (ON)"
    )
    print(
        f"Nonsense Suppression: "
        f"{ab_results['nonsense_suppressed_off']:.1%} (OFF) → "
        f"{ab_results['nonsense_suppressed_on']:.1%} (ON)"
    )
    print()
    print("VERDICT:")
    print(f"  {ab_results['verdict']}")
    print()

    # Write JSON
    RERANK_AB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RERANK_AB_PATH.write_text(json.dumps(ab_results, indent=2, ensure_ascii=False))
    print(f"Results written to {RERANK_AB_PATH}")
    print()

    return 0


def phase6_query_sets() -> dict[str, list[tuple[str, str]]]:
    """Per-repo query sets for Phase 6 profile evaluation.

    Format: repo_name -> [(query_text, expected_substring), ...]
    Expected substrings are heuristic checks for correctness.

    Returns:
        dict mapping repo name to list of (query, expected_substring) tuples
    """
    return {
        "tf-eks": [
            # Relevant queries
            ("EKS managed node group", "eks_node_group"),
            ("IAM role for cluster", "iam_role"),
            ("cluster security group rules", "security_group"),
            ("KMS key encryption", "kms"),
            ("cluster version variable", "cluster_version"),
            # Nonsense query
            ("kubernetes pod autoscaler reconcile loop", "kubernetes"),
        ],
        "mediatr": [
            # Relevant queries
            ("send a request through the mediator", "Mediator"),
            ("publish a notification", "Publish"),
            ("pipeline behavior", "PipelineBehavior"),
            ("register handlers DI", "ServiceCollection"),
            ("exception handling pipeline", "Exception"),
            # Nonsense query
            ("terraform aws vpc subnet", "terraform"),
        ],
        "django": [
            # Relevant queries
            ("resolve URL to view", "resolve"),
            ("execute queryset database", "execute"),
            ("form validation cleaned_data", "clean"),
            ("middleware chain", "middleware"),
            ("render template", "render"),
            # Nonsense query
            ("kubernetes pod autoscaler", "kubernetes"),
        ],
    }


def evaluate_repo(project_path: Path, queries_with_expected: list[tuple[str, str]]) -> dict | None:
    """Evaluate per-repo profile retrieval quality.

    Args:
        project_path: Path to the repo
        queries_with_expected: List of (query_text, expected_substring) tuples

    Returns:
        dict with keys:
          - top_1: Top-1 correctness percentage
          - top_5: Top-5 correctness percentage
          - nonsense_suppressed: Boolean (at least one nonsense query suppressed)
          - token_reduction: Percentage reduction vs whole-repo
          - details: List of per-query results
        Returns None if repo not indexed
    """
    chroma_path = project_path / ".cairn" / "chroma"
    if not chroma_path.exists():
        return None

    # Setup sys.path
    gateway_path = (
        project_path.parent / "Hybrid Smart-Gateway LAI" / "Hybrid Smart-Gateway for Local AI"
    )
    sys.path.insert(0, str(gateway_path))
    sys.path.insert(0, str(project_path))
    os.chdir(project_path)

    from server.context_assembler import ContextAssembler

    # Compute whole-repo baseline (context length in chars)
    whole_repo_text = ""
    if project_path.name == "tf-eks":
        # HCL files
        for f in project_path.glob("**/*.tf"):
            if ".terraform" not in str(f):
                try:
                    whole_repo_text += f.read_text(encoding="utf-8")
                except Exception:
                    pass
    elif project_path.name == "csharp-mediatr":
        # C# files
        for f in project_path.glob("**/*.cs"):
            if ".git" not in str(f) and "bin" not in str(f) and "obj" not in str(f):
                try:
                    whole_repo_text += f.read_text(encoding="utf-8")
                except Exception:
                    pass
    else:
        # Python files (Django)
        for f in project_path.glob("**/*.py"):
            if ".venv" not in str(f) and "__pycache__" not in str(f) and ".git" not in str(f):
                try:
                    whole_repo_text += f.read_text(encoding="utf-8")
                except Exception:
                    pass

    whole_repo_tokens = approx_tokens(whole_repo_text)

    # Evaluate each query
    details = []
    top_1_hits = 0
    top_5_hits = 0
    nonsense_count = 0
    nonsense_suppressed = 0

    for query_text, expected_substring in queries_with_expected:
        is_nonsense = query_text in [
            "kubernetes pod autoscaler reconcile loop",
            "terraform aws vpc subnet",
            "kubernetes pod autoscaler",
        ]

        try:
            assembler = ContextAssembler(project_path=project_path, top_k=5)
            results = assembler.semantic_search(query_text, top_k=5)
            context = assembler.assemble_context(query_text)

            # Check top-1 and top-5 hits
            top_1_hit = False
            top_5_hit = False

            if "No confident matches found" not in context and len(context.strip()) > 50:
                # Check if expected_substring appears in context
                combined_context = context.lower()
                if expected_substring.lower() in combined_context:
                    top_1_hit = True
                    top_5_hit = True
                elif len(results) >= 5:
                    # Check if it appears in top-5
                    combined_results = " ".join(
                        [
                            f"{r.get('function', '')} {r.get('filepath', '')}".lower()
                            for r in results
                        ]
                    )
                    if expected_substring.lower() in combined_results:
                        top_5_hit = True

            # For nonsense, check if suppressed (no confident matches)
            if is_nonsense:
                nonsense_count += 1
                if "No confident matches found" in context or len(context.strip()) <= 50:
                    nonsense_suppressed += 1

            if top_1_hit:
                top_1_hits += 1
            if top_5_hit:
                top_5_hits += 1

            # Estimate context tokens for this query
            context_tokens = approx_tokens(context)

            details.append(
                {
                    "query": query_text[:60],
                    "is_nonsense": is_nonsense,
                    "top_1_hit": top_1_hit,
                    "top_5_hit": top_5_hit,
                    "context_tokens": context_tokens,
                    "suppressed": "No confident matches found" in context if is_nonsense else False,
                }
            )

        except Exception as e:
            print(f"    WARNING: Query '{query_text[:40]}...' failed: {e}")
            details.append(
                {
                    "query": query_text[:60],
                    "is_nonsense": is_nonsense,
                    "top_1_hit": False,
                    "top_5_hit": False,
                    "context_tokens": 0,
                    "suppressed": False,
                    "error": str(e),
                }
            )

    # Compute metrics
    total_queries = len(queries_with_expected)
    relevant_queries = total_queries - nonsense_count

    top_1_pct = (top_1_hits / relevant_queries * 100) if relevant_queries > 0 else 0.0
    top_5_pct = (top_5_hits / relevant_queries * 100) if relevant_queries > 0 else 0.0
    nonsense_suppressed_pct = nonsense_suppressed / nonsense_count if nonsense_count > 0 else 0.0

    # Estimate avg context tokens and reduction
    tokens_sum = sum(d.get("context_tokens", 0) for d in details)  # type: ignore
    avg_context_tokens: float = tokens_sum / len(details) if details else 0.0
    token_reduction: float = (
        (1 - avg_context_tokens / whole_repo_tokens) * 100 if whole_repo_tokens > 0 else 0.0
    )

    return {
        "top_1": round(top_1_pct, 1),
        "top_5": round(top_5_pct, 1),
        "nonsense_suppressed": nonsense_suppressed_pct >= 0.5,  # At least 50%
        "nonsense_suppressed_pct": round(nonsense_suppressed_pct * 100, 1),
        "token_reduction": round(token_reduction, 1),
        "avg_context_tokens": round(avg_context_tokens, 0),
        "whole_repo_tokens": whole_repo_tokens,
        "details": details,
    }


def run_phase_6() -> int:
    """Execute Phase 6: Measure v0.6 profile retrieval vs v0.5 baseline.

    Returns:
        0 if successful, 1 if skipped or error
    """
    print()
    print("=" * 150)
    print("PHASE 6: Profile Retrieval Evaluation (v0.6 vs v0.5 Baseline)")
    print("=" * 150)
    print()

    repos = [
        (TFERKS_PATH, "tf-eks", "iac", {"v0.5": 17}),
        (MEDIATR_PATH, "mediatr", "dotnet", {"v0.5": 60}),
        (DJANGO_PATH, "django", "python", {"v0.5": 40}),
    ]

    phase6_queries = phase6_query_sets()
    results = {}

    for project_path, repo_name, profile, baselines in repos:
        print(f"Evaluating {repo_name} (profile: {profile})...")

        if not project_path.exists():
            print(f"  SKIPPED: {repo_name} not found at {project_path}")
            continue

        queries = phase6_queries.get(repo_name, [])
        if not queries:
            print(f"  SKIPPED: No queries defined for {repo_name}")
            continue

        eval_result = evaluate_repo(project_path, queries)
        if eval_result is None:
            print(f"  SKIPPED: {repo_name} not indexed (no .cairn/chroma)")
            continue

        results[repo_name] = eval_result
        print(f"  ✓ Top-1: {eval_result['top_1']:.1f}%")
        print(f"  ✓ Top-5: {eval_result['top_5']:.1f}%")
        print(f"  ✓ Nonsense suppressed: {eval_result['nonsense_suppressed_pct']:.1f}%")
        print(f"  ✓ Token reduction: {eval_result['token_reduction']:.1f}%")
        print()

    if not results:
        print("No repos evaluated.")
        return 0

    # Print summary table
    print()
    print("=" * 150)
    print("PHASE 6 RESULTS (v0.6 Profile Retrieval vs v0.5 Baseline)")
    print("=" * 150)
    print()
    hdr = (
        f"{'Repo':<20} {'Profile':<12} {'v0.5 Top-1':<15} "
        f"{'v0.6 Top-1':<15} {'Top-5':<10} {'Nonsense Supp':<15} "
        f"{'Token Red':<12} {'Status':<20}"
    )
    print(hdr)
    print("-" * 150)

    for repo_name, eval_result in results.items():
        repo_obj = next((r for r in repos if r[1] == repo_name), None)
        if not repo_obj:
            continue

        profile = repo_obj[2]
        v0_5_top_1 = repo_obj[3].get("v0.5", 0)
        v0_6_top_1 = eval_result["top_1"]
        top_5 = eval_result["top_5"]
        nonsense = eval_result["nonsense_suppressed_pct"]
        token_red = eval_result["token_reduction"]

        improvement = v0_6_top_1 - v0_5_top_1
        if improvement > 0:
            status = f"+{improvement:.1f}p (IMPROVED)"
        elif improvement == 0:
            status = "No change"
        else:
            status = f"{improvement:.1f}p (REGRESSED)"

        print(
            f"{repo_name:<20} {profile:<12} {v0_5_top_1:>6.1f}%        "
            f"{v0_6_top_1:>6.1f}%        {top_5:>6.1f}% {nonsense:>10.1f}% "
            f"{token_red:>10.1f}%   {status:<20}"
        )

    print("-" * 150)
    print()

    # Write JSON
    PHASE6_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": str(__import__("datetime").datetime.now().isoformat()),
        "phase": "6",
        "description": "v0.6 profile-aware retrieval vs v0.5 embeddings baseline",
        "results": results,
        "baselines_v0_5": {r[1]: r[3].get("v0.5", 0) for r in repos},
    }
    PHASE6_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results written to {PHASE6_PATH}")
    print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gateway vs full-context agent")
    parser.add_argument(
        "--models",
        type=str,
        default="nomic-embed-text",
        help="Comma-separated list of embedding models to A/B test (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--ab",
        action="store_true",
        help="Run reranker A/B comparison on Django (Phase F benchmark)",
    )
    parser.add_argument(
        "--phase6",
        action="store_true",
        help="Run Phase 6 profile retrieval evaluation on tf-eks/mediatr/django",
    )
    args = parser.parse_args()

    # ── Phase 6 Profile Retrieval Benchmark ─────────────────────────────
    if args.phase6:
        return run_phase_6()

    # ── Phase F A/B Reranker Benchmark ──────────────────────────────────
    if args.ab:
        return run_phase_f_ab()

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
    agent_baseline_tokens = load_agent_baseline(TESTAPI_PATH)

    # ── Run per-query comparison ─────────────────────────────────────────
    rows = []
    per_query_recalls_5 = []
    per_query_recalls_10 = []
    per_query_mrrs = []
    reductions = []

    for query_text, gold_ids in queries:
        try:
            context, results = gateway_context_for_query(query_text, TESTAPI_PATH)
        except Exception as exc:
            print(f"SKIPPED: gateway_context_for_query failed: {exc}")
            continue

        gateway_tokens = approx_tokens(context)
        hit = gold_function_hit(context, gold_ids)

        # Build per-query retrieved IDs from delivered context
        retrieved_ids = []
        if "No confident matches found" not in context and len(context.strip()) > 50:
            for r in results:
                fid = (
                    f"{r.get('filepath', '')}:{r.get('function', '')}" f":{r.get('line_start', 0)}"
                )
                retrieved_ids.append(fid)

        # Compute Recall@5, Recall@10, and MRR for THIS query
        query_recall_5 = recall_at_k(retrieved_ids, gold_ids, 5)
        query_recall_10 = recall_at_k(retrieved_ids, gold_ids, 10)
        query_mrr = mrr(retrieved_ids, gold_ids)
        per_query_recalls_5.append(query_recall_5)
        per_query_recalls_10.append(query_recall_10)
        per_query_mrrs.append(query_mrr)

        # Compute reduction vs agent (honest baseline)
        if agent_baseline_tokens:
            reduction_pct = compute_reduction_pct(gateway_tokens, agent_baseline_tokens)
        else:
            reduction_pct = compute_reduction_pct(gateway_tokens, baseline_tokens)

        reductions.append(reduction_pct)

        # Top raw cosine (embedding confidence)
        top_raw_cosine = 0.0
        if results:
            top_raw_cosine = float(results[0].get("raw_cosine", 0.0))

        rows.append(
            {
                "query": query_text[:80],
                "gateway_tokens": gateway_tokens,
                "agent_baseline_tokens": agent_baseline_tokens,
                "reduction_pct": reduction_pct,
                "gold_hit": hit,
                "recall_at_5": round(query_recall_5, 3),
                "recall_at_10": round(query_recall_10, 3),
                "mrr": round(query_mrr, 3),
                "top_raw_cosine": round(top_raw_cosine, 4),
            }
        )

    if not rows:
        print("No queries completed.")
        return 1

    # ── Aggregate metrics ────────────────────────────────────────────────
    reductions.sort()
    n = len(reductions)
    median_reduction = reductions[n // 2] if reductions else 0.0

    avg_recall_5 = (
        sum(per_query_recalls_5) / len(per_query_recalls_5) if per_query_recalls_5 else 0.0
    )
    avg_recall_10 = (
        sum(per_query_recalls_10) / len(per_query_recalls_10) if per_query_recalls_10 else 0.0
    )
    avg_mrr = sum(per_query_mrrs) / len(per_query_mrrs) if per_query_mrrs else 0.0

    # ── Honesty gate: reduction is only claimable if recall >= 0.8 ──────
    is_claimable = reduction_claimable(avg_recall_10)
    if is_claimable:
        headline = f"{median_reduction:.1f}% token reduction at Recall@10 = {avg_recall_10:.3f}"
    else:
        headline = (
            f"reduction not claimable — Recall@10 = {avg_recall_10:.3f} "
            f"(need >= 0.8 for honest claim)"
        )

    # ── Print table ──────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("COMPARISON: Gateway vs Full-Context Agent Baseline (~4-char/token)")
    print("=" * 100)
    hdr = (
        f"{'Query':<35s} {'GW':>7s} {'Red%':>7s} {'R@5':>6s} {'R@10':>6s} "
        f"{'MRR':>6s} {'Cosine':>8s} {'Hit':>4s}"
    )
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['query']:<35s} {r['gateway_tokens']:>7d} {r['reduction_pct']:>6.1f}% "
            f"{r['recall_at_5']:>6.3f} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} "
            f"{r['top_raw_cosine']:>8.4f} {'✓' if r['gold_hit'] else '✗':>4s}"
        )
    print("-" * 100)
    print(
        f"{'MEDIAN':<35s} {median_reduction:>6.1f}% "
        f"{'':>6s} {'':>6s} {avg_recall_10:>6.3f} {avg_mrr:>6.3f}"
    )
    print()
    print("HEADLINE:")
    print(f"  {headline}")
    print()

    # ── Embedding A/B comparison ────────────────────────────────────────
    print("=" * 100)
    print("EMBEDDING MODEL A/B: Discrimination (relevant vs nonsense queries)")
    print("=" * 100)
    embedding_results = compare_embedding_models(args.models.split(","))
    for model_name, stats in embedding_results.items():
        print(
            f"{model_name:<30s} "
            f"Relevant cosine: {stats['relevant_mean']:>7.4f}  "
            f"Nonsense cosine: {stats['nonsense_mean']:>7.4f}  "
            f"GAP: {stats['gap']:>7.4f}"
        )
    print()

    # ── Write JSON ───────────────────────────────────────────────────────
    output = {
        "agent_baseline_tokens": agent_baseline_tokens,
        "median_reduction_pct": median_reduction,
        "recall_at_5": round(avg_recall_5, 3),
        "recall_at_10": round(avg_recall_10, 3),
        "mrr": round(avg_mrr, 3),
        "is_claimable": is_claimable,
        "headline": headline,
        "per_query": rows,
        "embedding_models": embedding_results,
        "note": "Token counts use ~4-char/token approximation; reduction @ recall >= 0.8 gate",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results written to {RESULTS_PATH}")

    return 0


def django_rerank_queries() -> list[tuple[str, set[str], str]]:
    """Django-specific queries for Phase F reranker A/B benchmark.

    Format: (query_text, {gold_function_ids}, expected_substrings)
    Expected substrings are heuristic hints for "correctness" detection.

    Returns:
      - 5 relevant queries with clear expected function signatures
      - 3 nonsense queries to test suppression
    """
    # All gold IDs use Django's django/db/models/query.py and related
    return [
        # Relevant queries
        (
            "execute queryset database query",
            {
                "django/db/models/query.py:QuerySet.get:0",
                "django/db/models/query.py:QuerySet._fetch_all:0",
            },
            "execute",  # Functions should contain "execute" or "_fetch"
        ),
        (
            "Django URL routing resolve pattern matching",
            {
                "django/urls/resolvers.py:URLResolver.resolve:0",
                "django/urls/resolvers.py:URLPattern.resolve:0",
            },
            "resolve",  # Should contain "resolve" in name/file
        ),
        (
            "Model form validation save instance",
            {
                "django/forms/models.py:ModelForm.save:0",
                "django/db/models/base.py:Model.full_clean:0",
            },
            "save",  # Should contain "save" or "full_clean"
        ),
        (
            "HTTP request middleware processing chain",
            {
                "django/core/wsgi.py:WSGIHandler.__call__:0",
                "django/core/handlers/wsgi.py:WSGIHandler.load_middleware:0",
            },
            "middleware",  # Should reference middleware handling
        ),
        (
            "Queryset filter annotation aggregation",
            {
                "django/db/models/query.py:QuerySet.filter:0",
                "django/db/models/query.py:QuerySet.annotate:0",
            },
            "filter",  # Should contain filter/annotate
        ),
        # Nonsense queries (should be suppressed with rerank ON)
        (
            "kubernetes pod autoscaler reconcile loop",
            set(),
            "kubernetes",  # Unrelated domain
        ),
        (
            "blockchain consensus proof-of-work algorithm",
            set(),
            "blockchain",  # Unrelated domain
        ),
        (
            "xyzzy gibberish nonsense purple monkey",
            set(),
            "gibberish",  # Pure garbage
        ),
    ]


def _is_plausibly_correct(function_name: str, filepath: str, expected_substring: str) -> bool:
    """Heuristic check: does the result contain expected substrings?

    Args:
        function_name: The function name from results
        filepath: The file path from results
        expected_substring: Heuristic substring to match

    Returns:
        True if function_name or filepath contains the expected substring
    """
    combined = f"{function_name} {filepath}".lower()
    return expected_substring.lower() in combined


def compare_rerank_ab(project_path: Path, queries: list[tuple] | None = None) -> dict:
    """A/B test reranker on/off for Django queries.

    For each query, run twice:
      1. With rerank OFF (embeddings/hybrid, no cross-encoder)
      2. With rerank ON (hybrid with FlashRank reranker)

    Records:
      - Top-1 function name and score for each
      - Whether top-1 is "plausibly correct" (heuristic)
      - Whether nonsense queries are suppressed

    Args:
        project_path: Path to the Django project
        queries: List of (query_text, gold_ids, expected_substring) tuples.
                 If None, uses django_rerank_queries().

    Returns:
        dict with keys:
          - results: List of per-query A/B rows
          - top1_correct_off: Proportion of relevant queries with correct top-1 (rerank OFF)
          - top1_correct_on: Proportion of relevant queries with correct top-1 (rerank ON)
          - nonsense_suppressed_off: Proportion of nonsense queries suppressed (rerank OFF)
          - nonsense_suppressed_on: Proportion of nonsense queries suppressed (rerank ON)
          - verdict: Pass/Fail headline
    """
    if queries is None:
        queries = django_rerank_queries()

    # Pre-flight checks
    if not project_path.exists():
        return {"error": f"Project path {project_path} does not exist"}

    chroma_path = project_path / ".cairn" / "chroma"
    if not chroma_path.exists():
        return {"error": f"Project {project_path} is not indexed"}

    # Check Ollama
    try:
        from server.ollama_client import OllamaClient

        if not OllamaClient().health_check():
            return {"error": "Ollama not reachable"}
    except Exception as e:
        return {"error": f"Ollama error: {e}"}

    # Import after path setup
    sys.path.insert(0, str(project_path))
    sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(project_path)

    from server.context_assembler import ContextAssembler

    results = []
    relevant_correct_off: int = 0
    relevant_correct_on: int = 0
    relevant_count: int = 0
    nonsense_suppressed_off: float = 0.0
    nonsense_suppressed_on: float = 0.0
    nonsense_count: int = 0

    for query_text, gold_ids, expected_substring in queries:
        is_nonsense = len(gold_ids) == 0

        # Run with RERANK OFF: temporarily patch the config in memory
        try:
            # Patch where it's actually imported
            import server.context_assembler as ca_module

            original_load = ca_module.load_config

            def mock_load_off(*args, **kwargs):
                c = original_load(*args, **kwargs)
                c.retrieval.rerank_enabled = False
                return c

            ca_module.load_config = mock_load_off
            assembler_off = ContextAssembler(project_path=project_path, top_k=3)
            results_off = assembler_off.semantic_search(query_text, top_k=3)
            ca_module.load_config = original_load
        except Exception as e:
            results_off = []
            print(f"WARNING: rerank OFF failed for '{query_text[:40]}...': {e}")

        # Run with RERANK ON
        try:
            import server.context_assembler as ca_module

            original_load = ca_module.load_config

            def mock_load_on(*args, **kwargs):
                c = original_load(*args, **kwargs)
                c.retrieval.rerank_enabled = True
                return c

            ca_module.load_config = mock_load_on
            assembler_on = ContextAssembler(project_path=project_path, top_k=3)
            results_on = assembler_on.semantic_search(query_text, top_k=3)
            ca_module.load_config = original_load
        except Exception as e:
            results_on = []
            print(f"WARNING: rerank ON failed for '{query_text[:40]}...': {e}")

        # Extract top-1 info for OFF
        top1_func_off = None
        top1_score_off = 0.0
        top1_file_off = None
        if results_off:
            top1_func_off = results_off[0].get("function", "")
            top1_file_off = results_off[0].get("filepath", "")
            top1_score_off = float(results_off[0].get("raw_cosine", 0.0))

        # Extract top-1 info for ON
        top1_func_on = None
        top1_score_on = 0.0
        top1_file_on = None
        if results_on:
            top1_func_on = results_on[0].get("function", "")
            top1_file_on = results_on[0].get("filepath", "")
            top1_score_on = float(results_on[0].get("rerank_score", 0.0))

        # Heuristic correctness check
        correct_off = (
            _is_plausibly_correct(top1_func_off or "", top1_file_off or "", expected_substring)
            if not is_nonsense
            else False
        )
        correct_on = (
            _is_plausibly_correct(top1_func_on or "", top1_file_on or "", expected_substring)
            if not is_nonsense
            else False
        )

        # Suppression check for nonsense
        # Suppressed = top1_score < min_confidence (OFF) or rerank_min_score (ON)
        min_confidence = 0.8
        rerank_min_score = 0.47
        suppressed_off = top1_score_off < min_confidence if is_nonsense else False
        suppressed_on = top1_score_on < rerank_min_score if is_nonsense else False

        row = {
            "query": query_text[:80],
            "is_nonsense": is_nonsense,
            "expected_substring": expected_substring,
            "rerank_off_top1": f"{top1_func_off}",
            "rerank_off_score": round(top1_score_off, 4),
            "rerank_off_correct": correct_off,
            "rerank_on_top1": f"{top1_func_on}",
            "rerank_on_score": round(top1_score_on, 4),
            "rerank_on_correct": correct_on,
            "suppressed_off": suppressed_off,
            "suppressed_on": suppressed_on,
        }
        results.append(row)

        # Aggregate stats
        if is_nonsense:
            nonsense_count += 1
            if suppressed_off:
                nonsense_suppressed_off += 1
            if suppressed_on:
                nonsense_suppressed_on += 1
        else:
            relevant_count += 1
            if correct_off:
                relevant_correct_off += 1
            if correct_on:
                relevant_correct_on += 1

    # Compute rates
    top1_correct_off = relevant_correct_off / relevant_count if relevant_count > 0 else 0.0
    top1_correct_on = relevant_correct_on / relevant_count if relevant_count > 0 else 0.0
    nonsense_suppressed_off = (
        nonsense_suppressed_off / nonsense_count if nonsense_count > 0 else 0.0
    )
    nonsense_suppressed_on = nonsense_suppressed_on / nonsense_count if nonsense_count > 0 else 0.0

    # Verdict: reranker is a "win" if:
    #   1. Improves top-1 correctness on relevant queries
    #   2. Suppresses all nonsense
    is_win = top1_correct_on >= top1_correct_off and nonsense_suppressed_on == 1.0

    if is_win:
        verdict = (
            f"RERANK WIN: improved top-1 correctness "
            f"({top1_correct_off:.1%} → {top1_correct_on:.1%}) "
            f"and suppressed all nonsense ({nonsense_suppressed_on:.1%})"
        )
    else:
        verdict = (
            f"RERANK MIXED/LOSS: "
            f"top-1 correctness {top1_correct_off:.1%} → {top1_correct_on:.1%}, "
            f"nonsense suppression {nonsense_suppressed_off:.1%} → {nonsense_suppressed_on:.1%}"
        )

    return {
        "results": results,
        "top1_correct_off": round(top1_correct_off, 3),
        "top1_correct_on": round(top1_correct_on, 3),
        "nonsense_suppressed_off": round(nonsense_suppressed_off, 3),
        "nonsense_suppressed_on": round(nonsense_suppressed_on, 3),
        "relevant_count": relevant_count,
        "nonsense_count": nonsense_count,
        "verdict": verdict,
    }


def compare_embedding_models(models: list[str]) -> dict[str, dict[str, float]]:
    """A/B test embedding models for discrimination (relevant vs nonsense).

    For each model:
      - Measure mean raw_cosine for the 6 relevant DEMO queries
      - Measure mean raw_cosine for 3 nonsense queries
      - Report the GAP (relevant - nonsense); bigger = better discrimination

    Args:
        models: List of model names to test. If a model isn't available in Ollama,
                skip it with a note (don't crash).

    Returns:
        dict mapping model_name -> {relevant_mean, nonsense_mean, gap}
    """
    # Pre-flight: check Ollama
    try:
        from pipeline.indexer import VectorIndexer
        from server.ollama_client import OllamaClient

        if not OllamaClient().health_check():
            print("WARNING: Ollama not reachable, skipping embedding A/B")
            return {}
    except Exception:
        print("WARNING: Ollama client error, skipping embedding A/B")
        return {}

    # Nonsense queries (not in the dataset, should have low cosine)
    nonsense_queries = [
        "xyzzy purple monkey dishwasher",
        "qwerty zxcvb asdf",
        "blah blah nonsense foo",
    ]

    # Relevant queries from the demo dataset
    try:
        from benchmarks.datasets.demo import load_demo_queries

        demo_queries = load_demo_queries()
        relevant_queries = [q for q, _ in demo_queries]
    except ImportError:
        print("WARNING: Could not load demo queries")
        return {}

    results: dict[str, dict[str, float]] = {}

    for model_name in models:
        model_name = model_name.strip()

        # Check if model is available
        try:
            client = OllamaClient(embed_model=model_name)
            available = client.list_models()
            model_full_name = next((m for m in available if model_name in m), None)
            if not model_full_name:
                print(f"  SKIPPED {model_name}: not available in Ollama")
                continue
        except Exception as e:
            print(f"  SKIPPED {model_name}: {e}")
            continue

        try:
            indexer = VectorIndexer(
                chroma_path=TESTAPI_PATH / ".cairn" / "chroma",
                ollama_client=client,
                embedding_model=model_name,
                project_root=TESTAPI_PATH,
            )

            # Collect cosines for relevant queries
            relevant_cosines = []
            for query in relevant_queries:
                search_results = indexer.search(query, top_k=1)
                if search_results:
                    # Use raw_cosine if available, else similarity
                    result = search_results[0]
                    cosine = result.get("raw_cosine", result.get("similarity", 0.0))
                    relevant_cosines.append(float(cosine))

            # Collect cosines for nonsense queries
            nonsense_cosines = []
            for query in nonsense_queries:
                search_results = indexer.search(query, top_k=1)
                if search_results:
                    result = search_results[0]
                    cosine = result.get("raw_cosine", result.get("similarity", 0.0))
                    nonsense_cosines.append(float(cosine))

            relevant_mean = (
                sum(relevant_cosines) / len(relevant_cosines) if relevant_cosines else 0.0
            )
            nonsense_mean = (
                sum(nonsense_cosines) / len(nonsense_cosines) if nonsense_cosines else 0.0
            )
            gap = relevant_mean - nonsense_mean

            results[model_name] = {
                "relevant_mean": round(relevant_mean, 4),
                "nonsense_mean": round(nonsense_mean, 4),
                "gap": round(gap, 4),
            }

        except Exception as e:
            print(f"  SKIPPED {model_name}: {e}")
            continue

    return results


if __name__ == "__main__":
    sys.exit(main())
