#!/usr/bin/env python3
"""In-process benchmark: ChromaStore vs LanceStore on hardtest repos.

Indexes each repo with both backends, measures:
  - Index time, peak RSS, disk usage, resilience (crash detection)
  - Retrieval: hit rate on known-answer queries (vector A/B for subsets)

Write a markdown report to --report and print a summary to stdout.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Lazy imports for heavy deps
_psutil = None


def _get_psutil():
    """Lazy-load psutil."""
    global _psutil
    if _psutil is None:
        import psutil
        _psutil = psutil
    return _psutil


@dataclass
class BenchResult:
    """Result of one (repo, backend) index+retrieve run."""

    repo_name: str
    backend: str
    profile: str
    embeddings_enabled: bool
    blocks_indexed: int
    index_time_s: float
    peak_rss_mb: float
    disk_mb: float
    indexed_ok: bool
    index_error: Optional[str] = None
    retrieval_results: Optional[dict] = None  # {query: hit_rank}


def _ensure_clean_cairn(repo_path: Path) -> None:
    """Wipe .cairn directory (both chroma/ and lance/)."""
    cairn_dir = repo_path / ".cairn"
    if cairn_dir.exists():
        shutil.rmtree(cairn_dir, ignore_errors=True)
    cairn_dir.mkdir(parents=True, exist_ok=True)


def _get_disk_usage_mb(path: Path) -> float:
    """Compute du of a directory in MB."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total / (1024 * 1024)


def _build_cfg(repo_name: str, backend: str) -> object:
    """Build a Config object with embedding policy per repo."""
    from core.config import Config
    from core.profiles import get_profile

    cfg = Config()

    # Set backend
    cfg.indexing.store_backend = backend

    # Set profile and get patterns from it
    from tests.corpus.manifest import EXPECTED_PROFILE
    profile_name = EXPECTED_PROFILE.get(repo_name, "code")
    cfg.profile = profile_name
    profile = get_profile(profile_name)

    # Use profile's file patterns
    cfg.indexing.file_patterns = profile.file_patterns

    # Embedding policy per repo
    if repo_name in ("django", "cert-manager"):
        # Vector A/B repos: embeddings ON
        cfg.embeddings_enabled = True
        cfg.local_llm.enabled = True
        cfg.local_llm.backend = "ollama"
        cfg.local_llm.embed_model = "nomic-embed-text"
        cfg.indexing.embedding_model = "nomic-embed-text"
    else:
        # Resilience/IaC repos: embeddings OFF
        cfg.embeddings_enabled = False
        cfg.local_llm.enabled = False

    return cfg


def _index_repo(
    repo_path: Path, cfg: object, index_timeout_s: int
) -> tuple[int, float, float, bool, Optional[str]]:
    """Index a repo: parse + upsert + measure time/peak RSS.

    Returns:
        (blocks_indexed, index_time_s, peak_rss_mb, success, error_msg)
    """
    from core.repo import RepoManager, collect_source_files
    from pipeline.ast_parser import ASTParser
    from pipeline.store import blocks_from_ast, make_store

    try:
        _ensure_clean_cairn(repo_path)

        repo_mgr = RepoManager(repo_path)
        store = make_store(cfg, repo_mgr, project_root=repo_path)

        parser = ASTParser()

        # Collect files using reindex's logic
        filtered = collect_source_files(
            repo_path,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            cfg.indexing.source_roots,
        )

        # Track peak RSS
        psutil = _get_psutil()
        process = psutil.Process()
        peak_rss = 0

        blocks_total = 0
        start_time = time.time()

        for filepath in filtered:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > index_timeout_s:
                return (
                    blocks_total,
                    elapsed,
                    peak_rss / (1024 * 1024),
                    False,
                    f"Timeout after {elapsed:.1f}s",
                )

            try:
                ast_result = parser.parse_file(filepath)
                blocks = blocks_from_ast(ast_result)
                if blocks:
                    store.upsert_blocks(blocks, batch_size=2000)
                blocks_total += len(blocks)
            except Exception:
                # Log parse errors but continue
                pass

            # Sample RSS
            try:
                rss = process.memory_info().rss
                peak_rss = max(peak_rss, rss)
            except Exception:
                pass

        index_time = time.time() - start_time
        peak_rss_mb = peak_rss / (1024 * 1024)

        # Verify indexed count via store.count()
        final_count = store.count()

        return (final_count, index_time, peak_rss_mb, True, None)

    except Exception as e:
        return (0, 0.0, 0.0, False, str(e))


def _retrieve_results(
    repo_path: Path,
    store,
    queries: list[tuple[str, str]],
    embeddings_enabled: bool,
    backend: str,
) -> dict:
    """Run retrieval queries, record hit@8.

    Returns:
        {query_str: rank_or_none}
    """
    from pipeline.retrieval.bm25 import BM25Retriever

    results = {}

    for query_str, expected_substr in queries:
        try:
            if embeddings_enabled:
                # Use semantic search
                hits = store.search(query_str, top_k=8)
            else:
                # No embeddings: use hybrid_search (Lance) or lexical fallback
                # (Chroma)
                try:
                    hits = store.hybrid_search(query_str, top_k=8)
                except Exception:
                    # ChromaStore fallback: use BM25 over blocks
                    hits = []
                    try:
                        blocks = list(store.iter_blocks())
                        if blocks:
                            # Convert blocks to dicts for BM25
                            docs = []
                            for block in blocks:
                                # Note: iter_blocks returns dicts with "text" key
                                docs.append({
                                    "id": block.get("id", ""),
                                    "text": block.get("text", ""),
                                })
                            bm25 = BM25Retriever()
                            bm25.index(docs)
                            bm25_hits = bm25.search(query_str, top_k=8)
                            # Convert BM25 hits back to store hit format
                            hits = []
                            # Build lookup map
                            blocks_by_id = {b.get("id"): b for b in blocks}
                            for bm25_hit in bm25_hits:
                                block_id = bm25_hit["id"]
                                block = blocks_by_id.get(block_id)
                                if block:
                                    hits.append({
                                        "id": block.get("id", ""),
                                        "filepath": block.get("filepath", ""),
                                        "function": block.get("function", ""),
                                        "code": block.get("text", ""),
                                        "similarity": bm25_hit.get("score", 0),
                                    })
                    except Exception:
                        pass

            # Check if expected_substr appears in any hit
            rank = None
            for i, hit in enumerate(hits, start=1):
                filepath = hit.get("filepath", "")
                if expected_substr in filepath:
                    rank = i
                    break

            results[query_str] = rank

        except Exception:
            results[query_str] = None

    return results


def _run_benchmark(
    repos_root: Path,
    repo_filters: list[str],
    backends: list[str],
    report_path: Path,
    smoke_mode: bool,
    index_timeout_s: int,
) -> None:
    """Run full benchmark or smoke test."""

    # Map repo names to paths
    all_repos = {}
    for corpus_dir in (repos_root / "corpus", repos_root / "workspace"):
        if corpus_dir.exists():
            for repo_dir in corpus_dir.iterdir():
                if repo_dir.is_dir():
                    all_repos[repo_dir.name] = repo_dir

    # Filter repos
    if smoke_mode:
        # Smoke: only terragrunt-live
        repos_to_run = {k: v for k, v in all_repos.items()
                        if k == "terragrunt-live"}
    elif repo_filters:
        repos_to_run = {
            k: v for k, v in all_repos.items()
            if any(f in k for f in repo_filters)
        }
    else:
        repos_to_run = all_repos

    # Load manifest queries
    from tests.corpus.manifest import CORPUS_REPOS
    queries_map = {repo.name: repo.queries for repo in CORPUS_REPOS}

    # Run benchmarks
    results: list[BenchResult] = []

    for repo_name in sorted(repos_to_run.keys()):
        repo_path = repos_to_run[repo_name]
        queries = queries_map.get(repo_name, [])

        print(f"\n--- {repo_name} ---")

        for backend in backends:
            print(f"  {backend}...", end=" ", flush=True)

            cfg = _build_cfg(repo_name, backend)
            embeddings_enabled = cfg.embeddings_enabled and cfg.local_llm.enabled
            profile = cfg.profile

            # Index
            blocks, index_time, peak_rss, success, error_msg = _index_repo(
                repo_path, cfg, index_timeout_s
            )

            # Disk usage
            if backend == "lance":
                store_dir = repo_path / ".cairn" / "lance"
            else:
                store_dir = repo_path / ".cairn" / "chroma"

            disk_mb = _get_disk_usage_mb(store_dir) if store_dir.exists() else 0.0

            result = BenchResult(
                repo_name=repo_name,
                backend=backend,
                profile=profile,
                embeddings_enabled=embeddings_enabled,
                blocks_indexed=blocks,
                index_time_s=index_time,
                peak_rss_mb=peak_rss,
                disk_mb=disk_mb,
                indexed_ok=success,
                index_error=error_msg,
            )

            # Retrieval
            if success:
                try:
                    from core.repo import RepoManager
                    from pipeline.store import make_store

                    repo_mgr = RepoManager(repo_path)
                    store = make_store(cfg, repo_mgr, project_root=repo_path)
                    retrieval = _retrieve_results(
                        repo_path, store, queries, embeddings_enabled, backend
                    )
                    result.retrieval_results = retrieval
                except Exception:
                    result.retrieval_results = {}

            results.append(result)
            print(
                f"✓ {blocks} blocks in {index_time:.1f}s"
                f" ({peak_rss:.0f}MB peak RSS)"
            )

    # Write report
    _write_report(results, report_path)
    print(f"\nReport: {report_path}")


def _write_report(results: list[BenchResult], report_path: Path) -> None:
    """Write markdown report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cairn Store Benchmark Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "",
        "## Table 1: Index Metrics (all repos × backend)\n",
        "| Repo | Backend | Profile | Embeddings | Blocks | "
        "Index (s) | Peak RSS (MB) | Disk (MB) | OK |\n",
        "|------|---------|---------|------------|--------|"
        "-----------|-------------|----------|----|\n",
    ]

    for result in results:
        embeddings_str = "ON" if result.embeddings_enabled else "OFF"
        ok_str = "✓" if result.indexed_ok else "✗"
        error_suffix = (
            f" ({result.index_error[:30]})" if result.index_error else ""
        )

        lines.append(
            f"| {result.repo_name} | {result.backend} | "
            f"{result.profile} | {embeddings_str} | {result.blocks_indexed} | "
            f"{result.index_time_s:.2f} | {result.peak_rss_mb:.0f} | "
            f"{result.disk_mb:.1f} | {ok_str}{error_suffix} |\n"
        )

    lines.append("\n## Table 2: Retrieval Results (vector A/B + IaC)\n")
    lines.append("| Repo | Backend | Query | Hit@8 | Rank |\n")
    lines.append("|------|---------|-------|-------|------|\n")

    for result in results:
        if result.retrieval_results:
            for query, rank in result.retrieval_results.items():
                rank_str = str(rank) if rank is not None else "—"
                hit_str = "✓" if rank is not None else "✗"
                # Truncate query for readability
                query_short = (query[:40] + "...") if len(query) > 40 else query
                lines.append(
                    f"| {result.repo_name} | {result.backend} | "
                    f"{query_short} | {hit_str} | {rank_str} |\n"
                )

    lines.append("\n## Summary\n")

    # Summarize by backend and repo
    by_backend = {}
    for result in results:
        if result.backend not in by_backend:
            by_backend[result.backend] = []
        by_backend[result.backend].append(result)

    for backend, backend_results in sorted(by_backend.items()):
        total_blocks = sum(r.blocks_indexed for r in backend_results)
        total_time = sum(r.index_time_s for r in backend_results)
        total_disk = sum(r.disk_mb for r in backend_results)
        success_count = sum(1 for r in backend_results if r.indexed_ok)

        lines.append(
            f"**{backend.upper()}**: {total_blocks} blocks, "
            f"{total_time:.1f}s total, {total_disk:.1f}MB disk, "
            f"{success_count}/{len(backend_results)} OK\n"
        )

    lines.append("\n## Verdict\n")
    lines.append(
        "See above tables for detailed metrics. "
        "Compare index time, disk usage, resilience (OK count), "
        "and retrieval hit rates.\n"
    )

    report_path.write_text("".join(lines))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="In-process benchmark: ChromaStore vs LanceStore"
    )
    parser.add_argument(
        "--repos-root",
        type=Path,
        default=Path("/mnt/c/Users/alfre/Projects/cairn-hardtest"),
        help="Root directory containing corpus/ and workspace/",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repo_filters",
        default=[],
        help="Repo name filter (repeatable, e.g. --repo django --repo cert)",
    )
    parser.add_argument(
        "--store",
        choices=["chroma", "lance", "both"],
        default="both",
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/mnt/c/Users/alfre/Projects/cairn-hardtest/report/STORE_BENCH.md"
        ),
        help="Output report path",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: only terragrunt-live, embeddings off, both backends",
    )
    parser.add_argument(
        "--index-timeout",
        type=int,
        default=3600,
        help="Max seconds per index (default 3600)",
    )

    args = parser.parse_args()

    backends = (
        ["chroma", "lance"]
        if args.store == "both"
        else [args.store]
    )

    _run_benchmark(
        args.repos_root,
        args.repo_filters,
        backends,
        args.report,
        args.smoke,
        args.index_timeout,
    )


if __name__ == "__main__":
    main()
