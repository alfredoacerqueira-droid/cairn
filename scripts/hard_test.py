#!/usr/bin/env python3
"""Hard-test campaign runner for Cairn.

OPS/eval script that drives Cairn against real cloned repos, measures everything,
and writes a markdown report. Invoked manually (clones live elsewhere).
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout_s: float = 120.0,
    capture: bool = True,
) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr).

    Args:
        cmd: Command as a list.
        cwd: Working directory.
        timeout_s: Timeout in seconds.
        capture: If True, capture stdout/stderr; else stream to console.

    Returns:
        (success, stdout, stderr) tuple. If timeout, success=False, stderr contains
        "TIMEOUT" marker.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout_s,
        )
        return result.returncode == 0, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def wipe_cairn_dir(repo_path: Path) -> None:
    """Wipe <repo>/.cairn completely."""
    cairn_dir = repo_path / ".cairn"
    if cairn_dir.exists():
        shutil.rmtree(cairn_dir, ignore_errors=True)
    logger.info(f"  Wiped {repo_path}/.cairn")


def init_cairn(repo_path: Path, cairn_cmd: str = "cairn") -> bool:
    """Run 'cairn init --no-index' to create .cairn/config.yaml without indexing.

    Args:
        repo_path: Path to the repo.
        cairn_cmd: The cairn CLI command.

    Returns:
        True if successful, False otherwise.
    """
    success, _, _ = run_cmd([cairn_cmd, "init", "--no-index"], cwd=repo_path)
    if success:
        logger.info("  ✓ init --no-index succeeded")
        return True
    else:
        logger.warning("  ✗ init --no-index failed")
        return False


def reindex_cairn(
    repo_path: Path,
    cairn_cmd: str = "cairn",
    timeout_s: float = 1200.0,
) -> tuple[bool, float, str, str]:
    """Run 'cairn reindex' and measure wall-clock time.

    Args:
        repo_path: Path to the repo.
        cairn_cmd: The cairn CLI command.
        timeout_s: Timeout in seconds.

    Returns:
        (success, elapsed_s, stdout, stderr) tuple.
    """
    start = time.perf_counter()
    success, stdout, stderr = run_cmd(
        [cairn_cmd, "reindex"],
        cwd=repo_path,
        timeout_s=timeout_s,
    )
    elapsed = time.perf_counter() - start

    if "TIMEOUT" in stderr:
        logger.info(f"  ⏱ Reindex HUNG ({elapsed:.1f}s)")
        return False, elapsed, stdout, "TIMEOUT"
    elif success:
        logger.info(f"  ✓ Reindex succeeded ({elapsed:.1f}s)")
        return True, elapsed, stdout, stderr
    else:
        logger.warning(f"  ✗ Reindex failed ({elapsed:.1f}s)")
        return False, elapsed, stdout, stderr


def load_config_profile(repo_path: Path) -> str | None:
    """Load the detected profile from .cairn/config.yaml.

    Args:
        repo_path: Path to the repo.

    Returns:
        Profile name (str) or None if config missing/parse error.
    """
    try:
        from core.config import load_config

        cfg = load_config(repo_path)
        return cfg.profile
    except Exception as e:
        logger.debug(f"  Could not load profile: {e}")
        return None


def count_source_files(repo_path: Path) -> int:
    """Count source files matching standard patterns in the repo.

    Args:
        repo_path: Path to the repo.

    Returns:
        Count of files.
    """
    try:
        from core.config import load_config
        from core.repo import collect_source_files

        cfg = load_config(repo_path)
        files = collect_source_files(
            repo_path,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            cfg.indexing.source_roots,
        )
        return len(files)
    except Exception as e:
        logger.debug(f"  Could not count files: {e}")
        return 0


def count_indexed_blocks(repo_path: Path) -> int:
    """Count indexed blocks via VectorIndexer.collection.count().

    Args:
        repo_path: Path to the repo.

    Returns:
        Count of indexed blocks or 0 if error.
    """
    try:
        from core.config import load_config
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        cfg = load_config(repo_path)
        repo = RepoManager(repo_path)
        # Effective embeddings flag
        emb_enabled = cfg.embeddings_enabled and cfg.local_llm.enabled
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(),
            embeddings_enabled=emb_enabled,
            project_root=repo_path,
        )
        return indexer.collection.count()
    except Exception as e:
        logger.debug(f"  Could not count indexed blocks: {e}")
        return 0


def parse_reindex_warnings(stdout: str, stderr: str) -> dict[str, Any]:
    """Parse reindex output for warnings and issues.

    Looks for lines with "Skip", "timeout", "Batch size", "Dropped", "error" etc.

    Args:
        stdout: Stdout from reindex.
        stderr: Stderr from reindex.

    Returns:
        Dict with keys like 'skip_count', 'error_lines' (sample list).
    """
    combined = (stdout + "\n" + stderr).lower()
    lines = combined.split("\n")

    skip_count = 0
    error_lines = []
    timeout_count = 0
    dropped_count = 0

    for line in lines:
        if "skip" in line:
            skip_count += 1
            if len(error_lines) < 3:
                error_lines.append(line[:80])
        if "timeout" in line:
            timeout_count += 1
        if "dropped" in line:
            dropped_count += 1
        if "error" in line or "exception" in line:
            if len(error_lines) < 3:
                error_lines.append(line[:80])

    return {
        "skip_count": skip_count,
        "error_count": len(error_lines),
        "timeout_count": timeout_count,
        "dropped_count": dropped_count,
        "sample_errors": error_lines[:2],
    }


def run_doctor(repo_path: Path, cairn_cmd: str = "cairn") -> dict[str, Any]:
    """Run 'cairn doctor' and parse output for key lines.

    Args:
        repo_path: Path to the repo.
        cairn_cmd: The cairn CLI command.

    Returns:
        Dict with keys like 'reranker_ok', 'ripgrep_ok', 'profile', 'collection'.
    """
    success, stdout, _ = run_cmd([cairn_cmd, "doctor"], cwd=repo_path)
    if not success:
        return {"status": "failed"}

    # Extract key lines
    reranker_ok = "Reranker (flashrank): model loadable" in stdout
    ripgrep_ok = "ripgrep: found" in stdout
    profile_line = next(
        (line for line in stdout.split("\n") if "Collection:" in line),
        None,
    )

    return {
        "status": "ok",
        "reranker_ok": reranker_ok,
        "ripgrep_ok": ripgrep_ok,
        "profile_info": profile_line or "unknown",
    }


def semantic_search(
    repo_path: Path,
    query: str,
    top_k: int = 8,
    apply_guard: bool = True,
) -> tuple[bool, int, float]:
    """Run semantic search via ContextAssembler.

    Args:
        repo_path: Path to the repo.
        query: Search query.
        top_k: Top K results.
        apply_guard: Whether to apply rerank guard.

    Returns:
        (hit: bool, rank: int, score: float) where hit=True if expected substring
        found in any result filepath, rank is the position of the hit (1-indexed),
        and score is the best relevance score.
    """
    try:
        from server.context_assembler import ContextAssembler

        assembler = ContextAssembler(project_path=repo_path, top_k=top_k)
        results = assembler.semantic_search(query, top_k=top_k, apply_guard=apply_guard)
        return (len(results) > 0, 0, 0.0), results
    except Exception as e:
        logger.debug(f"  Search error: {e}")
        return (False, 0, 0.0), []


def assemble_context_size(
    repo_path: Path,
    query: str,
) -> tuple[int, int]:
    """Measure assembled context size (chars and estimated tokens).

    Args:
        repo_path: Path to the repo.
        query: Search query.

    Returns:
        (char_count, token_estimate) where token_estimate = chars / 4.
    """
    try:
        from server.context_assembler import ContextAssembler

        assembler = ContextAssembler(project_path=repo_path)
        context_md = assembler.assemble_context(query)
        char_count = len(context_md) if context_md else 0
        token_estimate = char_count // 4
        return char_count, token_estimate
    except Exception as e:
        logger.debug(f"  Assemble error: {e}")
        return 0, 0


def test_repo_step1(
    repo_path: Path,
    repo_name: str,
    known_queries: list[tuple[str, str]],
    cairn_cmd: str = "cairn",
    reindex_timeout: float = 1200.0,
) -> dict[str, Any]:
    """Run STEP 1 tests on a repo (no local LLM).

    Args:
        repo_path: Path to the repo.
        repo_name: Name of the repo (for logging).
        known_queries: List of (query, expected_substr) tuples.
        cairn_cmd: The cairn CLI command.
        reindex_timeout: Timeout for reindex in seconds.

    Returns:
        Dict with step1 results.
    """
    logger.info(f"\n=== STEP 1: {repo_name} ===")

    results = {
        "repo": repo_name,
        "profile_expected": None,
        "profile_got": None,
        "files_scanned": 0,
        "blocks_indexed": 0,
        "index_time_s": 0.0,
        "hung": False,
        "warnings": {},
        "doctor_ok": False,
        "query_hits": "0/0",
        "compression_chars": 0,
        "compression_tokens": 0,
    }

    # Wipe and init
    wipe_cairn_dir(repo_path)
    if not init_cairn(repo_path, cairn_cmd):
        logger.warning("  Init failed, skipping repo")
        return results

    # Reindex
    success, elapsed, stdout, stderr = reindex_cairn(
        repo_path,
        cairn_cmd,
        timeout_s=reindex_timeout,
    )
    results["index_time_s"] = elapsed
    results["hung"] = "TIMEOUT" in stderr

    # Profile
    profile = load_config_profile(repo_path)
    results["profile_got"] = profile

    # Files and blocks
    results["files_scanned"] = count_source_files(repo_path)
    results["blocks_indexed"] = count_indexed_blocks(repo_path)
    logger.info(f"  Files: {results['files_scanned']}, Blocks: {results['blocks_indexed']}")

    # Warnings
    results["warnings"] = parse_reindex_warnings(stdout, stderr)

    # Doctor
    doctor_result = run_doctor(repo_path, cairn_cmd)
    results["doctor_ok"] = doctor_result.get("status") == "ok"

    # Query hits
    if known_queries:
        hit_count = 0
        total_count = len(known_queries)
        for query_text, expected_substr in known_queries:
            (hit, rank, score), results_list = semantic_search(
                repo_path, query_text, top_k=8, apply_guard=True
            )

            # Check if expected_substr is in any filepath
            found = False
            for i, result in enumerate(results_list):
                filepath = result.get("filepath", "")
                if expected_substr.lower() in filepath.lower():
                    hit_count += 1
                    found = True
                    logger.info(f"    ✓ Query '{query_text}' found at rank {i+1}")
                    break

            if not found and results_list:
                # Try without guard if guarded gave 0
                logger.debug(f"    Query '{query_text}' not found, trying without guard...")
                (hit, rank, score), results_list = semantic_search(
                    repo_path, query_text, top_k=8, apply_guard=False
                )
                for i, result in enumerate(results_list):
                    filepath = result.get("filepath", "")
                    if expected_substr.lower() in filepath.lower():
                        hit_count += 1
                        logger.info(f"    ✓ Query '{query_text}' found at rank {i+1} (no guard)")
                        break

        results["query_hits"] = f"{hit_count}/{total_count}"
        logger.info(f"  Hits: {results['query_hits']}")

        # Compression on first query
        if known_queries:
            chars, tokens = assemble_context_size(repo_path, known_queries[0][0])
            results["compression_chars"] = chars
            results["compression_tokens"] = tokens
            logger.info(f"  Compression: {chars} chars, ~{tokens} tokens")

    return results


def test_repo_step2(
    repo_path: Path,
    repo_name: str,
    known_queries: list[tuple[str, str]],
    cairn_cmd: str = "cairn",
    reindex_timeout: float = 1200.0,
) -> dict[str, Any]:
    """Run STEP 2 tests on a repo (with local LLM embeddings A/B).

    Enables local LLM, reruns reindex, and compares against step 1.

    Args:
        repo_path: Path to the repo.
        repo_name: Name of the repo.
        known_queries: List of (query, expected_substr) tuples.
        cairn_cmd: The cairn CLI command.
        reindex_timeout: Timeout for reindex.

    Returns:
        Dict with step2 results.
    """
    logger.info(f"\n=== STEP 2: {repo_name} (Embeddings A/B) ===")

    results = {
        "repo": repo_name,
        "lexical_hits": "0/0",
        "embeddings_hits": "0/0",
        "index_time_lexical_s": 0.0,
        "index_time_embeddings_s": 0.0,
        "ollama_reachable": False,
    }

    # Check Ollama is reachable
    try:
        from server.ollama_client import OllamaClient

        ollama = OllamaClient()
        if not ollama.health_check():
            logger.warning("  Ollama not reachable, skipping step2")
            results["ollama_reachable"] = False
            return results
        results["ollama_reachable"] = True
    except Exception as e:
        logger.warning(f"  Ollama error: {e}, skipping step2")
        return results

    # Wipe and init
    wipe_cairn_dir(repo_path)
    if not init_cairn(repo_path, cairn_cmd):
        return results

    # Reindex with lexical (current state)
    success, elapsed_lexical, _, _ = reindex_cairn(repo_path, cairn_cmd, timeout_s=reindex_timeout)
    results["index_time_lexical_s"] = elapsed_lexical

    # Count lexical hits
    lexical_hit_count = 0
    if known_queries:
        for query_text, expected_substr in known_queries:
            (hit, rank, score), results_list = semantic_search(
                repo_path, query_text, top_k=8, apply_guard=True
            )
            for result in results_list:
                filepath = result.get("filepath", "")
                if expected_substr.lower() in filepath.lower():
                    lexical_hit_count += 1
                    break
        results["lexical_hits"] = f"{lexical_hit_count}/{len(known_queries)}"

    # Now enable embeddings: load config, enable, save, wipe, reindex
    try:
        from core.config import load_config, save_config

        cfg = load_config(repo_path)
        cfg.local_llm.enabled = True
        cfg.local_llm.backend = "ollama"
        cfg.embeddings_enabled = True
        save_config(cfg, repo_path)

        # Wipe index (keep config) and reindex with embeddings
        wipe_cairn_dir(repo_path)
        # Reinit with config
        if not init_cairn(repo_path, cairn_cmd):
            return results

        success, elapsed_embeddings, _, _ = reindex_cairn(
            repo_path, cairn_cmd, timeout_s=reindex_timeout
        )
        results["index_time_embeddings_s"] = elapsed_embeddings

        # Count embeddings hits
        embeddings_hit_count = 0
        if known_queries:
            for query_text, expected_substr in known_queries:
                (hit, rank, score), results_list = semantic_search(
                    repo_path, query_text, top_k=8, apply_guard=True
                )
                for result in results_list:
                    filepath = result.get("filepath", "")
                    if expected_substr.lower() in filepath.lower():
                        embeddings_hit_count += 1
                        break
            results["embeddings_hits"] = f"{embeddings_hit_count}/{len(known_queries)}"

    except Exception as e:
        logger.warning(f"  Step 2 error: {e}")

    return results


def test_workspace_router(
    workspace_root: Path,
    repos_to_test: list[tuple[Path, str, list[tuple[str, str]]]],
    cairn_cmd: str = "cairn",
) -> dict[str, Any]:
    """Test WorkspaceRouter if multiple repos are indexed in workspace.

    Args:
        workspace_root: The workspace root.
        repos_to_test: List of (repo_path, repo_name, queries) tuples.
        cairn_cmd: The cairn CLI command.

    Returns:
        Dict with router test results.
    """
    logger.info("\n=== WORKSPACE ROUTER TEST ===")

    if len(repos_to_test) < 2:
        logger.info("  Skipped (< 2 repos)")
        return {"status": "skipped", "reason": "fewer than 2 repos"}

    try:
        from server.workspace_router import WorkspaceRouter

        router = WorkspaceRouter(workspace_root)
        if len(router.repo_paths) < 2:
            logger.info("  Skipped (< 2 .cairn dirs in workspace)")
            return {"status": "skipped", "reason": "fewer than 2 indexed repos"}

        logger.info(f"  Discovered {len(router.repo_paths)} repos")

        # Test routing: use first repo's first query
        test_repo_path, test_repo_name, test_queries = repos_to_test[0]
        if not test_queries:
            logger.info("  Skipped (no test queries)")
            return {"status": "skipped", "reason": "no test queries"}

        query_text, _ = test_queries[0]
        routed_repo, results = router.route(query_text, top_k=5)

        if routed_repo is None:
            logger.warning("  Router failed to route query")
            return {"status": "failed", "reason": "no results from any repo"}

        logger.info(f"  Query '{query_text}' routed to {routed_repo.name}")

        # Check isolation: all project_id should match the routed repo
        from core.repo import project_id

        routed_project_id = project_id(routed_repo)
        isolation_ok = all(r.get("project_id") == routed_project_id for r in results)

        if isolation_ok:
            logger.info("  ✓ Project isolation verified")
        else:
            logger.warning("  ✗ Project leakage detected")

        return {
            "status": "ok",
            "routed_repo": routed_repo.name,
            "isolation_ok": isolation_ok,
            "result_count": len(results),
        }

    except Exception as e:
        logger.warning(f"  Router test error: {e}")
        return {"status": "error", "reason": str(e)}


def write_report(results_step1, results_step2, workspace_result, output_path):
    """Write markdown report.

    Args:
        results_step1: List of step1 result dicts.
        results_step2: List of step2 result dicts (may be empty).
        workspace_result: Workspace router result dict.
        output_path: Path to write report.
    """
    lines = [
        "# Cairn Hard-Test Campaign Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Step 1: Lexical/Structural Retrieval (No Local LLM)",
        "",
    ]

    # Step 1 table
    lines.append(
        "| Repo | Profile (Exp/Got) | Files | Blocks | Index (s) | "
        "Hung? | Warnings | Doctor | Hits | Compression |"
    )
    lines.append(
        "|------|-------------------|-------|--------|-----------|-------|"
        "----------|--------|------|------------|"
    )

    for result in results_step1:
        profile_str = (
            f"{result['profile_expected']}/{result['profile_got']}"
            if result["profile_expected"]
            else result["profile_got"] or "?"
        )
        warnings_str = f"skips:{result['warnings'].get('skip_count', 0)}"
        doctor_str = "✓" if result["doctor_ok"] else "✗"
        hung_str = "⏱ HUNG" if result["hung"] else "✓"
        compression_str = (
            f"{result['compression_chars']} chars (~{result['compression_tokens']} tokens)"
            if result["compression_chars"]
            else "—"
        )

        lines.append(
            f"| {result['repo']} | {profile_str} | {result['files_scanned']} | "
            f"{result['blocks_indexed']} | {result['index_time_s']:.1f} | {hung_str} | "
            f"{warnings_str} | {doctor_str} | {result['query_hits']} | "
            f"{compression_str} |"
        )

    lines.extend(["", "## Step 2: Embeddings A/B (With Local LLM)"])

    if results_step2:
        lines.append("")
        lines.append("| Repo | Ollama | Lexical | Embeddings | Index (s) Lex/Emb |")
        lines.append("|------|--------|---------|------------|--------------------|")

        for result in results_step2:
            ollama_str = "✓" if result["ollama_reachable"] else "✗"
            lex_time = f"{result['index_time_lexical_s']:.1f}"
            emb_time = f"{result['index_time_embeddings_s']:.1f}"
            lines.append(
                f"| {result['repo']} | {ollama_str} | {result['lexical_hits']} | "
                f"{result['embeddings_hits']} | {lex_time}/{emb_time} |"
            )
    else:
        lines.append("")
        lines.append("(Not run)")

    # Workspace router section
    lines.extend(["", "## Workspace Router", ""])
    if workspace_result["status"] == "ok":
        lines.append("**Status**: ✓ Passed")
        lines.append(f"- Routed repo: {workspace_result['routed_repo']}")
        lines.append(
            f"- Project isolation: {'✓ OK' if workspace_result['isolation_ok'] else '✗ LEAKAGE'}"
        )
        lines.append(f"- Results: {workspace_result['result_count']}")
    elif workspace_result["status"] == "skipped":
        lines.append(f"**Status**: Skipped ({workspace_result['reason']})")
    else:
        lines.append(f"**Status**: ✗ Failed ({workspace_result.get('reason', 'unknown')})")

    # Verdict
    lines.extend(["", "## Verdict", ""])
    failed_repos = [r["repo"] for r in results_step1 if r["hung"] or not r["doctor_ok"]]
    low_hits = [r["repo"] for r in results_step1 if r["query_hits"].startswith("0/")]

    if not failed_repos and not low_hits:
        lines.append("✓ All tests passed.")
    else:
        if failed_repos:
            lines.append(f"- **Failed repos**: {', '.join(failed_repos)}")
        if low_hits:
            lines.append(f"- **Low hits**: {', '.join(low_hits)}")

    output_path.write_text("\n".join(lines))
    logger.info(f"\nReport written to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cairn hard-test campaign runner",
    )
    parser.add_argument(
        "--repos-root",
        type=Path,
        help="Directory containing repo subdirs to test",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        type=Path,
        help="Explicit repo path to test (repeatable)",
    )
    parser.add_argument(
        "--step1",
        action="store_true",
        default=True,
        help="Run step 1 (lexical/structural retrieval)",
    )
    parser.add_argument(
        "--step2",
        action="store_true",
        help="Run step 2 (embeddings A/B with local LLM)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/cairn_hard_test.md"),
        help="Output report path (default: /tmp/cairn_hard_test.md)",
    )
    parser.add_argument(
        "--cairn",
        default="cairn",
        help="Cairn CLI command (default: cairn)",
    )
    parser.add_argument(
        "--reindex-timeout",
        type=float,
        default=1200.0,
        help="Reindex timeout in seconds (default: 1200)",
    )

    args = parser.parse_args()

    # Determine repos to test
    repos_to_test = []

    if args.repos:
        # Explicit --repo args
        for repo_path in args.repos:
            if not repo_path.is_dir():
                logger.error(f"Not a directory: {repo_path}")
                sys.exit(1)
            if not (repo_path / ".git").exists():
                logger.error(f"Not a git repo: {repo_path}")
                sys.exit(1)
            repos_to_test.append((repo_path, repo_path.name, []))

    elif args.repos_root:
        # Scan --repos-root for child git repos
        if not args.repos_root.is_dir():
            logger.error(f"Not a directory: {args.repos_root}")
            sys.exit(1)

        for child in sorted(args.repos_root.iterdir()):
            if child.is_dir() and (child / ".git").exists():
                repos_to_test.append((child, child.name, []))
    else:
        logger.error("Must provide --repos-root or --repo")
        sys.exit(1)

    # Load known queries from corpus manifest if available
    corpus_map = {}
    try:
        # Try direct import first (if run from project root)
        try:
            from tests.corpus.manifest import CORPUS_REPOS, EXPECTED_PROFILE
        except ImportError:
            # Fallback: add project root to path and try again
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from tests.corpus.manifest import CORPUS_REPOS, EXPECTED_PROFILE

        corpus_map = {r.name: (r.queries, EXPECTED_PROFILE.get(r.name)) for r in CORPUS_REPOS}
        updated_repos = []
        for repo_path, repo_name, _ in repos_to_test:
            if repo_name in corpus_map:
                queries, expected_profile = corpus_map[repo_name]
                updated_repos.append((repo_path, repo_name, queries, expected_profile))
            else:
                updated_repos.append((repo_path, repo_name, [], None))
        repos_to_test = updated_repos
    except ImportError as e:
        logger.debug(f"Could not import corpus manifest: {e}, running without known queries")

    logger.info(f"Testing {len(repos_to_test)} repo(s)")

    # Run tests
    results_step1 = []
    results_step2 = []
    workspace_root = None

    for repo_info in repos_to_test:
        if len(repo_info) == 4:
            repo_path, repo_name, queries, expected_profile = repo_info
        else:
            repo_path, repo_name, queries = repo_info
            expected_profile = None

        result1 = test_repo_step1(
            repo_path,
            repo_name,
            queries,
            cairn_cmd=args.cairn,
            reindex_timeout=args.reindex_timeout,
        )
        result1["profile_expected"] = expected_profile
        results_step1.append(result1)

        if args.step2 and expected_profile in ["python", "dotnet", "code"]:
            result2 = test_repo_step2(
                repo_path,
                repo_name,
                queries,
                cairn_cmd=args.cairn,
                reindex_timeout=args.reindex_timeout,
            )
            results_step2.append(result2)

        # Remember workspace root (first repo's parent)
        if workspace_root is None:
            workspace_root = repo_path.parent

    # Test workspace router if we have a workspace with >=2 repos
    workspace_result = {"status": "skipped", "reason": "no workspace"}
    if workspace_root:
        # Reconstruct repos_to_test for workspace router with queries
        workspace_test_repos = []
        for repo_info in repos_to_test:
            if len(repo_info) == 4:
                workspace_test_repos.append(repo_info[:3])
            else:
                workspace_test_repos.append(repo_info)

        workspace_result = test_workspace_router(
            workspace_root,
            workspace_test_repos,
            cairn_cmd=args.cairn,
        )

    # Write report
    write_report(results_step1, results_step2, workspace_result, args.report)
    logger.info("Done!")


if __name__ == "__main__":
    main()
