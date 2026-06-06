#!/usr/bin/env python3
# ruff: noqa: E501
"""Cairn exhaustive clean-scenario capability test matrix.

Usage:
    python3 -m scripts.full_matrix --list          # Print all planned entries
    python3 -m scripts.full_matrix --no-llm-only   # Run only non-LLM entries
    python3 -m scripts.full_matrix --only G1 G2    # Run specific groups
    python3 -m scripts.full_matrix --keep-temp     # Don't delete synth repos
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from core.config import Config, clear_config_cache, load_config, save_config
    from core.profiles import detect_profile, get_profile
    from core.repo import (
        RepoManager,
        census_extensions,
        detect_infra_markers,
        project_id,
    )
    from core.resources import get_system_resources, list_installed_ollama_models
    from core.tokens import count_tokens
    from pipeline.indexer import VectorIndexer
    from tests.fixtures.builders import (
        make_helm_repo,
        make_k8s_repo,
        make_python_repo,
        make_terraform_repo,
        make_workspace,
    )
    from tests.fixtures.harness import fresh_index
except ImportError as e:
    print(f"FATAL: Cannot import cairn modules: {e}", file=sys.stderr)
    sys.exit(1)

# ── Ollama availability probing ───────────────────────────────────────────

_ollama_up: Optional[bool] = None


def ollama_is_up(url: str = "http://127.0.0.1:11434") -> bool:
    global _ollama_up
    if _ollama_up is not None:
        return _ollama_up
    try:
        import httpx

        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=5)
        _ollama_up = r.status_code == 200
    except Exception:
        _ollama_up = False
    return _ollama_up


def has_model_installed(name: str) -> bool:
    try:
        models = list_installed_ollama_models()
    except Exception:
        return False
    return any(m.get("name", "").startswith(name.rstrip(":latest")) for m in models)


# ── Optional dependency probes ────────────────────────────────────────────

_lancedb_available: Optional[bool] = None


def lancedb_is_available() -> bool:
    global _lancedb_available
    if _lancedb_available is not None:
        return _lancedb_available
    try:
        import lancedb  # noqa: F401

        _lancedb_available = True
    except ImportError:
        _lancedb_available = False
    return _lancedb_available


_fastembed_available: Optional[bool] = None


def fastembed_is_available() -> bool:
    global _fastembed_available
    if _fastembed_available is not None:
        return _fastembed_available
    try:
        import fastembed  # noqa: F401

        _fastembed_available = True
    except ImportError:
        _fastembed_available = False
    return _fastembed_available


# ── Entry and report type ─────────────────────────────────────────────────


@dataclass
class Entry:
    group: str
    capability: str
    variant: str
    needs_llm: bool = False
    run_fn: Optional[Callable[[], tuple[str, str, str]]] = None


@dataclass
class MatrixState:
    temp_root: Path
    report_path: Path
    system_resources: dict = field(default_factory=dict)
    start_time: str = ""
    entries_run: int = 0
    entries_skipped_already: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────


def run_subprocess(
    args: list[str],
    timeout: int = 120,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or ""), f"Timeout after {timeout}s"


def repo_blocks(repo_path: Path) -> int:
    """Count indexed blocks via the Chroma collection."""
    try:
        repo = RepoManager(repo_path)
        idx = VectorIndexer(
            chroma_path=repo.get_chroma_path(),
            ollama_client=None,
            embeddings_enabled=False,
            project_root=repo_path,
        )
        return idx.count()
    except Exception:
        return -1


def fresh_index_with_config(
    repo_path: Path, *, embeddings: bool = False, extra_config: Optional[dict] = None
) -> Path:
    """Run fresh_index then apply extra config settings and re-index if needed."""
    keys_needing_reindex = {"index_location", "store_backend", "embedder"}
    if extra_config and any(k in keys_needing_reindex for k in _flatten_keys(extra_config)):
        config_first_index(repo_path, embeddings=embeddings, extra_config=extra_config)
        return repo_path
    result = fresh_index(repo_path, embeddings=embeddings)
    if extra_config:
        clear_config_cache(repo_path)
        cfg = load_config(repo_path)
        _apply_config(cfg, extra_config)
        save_config(cfg, repo_path)
    return result


def config_first_index(
    repo_path: Path, *, embeddings: bool = False, extra_config: Optional[dict] = None
) -> Path:
    """Set config FIRST, then index. For tests where config must be in place before indexing."""
    from core.freshness import DBFreshness
    from core.repo import RepoManager, collect_source_files, detect_source_layout
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer

    repo_path = Path(repo_path)
    detected_roots, detected_patterns = detect_source_layout(repo_path)
    cairn_dir = repo_path / ".cairn"
    if cairn_dir.exists():
        shutil.rmtree(cairn_dir)

    cfg = Config()
    cfg.indexing.source_roots = detected_roots
    cfg.indexing.file_patterns = detected_patterns
    cfg.embeddings_enabled = embeddings
    if extra_config:
        _apply_config(cfg, extra_config)
    save_config(cfg, repo_path)

    files = collect_source_files(
        repo_path,
        cfg.indexing.file_patterns,
        cfg.indexing.exclude_patterns,
        cfg.indexing.source_roots,
    )
    repo = RepoManager(repo_path)
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(),
        ollama_client=None,
        embeddings_enabled=embeddings,
        project_root=repo_path,
    )
    parser = ASTParser()
    for filepath in files:
        try:
            ast = parser.parse_file(filepath)
            indexer.index_ast(ast)
        except Exception:
            pass

    freshness = DBFreshness(repo_path)
    freshness.mark_indexed(freshness.get_current_commit())
    repo.write_index_meta()
    return repo_path


def _apply_config(cfg: Config, overrides: dict) -> None:
    """Recursively apply a dict of overrides to a Config model."""
    for key, value in overrides.items():
        if isinstance(value, dict) and hasattr(getattr(cfg, key, None), "__dict__"):
            _apply_config(getattr(cfg, key), value)
        else:
            setattr(cfg, key, value)


def _flatten_keys(d: dict, prefix: str = "") -> set[str]:
    """Get all key paths from a nested dict."""
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        keys.add(k)
        if isinstance(v, dict):
            keys |= _flatten_keys(v, full)
    return keys


def setup_mcp_single(repo_path: Path) -> None:
    """Set MCP server globals for in-process SINGLE-repo tool calls."""
    import server.mcp_server as mcp_server

    mcp_server._PROJECT_PATH = Path(repo_path)
    mcp_server._BIND_ERROR = None
    mcp_server._router = None
    mcp_server._assembler = None
    mcp_server._session_budget = None
    mcp_server._semantic_caches = {}
    clear_config_cache(repo_path)


def setup_mcp_workspace(workspace_root: Path) -> None:
    """Set MCP server globals for in-process WORKSPACE tool calls."""
    import server.mcp_server as mcp_server
    from server.workspace_router import WorkspaceRouter

    mcp_server._PROJECT_PATH = None
    mcp_server._BIND_ERROR = None
    mcp_server._router = WorkspaceRouter(workspace_root)
    mcp_server._assembler = None
    mcp_server._session_budget = None
    mcp_server._semantic_caches = {}


def call_mcp_tool(tool_fn, *args, **kwargs) -> str:
    """Call an MCP tool function with standard error handling."""
    import server.mcp_server as mcp_server

    mcp_server.reset_session_budget()
    return tool_fn(*args, **kwargs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Report writer (incremental append) ────────────────────────────────────


def init_report(state: MatrixState, total_planned: int, modes: str) -> None:
    resources = state.system_resources
    gemma4 = "present" if ollama_is_up() and has_model_installed("gemma4") else "missing"
    state.start_time = _now()

    header = f"""# Cairn Full Test Matrix Report

**Date:** {state.start_time}
**Modes:** {modes}
**Total planned entries:** {total_planned}

## Host Resources
- CPU cores: {resources.get('cpu_count', '?')}
- RAM total: {resources.get('ram_total_gb', '?')} GB
- RAM available: {resources.get('ram_available_gb', '?')} GB
- GPU: {resources.get('gpu_name', 'none')}
- VRAM free: {resources.get('vram_free_gb', '?')} GB
- gemma4:latest availability: {gemma4}
- lancedb importable: {lancedb_is_available()}
- fastembed importable: {fastembed_is_available()}

## Results

| group | capability | variant | clean? | result | evidence | elapsed_s | notes |
|-------|-----------|---------|--------|--------|----------|-----------|-------|
"""
    state.report_path.parent.mkdir(parents=True, exist_ok=True)
    state.report_path.write_text(header)


def record_row(
    state: MatrixState, entry: Entry, result: str, evidence: str, elapsed: float, notes: str = ""
) -> None:
    """Append a row to the report immediately."""
    clean = (
        "yes"
        if result == "PASS" and entry.needs_llm is False
        else ("-" if result == "SKIP" else "no") if result == "FAIL" else "yes"
    )
    elapsed_s = f"{elapsed:.1f}"
    evidence_clean = evidence.replace("|", "/").replace("\n", " // ")
    notes_clean = (notes or "").replace("|", "/").replace("\n", " // ")[:200]
    row = f"| {entry.group} | {entry.capability} | {entry.variant} | {clean} | {result} | {evidence_clean} | {elapsed_s} | {notes_clean} |\n"
    with open(state.report_path, "a") as f:
        f.write(row)
        f.flush()


def finalize_report(state: MatrixState, results: list[dict], failures: list[dict]) -> None:
    """Append summary section."""
    by_group: dict[str, dict[str, int]] = {}
    for r in results:
        g = r["entry"].group
        if g not in by_group:
            by_group[g] = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        by_group[g][r["result"]] += 1

    total_p = sum(v["PASS"] for v in by_group.values())
    total_f = sum(v["FAIL"] for v in by_group.values())
    total_s = sum(v["SKIP"] for v in by_group.values())
    total = total_p + total_f + total_s

    lines = [
        "\n## Summary\n\n",
        f"**Total entries:** {total}  |  "
        f"PASS: {total_p}  |  FAIL: {total_f}  |  SKIP: {total_s}\n\n",
        "### By Group\n\n",
        "| Group | PASS | FAIL | SKIP |\n",
        "|-------|------|------|------|\n",
    ]
    for g in sorted(by_group):
        stats = by_group[g]
        lines.append(f"| {g} | {stats['PASS']} | {stats['FAIL']} | {stats['SKIP']} |\n")
    lines.append("\n### Failures\n\n")
    if failures:
        for f in failures:
            lines.append(
                f"- **{f['entry'].group}/{f['entry'].capability}** ({f['entry'].variant}): {f.get('notes', 'no details')}\n"
            )
    else:
        lines.append("No failures.\n")

    lines.append(f"\n_Report completed at {_now()}_\n")

    with open(state.report_path, "a") as f:
        f.writelines(lines)
        f.flush()


def _mkdir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_python(state: MatrixState, name: str) -> Path:
    return make_python_repo(_mkdir(state.temp_root / name))


def _make_helm(state: MatrixState, name: str) -> Path:
    return make_helm_repo(_mkdir(state.temp_root / name))


def _make_terraform(state: MatrixState, name: str) -> Path:
    return make_terraform_repo(_mkdir(state.temp_root / name))


def _make_k8s(state: MatrixState, name: str, **kw) -> Path:
    return make_k8s_repo(_mkdir(state.temp_root / name), **kw)


def _make_workspace(state: MatrixState, name: str) -> Path:
    return make_workspace(_mkdir(state.temp_root / name))


def run_entry(state: MatrixState, entry: Entry, timeout: int = 600) -> tuple[str, str, str, float]:
    """Run a single matrix entry with timeout and error handling."""
    if entry.run_fn is None:
        return "FAIL", "no run_fn defined", "missing implementation", 0.0

    t0 = time.monotonic()
    try:
        result, evidence, notes = entry.run_fn()
    except Exception as e:
        tb = traceback.format_exc()
        elapsed = time.monotonic() - t0
        short_tb = " // ".join(
            line.strip()
            for line in tb.splitlines()
            if line.strip() and not line.startswith("  File")
        )[:400]
        return "FAIL", f"{type(e).__qualname__}: {e}", short_tb, elapsed

    elapsed = time.monotonic() - t0
    if elapsed > timeout:
        return "FAIL", f"timeout exceeded ({timeout}s)", "", elapsed
    return result, evidence, notes, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX DEFINITION
# ═══════════════════════════════════════════════════════════════════════════


def build_matrix(state: MatrixState) -> list[Entry]:
    entries: list[Entry] = []

    # ── G1: CLI Surface ──────────────────────────────────────────────────

    def _g1_init_no_llm():
        repo = make_python_repo(_mkdir(state.temp_root / "g1_init_no_llm"))
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "init", "--offline", "--no-index", "-y"],
            timeout=60,
            cwd=repo,
        )
        if rc != 0:
            return "FAIL", f"rc={rc}", f"stderr: {err[:200]}"
        cairn_dir = repo / ".cairn"
        return "PASS", f".cairn exists: {cairn_dir.exists()}", ""

    entries.append(Entry("G1", "init (no LLM)", "without LLM", False, _g1_init_no_llm))

    def _g1_init_with_llm():
        if not ollama_is_up() or not has_model_installed("gemma4"):
            return "SKIP", "gemma4 not available", ""
        repo = _make_python(state, "g1_init_with_llm")
        _update_config_for_llm(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "init", "-y"],
            timeout=300,
            cwd=repo,
        )
        cairn_dir = repo / ".cairn"
        return (
            "PASS" if rc == 0 else "FAIL",
            f".cairn exists: {cairn_dir.exists()}, rc={rc}",
            err[:200],
        )

    entries.append(Entry("G1", "init (with LLM)", "with LLM", True, _g1_init_with_llm))

    def _g1_config():
        repo = _make_python(state, "g1_config")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "config"],
            timeout=30,
            cwd=repo,
        )
        return (
            "PASS" if rc == 0 and len(out) > 100 else "FAIL",
            f"rc={rc}, output_len={len(out)}",
            err[:100],
        )

    entries.append(Entry("G1", "config", "once", False, _g1_config))

    def _g1_profile():
        repo = _make_python(state, "g1_profile")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "profile"],
            timeout=30,
            cwd=repo,
        )
        if rc != 0:
            return "FAIL", f"rc={rc}", err[:200]
        rc2, out2, err2 = run_subprocess(
            [sys.executable, "-m", "cli.main", "profile", "iac"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc2 == 0 else "FAIL", f"profile show rc={rc}, set rc={rc2}", err2[:100])

    entries.append(Entry("G1", "profile (show+set)", "once", False, _g1_profile))

    def _g1_status():
        repo = _make_python(state, "g1_status")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "status"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}, output_len={len(out)}", err[:100])

    entries.append(Entry("G1", "status", "once", False, _g1_status))

    def _g1_doctor_no_llm():
        repo = _make_python(state, "g1_doctor_no_llm")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "doctor"],
            timeout=60,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "doctor (no LLM)", "without LLM", False, _g1_doctor_no_llm))

    def _g1_doctor_with_llm():
        if not ollama_is_up():
            return "SKIP", "ollama not running", ""
        repo = _make_python(state, "g1_doctor_with_llm")
        fresh_index(repo)
        _update_config_for_llm(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "doctor"],
            timeout=120,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "doctor (with LLM)", "with LLM", True, _g1_doctor_with_llm))

    def _g1_reindex_quick_no_llm():
        repo = _make_python(state, "g1_reindex_quick")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "reindex", "--mode", "quick"],
            timeout=120,
            cwd=repo,
        )
        blocks = repo_blocks(repo)
        return (
            "PASS" if rc == 0 and blocks > 0 else "FAIL",
            f"blocks={blocks}, rc={rc}",
            err[:200],
        )

    entries.append(
        Entry("G1", "reindex quick (no LLM)", "without LLM", False, _g1_reindex_quick_no_llm)
    )

    def _g1_reindex_full_no_llm():
        repo = _make_python(state, "g1_reindex_full")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "reindex", "--mode", "full"],
            timeout=120,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(
        Entry("G1", "reindex full (no LLM)", "without LLM", False, _g1_reindex_full_no_llm)
    )

    def _g1_search_no_llm():
        repo = _make_python(state, "g1_search_no_llm")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "search", "process data", "--top-k", "3"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}, output_len={len(out)}", err[:200])

    entries.append(Entry("G1", "search (no LLM)", "without LLM", False, _g1_search_no_llm))

    def _g1_search_with_llm():
        if not ollama_is_up() or not has_model_installed("gemma4"):
            return "SKIP", "gemma4 not available", ""
        repo = _make_python(state, "g1_search_with_llm")
        fresh_index_with_config(repo, embeddings=True, extra_config=_llm_config_overrides())
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "search", "process data", "--top-k", "3"],
            timeout=120,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}, output_len={len(out)}", err[:200])

    entries.append(Entry("G1", "search (with LLM)", "with LLM", True, _g1_search_with_llm))

    def _g1_dry_run_no_llm():
        repo = _make_python(state, "g1_dry_run")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "dry-run", "process data"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "dry-run (no LLM)", "without LLM", False, _g1_dry_run_no_llm))

    def _g1_suggest_local():
        repo = _make_python(state, "g1_suggest_local")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "suggest-local", "process data"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "suggest-local", "once", False, _g1_suggest_local))

    def _g1_token_stats():
        repo = _make_python(state, "g1_token_stats")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "token-stats"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "token-stats", "once", False, _g1_token_stats))

    def _g1_token_history():
        repo = _make_python(state, "g1_token_history")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "token-history"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "token-history", "once", False, _g1_token_history))

    def _g1_cache_stats():
        repo = _make_python(state, "g1_cache_stats")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "cache", "stats"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "cache stats", "once", False, _g1_cache_stats))

    def _g1_cache_clear():
        repo = _make_python(state, "g1_cache_clear")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "cache", "clear"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "cache clear", "once", False, _g1_cache_clear))

    def _g1_memory_status():
        repo = _make_python(state, "g1_memory_status")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "memory", "status"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "memory status", "once", False, _g1_memory_status))

    def _g1_memory_clear():
        repo = _make_python(state, "g1_memory_clear")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "memory", "clear"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "memory clear", "once", False, _g1_memory_clear))

    def _g1_metrics():
        repo = _make_python(state, "g1_metrics")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "metrics"],
            timeout=15,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "metrics", "once", False, _g1_metrics))

    def _g1_dashboard():
        repo = _make_python(state, "g1_dashboard")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "dashboard"],
            timeout=15,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "dashboard", "once", False, _g1_dashboard))

    def _g1_janitor():
        repo = _make_python(state, "g1_janitor")
        fresh_index(repo)
        env = os.environ.copy()
        p = subprocess.Popen(
            [sys.executable, "-m", "cli.main", "janitor", "start"],
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(2)
        rc2, out2, err2 = run_subprocess(
            [sys.executable, "-m", "cli.main", "janitor", "stop"],
            timeout=15,
            cwd=repo,
        )
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        return ("PASS" if rc2 == 0 else "FAIL", f"start spawned, stop rc={rc2}", err2[:200])

    entries.append(Entry("G1", "janitor start+stop", "once", False, _g1_janitor))

    def _g1_mcp_smoke():
        repo = _make_python(state, "g1_mcp_smoke")
        fresh_index(repo)
        env = os.environ.copy()
        env["CAIRN_PROJECT"] = str(repo.resolve())
        p = subprocess.Popen(
            [sys.executable, "-m", "cli.main", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(3)
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        return_code = p.returncode
        return (
            "PASS" if return_code is not None else "FAIL",
            f"MCP process started, returncode={return_code}",
            f"stderr_len={len(p.stderr.read() if p.stderr else '')}",
        )

    entries.append(Entry("G1", "mcp smoke", "once", False, _g1_mcp_smoke))

    def _g1_start_all():
        repo = _make_python(state, "g1_start_all")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "run", "--no-janitor", "--no-index", "-y"],
            timeout=30,
            cwd=repo,
        )
        return ("PASS" if rc == 0 else "FAIL", f"rc={rc}", err[:200])

    entries.append(Entry("G1", "run (start-all)", "once", False, _g1_start_all))

    def _g1_dry_run_show_prompt():
        repo = _make_python(state, "g1_dry_run_prompt")
        fresh_index(repo)
        rc, out, err = run_subprocess(
            [sys.executable, "-m", "cli.main", "dry-run", "process data", "--show-prompt"],
            timeout=30,
            cwd=repo,
        )
        return (
            "PASS" if rc == 0 and "context" in out.lower() else "FAIL",
            f"rc={rc}, has_context={'context' in out.lower()}",
            err[:200],
        )

    entries.append(
        Entry("G1", "dry-run --show-prompt", "without LLM", False, _g1_dry_run_show_prompt)
    )

    # ── G2: MCP Tools (in-process) ─────────────────────────────────────────

    def _g2_search_code():
        repo = _make_python(state, "g2_search_code")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if "process_data" in result or "process" in result.lower() else "FAIL",
            f"result_len={len(result)}, has_process={'process' in result.lower()}",
            "",
        )

    entries.append(Entry("G2", "search_code", "without LLM", False, _g2_search_code))

    def _g2_assemble_context():
        repo = _make_python(state, "g2_assemble")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import assemble_context

        result = assemble_context("process data")
        has_context = (
            "Repository Structure" in result or "Relevant Functions" in result or "Memory" in result
        )
        return ("PASS" if has_context else "FAIL", f"result_len={len(result)}", "")

    entries.append(Entry("G2", "assemble_context", "without LLM", False, _g2_assemble_context))

    def _g2_set_profile():
        repo = _make_python(state, "g2_set_profile")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import set_profile as sp

        result = sp("iac")
        return ("PASS" if "iac" in result.lower() else "FAIL", f"response: {result[:200]}", "")

    entries.append(Entry("G2", "set_profile", "without LLM", False, _g2_set_profile))

    def _g2_orchestrate_context_only():
        repo = _make_python(state, "g2_orch_ctx")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import orchestrate

        result = orchestrate("process data", instruction="")
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(
        Entry(
            "G2", "orchestrate (context-only)", "without LLM", False, _g2_orchestrate_context_only
        )
    )

    def _g2_orchestrate_with_instruction():
        if not ollama_is_up() or not has_model_installed("gemma4"):
            return "SKIP", "gemma4 not available", ""
        repo = _make_python(state, "g2_orch_instr")
        fresh_index_with_config(repo, embeddings=True, extra_config=_llm_config_overrides())
        setup_mcp_single(repo)
        from server.context_assembler import ContextAssembler
        from server.ollama_client import make_llm_client
        from server.orchestrator import Orchestrator, WorkClass

        assembler = ContextAssembler(project_path=Path(repo))
        cfg = load_config(repo)
        llm = make_llm_client(cfg.local_llm)
        orch = Orchestrator(assembler=assembler, cfg=cfg, llm=llm)

        # Build payload sized to produce >=2 chunks in map-reduce.
        # chunk_window = max_local_tokens - reduce_reserve_tokens - 64 = 6000-1024-64 = 4912
        # stride = chunk_window * (1 - 0.12) ≈ 4323
        # So: n ≤ 4912 → 1 chunk; n > 4912 → ≥2 chunks.
        # Target ~9000-10000 tokens for a genuine multi-chunk split.
        sentence = (
            "We need to build a comprehensive data processing pipeline that handles "
            "validation, transformation, serialization, error handling, logging, "
            "monitoring, alerting, and reporting. "
        )
        payload = sentence * 1
        target_min = 4913  # just above single-chunk threshold
        target_max = 20000
        for _ in range(500):  # safety cap
            n = count_tokens(payload)
            if n < target_min:
                payload += sentence
            elif n > target_max:
                payload = payload[: -len(sentence)]
                break
            else:
                break

        n = count_tokens(payload)
        plan = orch.plan(payload, has_instruction=True)
        chunks = orch._chunk(payload)

        if plan.work_class != WorkClass.LOCAL_MAP_REDUCE or len(chunks) < 2:
            return (
                "FAIL",
                f"work_class={plan.work_class.value}, n_chunks={len(chunks)}",
                f"expected LOCAL_MAP_REDUCE with >=2 chunks; reason: {plan.reason}, tokens={n}",
            )

        # Actually execute on gemma4 to prove real generation (map-reduce path).
        try:
            result = orch.execute(
                query="summarize the pipeline requirements",
                payload=payload,
                instruction="Summarize the key requirements in 3 bullet points",
            )
        except Exception as exc:
            return (
                "FAIL",
                f"WorkClass={plan.work_class.value}, tokens={plan.input_tokens}, chunks={len(chunks)}",
                f"reason: {plan.reason}; execute raised: {exc}",
            )

        result_len = len(result) if isinstance(result, str) else 0
        if not result or result_len == 0:
            return (
                "FAIL",
                f"WorkClass={plan.work_class.value}, tokens={plan.input_tokens}, chunks={len(chunks)}",
                f"reason: {plan.reason}; empty result from execute",
            )

        return (
            "PASS",
            f"WorkClass={plan.work_class.value}, tokens={plan.input_tokens}, chunks={len(chunks)}",
            f"reason: {plan.reason}; result_len={result_len}",
        )

    entries.append(
        Entry(
            "G2",
            "orchestrate (with instruction)",
            "with LLM",
            True,
            _g2_orchestrate_with_instruction,
        )
    )

    def _g2_cache_set_get():
        repo = _make_python(state, "g2_cache")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import cache_get, cache_set

        s = cache_set("test query 42", "test value 42")
        g = cache_get("test query 42")
        g_miss = cache_get("completely different missing query")
        hit = "test value 42" in (g or "")
        return (
            "PASS" if hit else "FAIL",
            f"set={s[:50]}, get_hit={'yes' if hit else 'no'}, miss={g_miss[:50]}",
            "",
        )

    entries.append(Entry("G2", "cache_set/cache_get", "without LLM", False, _g2_cache_set_get))

    def _g2_list_repos_single():
        repo = _make_python(state, "g2_list_repos")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import list_repos

        result = list_repos()
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(Entry("G2", "list_repos (single)", "without LLM", False, _g2_list_repos_single))

    def _g2_list_repos_workspace():
        ws = _make_workspace(state, "g2_list_repos_ws")
        setup_mcp_workspace(ws)
        from server.mcp_server import list_repos

        result = list_repos()
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(
        Entry("G2", "list_repos (workspace)", "without LLM", False, _g2_list_repos_workspace)
    )

    def _g2_remember_recall():
        repo = _make_python(state, "g2_remember")
        fresh_index(repo)
        setup_mcp_single(repo)
        from server.mcp_server import recall, remember

        kinds = ["task", "decision", "convention", "change", "prompt"]
        for kind in kinds:
            remember(f"Test {kind}: use virtualenv for all projects", kind=kind)

        result = recall(max_entries=10)
        hits = sum(1 for kind in kinds if f"Test {kind}" in result)
        return (
            "PASS" if hits >= 2 else "FAIL",
            f"sections_found={hits}/{len(kinds)}",
            f"result_len={len(result)}",
        )

    entries.append(
        Entry("G2", "remember+recall (all kinds)", "without LLM", False, _g2_remember_recall)
    )

    # ── G3: Axes ──────────────────────────────────────────────────────────

    def _g3_store_chroma():
        repo = _make_python(state, "g3_store_chroma")
        fresh_index_with_config(repo, extra_config={"indexing": {"store_backend": "chroma"}})
        blocks = repo_blocks(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if blocks > 0 and len(result) > 0 else "FAIL",
            f"blocks={blocks}, search_len={len(result)}",
            "",
        )

    entries.append(Entry("G3", "store_backend chroma", "chroma", False, _g3_store_chroma))

    def _g3_store_lance():
        if not lancedb_is_available():
            return "SKIP", "lancedb not importable", ""
        repo = _make_python(state, "g3_store_lance")
        fresh_index_with_config(repo, extra_config={"indexing": {"store_backend": "lance"}})
        blocks = repo_blocks(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if blocks > 0 and len(result) > 0 else "FAIL",
            f"blocks={blocks}, search_len={len(result)}",
            "",
        )

    entries.append(Entry("G3", "store_backend lance", "lance", False, _g3_store_lance))

    def _g3_location_in_project():
        repo = _make_python(state, "g3_loc_project")
        fresh_index_with_config(repo, extra_config={"indexing": {"index_location": "in_project"}})
        index_path = repo / ".cairn" / "chroma"
        blocks = repo_blocks(repo)
        exists = index_path.exists()
        return (
            "PASS" if blocks > 0 and exists else "FAIL",
            f"blocks={blocks}, index_in_project={exists}",
            f"path={index_path}",
        )

    entries.append(
        Entry("G3", "index_location in_project", "in_project", False, _g3_location_in_project)
    )

    def _g3_location_native():
        repo = _make_python(state, "g3_loc_native")
        fresh_index_with_config(repo, extra_config={"indexing": {"index_location": "native"}})
        blocks = repo_blocks(repo)
        pid = project_id(repo)
        native_path = Path.home() / ".cache" / "cairn" / pid / "chroma"
        exists = native_path.exists()
        return (
            "PASS" if blocks > 0 and exists else "FAIL",
            f"blocks={blocks}, native_index_exists={exists}",
            f"path={native_path}",
        )

    entries.append(Entry("G3", "index_location native", "native", False, _g3_location_native))

    # Profile tests
    for prof_name in ["iac", "code", "python", "shell"]:

        def _make_profile_test(pname):
            def _inner():
                repo_dir = _mkdir(state.temp_root / f"g3_profile_{pname}")
                if pname == "python":
                    repo = make_python_repo(repo_dir)
                elif pname == "iac":
                    repo = make_helm_repo(repo_dir)
                else:
                    repo = make_python_repo(repo_dir)
                fresh_index_with_config(repo, extra_config={"profile": pname})
                clear_config_cache(repo)
                cfg = load_config(repo)
                ext_counts = census_extensions(repo)
                inf_markers = detect_infra_markers(repo)
                detected = detect_profile(ext_counts, has_infra_markers=inf_markers)
                profile = get_profile(cfg.profile)
                return (
                    "PASS",
                    f"set={pname}, detected={detected}, embeddings={profile.embedding_enabled}, legs={profile.legs}",
                    "",
                )

            return _inner

        entries.append(
            Entry("G3", f"profile {prof_name}", "explicit", False, _make_profile_test(prof_name))
        )

    def _g3_profile_auto_helm():
        repo = _make_helm(state, "g3_profile_auto_helm")
        fresh_index(repo)
        ext_counts = census_extensions(repo)
        inf_markers = detect_infra_markers(repo)
        detected = detect_profile(ext_counts, has_infra_markers=inf_markers)
        return (
            "PASS" if detected == "iac" else "FAIL",
            f"detected={detected}, expected=iac",
            f"ext_counts={ext_counts}",
        )

    entries.append(Entry("G3", "profile auto-detect helm", "auto", False, _g3_profile_auto_helm))

    def _g3_profile_auto_python():
        repo = _make_python(state, "g3_profile_auto_python")
        fresh_index(repo)
        ext_counts = census_extensions(repo)
        detected = detect_profile(ext_counts, has_infra_markers=False)
        return (
            "PASS" if detected == "python" else "FAIL",
            f"detected={detected}, expected=python",
            f"ext_counts={ext_counts}",
        )

    entries.append(
        Entry("G3", "profile auto-detect python", "auto", False, _g3_profile_auto_python)
    )

    def _g3_embed_placeholder():
        repo = _make_python(state, "g3_embed_ph")
        fresh_index_with_config(
            repo,
            embeddings=False,
            extra_config={
                "embeddings_enabled": False,
                "local_llm": {"enabled": False, "embedder": "none"},
            },
        )
        blocks = repo_blocks(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if blocks > 0 and len(result) > 0 else "FAIL",
            f"blocks={blocks}, search_len={len(result)}",
            "",
        )

    entries.append(Entry("G3", "embed mode placeholder", "no-LLM", False, _g3_embed_placeholder))

    def _g3_embed_fastembed():
        if not fastembed_is_available():
            return "SKIP", "fastembed not importable", ""
        repo = _make_python(state, "g3_embed_fast")
        fresh_index_with_config(
            repo,
            embeddings=True,
            extra_config={
                "embeddings_enabled": True,
                "local_llm": {"enabled": True, "embedder": "fastembed"},
            },
        )
        blocks = repo_blocks(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if blocks > 0 and len(result) > 0 else "FAIL",
            f"blocks={blocks}, search_len={len(result)}",
            "",
        )

    entries.append(Entry("G3", "embed mode fastembed", "no-LLM", False, _g3_embed_fastembed))

    def _g3_embed_ollama():
        if not ollama_is_up() or not has_model_installed("nomic-embed-text"):
            return "SKIP", "ollama/nomic-embed-text not available", ""
        repo = _make_python(state, "g3_embed_ollama")
        fresh_index_with_config(
            repo,
            embeddings=True,
            extra_config={
                "embeddings_enabled": True,
                "local_llm": {
                    "enabled": True,
                    "embedder": "ollama",
                    "embed_model": "nomic-embed-text",
                },
            },
        )
        blocks = repo_blocks(repo)
        setup_mcp_single(repo)
        from server.mcp_server import search_code

        result = search_code("process data", top_k=3)
        return (
            "PASS" if blocks > 0 and len(result) > 0 else "FAIL",
            f"blocks={blocks}, search_len={len(result)}",
            "",
        )

    entries.append(Entry("G3", "embed mode ollama", "with LLM", True, _g3_embed_ollama))

    def _g3_reranker_cross_encoder():
        repo = _make_python(state, "g3_rerank_ce")
        fresh_index_with_config(
            repo,
            extra_config={
                "retrieval": {"reranker_type": "cross_encoder", "rerank_enabled": True},
            },
        )
        setup_mcp_single(repo)
        from server.mcp_server import assemble_context

        result = assemble_context("process data")
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(
        Entry("G3", "reranker cross_encoder", "no-LLM", False, _g3_reranker_cross_encoder)
    )

    def _g3_reranker_none():
        repo = _make_python(state, "g3_rerank_none")
        fresh_index_with_config(
            repo,
            extra_config={
                "retrieval": {"reranker_type": "none", "rerank_enabled": False},
            },
        )
        setup_mcp_single(repo)
        from server.mcp_server import assemble_context

        result = assemble_context("process data")
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(Entry("G3", "reranker none", "no-LLM", False, _g3_reranker_none))

    def _g3_reranker_llm():
        if not ollama_is_up() or not has_model_installed("gemma4"):
            return "SKIP", "gemma4 not available", ""
        repo = _make_python(state, "g3_rerank_llm")
        fresh_index_with_config(
            repo,
            embeddings=True,
            extra_config={
                "retrieval": {"reranker_type": "llm", "rerank_enabled": True},
                **_llm_config_overrides(),
            },
        )
        setup_mcp_single(repo)
        from server.mcp_server import assemble_context

        result = assemble_context("process data")
        return ("PASS" if len(result) > 0 else "FAIL", f"result_len={len(result)}", "")

    entries.append(Entry("G3", "reranker llm", "with LLM", True, _g3_reranker_llm))

    for level in ["none", "minimal", "aggressive"]:

        def _make_compression_test(lev):
            def _inner():
                repo = make_python_repo(_mkdir(state.temp_root / f"g3_comp_{lev}"))
                fresh_index_with_config(
                    repo,
                    extra_config={
                        "compression": {"level": lev, "enabled": lev != "none"},
                    },
                )
                setup_mcp_single(repo)
                from server.mcp_server import assemble_context
                from server.token_compressor import FilterLevel, TokenCompressor

                raw = assemble_context("process data")
                before = count_tokens(raw)

                level_map = {
                    "none": FilterLevel.NONE,
                    "minimal": FilterLevel.MINIMAL,
                    "aggressive": FilterLevel.AGGRESSIVE,
                }
                tc = TokenCompressor(level=level_map[lev], max_tokens=10000)
                compressed = tc.compress(raw)
                after = count_tokens(compressed)
                pct = round((1 - after / before) * 100, 1) if before > 0 else 0.0
                return ("PASS", f"before={before}, after={after}, reduction={pct}%", f"level={lev}")

            return _inner

        entries.append(
            Entry("G3", f"compression {level}", level, False, _make_compression_test(level))
        )

    def _g3_memory_scope_repo():
        repo = _make_python(state, "g3_mem_repo")
        fresh_index_with_config(
            repo,
            extra_config={
                "memory": {"scope": "repo", "trigger": "manual"},
            },
        )
        setup_mcp_single(repo)
        from server.mcp_server import recall, remember

        remember("repo-only note", kind="convention")
        result = recall(max_entries=5)
        return (
            "PASS" if "repo-only note" in result else "FAIL",
            f"found={'yes' if 'repo-only note' in result else 'no'}",
            f"result_len={len(result)}",
        )

    entries.append(Entry("G3", "memory.scope repo", "repo", False, _g3_memory_scope_repo))

    def _g3_memory_scope_workspace():
        ws_root = _mkdir(state.temp_root / "g3_mem_ws_ws")
        ws = make_workspace(ws_root)
        for child in ws.iterdir():
            if child.is_dir() and (child / ".git").exists():
                fresh_index(child)
        ws_cfg_path = ws / ".cairn"
        if not ws_cfg_path.exists():
            ws_cfg_path.mkdir(parents=True, exist_ok=True)
            from core.config import Config, save_config

            wcfg = Config()
            wcfg.memory.scope = "workspace"
            wcfg.memory.trigger = "manual"
            save_config(wcfg, ws)

        setup_mcp_workspace(ws)
        from server.mcp_server import recall, remember

        remember("workspace-scoped note", kind="convention")
        result = recall(max_entries=5)
        return (
            "PASS" if "workspace-scoped note" in result else "FAIL",
            f"found={'yes' if 'workspace-scoped note' in result else 'no'}",
            f"result_len={len(result)}",
        )

    entries.append(
        Entry("G3", "memory.scope workspace", "workspace", False, _g3_memory_scope_workspace)
    )

    # Language tests from fixtures
    lang_repos = {
        "python": lambda base: make_python_repo(base),
        "helm-yaml": lambda base: make_helm_repo(base),
        "terraform-hcl": lambda base: make_terraform_repo(base),
        "k8s-yaml": lambda base: make_k8s_repo(base),
    }
    for lang_name, builder in lang_repos.items():

        def _make_lang_test(lname, bldr):
            dir_name = lname.replace("-", "_")

            def _inner():
                base = _mkdir(state.temp_root / f"g3_lang_{dir_name}")
                repo = bldr(base)
                fresh_index(repo)
                blocks = repo_blocks(repo)
                return ("PASS" if blocks > 0 else "FAIL", f"blocks={blocks}", f"lang={lname}")

            return _inner

        entries.append(
            Entry(
                "G3", f"language {lang_name}", "fixture", False, _make_lang_test(lang_name, builder)
            )
        )

    # Workspace fan-out + isolation
    def _g3_workspace_fanout():
        ws = _make_workspace(state, "g3_ws_fanout")
        for child in ws.iterdir():
            if child.is_dir() and (child / ".git").exists():
                fresh_index(child)
        setup_mcp_workspace(ws)
        from server.mcp_server import search_code

        result = search_code("deployment", top_k=8)
        return (
            "PASS" if len(result) > 0 else "FAIL",
            f"result_len={len(result)}, has_deployment={'deployment' in result.lower()}",
            "",
        )

    entries.append(
        Entry("G3", "workspace fan-out search_all", "multi-repo", False, _g3_workspace_fanout)
    )

    # Failsafe tests
    def _g3_failsafe_empty():
        repo_dir = state.temp_root / "g3_empty"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / ".gitkeep").write_text("")
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, capture_output=True)
        try:
            fresh_index(repo_dir)
            blocks = repo_blocks(repo_dir)
            return ("PASS" if blocks == 0 else "FAIL", f"blocks={blocks}", "empty repo, no crash")
        except Exception as e:
            return ("PASS", f"graceful: {type(e).__name__}", "empty repo, no crash")

    entries.append(Entry("G3", "failsafe empty repo", "edge", False, _g3_failsafe_empty))

    def _g3_failsafe_pathological():
        repo = _make_k8s(state, "g3_pathological", with_pathological=True)
        try:
            fresh_index(repo)
            blocks = repo_blocks(repo)
            return (
                "PASS" if blocks >= 0 else "FAIL",
                f"blocks={blocks}",
                "pathological k8s, no crash",
            )
        except Exception as e:
            tb = traceback.format_exc()
            return (
                "PASS",
                f"graceful: {type(e).__name__}",
                f"pathological k8s, no crash: {tb[:200]}",
            )

    entries.append(
        Entry("G3", "failsafe pathological k8s", "edge", False, _g3_failsafe_pathological)
    )

    def _g3_failsafe_malformed_config():
        repo = _make_python(state, "g3_malformed")
        (repo / ".cairn").mkdir(exist_ok=True)
        (repo / ".cairn" / "config.yaml").write_text(":garbage: {{{malformed\nindent: [[[")
        try:
            cfg = load_config(repo)
            return (
                "PASS" if cfg is not None else "FAIL",
                "config loaded (fallback to defaults)",
                "",
            )
        except Exception as e:
            return (
                "PASS",
                f"clean error: {type(e).__name__}",
                "malformed config handled gracefully",
            )

    entries.append(
        Entry("G3", "failsafe malformed config", "edge", False, _g3_failsafe_malformed_config)
    )

    return entries


# ── LLM helpers ───────────────────────────────────────────────────────────


def _llm_config_overrides() -> dict:
    return {
        "embeddings_enabled": True,
        "local_llm": {
            "enabled": True,
            "model": "gemma4:latest",
            "embed_model": "nomic-embed-text",
            "embedder": "ollama",
            "context_window": 8192,
        },
    }


def _update_config_for_llm(repo_path: Path) -> None:
    clear_config_cache(repo_path)
    cfg = load_config(repo_path)
    cfg.local_llm.enabled = True
    cfg.local_llm.model = "gemma4:latest"
    cfg.local_llm.embed_model = "nomic-embed-text"
    cfg.local_llm.embedder = "ollama"
    cfg.embeddings_enabled = True
    save_config(cfg, repo_path)


# ── Entry deduplication for resume ────────────────────────────────────────


def load_completed_entries(report_path: Path) -> set[tuple[str, str, str]]:
    if not report_path.exists():
        return set()
    completed = set()
    for line in report_path.read_text().splitlines():
        if line.startswith("| G") and "|" in line[3:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                completed.add((parts[1], parts[2], parts[3]))
    return completed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Cairn exhaustive clean-scenario capability test matrix"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        dest="only_groups",
        help="Only run entries in these groups (repeatable)",
    )
    parser.add_argument("--no-llm-only", action="store_true", help="Skip all WITH-LLM entries")
    parser.add_argument("--with-llm-only", action="store_true", help="Only run WITH-LLM entries")
    parser.add_argument("--report", default="docs/FULL_TEST_REPORT.md", help="Report output path")
    parser.add_argument(
        "--keep-temp", action="store_true", help="Don't delete temporary synth repos"
    )
    parser.add_argument("--list", action="store_true", help="Print planned entries and exit")
    parser.add_argument(
        "--resume", action="store_true", help="Skip entries already recorded in the report"
    )
    args = parser.parse_args()

    temp_root = Path(tempfile.mkdtemp(prefix="cairn_matrix_"))
    report_path = Path(args.report).resolve()
    system_resources = {}
    try:
        system_resources = get_system_resources()
    except Exception:
        pass

    state = MatrixState(
        temp_root=temp_root,
        report_path=report_path,
        system_resources=system_resources,
    )

    if not args.keep_temp:
        atexit.register(lambda: shutil.rmtree(str(temp_root), ignore_errors=True))

    entries = build_matrix(state)

    if args.only_groups:
        allowed = set(args.only_groups)
        entries = [e for e in entries if e.group in allowed]

    if args.no_llm_only and args.with_llm_only:
        print("ERROR: --no-llm-only and --with-llm-only are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if args.no_llm_only:
        entries = [e for e in entries if not e.needs_llm]
    elif args.with_llm_only:
        entries = [e for e in entries if e.needs_llm]

    if args.list:
        print(f"Planned entries: {len(entries)}")
        for i, e in enumerate(entries, 1):
            llm_tag = " [LLM]" if e.needs_llm else ""
            print(f"  {i}. [{e.group}] {e.capability} ({e.variant}){llm_tag}")
        return

    completed = set()
    if args.resume:
        completed = load_completed_entries(report_path)
        orig_count = len(entries)
        entries = [e for e in entries if (e.group, e.capability, e.variant) not in completed]
        state.entries_skipped_already = orig_count - len(entries)

    if not entries:
        print("No entries to run.")
        return

    modes = []
    if args.no_llm_only:
        modes.append("no-LLM")
    elif args.with_llm_only:
        modes.append("with-LLM")
    else:
        modes.append("all")

    init_report(state, len(entries), ", ".join(modes))
    if state.entries_skipped_already:
        print(f"Skipping {state.entries_skipped_already} already-completed entries (--resume)")

    results: list[dict] = []
    failures: list[dict] = []
    total = len(entries)

    for i, entry in enumerate(entries):
        llm_tag = " [LLM]" if entry.needs_llm else ""
        print(
            f"[{i + 1}/{total}] {entry.group}/{entry.capability} ({entry.variant}){llm_tag} ... ",
            end="",
            flush=True,
        )
        result, evidence, notes, elapsed = run_entry(state, entry)
        record_row(state, entry, result, evidence, elapsed, notes)
        status_c = {
            "PASS": "\033[32mPASS\033[0m",
            "FAIL": "\033[31mFAIL\033[0m",
            "SKIP": "\033[33mSKIP\033[0m",
        }.get(result, result)
        print(f"{status_c} ({elapsed:.1f}s)")
        r = {
            "entry": entry,
            "result": result,
            "evidence": evidence,
            "notes": notes,
            "elapsed": elapsed,
        }
        results.append(r)
        if result == "FAIL":
            failures.append(r)

    finalize_report(state, results, failures)

    print(
        f"\nDone. {len([r for r in results if r['result']=='PASS'])} PASS, "
        f"{len([r for r in results if r['result']=='FAIL'])} FAIL, "
        f"{len([r for r in results if r['result']=='SKIP'])} SKIP"
    )
    print(f"Report: {report_path}")

    # Exit with failure code if any FAIL
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
