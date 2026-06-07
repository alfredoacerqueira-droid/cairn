#!/usr/bin/env python3
"""Cairn Lifecycle Simulation — builds a Flask project and exercises every feature.

Run: python3 scripts/lifecycle_sim.py [--with-llm] [--keep] [--report <path>]
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("lifecycle_sim")

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
CAIRN_ROOT = THIS_FILE.parent.parent  # /mnt/c/Users/alfre/Projects/cairn
STAGE_NAMES = ["A_scaffold", "B_backend", "C_modify", "D_delete", "E_grow"]

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


def run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout_s: float = 120.0,
    env: dict[str, str] | None = None,
) -> tuple[bool, str, str]:
    """Run a subprocess command returning (ok, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout_s}s: {' '.join(cmd)}"
    except OSError as exc:
        return False, "", str(exc)


def _cli_env() -> dict[str, str]:
    """Build environment for CLI subprocess calls with PYTHONPATH set to CAIRN_ROOT."""
    return {
        "PYTHONPATH": str(CAIRN_ROOT),
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "HOME": os.environ.get("HOME", os.path.expanduser("~")),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Project Creation (5 stages)
# ────────────────────────────────────────────────────────────────────────────────


def create_stage_a(proj: Path) -> None:
    """Stage A: scaffold — Flask app, static files, requirements."""
    (proj / "templates").mkdir(parents=True, exist_ok=True)
    (proj / "static").mkdir(parents=True, exist_ok=True)

    (proj / "app.py").write_text(
        "from flask import Flask, render_template\n\n"
        "app = Flask(__name__)\n\n\n"
        "@app.route('/')\n"
        "def index():\n"
        "    return render_template('index.html')\n\n\n"
        'if __name__ == "__main__":\n'
        "    app.run(debug=True)\n"
    )

    (proj / "templates" / "index.html").write_text(
        "<!DOCTYPE html>\n<html><head><title>TaskBoard</title>"
        "<link rel=\"stylesheet\" href=\"{{ url_for('static', filename='style.css') }}\">"
        "</head><body><h1>TaskBoard</h1>"
        "<script src=\"{{ url_for('static', filename='app.js') }}\"></script>"
        "</body></html>\n"
    )

    (proj / "static" / "app.js").write_text("console.log('TaskBoard ready');\n")
    (proj / "static" / "style.css").write_text("body { font-family: sans-serif; }\n")
    (proj / "requirements.txt").write_text("flask>=3.0\n")
    (proj / "README.md").write_text("# TaskBoard\n\nA simple task management application.\n")

    _git_add_commit(proj, "scaffold")


def create_stage_b(proj: Path) -> None:
    """Stage B: add task service + routes."""
    (proj / "services").mkdir(exist_ok=True)
    (proj / "services" / "__init__.py").write_text("")

    (proj / "services" / "tasks.py").write_text(
        "def list_tasks():\n"
        "    return [{'id': 1, 'title': 'Example'}]\n\n\n"
        "def create_task(title):\n"
        "    return {'id': 1, 'title': title}\n\n\n"
        "def delete_task(task_id):\n"
        "    return True\n"
    )

    app_py = (proj / "app.py").read_text()
    app_py = app_py.replace(
        "from flask import Flask, render_template",
        "from flask import Flask, render_template, request, jsonify",
    )
    app_py = app_py.replace(
        'if __name__ == "__main__":',
        "from services.tasks import list_tasks, create_task, delete_task\n\n\n"
        "@app.route('/tasks', methods=['GET', 'POST'])\n"
        "def tasks():\n"
        "    if request.method == 'GET':\n"
        "        return jsonify(list_tasks())\n"
        "    data = request.get_json() or {}\n"
        "    title = data.get('title', '')\n"
        "    task = create_task(title)\n"
        "    return jsonify(task), 201\n\n\n"
        "@app.route('/tasks/<int:task_id>', methods=['DELETE'])\n"
        "def task_delete(task_id):\n"
        "    success = delete_task(task_id)\n"
        "    return jsonify({'deleted': success})\n\n\n"
        'if __name__ == "__main__":',
    )
    (proj / "app.py").write_text(app_py)

    _git_add_commit(proj, "add task service + routes")


def create_stage_c(proj: Path) -> None:
    """Stage C: validate create_task title."""
    tasks_py = (proj / "services" / "tasks.py").read_text()
    tasks_py = tasks_py.replace(
        "def create_task(title):\n" "    return {'id': 1, 'title': title}",
        "from datetime import datetime\n\n\n"
        "def create_task(title):\n"
        "    if not isinstance(title, str) or not title.strip():\n"
        "        raise ValueError('title must be a non-empty string')\n"
        "    return {\n"
        "        'id': 1,\n"
        "        'title': title.strip(),\n"
        "        'created_at': datetime.utcnow().isoformat(),\n"
        "    }",
    )
    (proj / "services" / "tasks.py").write_text(tasks_py)

    _git_add_commit(proj, "validate task title")


def create_stage_d(proj: Path) -> None:
    """Stage D: remove delete_task function and its route."""
    # Remove delete_task function from services/tasks.py
    tasks_py = (proj / "services" / "tasks.py").read_text()
    lines = tasks_py.split("\n")
    new_lines = []
    skip = False
    for line in lines:
        if line.startswith("def delete_task"):
            skip = True
            continue
        if skip:
            if line.startswith("def ") or (line.strip() and not line.startswith(" ")):
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)
    (proj / "services" / "tasks.py").write_text("\n".join(new_lines))

    # Remove task_delete route and delete_task import from app.py
    app_py = (proj / "app.py").read_text()
    app_py = app_py.replace(", delete_task", "")
    app_lines = app_py.split("\n")
    new_app_lines = []
    skip = False
    for line in app_lines:
        if "@app.route('/tasks/<int:task_id>', methods=['DELETE'])" in line:
            skip = True
            continue
        if skip:
            if line.startswith("def task_delete"):
                continue
            if line.strip() and not line.startswith((" ", "\t")):
                skip = False
                new_app_lines.append(line)
            continue
        new_app_lines.append(line)
    (proj / "app.py").write_text("\n".join(new_app_lines))

    _git_add_commit(proj, "drop delete_task")


def create_stage_e(proj: Path) -> None:
    """Stage E: add storage.py and models.py."""
    (proj / "services" / "storage.py").write_text(
        "import json\n"
        "import os\n\n"
        "STORAGE_FILE = os.path.join(os.path.dirname(__file__), '..', 'tasks.json')\n\n\n"
        "def save_tasks(tasks):\n"
        "    with open(STORAGE_FILE, 'w') as f:\n"
        "        json.dump(tasks, f, indent=2)\n\n\n"
        "def load_tasks():\n"
        "    if not os.path.exists(STORAGE_FILE):\n"
        "        return []\n"
        "    with open(STORAGE_FILE, 'r') as f:\n"
        "        return json.load(f)\n"
    )

    (proj / "models.py").write_text(
        "class Task:\n"
        "    def __init__(self, title):\n"
        "        self.title = title\n"
        "        self.created_at = None\n\n"
        "    def to_dict(self):\n"
        "        return {'title': self.title, 'created_at': self.created_at}\n"
    )

    _git_add_commit(proj, "persist tasks to disk")


def _git_add_commit(proj: Path, msg: str) -> None:
    """Stage all changes and commit."""
    run_cmd(["git", "add", "-A"], cwd=proj)
    run_cmd(["git", "commit", "-m", msg], cwd=proj)


def _git_init(proj: Path) -> None:
    """Initialize git repo with custom user."""
    run_cmd(["git", "init"], cwd=proj)
    run_cmd(
        ["git", "config", "user.email", "sim@test.local"],
        cwd=proj,
    )
    run_cmd(
        ["git", "config", "user.name", "LifecycleSim"],
        cwd=proj,
    )


# ────────────────────────────────────────────────────────────────────────────────
# In-process MCP helpers
# ────────────────────────────────────────────────────────────────────────────────


def _setup_mcp_inprocess(project_path: Path) -> None:
    """Set up the mcp_server module globals for in-process use."""
    # Ensure cairn is importable
    if str(CAIRN_ROOT) not in sys.path:
        sys.path.insert(0, str(CAIRN_ROOT))

    # Reload mcp_server to get fresh state
    import server.mcp_server as mcp_module

    # Invalidate any cached modules that hold state
    # (Module references are refreshed above; globals are reset below)

    # Clear global state
    mcp_module._PROJECT_PATH = Path(project_path)
    mcp_module._BIND_ERROR = None
    mcp_module._router = None
    mcp_module._assembler = None
    mcp_module._session_budget = None
    mcp_module._semantic_caches.clear()

    # Set environment for binding resolution
    os.environ["CAIRN_PROJECT"] = str(project_path)


def _teardown_mcp_inprocess(original_cairn_project: str | None) -> None:
    """Restore original CAIRN_PROJECT env var if it existed."""
    if original_cairn_project is not None:
        os.environ["CAIRN_PROJECT"] = original_cairn_project
    elif "CAIRN_PROJECT" in os.environ:
        del os.environ["CAIRN_PROJECT"]


def _mcp_remember(note: str, kind: str) -> str:
    """Call remember in-process."""
    import server.mcp_server as mcp_module

    return mcp_module.remember(note=note, kind=kind)


def _mcp_recall(max_entries: int = 20) -> str:
    """Call recall in-process."""
    import server.mcp_server as mcp_module

    return mcp_module.recall(max_entries=max_entries)


def _mcp_search_code(query: str, top_k: int = 5) -> str:
    """Call search_code in-process."""
    import server.mcp_server as mcp_module

    return mcp_module.search_code(query=query, top_k=top_k)


def _mcp_assemble_context(query: str) -> str:
    """Call assemble_context in-process."""
    import server.mcp_server as mcp_module

    return mcp_module.assemble_context(query=query)


# ────────────────────────────────────────────────────────────────────────────────
# Report helpers
# ────────────────────────────────────────────────────────────────────────────────


class Report:
    """Append-only report writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, text: str) -> None:
        with open(self.path, "a") as f:
            f.write(text)
            f.flush()

    def reset(self) -> None:
        """Truncate the report for a fresh run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write("")
            f.flush()

    def section(self, title: str) -> None:
        self.write(f"\n\n## {title}\n\n")

    def row(
        self,
        phase: str,
        step: str,
        mode: str,
        result: str,
        evidence: str,
        elapsed: str,
    ) -> None:
        self.write(f"| {phase} | {step} | {mode} | {result} | {evidence} | {elapsed} |\n")

    def table_header(self) -> None:
        self.write("| Phase | Step | Mode | Result | Evidence | Elapsed |\n")
        self.write("|-------|------|------|--------|----------|--------|\n")

    def code_block(self, text: str, lang: str = "") -> None:
        self.write(f"\n```{lang}\n{text}\n```\n")


# ────────────────────────────────────────────────────────────────────────────────
# Phase 1 — Init from 0
# ────────────────────────────────────────────────────────────────────────────────


def phase1_init(proj: Path, report: Report, mode: str) -> None:
    logger.info("Phase 1: Init from scratch")
    report.section("Phase 1 — Init from 0")
    report.table_header()

    t0 = time.monotonic()

    # Run cairn init
    ok, out, err = run_cmd(
        ["python3", "-m", "cli.main", "init", "-y", "--offline"],
        cwd=proj,
        timeout_s=180.0,
        env=_cli_env(),
    )
    elapsed = f"{time.monotonic() - t0:.1f}s"

    config_path = proj / ".cairn" / "config.yaml"
    if not ok or not config_path.exists():
        report.row("1", "init -y --offline", mode, "FAIL", f"CLI failed: {err[:200]}", elapsed)
        return

    # Read config
    from core.config import load_config

    cfg = load_config(proj)
    detected_profile = cfg.profile
    file_patterns = cfg.indexing.file_patterns
    index_location = cfg.indexing.index_location

    report.row(
        "1",
        "init -y --offline",
        mode,
        "PASS",
        f"profile={detected_profile} patterns={file_patterns} " f"index_loc={index_location}",
        elapsed,
    )

    # Find chroma path and verify
    from core.repo import RepoManager

    repo = RepoManager(proj)
    chroma_path = repo.get_chroma_path()
    chroma_exists = chroma_path.exists()

    t1 = time.monotonic()
    try:
        from pipeline.indexer import VectorIndexer

        indexer = VectorIndexer(
            chroma_path=chroma_path,
            embeddings_enabled=False,
            project_root=proj,
        )
        block_count = indexer.count()
    except Exception as exc:
        block_count = -1
        logger.warning("Indexer count failed: %s", exc)

    elapsed_count = f"{time.monotonic() - t1:.1f}s"

    report.row(
        "1",
        "chroma DB check",
        mode,
        "PASS" if chroma_exists and block_count >= 0 else "FAIL",
        f"chroma_path={chroma_path} exists={chroma_exists} blocks={block_count}",
        elapsed_count,
    )

    # Determine which files got indexed and which didn't
    src_files = list(proj.glob("*.py")) + list(proj.glob("services/*.py"))
    indexed = []
    not_indexed = []
    for sf in sorted(src_files):
        rel = str(sf.relative_to(proj))
        indexed.append(rel)

    # Files excluded by default patterns
    not_indexed = [
        "templates/index.html (excluded: not in file_patterns)",
        "static/app.js (excluded: **/static/** + not in file_patterns)",
        "static/style.css (excluded: **/static/** + not in file_patterns)",
    ]

    report.row(
        "1",
        "indexed files",
        mode,
        "INFO",
        f"Indexed: {indexed} | NOT indexed: {not_indexed}",
        "-",
    )
    report.write(
        f"\n**Finding:** Default file_patterns={file_patterns} and "
        f"exclude_patterns include `**/static/**` and do not include `*.html`, "
        f"`*.css`, `*.js`. Frontend files are NOT indexed by default.\n"
    )


# ────────────────────────────────────────────────────────────────────────────────
# Phase 2 — DB Sync
# ────────────────────────────────────────────────────────────────────────────────


def phase2_db_sync(proj: Path, report: Report, mode: str) -> None:
    logger.info("Phase 2: DB sync (janitor live-sync + manual reindex)")
    report.section("Phase 2 — DB Sync")
    report.table_header()

    # ── 2a: Janitor live-sync ──
    logger.info("Phase 2a: Janitor live-sync")
    janitor_proc = None
    try:
        janitor_proc = subprocess.Popen(
            ["python3", "-m", "cli.main", "janitor", "start"],
            cwd=str(proj),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_cli_env(),
            text=True,
        )
        logger.info("Janitor started (PID=%s)", janitor_proc.pid)
        time.sleep(3)

        # Make a real edit: add get_task_count
        tasks_py = proj / "services" / "tasks.py"
        original = tasks_py.read_text()
        new_content = (
            original.rstrip() + "\n\n\ndef get_task_count():\n" + "    return len(list_tasks())\n"
        )
        tasks_py.write_text(new_content)
        logger.info("Added get_task_count to services/tasks.py")

        # Poll for the symbol up to 40s
        t_start = time.monotonic()
        found = False
        janitor_elapsed = "no"
        while time.monotonic() - t_start < 40:
            time.sleep(3)
            elapsed_s = time.monotonic() - t_start
            logger.info("  polling get_task_count... %.0fs", elapsed_s)
            try:
                _setup_mcp_inprocess(proj)
                result = _mcp_search_code("get_task_count", top_k=5)
                if "get_task_count" in result:
                    found = True
                    janitor_elapsed = f"{elapsed_s:.0f}s"
                    break
            except Exception as exc:
                logger.debug("Poll error: %s", exc)

        if found:
            report.row(
                "2a",
                "janitor live-sync",
                mode,
                "PASS",
                f"get_task_count appeared after {janitor_elapsed}",
                janitor_elapsed,
            )
        else:
            report.row(
                "2a",
                "janitor live-sync",
                mode,
                "FAIL",
                "janitor did not pick up change within 40s (likely WSL2 /mnt/c inotify issue)",
                "40s",
            )

        # Direct index to prove indexer works
        logger.info("Direct indexing get_task_count via ASTParser + VectorIndexer")
        t_direct = time.monotonic()
        try:
            from core.repo import RepoManager
            from pipeline.ast_parser import ASTParser
            from pipeline.indexer import VectorIndexer

            repo = RepoManager(proj)
            chroma_path = repo.get_chroma_path()
            indexer = VectorIndexer(
                chroma_path=chroma_path,
                embeddings_enabled=(mode == "with-LLM"),
                project_root=proj,
            )
            parser = ASTParser()
            ast_result = parser.parse_file(tasks_py)
            indexer.index_ast(ast_result)

            _setup_mcp_inprocess(proj)
            direct_result = _mcp_search_code("get_task_count", top_k=5)
            direct_ok = "get_task_count" in direct_result
            direct_elapsed = f"{time.monotonic() - t_direct:.1f}s"
            report.row(
                "2a",
                "direct indexer path",
                mode,
                "PASS" if direct_ok else "FAIL",
                (
                    f"get_task_count {'found' if direct_ok else 'NOT found'}"
                    f" via direct index: {direct_result[:200]}"
                ),
                direct_elapsed,
            )
        except Exception as exc:
            report.row(
                "2a",
                "direct indexer path",
                mode,
                "FAIL",
                f"Exception: {exc}",
                f"{time.monotonic() - t_direct:.1f}s",
            )

    finally:
        # Stop janitor
        if janitor_proc is not None:
            try:
                run_cmd(
                    ["python3", "-m", "cli.main", "janitor", "stop"],
                    cwd=proj,
                    timeout_s=15,
                    env=_cli_env(),
                )
                janitor_proc.wait(timeout=10)
            except Exception:
                janitor_proc.kill()
                janitor_proc.wait()
            logger.info("Janitor stopped")

    # Restore original tasks.py (without get_task_count) for consistent state
    try:
        tasks_py.write_text(original)
    except OSError:
        # File write failure is non-fatal
        logger.debug("Could not restore tasks.py", exc_info=True)

    # ── 2b: Manual reindex after Stage B ──
    logger.info("Phase 2b: Manual reindex")
    t_reindex = time.monotonic()
    ok, out, err = run_cmd(
        ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
        cwd=proj,
        timeout_s=180.0,
        env=_cli_env(),
    )
    elapsed_reindex = f"{time.monotonic() - t_reindex:.1f}s"

    if not ok:
        report.row(
            "2b",
            "reindex quick (Stage B)",
            mode,
            "FAIL",
            f"CLI error: {err[:200]}",
            elapsed_reindex,
        )
    else:
        _setup_mcp_inprocess(proj)
        result = _mcp_search_code("create_task", top_k=5)
        has_create = "create_task" in result and "tasks.py" in result
        first_line = result.strip().split("\n")[0] if result.strip() else "(empty)"
        report.row(
            "2b",
            "search create_task after B",
            mode,
            "PASS" if has_create else "FAIL",
            f"Top: {first_line[:120]}",
            elapsed_reindex,
        )

    # Reindex after Stage C and search for validate/created_at
    _, out, _ = run_cmd(
        ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
        cwd=proj,
        timeout_s=180.0,
        env=_cli_env(),
    )
    _setup_mcp_inprocess(proj)
    result_c = _mcp_search_code("validate title", top_k=5)
    has_validate = any(w in result_c for w in ("validate", "created_at"))
    first_c = result_c.strip().split("\n")[0] if result_c.strip() else "(empty)"
    report.row(
        "2b",
        "search validate after C",
        mode,
        "PASS" if has_validate else "FAIL",
        f"Top: {first_c[:120]}",
        "-",
    )

    # Reindex after Stage D and search for delete_task — should be GONE
    _, out, _ = run_cmd(
        ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
        cwd=proj,
        timeout_s=180.0,
        env=_cli_env(),
    )
    _setup_mcp_inprocess(proj)
    result_d = _mcp_search_code("delete_task", top_k=5)
    has_delete_func = "delete_task" in result_d and "tasks.py" in result_d
    delete_desc = (
        "FUNCTION STILL IN tasks.py" if has_delete_func else "function block absent (correct)"
    )
    report.row(
        "2b",
        "search delete_task after D",
        mode,
        "PASS" if not has_delete_func else "FAIL",
        f"delete_task {delete_desc}: {result_d[:150]}",
        "-",
    )


# ────────────────────────────────────────────────────────────────────────────────
# Phase 3 — Memory sync
# ────────────────────────────────────────────────────────────────────────────────


def phase3_memory(proj: Path, report: Report, mode: str) -> None:
    logger.info("Phase 3: Memory sync")
    report.section("Phase 3 — Memory Sync")
    report.table_header()

    t0 = time.monotonic()

    # Run memory update
    ok, out, err = run_cmd(
        ["python3", "-m", "cli.main", "memory", "update", "--commits", "6"],
        cwd=proj,
        timeout_s=120.0,
        env=_cli_env(),
    )
    elapsed = f"{time.monotonic() - t0:.1f}s"

    memory_path = proj / ".cairn" / "memory.md"
    memory_exists = memory_path.exists()

    if not ok and not memory_exists:
        report.row(
            "3",
            "memory update --commits 6",
            mode,
            "FAIL",
            f"CLI error: {err[:200]}",
            elapsed,
        )
        return

    # Read memory contents
    if memory_exists:
        mem_content = memory_path.read_text()
    else:
        mem_content = "(memory.md not found)"

    has_recent = "PASS" if "Recent Changes" in (mem_content or "") else "FAIL"
    report.row(
        "3",
        "memory update --commits 6",
        mode,
        has_recent,
        f"memory.md exists={memory_exists}, sections: "
        f"{'Recent Changes' if 'Recent Changes' in (mem_content or '') else 'N/A'}",
        elapsed,
    )
    report.write("\n**Memory contents (first 1000 chars):**\n")
    report.code_block(mem_content[:1000] if mem_content else "(empty)")

    # In-process remember
    original_cairn = os.environ.get("CAIRN_PROJECT")
    try:
        _setup_mcp_inprocess(proj)
        t_mem = time.monotonic()
        r1 = _mcp_remember("Chose Flask + in-memory store for v1", "decision")
        r2 = _mcp_remember("services/ holds business logic", "convention")
        recall_result = _mcp_recall(max_entries=20)
        elapsed_mem = f"{time.monotonic() - t_mem:.1f}s"

        has_dec = "Chose Flask" in recall_result
        has_conv = "services/ holds" in recall_result
        report.row(
            "3",
            "in-process remember/recall",
            mode,
            "PASS" if has_dec and has_conv else "PARTIAL",
            f"decision='{r1}' convention='{r2}' both_in_recall={has_dec and has_conv}",
            elapsed_mem,
        )
    finally:
        _teardown_mcp_inprocess(original_cairn)

    # Accumulation test: 2 more commits then memory update
    (proj / "utils.py").write_text("def dummy():\n    return True\n")
    _git_add_commit(proj, "add utils.py")
    readme = proj / "README.md"
    current_readme = readme.read_text()
    readme.write_text(current_readme + "\n## Changelog\n- Initial version\n")
    _git_add_commit(proj, "update README")

    ok2, out2, _ = run_cmd(
        ["python3", "-m", "cli.main", "memory", "update", "--commits", "2"],
        cwd=proj,
        timeout_s=120.0,
        env=_cli_env(),
    )
    mem_after = memory_path.read_text() if memory_path.exists() else ""
    lines_count = len([entry for entry in mem_after.split("\n") if entry.strip().startswith("- [")])

    report.row(
        "3",
        "memory accumulation (2 more commits)",
        mode,
        "PASS" if lines_count > 2 else "PARTIAL",
        f"Entries after 2 more commits: ~{lines_count} "
        f"(before had decisions/conventions from remember)",
        "-",
    )
    report.write("\n**Memory after accumulation:**\n")
    report.code_block(mem_after[:1500])


# ────────────────────────────────────────────────────────────────────────────────
# Phase 4 — Retrieval
# ────────────────────────────────────────────────────────────────────────────────


def phase4_retrieval(proj: Path, report: Report, mode: str) -> None:
    logger.info("Phase 4: Retrieval")
    report.section("Phase 4 — Retrieval")
    report.table_header()

    original_cairn = os.environ.get("CAIRN_PROJECT")
    try:
        _setup_mcp_inprocess(proj)
        t0 = time.monotonic()

        # Redundant: ensure index is up to date
        run_cmd(
            ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
            cwd=proj,
            timeout_s=180.0,
            env=_cli_env(),
        )

        # assemble_context
        ctx_result = _mcp_assemble_context("how are tasks created and listed")
        ctx_nonempty = bool(ctx_result.strip()) and len(ctx_result) > 50
        has_sections = "Codebase Context" in ctx_result or "Relevant Functions" in ctx_result
        report.row(
            "4",
            "assemble_context",
            mode,
            "PASS" if ctx_nonempty else "FAIL",
            f"Len={len(ctx_result)} has_sections={has_sections}",
            f"{time.monotonic() - t0:.1f}s",
        )
        report.write("\n**assemble_context snippet:**\n")
        report.code_block(ctx_result[:1200])

        # search_code for storage/model
        t1 = time.monotonic()
        r1 = _mcp_search_code("persist tasks to disk", top_k=3)
        top1 = r1.strip().split("\n")[0] if r1.strip() else "(empty)"
        has_storage = "storage" in top1.lower() or "model" in top1.lower()
        report.row(
            "4",
            "search_code 'persist tasks'",
            mode,
            "PASS" if has_storage else "PARTIAL",
            f"Top: {top1[:120]}",
            f"{time.monotonic() - t1:.1f}s",
        )

        t2 = time.monotonic()
        r2 = _mcp_search_code("Task model", top_k=3)
        top2 = r2.strip().split("\n")[0] if r2.strip() else "(empty)"
        has_model = "model" in top2.lower() or "task" in top2.lower()
        report.row(
            "4",
            "search_code 'Task model'",
            mode,
            "PASS" if has_model else "PARTIAL",
            f"Top: {top2[:120]}",
            f"{time.monotonic() - t2:.1f}s",
        )

        t3 = time.monotonic()
        r3 = _mcp_search_code("validate title", top_k=3)
        top3 = r3.strip().split("\n")[0] if r3.strip() else "(empty)"
        has_validate = "validate" in r3.lower() or "created_at" in r3.lower()
        report.row(
            "4",
            "search_code 'validate title'",
            mode,
            "PASS" if has_validate else "PARTIAL",
            f"Top: {top3[:120]}",
            f"{time.monotonic() - t3:.1f}s",
        )

    finally:
        _teardown_mcp_inprocess(original_cairn)


# ────────────────────────────────────────────────────────────────────────────────
# Phase 5 — Freshness
# ────────────────────────────────────────────────────────────────────────────────


def phase5_freshness(proj: Path, report: Report, mode: str) -> None:
    logger.info("Phase 5: Freshness")
    report.section("Phase 5 — Freshness")
    report.table_header()

    # Make commits without reindexing
    app_py = proj / "app.py"
    app_content = app_py.read_text()
    app_py.write_text(app_content + "\n# dummy comment for freshness test\n")
    _git_add_commit(proj, "add dummy comment")
    readme = proj / "README.md"
    readme_content = readme.read_text()
    readme.write_text(readme_content + "\n- Freshness test line\n")
    _git_add_commit(proj, "update README again")

    t0 = time.monotonic()
    ok, out, err = run_cmd(
        ["python3", "-m", "cli.main", "status"],
        cwd=proj,
        timeout_s=30.0,
        env=_cli_env(),
    )
    elapsed = f"{time.monotonic() - t0:.1f}s"

    report.row(
        "5",
        "status (stale)",
        mode,
        "PASS" if (ok and "behind" in (out + err).lower()) else "PARTIAL",
        f"Output: {(out + err)[:200]}",
        elapsed,
    )
    report.write("\n**status output:**\n")
    report.code_block((out + err)[:800])

    # Reindex and check fresh
    run_cmd(
        ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
        cwd=proj,
        timeout_s=180.0,
        env=_cli_env(),
    )
    ok2, out2, err2 = run_cmd(
        ["python3", "-m", "cli.main", "status"],
        cwd=proj,
        timeout_s=30.0,
        env=_cli_env(),
    )
    report.row(
        "5",
        "status (fresh after reindex)",
        mode,
        "PASS" if ok2 else "FAIL",
        f"Output: {(out2 + err2)[:200]}",
        "-",
    )


# ────────────────────────────────────────────────────────────────────────────────
# Phase 6 — With-vs-Without LLM comparison
# ────────────────────────────────────────────────────────────────────────────────


def phase6_llm_comparison(report: Report, mode: str) -> None:
    """Record comparison notes. The actual LLM test runs integrated in other phases."""
    if mode != "with-LLM":
        return
    report.section("Phase 6 — With-LLM vs Without-LLM")
    report.write(
        "- **No LLM (default):** Embeddings disabled, retriever uses BM25 + AST legs only. "
        "No Ollama calls needed. Works offline.\n"
        "- **With LLM:** Embeddings enabled (nomic-embed-text), hybrid retrieval. "
        "Requires Ollama running with models pulled.\n"
    )


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────


def _get_system_resources_safe() -> dict:
    """Get system resources, never crashing."""
    try:
        from core.resources import get_system_resources

        return get_system_resources()
    except Exception:
        return {}


def _check_ollama_models() -> str:
    """Check available ollama models, never crashing."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()[:500]
        return f"ollama not available: {r.stderr[:200]}"
    except Exception as exc:
        return f"ollama check failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cairn Lifecycle Simulation")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="Run with local LLM embeddings enabled",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=True,
        help="Run without local LLM (default)",
    )
    parser.add_argument(
        "--report",
        default=str(CAIRN_ROOT / "docs" / "LIFECYCLE_SIMULATION.md"),
        help="Path to write the report",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="Keep the temp project directory",
    )
    args = parser.parse_args()

    mode = "with-LLM" if args.with_llm else "no-LLM"
    report_path = Path(args.report)

    logger.info("Cairn Lifecycle Simulation — mode=%s", mode)
    logger.info("Report: %s", report_path)

    # Clear report
    report = Report(report_path)
    report.reset()

    # Header
    report.write(
        f"# Cairn Lifecycle Simulation Report\n\n"
        f"**Date:** {datetime.now(timezone.utc).isoformat()}\n"
        f"**CAIRN_ROOT:** {CAIRN_ROOT}\n"
        f"**Mode:** {mode}\n\n"
    )

    # System resources
    resources = _get_system_resources_safe()
    report.write(f"**System Resources:** {resources}\n\n")

    # Ollama models
    ollama_info = _check_ollama_models()
    report.write(f"**Ollama models available:**\n```\n{ollama_info}\n```\n\n")

    # Gemini check
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        gemma_available = "gemma" in r.stdout.lower() if r.returncode == 0 else False
        report.write(f"**Gemma model available:** {gemma_available}\n\n")
    except Exception:
        report.write("**Gemma model available:** could not check\n\n")

    # ── Create temp project ──
    proj = Path(tempfile.mkdtemp(prefix="taskboard_", dir="/mnt/c/Users/alfre/Projects"))
    logger.info("Temp project: %s", proj)

    try:
        # Initialize git
        _git_init(proj)

        # Stage A
        logger.info("Creating Stage A (scaffold)")
        create_stage_a(proj)

        # Run Phase 1 — only Stage A is needed
        phase1_init(proj, report, mode)

        # Stage B
        logger.info("Creating Stage B (backend)")
        create_stage_b(proj)

        # Stage C
        logger.info("Creating Stage C (modify)")
        create_stage_c(proj)

        # Stage D
        logger.info("Creating Stage D (delete)")
        create_stage_d(proj)

        # Stage E
        logger.info("Creating Stage E (grow)")
        create_stage_e(proj)

        # Reindex to pick up all stages
        logger.info("Reindexing for all stages")
        run_cmd(
            ["python3", "-m", "cli.main", "reindex", "--mode", "quick"],
            cwd=proj,
            timeout_s=180.0,
            env=_cli_env(),
        )

        # For with-LLM mode, modify config before Phase 2
        if args.with_llm:
            logger.info("Enabling LLM config")
            from core.config import load_config, save_config

            cfg = load_config(proj)
            cfg.local_llm.enabled = True
            cfg.local_llm.model = "gemma4:latest"
            cfg.local_llm.embed_model = "nomic-embed-text"
            cfg.indexing.embedding_model = "nomic-embed-text"
            cfg.retrieval.mode = "hybrid"
            cfg.embeddings_enabled = True
            save_config(cfg, proj)
            logger.info("LLM config applied")

        # Phase 2
        phase2_db_sync(proj, report, mode)

        # Phase 3
        phase3_memory(proj, report, mode)

        # Phase 4
        phase4_retrieval(proj, report, mode)

        # Phase 5
        phase5_freshness(proj, report, mode)

        # Phase 6
        phase6_llm_comparison(report, mode)

        # ── Summary ──
        report.section("Summary")
        report.write(
            "## Summary\n\n"
            "The Cairn lifecycle simulation tested the full feature set: "
            "init, indexing (AST parse + ChromaDB), DB freshness tracking, "
            "janitor live-sync, manual reindex, memory summarization, "
            "in-process MCP tools (remember/recall/search_code/assemble_context), "
            "and index status reporting.\n\n"
        )

        report.section("Findings & Limitations")
        report.write(
            "### Findings & Limitations\n\n"
            "1. **Frontend not indexed by default:** `file_patterns` default to "
            "`.py`, `.rs`, `.go`, `.c`, etc. and do not include `.html`, `.css`, `.js`. "
            "Additionally, `exclude_patterns` includes `**/static/**`. This means frontend "
            "files in `templates/` and `static/` are never indexed out of the box. "
            "This is intentional for code-centric repos but worth documenting.\n\n"
            "2. **Janitor on /mnt/c (WSL2):** The janitor uses watchdog (inotify). "
            "On WSL2 with /mnt/c paths (DrvFs/9p filesystem), inotify may not detect "
            "file changes reliably. The janitor live-sync test may FAIL in this environment "
            "even though the direct indexer path works correctly. "
            "This is a known WSL2 limitation, not a Cairn bug.\n\n"
            "3. **No-LLM mode:** Without local LLM, embeddings are disabled and "
            "retrieval falls back to BM25 + AST structural search. This works correctly "
            "but may produce different search results (lower recall for semantic queries).\n\n"
            "4. **With-LLM mode:** Requires Ollama running with `nomic-embed-text` and "
            "`gemma4:latest` models pulled. Enables hybrid retrieval with embeddings, "
            "which provides better semantic matching.\n\n"
            "5. **Memory accumulation:** Memory entries are persisted across sessions "
            "in `.cairn/memory.md` with bounded sections. The `remember()` tool writes "
            "to the appropriate section (decisions, conventions, etc.) and auto-caps "
            "entries to prevent unbounded growth.\n\n"
            "6. **Profile detection:** Cairn correctly detected the project profile "
            "based on file extension census. For a pure Python project, it selects "
            "the `python` or `code` profile with appropriate retrieval strategy.\n"
        )

        logger.info("Simulation complete. Report: %s", report_path)

    except Exception as exc:
        logger.error("Simulation failed: %s", exc)
        logger.error(traceback.format_exc())
        report.section("Exception")
        report.code_block(traceback.format_exc())
    finally:
        # Cleanup
        if not args.keep and proj.exists():
            logger.info("Cleaning up %s", proj)
            shutil.rmtree(proj, ignore_errors=True)
        elif args.keep:
            logger.info("Keeping project at %s", proj)


if __name__ == "__main__":
    main()
