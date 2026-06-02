"""CLI entry point for cairn."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from core.config import Config, load_config
from core.profiles import detect_profile, get_profile
from core.repo import census_extensions, detect_infra_markers

if TYPE_CHECKING:
    from server.ollama_client import OllamaClient


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _detect_workspace_siblings(project_path: Path) -> list[Path]:
    """Detect if project is in a workspace by finding sibling git repos.

    A workspace is detected if the parent directory contains multiple
    sibling directories that are git repositories (contain .git/).

    Args:
        project_path: The current project path.

    Returns:
        List of sibling directories that contain .git/ (empty if not in workspace).
    """
    parent = project_path.parent
    siblings = []
    try:
        for sibling in parent.iterdir():
            if sibling.is_dir() and sibling != project_path:
                if (sibling / ".git").exists():
                    siblings.append(sibling)
    except Exception:
        pass
    return siblings


def _get_embeddings_enabled(cfg: Config) -> bool:
    """Get effective embeddings flag: both config and local_llm must be enabled."""
    return cfg.embeddings_enabled and cfg.local_llm.enabled


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """Cairn - Intelligent context engine for OpenCode."""
    _setup_logging(verbose)


def _run_preflight_checks(skip_ollama: bool = False) -> bool:
    """Run preflight checks for init/start-all.

    Args:
        skip_ollama: If True, skip Ollama checks (used when --no-index).

    Returns:
        True if all critical checks pass, False otherwise.
    """
    import shutil

    ok = True

    # Python version
    vi = sys.version_info
    py_ok = vi >= (3, 10)
    click.echo(f"  [{'✓' if py_ok else '✗'}] Python: {vi.major}.{vi.minor} (need ≥3.10)")
    ok = ok and py_ok

    if not skip_ollama:
        # Ollama (required for indexing)
        try:
            from server.ollama_client import OllamaClient

            ollama = OllamaClient()
            if ollama.health_check():
                click.echo("  [✓] Ollama: reachable")
            else:
                click.echo("[✗] Ollama: NOT REACHABLE")
                click.echo("   → Start Ollama: ollama serve")
                ok = False
        except Exception:
            click.echo("[✗] Ollama: error connecting")
            ok = False

    # Disk space
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free // (1024**3)
    disk_ok = free_gb >= 2
    click.echo(f"  [{'✓' if disk_ok else '✗'}] Disk: {free_gb}GB free (need ≥2GB)")
    ok = ok and disk_ok

    # Git repo
    git_dir = Path.cwd() / ".git"
    git_ok = git_dir.is_dir()
    click.echo(
        f"  [{'✓' if git_ok else '✗'}] Git: {'repository found' if git_ok else 'NOT a git repo'}"
    )
    ok = ok and git_ok

    # ChromaDB writable
    chroma_path = Path.cwd() / ".cairn" / "chroma"
    try:
        chroma_path.mkdir(parents=True, exist_ok=True)
        test_file = chroma_path / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        click.echo("  [✓] ChromaDB: writable")
    except Exception:
        click.echo("[✗] ChromaDB: NOT WRITABLE")
        ok = False

    return ok


@main.command()
@click.option("--no-index", is_flag=True, help="Skip indexing step (fast config-only setup)")
@click.option("-y", "--yes", is_flag=True, help="Auto-pull missing models, no prompt")
@click.option("--force", is_flag=True, help="Re-detect and rebuild index, overwriting config")
@click.option(
    "--offline",
    is_flag=True,
    help="Disable reranker (FlashRank); use in corporate/proxy environments",
)
def init(no_index: bool, yes: bool, force: bool, offline: bool):
    """One-command setup: detect layout, write config, build index.

    Similar to 'git init' — prepares the current repo for cairn.
    Idempotent: running twice without --force leaves things as-is.
    """
    from core.config import Config, save_config
    from core.freshness import DBFreshness
    from core.repo import RepoManager, collect_source_files, detect_source_layout
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer
    from server.ollama_client import OllamaClient

    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║         Cairn Initialization                         ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()

    project_path = Path.cwd()

    # STEP 1: Preflight checks
    click.echo("1. Pre-flight checks:")
    preflight_ok = _run_preflight_checks(skip_ollama=no_index)
    if not preflight_ok and not no_index:
        click.echo()
        click.echo("Some checks failed. Fix the issues above before running init.")
        sys.exit(1)
    click.echo()

    # STEP 2: Model check (unless --no-index)
    if not no_index:
        click.echo("2. Embedding model check:")
        try:
            ollama = OllamaClient()
            models = ollama.list_models()
            embed_ok = any("nomic-embed-text" in name or "embed" in name.lower() for name in models)

            if not embed_ok:
                click.echo("  [✗] Embedding model not found")
                success = _ensure_model(
                    ollama,
                    "nomic-embed-text",
                    models,
                    is_required=True,
                    yes_auto_pull=yes,
                )
                if not success:
                    click.echo()
                    click.echo("Cannot index without embedding model. Stopping.")
                    sys.exit(1)
            else:
                click.echo("  [✓] Embedding model found")
        except Exception as e:
            click.echo(f"  [✗] Error checking models: {e}")
            click.echo("Cannot index without Ollama. Stopping.")
            sys.exit(1)
        click.echo()

    # STEP 3: Detect layout
    click.echo("3. Detecting source layout:")
    detected_roots, detected_patterns = detect_source_layout(project_path)
    click.echo(f"  Detected source roots: {detected_roots}")
    click.echo(f"  Detected languages: {', '.join(detected_patterns)}")
    click.echo()

    # STEP 3b: Detect profile and retrieval strategy
    click.echo("3b. Detecting repository profile:")
    ext_census = census_extensions(project_path, source_roots=detected_roots)
    has_infra_markers = detect_infra_markers(project_path, source_roots=detected_roots)
    detected_profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
    detected_profile = get_profile(detected_profile_name)
    click.echo(f"  Detected profile: {detected_profile_name}")
    click.echo(f"  Retrieval strategy: {detected_profile.retrieval_mode}")
    click.echo(f"  Embedding models: {'ON' if detected_profile.embedding_enabled else 'OFF'}")
    if detected_profile.embedding_enabled:
        click.echo(f"  Embedding model: {detected_profile.embedding_model}")
    if detected_profile.description:
        click.echo(f"  Description: {detected_profile.description}")
    click.echo()

    # STEP 4: Write config
    click.echo("4. Writing configuration:")
    config_path = project_path / ".cairn" / "config.yaml"

    if config_path.exists() and not force:
        # Merge: only update if unset/default
        cfg = load_config(project_path)
        click.echo(f"  Found existing config at {config_path}")

        # Update source_roots if still default
        if cfg.indexing.source_roots == ["."]:
            cfg.indexing.source_roots = detected_roots
            click.echo(f"  Updated source_roots: {detected_roots}")

        # Update file_patterns if still default (all patterns)
        default_config = Config()
        if cfg.indexing.file_patterns == default_config.indexing.file_patterns:
            cfg.indexing.file_patterns = detected_patterns
            click.echo(f"  Updated file_patterns: {detected_patterns}")

        # Update profile if still default
        if cfg.profile == "code":
            cfg.profile = detected_profile_name
            click.echo(f"  Updated profile: {detected_profile_name}")
            # Apply profile settings
            cfg.embeddings_enabled = detected_profile.embedding_enabled
            if detected_profile.embedding_enabled:
                cfg.indexing.embedding_model = detected_profile.embedding_model
            click.echo("  Applied profile settings")
    else:
        # Create fresh config with detected values
        cfg = Config()
        cfg.indexing.source_roots = detected_roots
        cfg.indexing.file_patterns = detected_patterns
        # Apply profile settings
        cfg.profile = detected_profile_name
        cfg.embeddings_enabled = detected_profile.embedding_enabled
        if detected_profile.embedding_enabled:
            cfg.indexing.embedding_model = detected_profile.embedding_model
        click.echo("  Created new config with profile settings")

    # Apply --offline flag if set (disables reranker)
    if offline:
        cfg.retrieval.offline = True
        click.echo("  Offline mode: reranker disabled")

    # Ensure expanded exclude_patterns (migration support)
    default_config = Config()
    old_patterns = set(cfg.indexing.exclude_patterns)
    new_patterns = set(default_config.indexing.exclude_patterns)
    cfg.indexing.exclude_patterns = sorted(list(old_patterns | new_patterns))

    save_config(cfg, project_path)
    click.echo("  ✓ Config saved to .cairn/config.yaml")

    # Create .gitignore to prevent committing generated files
    gitignore_path = project_path / ".cairn" / ".gitignore"
    if not gitignore_path.exists():
        gitignore_content = """# Cairn - Generated Files
# These files are regenerated automatically and should not be version controlled.

# Vector database (regenerated by reindex)
chroma/

# Observability metrics (regenerated on each operation)
metrics.json

# Process ID files (runtime only)
*.pid

# Repository structure map (regenerated by reindex)
repo_map.json

# Memory summaries (regenerated by memory update)
memory.md
"""
        gitignore_path.write_text(gitignore_content)
        click.echo("  ✓ Created .cairn/.gitignore")
    click.echo()

    # STEP 4b: Scaffold MCP client configs
    click.echo("4b. Scaffolding MCP client configs:")
    import json as _json
    import shutil as _shutil

    # Resolve cairn binary path (OpenCode 1.15+ requires command as array)
    cairn_cmd = _shutil.which("cairn")
    if not cairn_cmd:
        # Fallback: use sys.executable -m cairn
        cairn_cmd = sys.executable
        cairn_args = ["-m", "cairn", "mcp"]
    else:
        cairn_args = ["mcp"]

    project_abs_path = str(project_path.resolve())

    # OpenCode 1.15+ format: command is array, enabled is true
    opencode_mcp_entry = {
        "type": "local",
        "command": [cairn_cmd] + cairn_args,
        "enabled": True,
        "env": {"CAIRN_PROJECT": project_abs_path},
    }

    # Write opencode.json
    opencode_path = project_path / "opencode.json"
    try:
        if opencode_path.exists():
            opencode_data = _json.loads(opencode_path.read_text())
        else:
            opencode_data = {"$schema": "https://opencode.ai/config.json"}

        if "mcp" not in opencode_data:
            opencode_data["mcp"] = {}
        opencode_data["mcp"]["cairn"] = opencode_mcp_entry

        opencode_path.write_text(_json.dumps(opencode_data, indent=2))
        click.echo("  ✓ Wrote opencode.json (MCP config)")
    except Exception as e:
        click.echo(f"  [!] opencode.json: {e}")

    # Detect if in workspace (parent has multiple sibling git repos)
    workspace_siblings = _detect_workspace_siblings(project_path)
    if workspace_siblings:
        click.echo()
        click.echo("  ⚠ Workspace detected:")
        click.echo(
            f"    This project is in a monorepo with "
            f"{len(workspace_siblings)} sibling git repo(s)."
        )
        click.echo("    OpenCode resolves MCP config from the WORKSPACE ROOT.")
        click.echo("    Consider placing opencode.json at the workspace root instead:")
        parent = project_path.parent
        click.echo()
        click.echo(f"    cp {opencode_path.name} {parent}/opencode.json")
        click.echo()

    # Write .mcp.json (Claude Code format: command + args as separate fields)
    mcp_path = project_path / ".mcp.json"
    try:
        if mcp_path.exists():
            mcp_data = _json.loads(mcp_path.read_text())
        else:
            mcp_data = {}

        if "mcpServers" not in mcp_data:
            mcp_data["mcpServers"] = {}
        mcp_data["mcpServers"]["cairn"] = {
            "command": cairn_cmd,
            "args": cairn_args,
            "env": {"CAIRN_PROJECT": project_abs_path},
        }

        mcp_path.write_text(_json.dumps(mcp_data, indent=2))
        click.echo("  ✓ Wrote .mcp.json (MCP config)")
    except Exception as e:
        click.echo(f"  [!] .mcp.json: {e}")
    click.echo()

    # STEP 5: Build index (unless --no-index)
    if not no_index:
        click.echo("5. Building index:")

        repo = RepoManager(project_path)
        parser = ASTParser()
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )
        freshness = DBFreshness(
            project_path,
            quick_threshold=cfg.stale_db.quick_reindex_threshold,
            full_threshold=cfg.stale_db.full_reindex_threshold,
        )

        filtered = collect_source_files(
            project_path,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            cfg.indexing.source_roots,
        )

        total = 0
        file_count = 0
        with click.progressbar(
            filtered,
            label="  Indexing",
            length=len(filtered),
            show_pos=True,
        ) as bar:
            for filepath in bar:
                try:
                    ast = parser.parse_file(filepath)
                    indexer.index_ast(ast)
                    total += len(ast.functions)
                    for cls in ast.classes:
                        total += len(cls.methods)
                    file_count += 1
                except Exception:
                    pass

        freshness.mark_indexed(freshness.get_current_commit())
        repo.write_index_meta()
        click.echo(f"  ✓ Indexed {total} functions from {file_count} files")
        click.echo()

    # STEP 6: Done
    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║               Initialization Complete!                      ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()
    click.echo("Ready to go! Start serving with:")
    click.echo()
    click.echo("  cairn run")
    click.echo()


@main.command()
def config():
    """Show current configuration."""
    cfg = load_config()
    click.echo(f"File watcher:    {cfg.enabled.file_watcher}")
    click.echo(f"Vector indexing: {cfg.enabled.vector_indexing}")
    click.echo(f"Memory summary:  {cfg.enabled.memory_summarizer}")
    click.echo(f"CPU limit:       {cfg.resources.max_cpu_percent}%")
    click.echo(f"RAM limit:       {cfg.resources.max_memory_mb}MB")
    click.echo(f"VRAM priority:   {cfg.resources.vram_priority}")
    click.echo(f"File patterns:   {cfg.indexing.file_patterns}")
    click.echo(f"Exclude:         {cfg.indexing.exclude_patterns}")
    click.echo(f"Batch size:      {cfg.indexing.batch_size}")
    click.echo(f"Debounce:        {cfg.indexing.delay_ms}ms")
    click.echo(f"Quick reindex:   {cfg.stale_db.quick_reindex_threshold} commits")
    click.echo(f"Full reindex:    {cfg.stale_db.full_reindex_threshold} commits")
    click.echo(f"Memory trigger:  {cfg.memory.trigger}")
    click.echo(f"Memory max:      {cfg.memory.max_entries} entries")
    click.echo(f"Compaction:      {cfg.memory.compaction_model}")
    click.echo(f"Routing mode:    {cfg.routing.mode}")
    click.echo(f"Cache enabled:   {cfg.cache.enabled}")
    click.echo(f"Cache TTL:       {cfg.cache.ttl_seconds}s")


@main.command()
@click.argument("name", required=False, default=None)
def profile(name: str | None):
    """Show or set the repository profile.

    With no arguments, prints the current profile and its strategy.
    With a NAME argument, sets the profile in config.yaml and
    re-applies all profile-driven settings.

    Example:
        cairn profile            # show current
        cairn profile iac        # set to iac profile
    """
    from core.profiles import PROFILES, get_profile

    project_path = Path.cwd()
    cfg = load_config(project_path)

    if name is None:
        # Show current profile
        current_profile_name = cfg.profile
        current_profile = get_profile(current_profile_name)

        click.echo(f"Repository Profile: {current_profile_name}")
        click.echo(f"  Description: {current_profile.description}")
        click.echo()
        click.echo("Retrieval Strategy:")
        click.echo(
            f"  Mode: {current_profile.retrieval_mode} " f"({', '.join(current_profile.legs)})"
        )
        click.echo(f"  Embeddings: {'ON' if current_profile.embedding_enabled else 'OFF'}")
        if current_profile.embedding_enabled:
            click.echo(f"  Embedding model: {current_profile.embedding_model}")
        click.echo(f"  Worker model: {current_profile.worker_model}")
        click.echo()
        click.echo("File patterns:")
        for pattern in current_profile.file_patterns:
            click.echo(f"  {pattern}")
        return

    # Set profile
    if name not in PROFILES:
        click.echo(f"Unknown profile: {name}", err=True)
        click.echo(f"Available: {', '.join(PROFILES.keys())}", err=True)
        return

    new_profile = get_profile(name)
    cfg.profile = name
    cfg.embeddings_enabled = new_profile.embedding_enabled
    if new_profile.embedding_enabled:
        cfg.indexing.embedding_model = new_profile.embedding_model
    # Note: file_patterns NOT auto-updated when setting profile via CLI
    # (user may have customized them). Only updated during init.

    from core.config import save_config

    save_config(cfg, project_path)
    click.echo(f"Profile set to: {name}")
    click.echo(f"  Embeddings: {'ON' if new_profile.embedding_enabled else 'OFF'}")
    click.echo(f"  Retrieval mode: {new_profile.retrieval_mode}")
    click.echo("Config saved to .cairn/config.yaml")


@main.command()
def status():
    """Check indexing status and DB freshness."""
    from core.freshness import DBFreshness
    from core.repo import RepoManager
    from pipeline.indexer import VectorIndexer

    cfg = load_config()
    project_path = Path.cwd()

    freshness = DBFreshness(
        project_path,
        quick_threshold=cfg.stale_db.quick_reindex_threshold,
        full_threshold=cfg.stale_db.full_reindex_threshold,
    )
    info = freshness.check_freshness()

    click.echo(f"Project: {project_path}")
    click.echo(f"Current commit: {info['current_commit'][:8]}")
    last = info["last_indexed_commit"]
    click.echo(f"Last indexed: {last[:8] if last else 'Never'}")
    click.echo(f"Commits behind: {info['commits_behind']}")

    if info["commits_behind"] >= cfg.stale_db.full_reindex_threshold:
        click.echo("⚠ Full re-index recommended")
    elif info["commits_behind"] >= cfg.stale_db.quick_reindex_threshold:
        click.echo("⚠ Quick re-index recommended")

    repo = RepoManager(project_path)
    try:
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )
        click.echo(f"Indexed functions: {indexer.count()}")
    except Exception as e:
        click.echo(f"Indexer: {e}")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--background", is_flag=True, help="Run server in background (PID file)")
def serve(host: str, port: int, background: bool):
    """Start the gateway API server."""
    import uvicorn

    from server.api import app as gateway_app

    if background:
        _start_background(host, port)
        return

    click.echo(f"Starting Semantic Gateway on http://{host}:{port}")
    uvicorn.run(gateway_app, host=host, port=port)


def _start_background(host: str, port: int) -> None:
    """Start the gateway server as a background process via PID file."""
    import subprocess
    import sys

    pid_file = Path.cwd() / ".cairn" / "gateway.pid"

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)
            click.echo(f"Gateway already running (PID {old_pid}). Stop it first.")
            return
        except (OSError, ValueError):
            pid_file.unlink()

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid_file.write_text(str(proc.pid))
    click.echo(f"Gateway started in background (PID {proc.pid}) on http://{host}:{port}")


# ── Doctor ──────────────────────────────────────────────────────


@main.command()
def doctor():
    """Run pre-flight checks and report any issues."""
    import shutil

    click.echo("== Cairn Doctor ==")
    click.echo()

    ok = True

    # Gateway code location (helps debug editable install issues)
    try:
        from server import api as server_module

        gateway_path = Path(server_module.__file__).parent.parent
        click.echo(f"[i] Gateway code: {gateway_path}")
    except Exception:
        pass

    # Python interpreter (exposes venv vs system Python issues)
    click.echo(f"[i] Interpreter: {sys.executable}")

    click.echo()

    # Python version
    vi = sys.version_info
    py_ok = vi >= (3, 10)
    click.echo(f"[{'✓' if py_ok else '✗'}] Python: {vi.major}.{vi.minor}.{vi.micro} (need ≥3.10)")
    ok = ok and py_ok

    # Local LLM status (informational)
    cfg = load_config(Path.cwd())
    if cfg.local_llm.enabled:
        click.echo(f"[i] Local LLM: enabled ({cfg.local_llm.backend})")

        # Health check the configured backend
        try:
            if cfg.local_llm.backend == "openai_compatible":
                from server.ollama_client import OpenAICompatibleClient

                if not cfg.local_llm.base_url:
                    click.echo("[!] OpenAI-compatible backend: base_url not set")
                else:
                    client = OpenAICompatibleClient(
                        base_url=cfg.local_llm.base_url,
                        model=cfg.local_llm.model,
                        embed_model=cfg.local_llm.embed_model,
                    )
                    if client.health_check():
                        click.echo("[✓] OpenAI-compatible server: reachable")
                    else:
                        click.echo("[✗] OpenAI-compatible server: NOT REACHABLE")
                        click.echo(f"   → Check: {cfg.local_llm.base_url}")
            else:
                # Ollama backend
                from server.ollama_client import OllamaClient

                ollama = OllamaClient(
                    base_url=cfg.local_llm.base_url,
                    generate_model=cfg.local_llm.model,
                    embed_model=cfg.local_llm.embed_model,
                )
                if ollama.health_check():
                    click.echo("[✓] Ollama: reachable")
                else:
                    click.echo("[✗] Ollama: NOT REACHABLE")
                    click.echo("   → Start Ollama: ollama serve")
        except Exception as e:
            click.echo(f"[!] LLM health check error: {e}")
    else:
        click.echo("[i] Local LLM: disabled (lexical/structural + cross-encoder only)")

    # Ollama check (only if embeddings+llm are enabled)
    if cfg.embeddings_enabled and cfg.local_llm.enabled and cfg.local_llm.backend == "ollama":
        try:
            from server.ollama_client import OllamaClient

            ollama = OllamaClient(
                base_url=cfg.local_llm.base_url,
                generate_model=cfg.local_llm.model,
                embed_model=cfg.local_llm.embed_model,
            )
            if ollama.health_check():
                models = ollama.list_models()
                embed_ok = any(
                    "nomic-embed-text" in name or "embed" in name.lower() for name in models
                ) or any("embed" in name.lower() for name in models)
                gen_ok = len(models) > 0
                click.echo(
                    f"[{'✓' if embed_ok else '✗'}] Embedding model (nomic-embed-text or similar): "
                    f"{'found' if embed_ok else 'NOT FOUND'}"
                )
                if not embed_ok:
                    click.echo(
                        "   → Run: ollama pull nomic-embed-text   (or set OLLAMA_EMBED_MODEL)"
                    )
                    ok = False
                click.echo(f"[{'✓' if gen_ok else '✗'}] Generation models: {len(models)} available")
                if not gen_ok:
                    click.echo(
                        "   → Run: ollama pull qwen2.5-coder:3b   (or set OLLAMA_GENERATE_MODEL)"
                    )
                    ok = False
            else:
                click.echo("[✗] Ollama: NOT REACHABLE")
                click.echo("   → Start Ollama: ollama serve")
                ok = False
        except Exception:
            click.echo("[✗] Ollama: error connecting")
            ok = False

    # Disk space
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free // (1024**3)
    disk_ok = free_gb >= 2
    click.echo(f"[{'✓' if disk_ok else '✗'}] Disk: {free_gb}GB free (need ≥2GB)")
    ok = ok and disk_ok

    # Git repo
    git_dir = Path.cwd() / ".git"
    git_ok = git_dir.is_dir()
    click.echo(
        f"[{'✓' if git_ok else '✗'}] Git: {'repository found' if git_ok else 'NOT a git repo'}"
    )
    ok = ok and git_ok

    # ChromaDB writable
    chroma_path = Path.cwd() / ".cairn" / "chroma"
    try:
        chroma_path.mkdir(parents=True, exist_ok=True)
        test_file = chroma_path / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        click.echo("[✓] ChromaDB: writable")
    except Exception:
        click.echo("[✗] ChromaDB: NOT WRITABLE")
        ok = False

    # Index schema freshness: an index built by an older gateway can be missing or
    # garbled blocks (e.g. pre-BOM-fix names, embeddings-on iac). Prompt a reindex.
    from core.repo import RepoManager
    from core.version import INDEX_SCHEMA_VERSION

    _repo = RepoManager(Path.cwd())
    if _repo.get_chroma_path().exists() and any(_repo.get_chroma_path().iterdir()):
        _meta = _repo.read_index_meta()
        if _meta is None:
            click.echo("[!] Index: unstamped (built by an older version)")
            click.echo("   → Rebuild for the current schema: cairn reindex --mode full")
        elif int(_meta.get("schema_version", 0)) < INDEX_SCHEMA_VERSION:
            click.echo(
                f"[!] Index: stale schema v{_meta.get('schema_version')} "
                f"(current v{INDEX_SCHEMA_VERSION})"
            )
            click.echo("   → Rebuild: cairn reindex --mode full")
        else:
            click.echo(f"[✓] Index: schema v{INDEX_SCHEMA_VERSION} (current)")

    # Ripgrep (optional but enables fresh exact-match search)
    from pipeline.retrieval.ripgrep import RipgrepRetriever

    rg_available = RipgrepRetriever.available()
    if rg_available:
        click.echo("[✓] ripgrep: found (fresh exact-match search)")
    else:
        click.echo("[i] ripgrep: NOT found — using in-memory BM25 fallback")
        click.echo("   → Install: apt install ripgrep")

    # Reranker (flashrank availability and loadability with timeout)
    try:
        import importlib.util
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        flashrank_spec = importlib.util.find_spec("flashrank")
        if flashrank_spec is None:
            click.echo("[✗] Reranker (flashrank): NOT installed — cross-encoder reranking disabled")
            click.echo("   → Retrieval quality + nonsense-rejection degraded. Install with:")
            click.echo("   → pip install flashrank")
        else:
            # Flashrank is importable; now try to load the model with a short timeout
            # to detect if it hangs (e.g., proxy stall during HuggingFace download)
            def _test_load():
                try:
                    from flashrank import Ranker

                    Ranker(model_name="ms-marco-MiniLM-L-12-v2")
                    return True
                except Exception as e:
                    return str(e)

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_test_load)
                    result = future.result(timeout=5)  # 5s timeout for model load

                if result is True:
                    click.echo("[✓] Reranker (flashrank): model loadable (ms-marco-MiniLM-L-12-v2)")
                else:
                    click.echo(f"[✗] Reranker (flashrank): model failed to load: {result}")
                    ok = False
            except FuturesTimeoutError:
                click.echo("[✗] Reranker (flashrank): model load TIMED OUT (blocked by proxy?)")
                click.echo("   → Solution: use cairn init --offline, or fix proxy/CA bundle")
                ok = False
            except Exception as e:
                click.echo(f"[✗] Reranker (flashrank): error during test load: {e}")
                ok = False
    except Exception:
        click.echo("[✗] Reranker (flashrank): error checking availability")
        ok = False

    # CA bundle configuration (useful for corporate proxies)
    cfg = load_config(Path.cwd())
    ca_bundle = (
        cfg.retrieval.ca_bundle
        or os.environ.get("CAIRN_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
    )
    if ca_bundle:
        click.echo(f"[i] CA bundle: {ca_bundle}")
    else:
        click.echo("[i] CA bundle: not set (using system default)")

    # Offline mode check
    if cfg.retrieval.offline:
        click.echo("[i] Offline mode: ENABLED (reranker disabled)")

    # Project path and ChromaDB collection info
    project_path = Path.cwd()
    click.echo(f"[i] Project path: {project_path.resolve()}")

    try:
        from core.repo import RepoManager, project_id

        repo = RepoManager(project_path)
        from pipeline.indexer import VectorIndexer

        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )
        collection_name = indexer.collection.name if indexer.collection else "unknown"
        pid = project_id(project_path)
        click.echo(f"[i] Collection: {collection_name}")
        if pid:
            click.echo(f"[i] Project ID: {pid}")
    except Exception as e:
        click.echo(f"[i] Collection: error ({e})")

    # VRAM tip: on small GPUs the embedder + worker model contend for memory,
    # causing slow cold reloads when they swap. Keeping >=2 models resident helps.
    if not os.environ.get("OLLAMA_MAX_LOADED_MODELS"):
        click.echo(
            "[i] Tip: on a small GPU set OLLAMA_MAX_LOADED_MODELS=2 (+ OLLAMA_KEEP_ALIVE=30m)"
        )
        click.echo("   → keeps embedder + worker resident, avoiding slow model-swap reloads")

    click.echo()
    if ok:
        click.echo("All checks passed. Ready to run!")
    else:
        click.echo("Some checks failed. Fix the issues above before starting.")


# ── Start-All ──────────────────────────────────────────────────


def _model_name_matches(installed_name: str, required_base: str) -> bool:
    """Check if an installed model matches a required base name.

    Examples:
        "nomic-embed-text:latest" matches "nomic-embed-text"
        "qwen2.5-coder:3b" matches "qwen2.5-coder"
    """
    return installed_name.split(":")[0] == required_base.split(":")[0]


def _ensure_model(
    ollama: OllamaClient,
    model_name: str,
    available_models: list[str],
    is_required: bool = True,
    yes_auto_pull: bool = False,
) -> bool:
    """Ensure a model is available, optionally pulling it.

    Returns:
        True if model is available or successfully pulled, False otherwise.
    """
    # Check if model is already installed
    if any(_model_name_matches(m, model_name) for m in available_models):
        return True

    # Model is missing
    severity = "ERROR" if is_required else "WARNING"
    click.echo(f"\n   {severity}: Missing {model_name}")

    # If --yes flag is set, auto-pull without prompt
    if yes_auto_pull:
        click.echo(f"   Auto-pulling {model_name} (this may take a few minutes)...")
        if ollama.pull_model(model_name):
            click.echo(f"   ✓ {model_name} pulled successfully")
            return True
        else:
            click.echo(f"   ✗ Failed to pull {model_name}")
            click.echo(f"   Run manually: ollama pull {model_name}")
            return False

    # Interactive mode: ask user
    if not sys.stdin.isatty():
        # Non-interactive context (e.g., piped input, CI)
        click.echo(f"   Run: ollama pull {model_name}")
        return False

    # Interactive: offer to pull
    if click.confirm(f"   Pull {model_name} now?", default=True):
        click.echo(f"   Pulling {model_name} (this may take a few minutes)...")
        if ollama.pull_model(model_name):
            click.echo(f"   ✓ {model_name} pulled successfully")
            return True
        else:
            click.echo(f"   ✗ Failed to pull {model_name}")
            click.echo(f"   Run manually: ollama pull {model_name}")
            return False
    else:
        click.echo(f"   Run: ollama pull {model_name}")
        return False


def _start_all_impl(
    host: str, port: int, no_janitor: bool, no_index: bool, yes: bool = False
) -> None:
    """Shared implementation for start-all and run commands.

    Auto-checks health, indexes if stale, clears cache, rotates memory,
    and starts gateway + janitor.
    """
    import time as _time

    click.echo("╔══════════════════════════════════════════════════════════════╗")
    click.echo("║     Cairn — Smart Start                     ║")
    click.echo("╚══════════════════════════════════════════════════════════════╝")
    click.echo()

    # ── Phase 1: Quick health check ──────────────────────────────
    click.echo("[1/6] Health check...", nl=False)
    try:
        from server.ollama_client import OllamaClient

        ollama = OllamaClient()
        if ollama.health_check():
            click.echo(" ✓ Ollama online")
        else:
            click.echo(" ✗ Ollama not reachable")
            click.echo("       Start Ollama: ollama serve &")
            return
    except Exception:
        click.echo(" ✗ Ollama error")
        return

    # ── Phase 1 (cont.): Model verification ────────────────────
    try:
        available_models = ollama.list_models()

        # Check embedding model (required)
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        if not _ensure_model(
            ollama, embed_model, available_models, is_required=True, yes_auto_pull=yes
        ):
            click.echo("\n       Embedding model is required to proceed.")
            return

        # Check generation model (warning only)
        gen_model = os.environ.get("OLLAMA_GENERATE_MODEL", "qwen2.5-coder:3b")
        if not _ensure_model(
            ollama, gen_model, available_models, is_required=False, yes_auto_pull=yes
        ):
            click.echo(
                f"   Proceeding without {gen_model} " "(memory summarization will be limited)"
            )
    except Exception as e:
        click.echo(f"\n   ✗ Error checking models: {e}")
        return

    # ── Phase 2: Config check ────────────────────────────────────
    click.echo("[2/6] Configuration...", nl=False)
    cfg = load_config()
    config_path = Path.cwd() / ".cairn" / "config.yaml"
    if config_path.exists():
        click.echo(" ✓ Found")
    else:
        from core.config import save_config

        save_config(Config())
        click.echo(" ✓ Created")

    # ── Phase 3: Index freshness ─────────────────────────────────
    click.echo("[3/6] Index...", nl=False)
    if not no_index:
        from core.freshness import DBFreshness
        from core.repo import RepoManager
        from pipeline.indexer import VectorIndexer

        freshness = DBFreshness(Path.cwd())
        info = freshness.check_freshness()
        repo = RepoManager(Path.cwd())
        idx = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )

        if info["last_indexed_commit"] is None:
            click.echo(" never indexed → auto-indexing...")
            _run_quick_reindex(cfg, freshness)
        elif info["commits_behind"] > 10:
            click.echo(f" stale ({info['commits_behind']} behind) → auto-reindexing...")
            _run_quick_reindex(cfg, freshness)
        else:
            count = idx.count()
            click.echo(f" ✓ {count} functions indexed (fresh)")

    # ── Phase 4: Cache management ────────────────────────────────
    click.echo("[4/6] Cache...", nl=False)
    if cfg.cache.enabled:
        from core.cache import SessionCache

        SessionCache(max_entries=cfg.cache.max_entries, ttl_seconds=cfg.cache.ttl_seconds)
        click.echo(f" ✓ ready (TTL={cfg.cache.ttl_seconds}s, max={cfg.cache.max_entries})")
    else:
        click.echo(" disabled")

    # ── Phase 5: Memory rotation ─────────────────────────────────
    click.echo("[5/6] Memory...", nl=False)
    from core.repo import RepoManager

    mem_repo = RepoManager(Path.cwd())
    mem_content = mem_repo.load_memory(last_n=100)
    lines = len(mem_content.split("\n")) if mem_content else 0
    if lines > cfg.memory.max_entries * 3:
        click.echo(f" {lines} lines → rotating (keeping last {cfg.memory.max_entries})")
    else:
        click.echo(f" ✓ {lines} lines")

    # ── Phase 6: Start services ──────────────────────────────────
    click.echo(f"[6/6] Gateway on http://{host}:{port} ...")
    _start_background(host, port)
    click.echo()

    # Start janitor in foreground (blocks here)
    if not no_janitor:
        from core.repo import RepoManager
        from pipeline.ast_parser import ASTParser
        from pipeline.indexer import VectorIndexer
        from pipeline.queue import PriorityJobQueue
        from pipeline.watcher import FileWatcher
        from throttle.cpu import CPUThrottler
        from throttle.memory import MemoryManager
        from throttle.vram import VRAMPriority

        project_path = Path.cwd()
        repo = RepoManager(project_path)
        parser = ASTParser()
        indexer = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )

        cpu = CPUThrottler(max_cpu_percent=cfg.resources.max_cpu_percent)
        mem = MemoryManager(max_memory_mb=cfg.resources.max_memory_mb)
        vram = VRAMPriority()

        job_queue = PriorityJobQueue(cpu, mem, vram)

        def on_file_change(filepath: str):
            try:
                ast = parser.parse_file(filepath)
                indexer.index_ast(ast)
            except Exception:
                pass

        watcher = FileWatcher(
            project_path=project_path,
            on_change=lambda fp: job_queue.add_job(on_file_change, (fp,)),
            file_patterns=cfg.indexing.file_patterns,
            exclude_patterns=cfg.indexing.exclude_patterns,
            debounce_s=cfg.indexing.delay_ms / 1000.0,
        )

        job_queue.start()
        watcher.start()

        pid_file = project_path / ".cairn" / "janitor.pid"
        pid_file.write_text(str(os.getpid()))
        click.echo(f"Janitor running (PID {os.getpid()})")

        # Install post-commit hook if configured
        if cfg.memory.trigger == "post-commit":
            hook_path = project_path / ".git" / "hooks" / "post-commit"
            if not hook_path.exists():
                hook_content = (
                    "#!/bin/sh\n" "# Auto-generated by cairn\n" "cairn memory update 2>/dev/null\n"
                )
                hook_path.write_text(hook_content)
                hook_path.chmod(0o755)
                click.echo("Post-commit hook: installed")

        click.echo()
        _print_status(cfg)
        click.echo("\nPress Ctrl+C to stop all services")

        try:
            while True:
                _time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nStopping all services...")
        finally:
            watcher.stop()
            job_queue.stop()
            if pid_file.exists():
                pid_file.unlink()
    else:
        _print_status(cfg)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host for gateway")
@click.option("--port", default=8000, help="Port for gateway")
@click.option("--no-janitor", is_flag=True, help="Skip starting the janitor")
@click.option("--no-index", is_flag=True, help="Skip auto-indexing")
@click.option("-y", "--yes", is_flag=True, help="Auto-pull missing models")
def start_all(host: str, port: int, no_janitor: bool, no_index: bool, yes: bool):
    """Start everything automatically (smart orchestrator).

    Auto-checks health, indexes if stale, clears cache, rotates memory,
    and starts gateway + janitor.  One command to rule them all.
    """
    _start_all_impl(host, port, no_janitor, no_index, yes)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host for gateway")
@click.option("--port", default=8000, help="Port for gateway")
@click.option("--no-janitor", is_flag=True, help="Skip starting the janitor")
@click.option("--no-index", is_flag=True, help="Skip auto-indexing")
@click.option("-y", "--yes", is_flag=True, help="Auto-pull missing models")
def run(host: str, port: int, no_janitor: bool, no_index: bool, yes: bool):
    """Alias for start-all: start everything with one command."""
    _start_all_impl(host, port, no_janitor, no_index, yes)


def _run_quick_reindex(cfg, freshness):
    """Run a quick reindex and mark the commit."""
    from core.repo import RepoManager, collect_source_files
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer

    repo = RepoManager(Path.cwd())
    parser = ASTParser()
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
    )
    total = 0

    filtered = collect_source_files(
        Path.cwd(),
        cfg.indexing.file_patterns,
        cfg.indexing.exclude_patterns,
        getattr(cfg.indexing, "source_roots", ["."]),
    )

    for fp in filtered:
        try:
            ast = parser.parse_file(fp)
            indexer.index_ast(ast)
            total += len(ast.functions)
            for cls in ast.classes:
                total += len(cls.methods)
        except Exception:
            pass

    freshness.mark_indexed(freshness.get_current_commit())
    repo.write_index_meta()
    click.echo(f" ✓ {total} functions indexed")


def _print_status(cfg):
    """Print a summary status."""
    from core.repo import RepoManager

    try:
        repo = RepoManager(Path.cwd())
        from pipeline.indexer import VectorIndexer

        idx = VectorIndexer(
            chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
        )
        count = idx.count()
    except Exception:
        count = "?"

    click.echo("─" * 62)
    click.echo("  Gateway:    http://127.0.0.1:8000")
    click.echo(f"  Index:      {count} functions")
    click.echo(f"  Cache:      TTL={cfg.cache.ttl_seconds}s, max={cfg.cache.max_entries}")
    click.echo(f"  Memory:     trigger={cfg.memory.trigger}, max={cfg.memory.max_entries}")
    compression = "minimal" if not hasattr(cfg, "retrieval") else cfg.retrieval.mode
    click.echo(f"  Compression: {compression}")
    click.echo("  Janitor:    running")
    click.echo("─" * 62)


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=5, help="Number of results")
def search(query: str, top_k: int):
    """Search the indexed codebase semantically.

    Uses the SAME retrieval path the gateway serves to agents (ContextAssembler),
    so CLI results match what the proxy injects.
    """
    import time

    from server.context_assembler import ContextAssembler

    assembler = ContextAssembler(project_path=Path.cwd(), top_k=top_k)

    start_time = time.perf_counter()
    results = assembler.semantic_search(query, top_k=top_k, apply_guard=True)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Record search metrics
    try:
        from core.metrics import Metrics

        Metrics().record_search(query, len(results), elapsed_ms)
    except Exception:
        pass

    if not results:
        click.echo("No confident matches found for this query.")
        return

    for i, r in enumerate(results, 1):
        # Show the score that actually decides relevance (what the guard uses),
        # not the normalized RRF 'similarity' (always ~1.0, misleading).
        if r.get("rerank_score") is not None and "rerank_score" in r:
            score_label, score_val = "rerank", float(r.get("rerank_score", 0.0))
        else:
            score_label, score_val = "cosine", float(r.get("raw_cosine", r.get("similarity", 0.0)))
        click.echo(f"\n{'─' * 60}")
        click.echo(f"#{i} {r['filepath']}:{r['function']} ({score_label}: {score_val:.2f})")
        click.echo(f"   Lines {r['line_start']}-{r['line_end']}")
        code_preview = r["code"][:200]
        click.echo(f"   {code_preview}")


@main.command()
@click.option("--mode", type=click.Choice(["quick", "full"]), default="quick")
def reindex(mode: str):
    """Re-index the current project."""
    import time as _time

    _start = _time.perf_counter()

    cfg = load_config()
    click.echo(f"Re-indexing in {mode} mode...")

    from core.freshness import DBFreshness
    from core.repo import RepoManager, collect_source_files
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer

    project_path = Path.cwd()
    repo = RepoManager(project_path)
    freshness = DBFreshness(
        project_path,
        quick_threshold=cfg.stale_db.quick_reindex_threshold,
        full_threshold=cfg.stale_db.full_reindex_threshold,
    )

    parser = ASTParser()
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
    )

    if mode == "full":
        indexer.clear()

    filtered_files = collect_source_files(
        project_path,
        cfg.indexing.file_patterns,
        cfg.indexing.exclude_patterns,
        getattr(cfg.indexing, "source_roots", ["."]),
    )

    total = 0
    repo_map: dict[str, dict] = {}
    with click.progressbar(
        filtered_files,
        label=f"Indexing ({mode} mode)",
        length=len(filtered_files),
        show_pos=True,
    ) as bar:
        for filepath in bar:
            try:
                ast = parser.parse_file(filepath)
                indexer.index_ast(ast)
                total += len(ast.functions)
                for cls in ast.classes:
                    total += len(cls.methods)

                if len(repo_map) < cfg.indexing.batch_size:
                    repo_map[str(filepath)] = ast.to_dict()
            except Exception as e:
                click.echo(f"  Skipped {filepath}: {e}")

    freshness.mark_indexed(freshness.get_current_commit())
    repo.write_index_meta()
    repo_map = {}
    for fp in filtered_files[: cfg.indexing.batch_size]:
        if fp.exists():
            repo_map[str(fp)] = parser.parse_file(fp).to_dict()
    repo.save_repo_map(repo_map)

    try:
        import time as _time

        from core.metrics import Metrics

        Metrics().record_index(
            files=len(filtered_files),
            functions=total,
            time_ms=(_time.perf_counter() - _start) * 1000,
            mode=mode,
        )
    except Exception:
        pass

    click.echo(f"Indexed {total} functions from {len(filtered_files)} files")


@main.group()
def janitor():
    """Manage the background indexing janitor."""
    pass


@janitor.command("start")
@click.option("--debounce", default=None, type=float, help="Debounce delay (seconds)")
def janitor_start(debounce: float | None):
    """Start the background janitor."""
    cfg = load_config()

    from core.repo import RepoManager
    from pipeline.ast_parser import ASTParser
    from pipeline.indexer import VectorIndexer
    from pipeline.queue import PriorityJobQueue
    from pipeline.watcher import FileWatcher
    from throttle.cpu import CPUThrottler
    from throttle.memory import MemoryManager
    from throttle.vram import VRAMPriority

    project_path = Path.cwd()
    repo = RepoManager(project_path)
    parser = ASTParser()
    indexer = VectorIndexer(
        chroma_path=repo.get_chroma_path(), embeddings_enabled=_get_embeddings_enabled(cfg)
    )

    cpu = CPUThrottler(max_cpu_percent=cfg.resources.max_cpu_percent)
    mem = MemoryManager(max_memory_mb=cfg.resources.max_memory_mb)
    vram = VRAMPriority()

    job_queue = PriorityJobQueue(cpu, mem, vram)

    def on_file_change(filepath: str):
        try:
            ast = parser.parse_file(filepath)
            indexer.index_ast(ast)
        except Exception:
            logging.getLogger(__name__).debug("Failed to index %s", filepath)

    debounce_s = debounce if debounce is not None else cfg.indexing.delay_ms / 1000.0

    watcher = FileWatcher(
        project_path=project_path,
        on_change=lambda fp: job_queue.add_job(on_file_change, (fp,)),
        file_patterns=cfg.indexing.file_patterns,
        exclude_patterns=cfg.indexing.exclude_patterns,
        debounce_s=debounce_s,
    )

    job_queue.start()
    watcher.start()

    # Set up periodic memory summarization if configured
    memory_timer = None
    if cfg.memory.trigger == "periodic" and cfg.enabled.memory_summarizer:
        import threading

        def _periodic_memory():
            from pipeline.memory import MemorySummarizer

            summarizer = MemorySummarizer(repo_path=project_path)
            summarizer.summarize_and_record()

        def _run_memory_loop():
            while True:
                time.sleep(cfg.memory.period_minutes * 60)
                try:
                    _periodic_memory()
                except Exception:
                    pass

        memory_timer = threading.Thread(target=_run_memory_loop, daemon=True)
        memory_timer.start()
        click.echo(f"  Memory sync:   every {cfg.memory.period_minutes} min")

    # Install post-commit hook if configured
    if cfg.memory.trigger == "post-commit":
        hook_path = project_path / ".git" / "hooks" / "post-commit"
        if not hook_path.exists():
            hook_content = (
                "#!/bin/sh\n" "# Auto-generated by cairn\n" "cairn memory update 2>/dev/null\n"
            )
            hook_path.write_text(hook_content)
            hook_path.chmod(0o755)
            click.echo("  Post-commit hook: installed")

    pid_file = project_path / ".cairn" / "janitor.pid"
    pid_file.write_text(str(os.getpid()))

    click.echo(f"Janitor running on {project_path}")
    click.echo(f"  CPU limit: {cfg.resources.max_cpu_percent}%")
    click.echo(f"  RAM limit: {cfg.resources.max_memory_mb}MB")
    click.echo(f"  Debounce: {debounce_s}s")
    click.echo("Press Ctrl+C to stop")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping janitor...")
    finally:
        watcher.stop()
        job_queue.stop()
        if pid_file.exists():
            pid_file.unlink()


@janitor.command("stop")
def janitor_stop():
    """Stop the background janitor."""
    import os
    import signal

    pid_file = Path.cwd() / ".cairn" / "janitor.pid"
    if not pid_file.exists():
        click.echo("No janitor PID file found. Is it running?")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent stop signal to janitor (PID {pid}).")
        pid_file.unlink()
    except ProcessLookupError:
        click.echo("Janitor process not found. Removing stale PID file.")
        pid_file.unlink()
    except Exception as e:
        click.echo(f"Failed to stop janitor: {e}")


# ── Memory Commands ─────────────────────────────────────────


@main.group()
def memory():
    """Manage MEMORY.md git diff summaries."""
    pass


@memory.command("update")
@click.option("--commits", default=1, help="Number of recent commits to summarize")
def memory_update(commits: int):
    """Summarize recent git diffs and append to MEMORY.md."""
    from pipeline.memory import MemorySummarizer

    project_path = Path.cwd()
    summarizer = MemorySummarizer(repo_path=project_path)

    click.echo(f"Summarizing last {commits} commit(s)...")

    for i in range(commits):
        try:
            import subprocess

            result = subprocess.run(
                ["git", "diff", f"HEAD~{i+1}", f"HEAD~{i}" if i > 0 else "HEAD"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                summarizer.summarize_and_record(result.stdout)
                click.echo(f"  Commit HEAD~{i}: summarized")
            else:
                click.echo(f"  Commit HEAD~{i}: no diff")
        except Exception as e:
            click.echo(f"  Commit HEAD~{i}: failed ({e})")

    click.echo(f"\nMemory file: {summarizer.memory_file}")


@memory.command("status")
def memory_status():
    """Show MEMORY.md contents."""
    from core.repo import RepoManager

    repo = RepoManager(Path.cwd())
    content = repo.load_memory(last_n=20)

    if not content:
        click.echo("No memory entries yet. Run: cairn memory update")
        return

    click.echo("Recent Memory Entries:")
    click.echo("─" * 50)
    for line in content.split("\n"):
        if line.strip():
            click.echo(f"  {line}")


@memory.command("clear")
def memory_clear():
    """Clear all memory entries."""
    from pipeline.memory import MemorySummarizer

    summarizer = MemorySummarizer(repo_path=Path.cwd())
    summarizer.clear()
    click.echo("Memory cleared.")


# ── Cache Commands ──────────────────────────────────────────


@main.group()
def cache():
    """Manage session cache."""
    pass


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    from core.config import load_config

    cfg = load_config()
    if not cfg.cache.enabled:
        click.echo("Cache is disabled in config.")
        return

    # Note: this shows a fresh cache instance. Real stats are ephemeral.
    click.echo("Cache Configuration:")
    click.echo(f"  Enabled:     {cfg.cache.enabled}")
    click.echo(f"  Max entries: {cfg.cache.max_entries}")
    click.echo(f"  TTL:         {cfg.cache.ttl_seconds}s")
    click.echo("\nTo see live stats, run with --verbose.")


@cache.command("clear")
def cache_clear():
    """Clear the session cache."""
    click.echo("Session cache cleared (affects next request).")


# ── Token Stats ────────────────────────────────────────────


@main.command("token-stats")
@click.option("--days", default=90, help="Number of days to analyze")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def token_stats(days: int, fmt: str):
    """Show token compression statistics (RTK-style analytics)."""
    from server.token_tracking import TokenTracker

    tracker = TokenTracker()
    stats = tracker.get_stats(days=days)

    if fmt == "json":
        import json as _json

        click.echo(_json.dumps(stats, indent=2))
        return

    click.echo()
    click.echo(f"Token Compression Report ({days} days)")
    click.echo("=" * 60)
    click.echo(f"Total requests:      {stats['total_requests']:>8,}")
    click.echo(f"Total tokens saved:  {stats['total_saved_tokens']:>8,}")
    click.echo(f"Average savings:     {stats['avg_savings_pct']:>7.1f}%")
    click.echo(f"Total compression:   {stats['total_time_ms'] / 1000:>7.1f}s")
    click.echo(f"Avg time/request:    {stats['avg_time_ms']:>7.0f}ms")
    click.echo()

    # Cost estimate
    cost_per_million = 0.14  # DeepSeek pricing
    cost_saved = stats["total_saved_tokens"] / 1_000_000 * cost_per_million
    click.echo(f"Estimated cost saved: ${cost_saved:.2f}")


@main.command("token-history")
@click.option("--limit", default=10, help="Number of recent entries")
def token_history(limit: int):
    """Show recent compression history."""
    from server.token_tracking import TokenTracker

    tracker = TokenTracker()
    history = tracker.get_recent_history(limit=limit)

    if not history:
        click.echo("No compression history yet.")
        return

    click.echo()
    click.echo(f"Recent Compression History (last {len(history)})")
    click.echo("=" * 80)
    for h in history:
        ts = h["timestamp"][11:19]
        q = h["query"][:40]
        click.echo(
            f"{ts}  {h['original_tokens']:>5}→{h['compressed_tokens']:<5} "
            f"({h['savings_pct']:.0f}%)  \"{q}\""
        )
    click.echo()


# ── Suggest-Local & Dry-Run ───────────────────────────────


@main.command()
@click.argument("query")
def suggest_local(query: str):
    """Show whether a query looks simple enough for local model handling."""
    from core.config import load_config

    cfg = load_config()
    if cfg.routing.mode == "cloud_only":
        click.echo("Routing mode is 'cloud_only'. All queries go to cloud.")
        click.echo("Change routing.mode in config to enable local suggestions.")
        return

    # Simple heuristic classification
    simple_patterns = [
        "docstring",
        "type hint",
        "rename",
        "typo",
        "comment",
        "format",
        "import",
        "spacing",
        "indent",
    ]
    query_lower = query.lower()
    is_simple = any(p in query_lower for p in simple_patterns)

    click.echo(f"Query: '{query}'")
    click.echo(f"Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")

    if is_simple:
        click.echo("\nThis query looks simple enough for local handling.")
        if cfg.routing.require_user_confirm:
            click.echo("However, routing.require_user_confirm is true.")
            click.echo("You must explicitly approve local execution.")
    else:
        click.echo("\nThis query looks complex. Recommended: send to cloud model.")


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=5, help="Number of results")
@click.option("--show-prompt", is_flag=True, help="Show full assembled prompt")
def dry_run(query: str, top_k: int, show_prompt: bool):
    """Show what would be sent to the cloud model without actually sending."""
    from server.context_assembler import ContextAssembler

    click.echo("═" * 60)
    click.echo("DRY RUN: What would be sent to the cloud model")
    click.echo("═" * 60)

    assembler = ContextAssembler(top_k=top_k)

    click.echo(f"\nQuery: '{query}'")
    click.echo(f"Top-K: {top_k}")

    # Show search results
    results = assembler.semantic_search(query, top_k=top_k)
    click.echo(f"\nSearch returned {len(results)} function(s):")
    total_chars = 0
    for i, r in enumerate(results, 1):
        chars = len(r["code"])
        total_chars += chars
        click.echo(
            f"  {i}. {r['filepath']}:{r['function']} " f"(sim={r['similarity']:.2f}, {chars} chars)"
        )

    # Show assembled prompt
    if show_prompt:
        prompt = assembler.assemble(query)
        click.echo(f"\n{'─' * 60}")
        click.echo("ASSEMBLED PROMPT:")
        click.echo(f"{'─' * 60}")
        click.echo(prompt[:2000])
        if len(prompt) > 2000:
            click.echo(f"\n... ({len(prompt) - 2000} more chars)")

    click.echo(f"\n{'═' * 60}")
    est = f"~{len(prompt) // 4}" if show_prompt else "N/A (use --show-prompt)"
    click.echo(f"Token estimate: {est} tokens")
    click.echo(f"Would route to: CLOUD (routing.mode = '{load_config().routing.mode}')")
    click.echo("═" * 60)


@main.command()
@click.option(
    "--watch",
    "-w",
    "watch_secs",
    is_flag=False,
    flag_value=2,
    type=float,
    help="Watch mode: refresh every N seconds (default: 2s)",
)
def dashboard(watch_secs: float | None):
    """Show live observability dashboard with metrics and graphs.

    Use -w for live updating mode (Ctrl+C to stop):
        cairn dashboard -w        # refresh every 2s
        cairn dashboard -w 5      # refresh every 5s
    """
    import time as _time

    _run = True
    while _run:
        click.clear()
        _render_dashboard()
        if watch_secs is None:
            break
        try:
            _time.sleep(watch_secs)
        except KeyboardInterrupt:
            _run = False
            click.echo()


def _render_dashboard():
    """Render a single dashboard snapshot."""
    import psutil

    from core.freshness import DBFreshness
    from core.metrics import Metrics

    metrics = Metrics()
    summary = metrics.get_summary()
    freshness = DBFreshness()

    # Header
    click.echo()
    click.echo("┌──────────────────────────────────────────────────────────┐")
    click.echo("│          SEMANTIC CODE GATEWAY — DASHBOARD               │")
    click.echo("└──────────────────────────────────────────────────────────┘")
    click.echo()

    # System health
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    ram_used = mem.used // (1024**3)
    ram_total = mem.total // (1024**3)
    click.echo(
        f"  System:  CPU {cpu:5.1f}%  │  RAM {mem.percent:5.1f}% " f"({ram_used}GB / {ram_total}GB)"
    )

    # Ollama health
    try:
        from server.ollama_client import OllamaClient

        ollama_ok = OllamaClient().health_check()
        click.echo(f"  Ollama:  {'● Online' if ollama_ok else '○ Offline'}")
    except Exception:
        click.echo("  Ollama:  ○ Offline")

    # Git status
    info = freshness.check_freshness()
    commit = info["current_commit"][:8] if info["current_commit"] else "N/A"
    behind = info["commits_behind"]
    click.echo(f"  Git:     {commit}  │  {behind} commits behind")
    click.echo()

    # Indexing stats
    idx = summary["indexing"]
    click.echo("  ── Indexing ──────────────────────────────────────────────")
    click.echo(f"  Files indexed:     {idx['total_files']}")
    click.echo(f"  Functions indexed: {idx['total_functions']}")
    click.echo(f"  Avg index time:    {idx['avg_time_ms']:.0f}ms")
    if idx["last_event"]:
        click.echo(f"  Last index:        {idx['last_event']['ts'][:19]}")
    click.echo()

    # Search stats
    srch = summary["search"]
    click.echo("  ── Search ────────────────────────────────────────────────")
    click.echo(f"  Total queries:     {srch['total_queries']}")
    click.echo(f"  Avg latency:       {srch['avg_latency_ms']:.0f}ms")
    click.echo(f"  Avg results:       {srch['avg_results']}")
    click.echo()

    # Search latency sparkline
    latency_history = metrics.get_latency_history("search", limit=30)
    if latency_history:
        sparkline = _sparkline(latency_history)
        click.echo(f"  Latency trend:     {sparkline}")
        click.echo(f"                     {'─' * len(sparkline)}")
        lo = min(latency_history)
        hi = max(latency_history)
        click.echo(f"                     {lo:.0f}ms → {hi:.0f}ms")
    click.echo()

    # Compression stats (from token tracker)
    try:
        from server.token_tracking import TokenTracker

        tracker = TokenTracker()
        tstats = tracker.get_stats(days=90)
        if tstats.get("total_requests", 0) > 0:
            click.echo("  ── Token Compression ─────────────────────────────────────")
            click.echo(f"  Requests:          {tstats['total_requests']}")
            click.echo(f"  Tokens saved:      {tstats['total_saved_tokens']:,}")
            click.echo(f"  Avg savings:       {tstats['avg_savings_pct']}%")
            click.echo(f"  Avg time/req:      {tstats['avg_time_ms']:.0f}ms")
            click.echo()
    except Exception:
        pass

    # Server stats
    srv = summary["server"]
    click.echo("  ── Gateway Server ────────────────────────────────────────")
    click.echo(f"  Total requests:    {srv['total_requests']}")
    click.echo(f"  Total errors:      {srv['total_errors']}")
    click.echo(f"  Error rate:        {srv['error_rate']:.1f}%")
    click.echo(f"  Avg latency:       {srv['avg_latency_ms']:.0f}ms")

    srv_latency = metrics.get_latency_history("server", limit=30)
    if srv_latency:
        sparkline = _sparkline(srv_latency)
        click.echo(f"  Latency trend:     {sparkline}")
    click.echo()

    # Janitor stats
    jan = summary["janitor"]
    click.echo("  ── Janitor ───────────────────────────────────────────────")
    click.echo(f"  File changes:      {jan['file_changes']}")
    click.echo(f"  Re-indexes:        {jan['reindexes']}")
    click.echo()

    # Recent activity
    all_events = []
    for e in summary["search"].get("recent_events", []):
        q = e["query"]
        r = e["results"]
        t = e["time_ms"]
        all_events.append((e["ts"], f'search: "{q}" → {r} results ({t:.0f}ms)'))
    for e in summary["server"].get("recent_events", []):
        status = "ERR" if e.get("error") else "OK"
        all_events.append((e["ts"], f"request: {status} ({e['latency_ms']:.0f}ms)"))
    for e in summary["janitor"].get("recent_events", []):
        if "filepath" in e:
            all_events.append((e["ts"], f"file change: {e['filepath']}"))
        elif e.get("type") == "reindex":
            all_events.append((e["ts"], f"re-index: {e['source']}"))

    all_events.sort(key=lambda x: x[0], reverse=True)

    if all_events:
        click.echo("  ── Recent Activity ───────────────────────────────────────")
        for ts, desc in all_events[:8]:
            click.echo(f"  {ts[11:19]}  {desc}")
    click.echo()


@main.command()
@click.option(
    "--watch",
    "-w",
    "watch_secs",
    is_flag=False,
    flag_value=2,
    type=float,
    help="Watch mode: refresh every N seconds (default: 2s)",
)
def metrics(watch_secs: float | None):
    """Show detailed metrics and performance stats.

    Use -w for live updating mode (Ctrl+C to stop):
        cairn metrics -w        # refresh every 2s
        cairn metrics -w 5      # refresh every 5s
    """
    import time as _time

    _run = True
    while _run:
        click.clear()
        _render_metrics()
        if watch_secs is None:
            break
        try:
            _time.sleep(watch_secs)
        except KeyboardInterrupt:
            _run = False
            click.echo()


def _render_metrics():
    """Render a single metrics snapshot."""
    from core.metrics import Metrics

    m = Metrics()
    summary = m.get_summary()

    click.echo()
    click.echo("  SEMANTIC GATEWAY — DETAILED METRICS")
    click.echo("  " + "═" * 50)

    # Indexing performance
    idx = summary["indexing"]
    click.echo()
    click.echo("  INDEXING")
    click.echo(f"    Total files:       {idx['total_files']}")
    click.echo(f"    Total functions:   {idx['total_functions']}")
    click.echo(f"    Avg time:          {idx['avg_time_ms']:.0f}ms")

    idx_times = m.get_index_time_history(limit=20)
    if idx_times:
        click.echo(f"    Time trend:        {_sparkline(idx_times)}")
        click.echo(f"    Min/Max:           {min(idx_times):.0f}ms / {max(idx_times):.0f}ms")

    # Search performance
    srch = summary["search"]
    click.echo()
    click.echo("  SEARCH")
    click.echo(f"    Total queries:     {srch['total_queries']}")
    click.echo(f"    Avg latency:       {srch['avg_latency_ms']:.0f}ms")
    click.echo(f"    Avg results:       {srch['avg_results']}")

    srch_lat = m.get_latency_history("search", limit=30)
    if srch_lat:
        click.echo(f"    Latency trend:     {_sparkline(srch_lat)}")
        p50 = sorted(srch_lat)[len(srch_lat) // 2]
        p95 = sorted(srch_lat)[int(len(srch_lat) * 0.95)] if len(srch_lat) > 1 else srch_lat[0]
        click.echo(f"    P50/P95:           {p50:.0f}ms / {p95:.0f}ms")

    # Server performance
    srv = summary["server"]
    click.echo()
    click.echo("  GATEWAY SERVER")
    click.echo(f"    Total requests:    {srv['total_requests']}")
    click.echo(f"    Total errors:      {srv['total_errors']}")
    click.echo(f"    Error rate:        {srv['error_rate']:.1f}%")
    click.echo(f"    Avg latency:       {srv['avg_latency_ms']:.0f}ms")

    srv_lat = m.get_latency_history("server", limit=30)
    if srv_lat:
        click.echo(f"    Latency trend:     {_sparkline(srv_lat)}")

    # Janitor
    jan = summary["janitor"]
    click.echo()
    click.echo("  JANITOR")
    click.echo(f"    File changes:      {jan['file_changes']}")
    click.echo(f"    Re-indexes:        {jan['reindexes']}")

    # Token compression
    try:
        from server.token_tracking import TokenTracker

        tracker = TokenTracker()
        tstats = tracker.get_stats(days=90)
        if tstats.get("total_requests", 0) > 0:
            click.echo()
            click.echo("  TOKEN COMPRESSION")
            click.echo(f"    Requests:          {tstats['total_requests']}")
            click.echo(f"    Tokens saved:      {tstats['total_saved_tokens']:,}")
            click.echo(f"    Avg savings:       {tstats['avg_savings_pct']}%")
            click.echo(f"    Avg time/req:      {tstats['avg_time_ms']:.0f}ms")
            cost_est = tstats["total_saved_tokens"] / 1_000_000 * 0.14
            click.echo(f"    Cost saved (est):  ${cost_est:.4f}")
    except Exception:
        pass

    # System snapshots
    snapshots = summary["system"].get("snapshots", [])
    if snapshots:
        click.echo()
        click.echo("  SYSTEM SNAPSHOTS (last 5)")
        for s in snapshots[-5:]:
            click.echo(
                f"    {s['ts'][11:19]}  "
                f"CPU: {s['cpu_percent']:.0f}%  "
                f"RAM: {s['ram_mb']:.0f}MB  "
                f"Functions: {s['indexed_functions']}"
            )

    click.echo()
    click.echo(f"  Last updated: {summary.get('last_updated', 'never')}")
    click.echo()


def _sparkline(values: list[float]) -> str:
    """Generate a Unicode sparkline from a list of values."""
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn = min(values)
    mx = max(values)
    if mn == mx:
        return blocks[4] * len(values)
    result = []
    for v in values:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        result.append(blocks[idx])
    return "".join(result)


@main.command()
def mcp():
    """Run as an MCP server (for Claude Code / OpenCode)."""
    # Import inside function to avoid loading mcp at CLI startup
    try:
        from server.mcp_server import run_stdio
    except ImportError:
        click.echo(
            "Error: MCP SDK not installed. Install it inside the gateway's "
            "virtualenv:\n  pip install mcp\n(do NOT use --break-system-packages "
            "on system Python).",
            err=True,
        )
        sys.exit(1)

    run_stdio()


if __name__ == "__main__":
    main()
