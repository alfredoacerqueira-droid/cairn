"""MCP (Model Context Protocol) server exposing the gateway's retrieval engine.

This module exposes the semantic search and context assembly capabilities as
native MCP tools, allowing Claude Code and OpenCode to use the gateway directly
without needing to route through OpenAI-compatible endpoints.

Supports both single-repo and workspace modes:
  - SINGLE: CAIRN_PROJECT points to a repo with .cairn/ directory
  - WORKSPACE: CAIRN_PROJECT points to a parent directory with indexed child repos
  - UNBOUND: No valid binding found
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server import FastMCP

from core.semantic_cache import SemanticCache
from server.context_assembler import ContextAssembler
from server.orchestrator import Orchestrator, SessionBudget, emit
from server.workspace_router import WorkspaceRouter

logger = logging.getLogger(__name__)

# Fail-closed project binding: strict resolution without fallback to cwd.
_PROJECT_PATH: Path | None = None
_BIND_ERROR: str | None = None
_assembler: ContextAssembler | None = None
_router: WorkspaceRouter | None = None

# Per-session token budget (one per MCP process).
_session_budget: SessionBudget | None = None

# Per-project semantic caches, keyed by resolved project path.
_semantic_caches: dict[Path, SemanticCache] = {}


def _classify_binding() -> tuple[str, Path | None, str | None]:
    """Classify CAIRN_PROJECT/GATEWAY_PROJECT as SINGLE, WORKSPACE, or UNBOUND.

    Robustly handles the case where a path has both .cairn/ (could be SINGLE)
    AND indexed child repos (WORKSPACE). Prefers WORKSPACE when >=2 child repos
    are discovered, to better support multi-repo workspaces.

    Returns:
        (mode, path, error) where:
          - SINGLE: path has .cairn/ and no indexed children, error is None
          - WORKSPACE: path has >=2 indexed child repos (even if root has .cairn/), error is None
          - UNBOUND: path is None, error is an explanatory message
    """
    env_path = os.getenv("CAIRN_PROJECT") or os.getenv("GATEWAY_PROJECT")
    if not env_path:
        return (
            "UNBOUND",
            None,
            (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/) or a workspace root."
            ),
        )

    path = Path(env_path).resolve()
    if not path.exists():
        return "UNBOUND", None, f"CAIRN_PROJECT path does not exist: {env_path}"

    # Check for WORKSPACE mode first: this path has >=2 indexed children
    discovered = WorkspaceRouter.discover_repos(path)
    if len(discovered) >= 2:
        return "WORKSPACE", path, None

    # Check for SINGLE mode: this path itself has .cairn/
    cairn_dir = path / ".cairn"
    if cairn_dir.exists():
        return "SINGLE", path, None

    # Check for WORKSPACE mode with 1 child (edge case: single repo in a folder)
    if discovered:
        return "WORKSPACE", path, None

    # Neither SINGLE nor WORKSPACE
    return (
        "UNBOUND",
        None,
        f"CAIRN_PROJECT path is neither indexed (no .cairn/) nor a workspace "
        f"(no indexed child repos): {env_path}",
    )


def _resolve_project_path() -> tuple[Path | None, str | None]:
    """Resolve project path from environment (SINGLE mode only).

    Deprecated: use _classify_binding() for full mode detection.
    Kept for backward compatibility in tests.

    Returns:
        (Path, None) if bound, or (None, error_msg) if unbound.
    """
    mode, path, error = _classify_binding()
    if mode == "SINGLE":
        return path, None
    return None, error or "Unbound project"


def _get_session_budget(cfg) -> SessionBudget:
    """Get or create the per-process session budget.

    Lazily creates a SessionBudget with capacity = session_window * session_pct
    (e.g. 200_000 * 0.18 = 36_000 tokens by default).

    Args:
        cfg: Config object with .budget attributes.

    Returns:
        The global SessionBudget instance.
    """
    global _session_budget
    if _session_budget is None:
        cap = int(cfg.budget.session_window * cfg.budget.session_pct)
        _session_budget = SessionBudget(cap)
    return _session_budget


def _emit(text: str, cfg) -> str:
    """Budget-wrap tool output: per-tool cap + session budget charge.

    Args:
        text: Text to emit.
        cfg: Config object with .budget.tool_max_tokens.

    Returns:
        Text truncated to per-tool cap, then charged to session budget.
    """
    session_budget = _get_session_budget(cfg)
    return emit(text, cfg.budget.tool_max_tokens, session_budget)


def _get_cache(project_path: Path, cfg) -> SemanticCache:
    """Get or create a SemanticCache for a project.

    Caches instances by resolved project_path so we reuse one instance per
    project across multiple tool calls.

    Args:
        project_path: The project root path.
        cfg: Config object.

    Returns:
        SemanticCache instance (cached).
    """
    global _semantic_caches
    project_path = project_path.resolve()
    if project_path not in _semantic_caches:
        from pipeline.store.embedders import make_embedder

        cache_dir = project_path / ".cairn" / "cache" / "semantic"
        embedder = make_embedder(cfg)
        _semantic_caches[project_path] = SemanticCache(
            cache_dir,
            embedder,
            ttl_seconds=cfg.cache.semantic_ttl_seconds,
        )
    return _semantic_caches[project_path]


def reset_session_budget() -> None:
    """Reset the session budget (for test isolation).

    This is a test-only helper; never called in production.
    """
    global _session_budget
    _session_budget = None


def _get_assembler() -> ContextAssembler | None:
    """Get or create the shared ContextAssembler instance, or None if unbound."""
    global _assembler, _PROJECT_PATH, _BIND_ERROR
    if _BIND_ERROR is not None:
        return None
    if _assembler is None and _PROJECT_PATH is not None:
        _assembler = ContextAssembler(project_path=_PROJECT_PATH)
    return _assembler


# Create the FastMCP server
mcp = FastMCP("cairn")


@mcp.tool(description="Search for relevant code by semantic similarity")
def search_code(query: str, top_k: int = 5) -> str:
    """Search the codebase for functions semantically similar to the query.

    Args:
        query: The search query (e.g., "how do we validate user input?")
        top_k: Number of results to return (default: 5)

    Returns:
        A formatted string with matched functions, their locations, and code.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR
        # If workspace router is bound, use multi-repo search
        if _router is not None:
            from core.config import load_config

            result = _router.search_all(query, top_k=top_k)
            # For budget wrapping, use the first repo's config (or workspace root)
            # They share the same budget config in practice
            if _router.repo_paths:
                cfg = load_config(_router.repo_paths[0])
            else:
                cfg = load_config(_router.workspace_root)
            return _emit(result, cfg)
        # Otherwise single-repo mode
        assembler = _get_assembler()
        if assembler is None:
            return (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/)."
            )
        from core.config import load_config

        cfg = load_config(_PROJECT_PATH)
        # apply_guard=True so off-topic queries return "no confident matches"
        # rather than low-confidence noise (same gate as assemble_context).
        results = assembler.semantic_search(query, top_k=top_k, apply_guard=True)

        if not results:
            result = "No confident matches found for this query."
            return _emit(result, cfg)

        lines = []
        for i, result in enumerate(results, 1):
            filepath = result.get("filepath", "unknown")
            function = result.get("function", "unknown")
            line_start = result.get("line_start", "?")
            code = result.get("code", "")
            # Report the score that decides relevance (rerank if present, else raw
            # cosine) — not the normalized 'similarity' which is always ~1.0.
            if "rerank_score" in result:
                score_label, score_val = "relevance", float(result.get("rerank_score", 0.0))
            else:
                score_label = "relevance"
                score_val = float(result.get("raw_cosine", result.get("similarity", 0.0)))

            lines.append(f"{i}. {filepath}:{function} (line {line_start})")
            lines.append(f"   {score_label}: {score_val:.3f}")
            if code:
                code_preview = code[:200].replace("\n", "\n   ")
                lines.append(f"   Code: {code_preview}")
            lines.append("")

        return _emit("\n".join(lines), cfg)
    except Exception as e:
        return f"Search error: {str(e)}"


@mcp.tool(description="Assemble surgical context for a code query")
def assemble_context(query: str) -> str:
    """Assemble the complete surgical context for a query.

    Combines semantic search, repo map, and memory to create a token-compressed
    context suitable for sending to an LLM. Compression reduces token usage by
    20-90% depending on compression level (config.compression.level or
    COMPRESSION_LEVEL env). All consumers (CLI search, MCP tools, proxy) get
    the compressed output automatically.

    Args:
        query: The user's code query or request

    Returns:
        The assembled context as markdown, token-compressed.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR
        # If workspace router is bound, use multi-repo assembly
        if _router is not None:
            from core.config import load_config

            result = _router.assemble_all(query)
            # For budget wrapping, use the first repo's config
            if _router.repo_paths:
                cfg = load_config(_router.repo_paths[0])
            else:
                cfg = load_config(_router.workspace_root)
            return _emit(result, cfg)
        # Otherwise single-repo mode
        assembler = _get_assembler()
        if assembler is None:
            return (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/)."
            )
        from core.config import load_config

        cfg = load_config(_PROJECT_PATH)
        result = assembler.assemble_context(query)
        return _emit(result, cfg)
    except Exception as e:
        return f"Context assembly error: {str(e)}"


@mcp.tool(description="Set the repository profile for retrieval strategy")
def set_profile(profile_name: str) -> str:
    """Set the repository profile (auto-detects or validates the choice).

    Profiles determine which retrieval legs are active and whether embeddings
    are enabled. Options: 'iac', 'dotnet', 'python', 'code', 'shell', 'auto'.

    Note: set_profile requires a single-repo binding (not a workspace).

    Args:
        profile_name: The profile to set ('iac', 'dotnet', 'python', 'code',
                      'shell', or 'auto' for detection)

    Returns:
        Confirmation message with the set profile and its strategy.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR
        # set_profile only works in single-repo mode
        if _router is not None:
            return "set_profile requires a single-repo binding, not a workspace."
        if _PROJECT_PATH is None:
            return (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/)."
            )
        from core.config import load_config, save_config
        from core.profiles import PROFILES, detect_profile, get_profile
        from core.repo import census_extensions, detect_infra_markers, detect_source_layout

        if profile_name not in PROFILES and profile_name != "auto":
            return (
                f"Unknown profile: {profile_name}. "
                f"Available: {', '.join(PROFILES.keys())}, auto"
            )

        cfg = load_config(_PROJECT_PATH)

        if profile_name == "auto":
            # Auto-detect
            detected_roots, _ = detect_source_layout(_PROJECT_PATH)
            ext_census = census_extensions(_PROJECT_PATH, source_roots=detected_roots)
            has_infra_markers = detect_infra_markers(_PROJECT_PATH, source_roots=detected_roots)
            detected_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
            profile = get_profile(detected_name)
            cfg.profile = detected_name
            result_msg = (
                f"Auto-detected profile: {detected_name}. "
                f"Retrieval: {profile.retrieval_mode} "
                f"({', '.join(profile.legs)}). "
                f"Embeddings: {'ON' if profile.embedding_enabled else 'OFF'}."
            )
        else:
            # Explicit profile
            profile = get_profile(profile_name)
            cfg.profile = profile_name
            result_msg = (
                f"Profile set to: {profile_name}. "
                f"Retrieval: {profile.retrieval_mode} "
                f"({', '.join(profile.legs)}). "
                f"Embeddings: {'ON' if profile.embedding_enabled else 'OFF'}."
            )

        # Apply profile settings to config
        cfg.embeddings_enabled = profile.embedding_enabled
        if profile.embedding_enabled:
            cfg.indexing.embedding_model = profile.embedding_model

        save_config(cfg, _PROJECT_PATH)
        result = result_msg + " Config updated."
        return _emit(result, cfg)

    except Exception as e:
        return f"Error setting profile: {str(e)}"


@mcp.tool(description="Execute a query with orchestrated context assembly and LLM routing")
def orchestrate(query: str, instruction: str = "", payload: str = "") -> str:
    """Execute a query with smart context assembly and optional local LLM processing.

    Routes work to one of four execution paths based on token budget and LLM
    availability:
      - CONTEXT_ONLY: return enriched context (no LLM call)
      - LOCAL_ONE_SHOT: single local-LLM call
      - LOCAL_MAP_REDUCE: split + map-reduce execution
      - DEFER_TO_CLOUD: too big for local → context-only with marker

    All output is token-budget-capped per tool and per session.

    Args:
        query: The semantic search query.
        instruction: Optional task instruction (e.g., "summarize", "extract").
                    If empty, returns context-only.
        payload: Optional explicit payload (overrides assembled context).

    Returns:
        LLM output (if instruction provided and local LLM enabled) or
        context-only output (budget-capped).
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        # Resolve project path and assembler (workspace or single)
        if _router is not None:
            # Workspace mode: use multi-repo context (assemble_all)
            # Note: for LLM execution, we route to best repo for efficiency.
            # For context-only, we return merged multi-repo context.
            best_repo, _ = _router.route(query, top_k=5)
            if best_repo is None:
                return (
                    "Could not confidently determine which repo answers this query "
                    "(no confident match in any workspace repo)."
                )
            # Use best repo's assembler (for consistent execution)
            assembler = _router.assembler_for(best_repo)
            cfg_path = best_repo
        else:
            assembler = _get_assembler()
            if assembler is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )
            cfg_path = _PROJECT_PATH

        from core.config import load_config

        cfg = load_config(cfg_path)

        # Build LLM client if local_llm is enabled
        llm = None
        if cfg.local_llm.enabled:
            from server.ollama_client import make_llm_client

            llm = make_llm_client(cfg.local_llm)

        # Execute via orchestrator
        orch = Orchestrator(assembler, cfg, llm)
        result = orch.execute(query, payload or None, instruction or None)
        return _emit(result, cfg)
    except Exception as e:
        return f"Orchestration error: {str(e)}"


@mcp.tool(description="Retrieve a cached value by query")
def cache_get(query: str) -> str:
    """Retrieve a cached value by query (exact or semantic match).

    First tries EXACT match (O(1) filename probe). If miss, tries SEMANTIC
    match by scanning non-expired entries and comparing embeddings.

    Args:
        query: The prompt/query string.

    Returns:
        Cached value if found and not expired, or "CACHE_MISS" literal.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        # Resolve project path (SINGLE: _PROJECT_PATH; WORKSPACE: route to best repo)
        if _router is not None:
            best_repo, _ = _router.route(query, top_k=5)
            if best_repo is None:
                return "No repo matched for cache lookup."
            project_path = best_repo
        else:
            if _PROJECT_PATH is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )
            project_path = _PROJECT_PATH

        from core.config import load_config

        cfg = load_config(project_path)
        cache = _get_cache(project_path, cfg)
        hit = cache.get(query)
        if hit is None:
            return "CACHE_MISS"
        return _emit(hit, cfg)
    except Exception as e:
        return f"Cache lookup error: {str(e)}"


@mcp.tool(description="Store a value in the semantic cache")
def cache_set(query: str, value: str, ttl_seconds: int = 0) -> str:
    """Store a query-value pair in the semantic cache.

    Args:
        query: The prompt/query string.
        value: The response/cached value.
        ttl_seconds: Optional TTL override (0 = use cache default).

    Returns:
        Confirmation message "cached" or error.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        # Resolve project path (SINGLE: _PROJECT_PATH; WORKSPACE: route to best repo)
        if _router is not None:
            best_repo, _ = _router.route(query, top_k=5)
            if best_repo is None:
                return "No repo matched for cache storage."
            project_path = best_repo
        else:
            if _PROJECT_PATH is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )
            project_path = _PROJECT_PATH

        from core.config import load_config

        cfg = load_config(project_path)
        cache = _get_cache(project_path, cfg)
        cache.set(query, value, ttl_seconds or None)
        return "cached"
    except Exception as e:
        return f"Cache write error: {str(e)}"


@mcp.tool(
    description="List repos in the workspace with profiles and indexed block counts"
)
def list_repos() -> str:
    """List the repos in the current workspace with their profile and indexed block count.

    In WORKSPACE mode, shows all discovered indexed repos with their profile
    and block count. In SINGLE mode, shows just the bound repo. The agent can
    use this to understand the workspace scope and target searches appropriately.

    Returns:
        Formatted text listing repos: one line per repo with name, profile,
        block count, and path. Prefixed with workspace info if in WORKSPACE mode.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        if _router is not None:
            # WORKSPACE mode: list all discovered repos
            overview = _router.overview()

            if not overview:
                return (
                    f"Workspace at {_router.workspace_root} has no indexed repos "
                    "(no .cairn/ directories found in children)."
                )

            lines = [f"Workspace: {_router.workspace_root} ({len(overview)} repos)"]
            lines.append("")

            for item in overview:
                name = item.get("name", "unknown")
                profile = item.get("profile", "unknown")
                blocks = item.get("blocks", 0)
                path = item.get("path", "")
                lines.append(f"{name}  profile={profile}  blocks={blocks}  ({path})")

            result = "\n".join(lines)
            from core.config import load_config

            # Use first repo's config for budget wrapping (or workspace root)
            if _router.repo_paths:
                cfg = load_config(_router.repo_paths[0])
            else:
                cfg = load_config(_router.workspace_root)
            return _emit(result, cfg)

        else:
            # SINGLE mode: list the bound repo
            if _PROJECT_PATH is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )

            from core.config import load_config

            cfg = load_config(_PROJECT_PATH)
            assembler = _get_assembler()
            if assembler is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )

            try:
                blocks = assembler.store.count()
            except Exception as e:
                logger.warning("Error getting block count: %s", e)
                blocks = 0

            result = (
                f"{_PROJECT_PATH.name}  profile={cfg.profile}  "
                f"blocks={blocks}  (single-repo mode)"
            )
            return _emit(result, cfg)

    except Exception as e:
        return f"Error listing repos: {str(e)}"


@mcp.tool(description="Record a durable note to Cairn memory for this session")
def remember(note: str) -> str:
    """Record a durable note to Cairn memory (continuous memory across the workspace/repo).

    In WORKSPACE mode, writes to the workspace-level memory at <workspace>/.cairn/memory.md.
    In SINGLE mode, writes to the repo-level memory at <repo>/.cairn/memory.md.
    These notes persist across sessions and can be recalled with the recall() tool.

    Args:
        note: The memory note to record.

    Returns:
        Confirmation message indicating where the note was stored.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        if _router is not None:
            # WORKSPACE mode: write to workspace memory
            _router.write_memory(note)
            return "remembered (workspace)"
        else:
            # SINGLE mode: write to repo memory
            if _PROJECT_PATH is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )
            from core.repo import RepoManager

            repo = RepoManager(_PROJECT_PATH)
            repo.append_memory(note)
            return f"remembered (repo: {_PROJECT_PATH.name})"
    except Exception as e:
        return f"Memory write error: {str(e)}"


@mcp.tool(description="Recall recent Cairn memory (workspace + per-repo per memory.scope)")
def recall(max_entries: int = 10) -> str:
    """Recall recent Cairn memory entries.

    In WORKSPACE mode, returns memory entries according to memory.scope config:
      - 'workspace': workspace memory only
      - 'repo': per-repo memories across all repos
      - 'both' (default): workspace memory + per-repo memories
    In SINGLE mode, returns the bound repo's memory.

    Args:
        max_entries: Maximum number of memory entries to recall (hint, not strict).

    Returns:
        Recent memory entries formatted with headers, or empty string if no memory.
    """
    try:
        if _BIND_ERROR is not None:
            return _BIND_ERROR

        if _router is not None:
            # WORKSPACE mode: read per scope, with token cap
            from core.config import load_config

            if _router.repo_paths:
                cfg = load_config(_router.repo_paths[0])
            else:
                cfg = load_config(_router.workspace_root)
            result = _router.read_memory(max_tokens=cfg.budget.tool_max_tokens)
            return _emit(result, cfg) if result else ""
        else:
            # SINGLE mode: read repo memory
            if _PROJECT_PATH is None:
                return (
                    "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                    "indexed repo (a dir containing .cairn/)."
                )
            from core.config import load_config
            from core.repo import RepoManager

            cfg = load_config(_PROJECT_PATH)
            repo = RepoManager(_PROJECT_PATH)
            result = repo.load_memory(
                last_n=max_entries, max_tokens=cfg.budget.tool_max_tokens
            )
            if result:
                result = f"## Repo memory\n{result}"
            return _emit(result, cfg) if result else ""
    except Exception as e:
        return f"Memory recall error: {str(e)}"


def run_stdio() -> None:
    """Run the MCP server over stdio (for Claude Code / OpenCode).

    Resolves the binding mode (SINGLE, WORKSPACE, or UNBOUND) from
    CAIRN_PROJECT/GATEWAY_PROJECT and initializes accordingly.

    Logging is configured to go to stderr or file (NEVER stdout) to preserve
    the JSON-RPC protocol on stdout.
    """
    from core.logging_setup import configure_logging

    global _PROJECT_PATH, _BIND_ERROR, _router

    # Configure logging (respects CAIRN_DEBUG env var)
    configure_logging()

    mode, path, error = _classify_binding()

    if mode == "SINGLE":
        _PROJECT_PATH = path
        _BIND_ERROR = None
        logger.info("Cairn MCP bound to single repo: %s", _PROJECT_PATH)
    elif mode == "WORKSPACE":
        _router = WorkspaceRouter(path)
        discovered_names = [p.name for p in _router.repo_paths]
        logger.info(
            "Cairn MCP bound to workspace %s (%d repos: %s)",
            path,
            len(_router.repo_paths),
            ", ".join(discovered_names),
        )
    else:  # UNBOUND
        _BIND_ERROR = error
        logger.error("Cairn MCP: %s", _BIND_ERROR)

    mcp.run()


if __name__ == "__main__":
    run_stdio()
