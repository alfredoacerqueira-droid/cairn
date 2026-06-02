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

from server.context_assembler import ContextAssembler
from server.workspace_router import WorkspaceRouter

logger = logging.getLogger(__name__)

# Fail-closed project binding: strict resolution without fallback to cwd.
_PROJECT_PATH: Path | None = None
_BIND_ERROR: str | None = None
_assembler: ContextAssembler | None = None
_router: WorkspaceRouter | None = None


def _classify_binding() -> tuple[str, Path | None, str | None]:
    """Classify CAIRN_PROJECT/GATEWAY_PROJECT as SINGLE, WORKSPACE, or UNBOUND.

    Returns:
        (mode, path, error) where:
          - SINGLE: path has .cairn/, error is None
          - WORKSPACE: path has no .cairn/ but children do, error is None
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

    # Check for SINGLE mode: this path itself has .cairn/
    cairn_dir = path / ".cairn"
    if cairn_dir.exists():
        return "SINGLE", path, None

    # Check for WORKSPACE mode: this path has children with .cairn/
    discovered = WorkspaceRouter.discover_repos(path)
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
        # If workspace router is bound, delegate to it
        if _router is not None:
            return _router.search(query, top_k=top_k)
        # Otherwise single-repo mode
        assembler = _get_assembler()
        if assembler is None:
            return (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/)."
            )
        # apply_guard=True so off-topic queries return "no confident matches"
        # rather than low-confidence noise (same gate as assemble_context).
        results = assembler.semantic_search(query, top_k=top_k, apply_guard=True)

        if not results:
            return "No confident matches found for this query."

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

        return "\n".join(lines)
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
        # If workspace router is bound, delegate to it
        if _router is not None:
            return _router.assemble(query)
        # Otherwise single-repo mode
        assembler = _get_assembler()
        if assembler is None:
            return (
                "Cairn MCP server has no bound project. Set CAIRN_PROJECT to an "
                "indexed repo (a dir containing .cairn/)."
            )
        return assembler.assemble_context(query)
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
        return result_msg + " Config updated."

    except Exception as e:
        return f"Error setting profile: {str(e)}"


def run_stdio() -> None:
    """Run the MCP server over stdio (for Claude Code / OpenCode).

    Resolves the binding mode (SINGLE, WORKSPACE, or UNBOUND) from
    CAIRN_PROJECT/GATEWAY_PROJECT and initializes accordingly.
    """
    global _PROJECT_PATH, _BIND_ERROR, _router
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
