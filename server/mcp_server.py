"""MCP (Model Context Protocol) server exposing the gateway's retrieval engine.

This module exposes the semantic search and context assembly capabilities as
native MCP tools, allowing Claude Code and OpenCode to use the gateway directly
without needing to route through OpenAI-compatible endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server import FastMCP

from server.context_assembler import ContextAssembler

# Bind project path from environment or use cwd
_PROJECT_PATH = Path(os.getenv("CAIRN_PROJECT") or os.getenv("GATEWAY_PROJECT") or ".").resolve()
_assembler: ContextAssembler | None = None


def _get_assembler() -> ContextAssembler:
    """Get or create the shared ContextAssembler instance."""
    global _assembler
    if _assembler is None:
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
        assembler = _get_assembler()
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
        assembler = _get_assembler()
        return assembler.assemble_context(query)
    except Exception as e:
        return f"Context assembly error: {str(e)}"


@mcp.tool(description="Set the repository profile for retrieval strategy")
def set_profile(profile_name: str) -> str:
    """Set the repository profile (auto-detects or validates the choice).

    Profiles determine which retrieval legs are active and whether embeddings
    are enabled. Options: 'iac', 'dotnet', 'python', 'code', 'shell', 'auto'.

    Args:
        profile_name: The profile to set ('iac', 'dotnet', 'python', 'code',
                      'shell', or 'auto' for detection)

    Returns:
        Confirmation message with the set profile and its strategy.
    """
    try:
        from core.config import load_config, save_config
        from core.profiles import PROFILES, detect_profile, get_profile
        from core.repo import census_extensions

        if profile_name not in PROFILES and profile_name != "auto":
            return (
                f"Unknown profile: {profile_name}. "
                f"Available: {', '.join(PROFILES.keys())}, auto"
            )

        cfg = load_config(_PROJECT_PATH)

        if profile_name == "auto":
            # Auto-detect
            from core.repo import detect_source_layout

            detected_roots, _ = detect_source_layout(_PROJECT_PATH)
            ext_census = census_extensions(_PROJECT_PATH, source_roots=detected_roots)
            detected_name = detect_profile(ext_census)
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
    """Run the MCP server over stdio (for Claude Code / OpenCode)."""
    mcp.run()


if __name__ == "__main__":
    run_stdio()
