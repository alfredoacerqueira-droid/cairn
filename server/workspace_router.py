"""Workspace router for multi-repo MCP binding.

Allows a single Cairn MCP server to be bound to a workspace root and route queries
to the appropriate sibling repo based on semantic relevance. Each repo remains
hard-isolated with its own .cairn/ directory and project_id.
"""

from __future__ import annotations

import logging
from pathlib import Path

from server.context_assembler import ContextAssembler

logger = logging.getLogger(__name__)


class WorkspaceRouter:
    """Routes queries across multiple indexed repos to the most relevant one.

    Discovers sibling repos by looking for .cairn/ directories in immediate
    children of the workspace root. Each repo is lazily initialized with its own
    ContextAssembler, ensuring hard project isolation.
    """

    @staticmethod
    def discover_repos(workspace_root: Path) -> list[Path]:
        """Discover indexed repos in a workspace.

        Scans immediate children of workspace_root for directories containing
        a .cairn/ subdirectory (indicating they have been indexed).

        Args:
            workspace_root: The workspace root directory.

        Returns:
            List of child directories that contain .cairn/, ordered by name.
            Does NOT include the workspace_root itself.
        """
        repos = []
        try:
            for child in sorted(workspace_root.iterdir()):
                if child.is_dir() and (child / ".cairn").exists():
                    repos.append(child)
        except (OSError, PermissionError):
            pass
        return repos

    def __init__(self, workspace_root: Path):
        """Initialize the workspace router.

        Args:
            workspace_root: The workspace root directory.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.repo_paths = self.discover_repos(self.workspace_root)
        self._assemblers: dict[Path, ContextAssembler] = {}
        self._workspace_repo = None  # Lazy-loaded workspace-level RepoManager

    def _get_assembler(self, repo_path: Path) -> ContextAssembler:
        """Get or lazily construct the ContextAssembler for a repo.

        Args:
            repo_path: The repo path.

        Returns:
            The ContextAssembler for that repo (cached).
        """
        if repo_path not in self._assemblers:
            self._assemblers[repo_path] = ContextAssembler(project_path=repo_path)
        return self._assemblers[repo_path]

    def assembler_for(self, repo_path: Path) -> ContextAssembler:
        """Get the lazily-built ContextAssembler for a given repo.

        Public API for external callers (e.g., MCP tools).

        Args:
            repo_path: The repo path.

        Returns:
            The ContextAssembler for that repo (cached).
        """
        return self._get_assembler(repo_path)

    def route(self, query: str, top_k: int = 5) -> tuple[Path | None, list[dict]]:
        """Route a query to the most relevant repo.

        Searches each repo using its assembler, scores by the top result's
        relevance, and returns the winning repo and its results.

        Args:
            query: The search query.
            top_k: Number of results per repo.

        Returns:
            (best_repo_path, results) where best_repo_path is the repo with the
            highest-scoring top result, or (None, []) if all repos returned empty.
        """
        best_repo = None
        best_score = -1.0
        best_results = []

        for repo_path in self.repo_paths:
            try:
                assembler = self._get_assembler(repo_path)
                results = assembler.semantic_search(query, top_k=top_k, apply_guard=True)

                # Score this repo by its top result's relevance
                if results:
                    top_result = results[0]
                    # Use rerank_score if available, else raw_cosine, else similarity
                    if "rerank_score" in top_result and top_result["rerank_score"] > 0:
                        score = float(top_result["rerank_score"])
                    else:
                        score = float(
                            top_result.get("raw_cosine", top_result.get("similarity", 0.0))
                        )

                    if score > best_score:
                        best_score = score
                        best_repo = repo_path
                        best_results = results
            except Exception as e:
                logger.warning("Error searching repo %s: %s", repo_path, e)
                continue

        return best_repo, best_results

    def search(self, query: str, top_k: int = 5) -> str:
        """Search across all repos and return formatted results.

        Routes the query to the best-matching repo and formats results with
        a repo header showing which repo was selected.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Formatted search results, or a fail-closed message if no repo matched.
        """
        best_repo, results = self.route(query, top_k=top_k)

        if best_repo is None or not results:
            return (
                "Could not confidently determine which repo answers this query "
                "(no confident match in any workspace repo)."
            )

        # Format like mcp_server.search_code, but with repo prefix
        repo_name = best_repo.name
        lines = [f"Repo: {repo_name}"]
        lines.append("")

        for i, result in enumerate(results, 1):
            filepath = result.get("filepath", "unknown")
            function = result.get("function", "unknown")
            line_start = result.get("line_start", "?")
            code = result.get("code", "")

            # Report the score that decides relevance
            if "rerank_score" in result and result.get("rerank_score", 0.0) > 0:
                score_label, score_val = "relevance", float(result.get("rerank_score", 0.0))
            else:
                score_label, score_val = "relevance", float(
                    result.get("raw_cosine", result.get("similarity", 0.0))
                )

            lines.append(f"{i}. {filepath}:{function} (line {line_start})")
            lines.append(f"   {score_label}: {score_val:.3f}")
            if code:
                code_preview = code[:200].replace("\n", "\n   ")
                lines.append(f"   Code: {code_preview}")
            lines.append("")

        return "\n".join(lines)

    def assemble(self, query: str) -> str:
        """Assemble surgical context from the best-matching repo.

        Routes the query to the best repo and returns its assembled context,
        prefixed with a repo header.

        Args:
            query: The search query.

        Returns:
            Assembled context with repo header, or fail-closed message.
        """
        best_repo, _ = self.route(query, top_k=5)

        if best_repo is None:
            return (
                "Could not confidently determine which repo answers this query "
                "(no confident match in any workspace repo)."
            )

        repo_name = best_repo.name
        assembler = self._get_assembler(best_repo)
        assembled = assembler.assemble_context(query)

        # Prefix with repo name
        return f"# Repo: {repo_name}\n\n{assembled}"

    def route_multi(self, query: str, top_k: int = 8) -> list[dict]:
        """Fan out to all repos and return merged, ranked results.

        Searches each repo and merges results across repos, then ranks by
        relevance score (rerank_score > 0, else raw_cosine, else similarity).
        Each result is tagged with 'repo' (repo name) and 'repo_path' (full path).
        Hard isolation: each repo searched independently; merge only in memory.

        Args:
            query: The search query.
            top_k: Number of top results to return across all repos.

        Returns:
            List of result dicts, sorted by score descending, or empty list if
            no confident matches anywhere (fail-closed).
        """
        merged = []
        repo_scores = {}  # Track best score per repo for ordering repos in output

        for repo_path in self.repo_paths:
            try:
                assembler = self._get_assembler(repo_path)
                results = assembler.semantic_search(query, top_k=top_k, apply_guard=True)

                for result in results:
                    # Tag with repo info
                    result["repo"] = repo_path.name
                    result["repo_path"] = str(repo_path)
                    merged.append(result)

                    # Track best score for this repo
                    if results:
                        top_result = results[0]
                        if "rerank_score" in top_result and top_result["rerank_score"] > 0:
                            score = float(top_result["rerank_score"])
                        else:
                            score = float(
                                top_result.get("raw_cosine", top_result.get("similarity", 0.0))
                            )
                        if repo_path not in repo_scores or score > repo_scores[repo_path]:
                            repo_scores[repo_path] = score
            except Exception as e:
                logger.warning("Error searching repo %s: %s", repo_path, e)
                continue

        if not merged:
            return []

        # Sort merged results by score descending
        def get_score(r: dict) -> float:
            if "rerank_score" in r and r["rerank_score"] > 0:
                return float(r["rerank_score"])
            return float(r.get("raw_cosine", r.get("similarity", 0.0)))

        merged.sort(key=get_score, reverse=True)
        return merged[:top_k]

    def search_all(self, query: str, top_k: int = 8) -> str:
        """Search all repos and return merged formatted results.

        Combines results from all repos with repo labels, ranked by relevance.
        Each result includes [repo_name] tag for LLM provenance tracking.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Formatted multi-repo search results, or fail-closed message if no
            confident matches anywhere.
        """
        merged = self.route_multi(query, top_k=top_k)

        if not merged:
            return (
                "Could not confidently determine which repo answers this query "
                "(no confident match in any workspace repo)."
            )

        # Gather unique repo names from results (in order of appearance)
        repo_names = []
        seen_repos = set()
        for r in merged:
            repo = r.get("repo", "unknown")
            if repo not in seen_repos:
                repo_names.append(repo)
                seen_repos.add(repo)

        lines = [f"Searched {len(self.repo_paths)} repos: {', '.join(repo_names)}"]
        lines.append("")

        for i, result in enumerate(merged, 1):
            repo = result.get("repo", "unknown")
            filepath = result.get("filepath", "unknown")
            function = result.get("function", "unknown")
            line_start = result.get("line_start", "?")
            code = result.get("code", "")

            # Report the score that decides relevance
            if "rerank_score" in result and result.get("rerank_score", 0.0) > 0:
                score_label, score_val = "relevance", float(result.get("rerank_score", 0.0))
            else:
                score_label, score_val = "relevance", float(
                    result.get("raw_cosine", result.get("similarity", 0.0))
                )

            lines.append(f"{i}. [{repo}] {filepath}:{function} (line {line_start})")
            lines.append(f"   {score_label}: {score_val:.3f}")
            if code:
                code_preview = code[:200].replace("\n", "\n   ")
                lines.append(f"   Code: {code_preview}")
            lines.append("")

        return "\n".join(lines)

    def assemble_all(self, query: str, top_k: int = 8) -> str:
        """Assemble multi-repo context from all matching repos.

        Groups merged results by repo and formats as markdown sections.
        Prepends workspace memory (if enabled) before per-repo sections.
        Does NOT call full assemble_context (too expensive); formats the
        merged search results with repo headers.

        Args:
            query: The search query.
            top_k: Number of results per repo (soft limit, sorted globally).

        Returns:
            Markdown with optional workspace memory, '## Repo: name' headers,
            and code blocks, or fail-closed.
        """
        lines = []

        # Prepend workspace memory if scope includes workspace
        scope = self.resolve_scope()
        if scope in ("workspace", "both"):
            ws_mem = self.read_memory(scope="workspace", max_tokens=2000)
            if ws_mem:
                lines.append(ws_mem)
                lines.append("")

        merged = self.route_multi(query, top_k=top_k)

        if not merged:
            fail_msg = (
                "Could not confidently determine which repo answers this query "
                "(no confident match in any workspace repo)."
            )
            if lines:
                # If we have workspace memory, include fail-closed msg after it
                lines.append(fail_msg)
                return "\n".join(lines)
            return fail_msg

        # Group by repo, preserving order of first appearance
        repos_dict: dict[str, list[dict]] = {}
        repo_order = []
        for result in merged:
            repo = result.get("repo", "unknown")
            if repo not in repos_dict:
                repos_dict[repo] = []
                repo_order.append(repo)
            repos_dict[repo].append(result)

        for repo in repo_order:
            results = repos_dict[repo]
            lines.append(f"## Repo: {repo}")
            lines.append("")

            for result in results:
                filepath = result.get("filepath", "unknown")
                function = result.get("function", "unknown")
                line_start = result.get("line_start", "?")
                code = result.get("code", "")

                lines.append(f"### {filepath}:{function} (line {line_start})")
                lines.append("")
                if code:
                    lines.append("```")
                    lines.append(code[:500])  # Longer preview for assemble
                    lines.append("```")
                    lines.append("")

            lines.append("")

        return "\n".join(lines)

    def overview(self) -> list[dict]:
        """List all repos in the workspace with their profiles and block counts.

        Returns a cheap overview of each repo: name, path, configured profile,
        and indexed block count. Used by list_repos() MCP tool.

        Returns:
            List of dicts with keys: 'name', 'path', 'profile', 'blocks'.
            On error for a repo, 'blocks' is 0 and profile is 'unknown'.
            Always succeeds (fail-closed).
        """
        from core.config import load_config

        overview_list = []
        for repo_path in self.repo_paths:
            profile = "unknown"
            blocks = 0
            try:
                cfg = load_config(repo_path)
                profile = cfg.profile
                assembler = self._get_assembler(repo_path)
                blocks = assembler.store.count()
            except Exception as e:
                logger.warning("Error loading overview for repo %s: %s", repo_path, e)

            overview_list.append(
                {
                    "name": repo_path.name,
                    "path": str(repo_path),
                    "profile": profile,
                    "blocks": blocks,
                }
            )

        return overview_list

    def _get_workspace_repo(self):
        """Get or lazily construct the workspace-level RepoManager.

        The workspace memory lives at <workspace_root>/.cairn/memory.md,
        reusing the same RepoManager format and rotation logic.

        Returns:
            RepoManager instance for the workspace root (cached).
        """
        # Handle case where object was constructed without __init__ (test stubs)
        if not hasattr(self, "_workspace_repo"):
            self._workspace_repo = None

        if self._workspace_repo is None:
            from core.repo import RepoManager

            self._workspace_repo = RepoManager(self.workspace_root)
        return self._workspace_repo

    def resolve_scope(self) -> str:
        """Resolve the configured memory scope for this workspace.

        Reads the config from the workspace root and returns the scope.
        If scope is 'auto', returns 'both' (we ARE a workspace).
        Otherwise returns the configured value.

        Returns:
            'workspace', 'repo', or 'both'.
        """
        from core.config import load_config

        # Handle case where object was constructed without __init__ (test stubs)
        if not hasattr(self, "workspace_root"):
            return "auto"

        try:
            cfg = load_config(self.workspace_root)
            scope = getattr(cfg.memory, "scope", "auto")
        except Exception:
            scope = "auto"

        # 'auto' means 'both' when we're in a workspace
        if scope == "auto":
            return "both"
        return scope

    def write_memory(self, note: str, scope: str | None = None) -> None:
        """Record a durable note to workspace memory.

        Args:
            note: The memory note to append.
            scope: Memory scope override. If None, uses resolve_scope().
                   For simplicity: workspace/both -> workspace memory.
        """
        if scope is None:
            scope = self.resolve_scope()

        # Both 'workspace' and 'both' scopes write to workspace memory
        if scope in ("workspace", "both"):
            workspace_repo = self._get_workspace_repo()
            workspace_repo.append_memory(note)
        elif scope == "repo":
            # Per-repo scope in workspace mode: not clearly bound to a single repo.
            # Log and write to workspace as safe default.
            logger.warning(
                "write_memory called with 'repo' scope in workspace mode; "
                "writing to workspace memory as default"
            )
            workspace_repo = self._get_workspace_repo()
            workspace_repo.append_memory(note)

    def read_memory(
        self, max_tokens: int = 4000, scope: str | None = None
    ) -> str:
        """Read recent memory entries within token budget.

        Args:
            max_tokens: Total token budget for the returned memory string.
            scope: Memory scope override. If None, uses resolve_scope().

        Returns:
            Token-budgeted memory string with headers (empty if no entries).
        """
        from core.tokens import count_tokens, truncate_to_tokens

        if scope is None:
            scope = self.resolve_scope()

        if scope == "workspace":
            # Workspace memory only
            workspace_repo = self._get_workspace_repo()
            mem = workspace_repo.load_memory(last_n=20, max_tokens=max_tokens)
            if not mem:
                return ""
            return f"## Workspace memory\n{mem}"

        elif scope == "repo":
            # Per-repo memories (cross all repos)
            lines = []
            remaining_tokens = max_tokens
            for repo_path in self.repo_paths:
                from core.repo import RepoManager

                repo = RepoManager(repo_path)
                # Allocate tokens evenly across repos (simple approach)
                per_repo_budget = remaining_tokens // max(1, len(self.repo_paths) - len(lines) + 1)
                if per_repo_budget <= 0:
                    break
                mem = repo.load_memory(last_n=10, max_tokens=per_repo_budget)
                if mem:
                    lines.append(f"## Repo: {repo_path.name}")
                    lines.append(mem)
                    lines.append("")
                    remaining_tokens -= count_tokens(mem)
            return "\n".join(lines).strip()

        else:  # scope == "both"
            # Workspace memory + per-repo memories, within total max_tokens
            lines = []
            total_so_far = 0

            # Workspace memory first (e.g., 2000 tokens max)
            workspace_cap = max_tokens // 2
            workspace_repo = self._get_workspace_repo()
            ws_mem = workspace_repo.load_memory(last_n=20, max_tokens=workspace_cap)
            if ws_mem:
                lines.append("## Workspace memory")
                lines.append(ws_mem)
                lines.append("")
                total_so_far = count_tokens(ws_mem)

            # Per-repo memories (remaining budget)
            remaining = max_tokens - total_so_far
            for repo_path in self.repo_paths:
                if remaining <= 0:
                    break
                from core.repo import RepoManager

                repo = RepoManager(repo_path)
                per_repo_budget = remaining // max(1, len(self.repo_paths))
                mem = repo.load_memory(last_n=10, max_tokens=per_repo_budget)
                if mem:
                    lines.append(f"## Repo: {repo_path.name}")
                    lines.append(mem)
                    lines.append("")
                    total_so_far += count_tokens(mem)
                    remaining = max_tokens - total_so_far

            # Final cap to ensure we stay within budget
            full_text = "\n".join(lines).strip()
            if count_tokens(full_text) > max_tokens:
                full_text = truncate_to_tokens(full_text, max_tokens)

            return full_text
