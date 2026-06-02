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
