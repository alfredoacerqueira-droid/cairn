"""Fast unit tests for multi-repo search in WorkspaceRouter.

Uses fake assemblers (no indexing, no embeddings, no flashrank).
Must run in <5s.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from server.workspace_router import WorkspaceRouter


class FakeStore:
    """Fake store with a .count() method for testing overview()."""

    def __init__(self, block_count: int = 0):
        """Initialize with a block count.

        Args:
            block_count: Number of indexed blocks to report.
        """
        self.block_count = block_count

    def count(self) -> int:
        """Return the block count."""
        return self.block_count


class FakeAssembler:
    """Fake ContextAssembler that returns canned results without any indexing."""

    def __init__(self, results: list[dict], block_count: int = 0):
        """Initialize with pre-canned results and optional block count.

        Args:
            results: List of result dicts (with 'filepath', 'function', 'code', etc.)
            block_count: Number of indexed blocks (for overview testing).
        """
        self.results = results
        self.store = FakeStore(block_count)

    def semantic_search(self, query: str, top_k: int = 5, apply_guard: bool = False):
        """Return canned results, optionally filtered by guard.

        Args:
            query: Ignored (we return the same results for any query).
            top_k: Return up to this many results.
            apply_guard: If True, apply a simple confidence guard.

        Returns:
            List of canned results (up to top_k).
        """
        results = self.results[:top_k]
        # Simple guard: if apply_guard and no results pass confidence, return []
        if apply_guard and results and results[0].get("rerank_score", 0) <= 0:
            return []
        return results


class TestRouteMulti:
    """Test the route_multi method."""

    def test_route_multi_single_repo_results(self):
        """route_multi with one repo returns its results tagged."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        router._assemblers = {repo_path: FakeAssembler(web_results)}

        merged = router.route_multi("checkout", top_k=5)

        assert len(merged) == 1
        assert merged[0]["repo"] == "web"
        assert merged[0]["repo_path"] == str(repo_path)
        assert merged[0]["function"] == "renderCheckout"

    def test_route_multi_multiple_repos_merged_ranked(self):
        """route_multi merges and ranks results from multiple repos."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        payments_path = Path("/ws/payments")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, payments_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        payments_results = [
            {
                "filepath": "billing.py",
                "function": "charge_card",
                "line_start": 20,
                "code": "def charge_card(): pass",
                "rerank_score": 0.9,
            }
        ]
        router._assemblers = {
            web_path: FakeAssembler(web_results),
            payments_path: FakeAssembler(payments_results),
        }

        merged = router.route_multi("payment", top_k=5)

        # Should have both results
        assert len(merged) == 2
        # charge_card (0.9) should rank before renderCheckout (0.6)
        assert merged[0]["function"] == "charge_card"
        assert merged[0]["repo"] == "payments"
        assert merged[1]["function"] == "renderCheckout"
        assert merged[1]["repo"] == "web"

    def test_route_multi_empty_results(self):
        """route_multi returns [] if all repos have no confident results."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        # Empty results (guard applied, nothing confident)
        router._assemblers = {repo_path: FakeAssembler([])}

        merged = router.route_multi("nonsense", top_k=5)

        assert merged == []

    def test_route_multi_one_repo_empty_one_has_results(self):
        """route_multi skips empty repos, returns results from non-empty."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        auth_path = Path("/ws/auth")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, auth_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        router._assemblers = {
            web_path: FakeAssembler(web_results),
            auth_path: FakeAssembler([]),
        }

        merged = router.route_multi("checkout", top_k=5)

        # Only web's result
        assert len(merged) == 1
        assert merged[0]["repo"] == "web"

    def test_route_multi_respects_top_k(self):
        """route_multi returns at most top_k results globally."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        payments_path = Path("/ws/payments")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, payments_path]

        web_results = [
            {
                "filepath": f"file{i}.py",
                "function": f"func{i}",
                "line_start": i,
                "code": "",
                "rerank_score": 0.5 + i * 0.01,
            }
            for i in range(5)
        ]
        payments_results = [
            {
                "filepath": f"pay{i}.py",
                "function": f"pay_func{i}",
                "line_start": i,
                "code": "",
                "rerank_score": 0.7 + i * 0.01,
            }
            for i in range(5)
        ]
        router._assemblers = {
            web_path: FakeAssembler(web_results),
            payments_path: FakeAssembler(payments_results),
        }

        merged = router.route_multi("anything", top_k=3)

        # Should return exactly 3 results (top_k), not all 10
        assert len(merged) == 3


class TestSearchAll:
    """Test the search_all method."""

    def test_search_all_single_repo(self):
        """search_all with one repo includes repo header and name."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        router._assemblers = {repo_path: FakeAssembler(web_results)}

        result = router.search_all("checkout", top_k=5)

        # Should include header with 1 repo
        assert "Searched 1 repos:" in result or "Searched 1 repos" in result
        # Should include repo name
        assert "[web]" in result
        # Should include the function
        assert "renderCheckout" in result
        # Should include relevance score
        assert "relevance: 0.600" in result

    def test_search_all_multiple_repos(self):
        """search_all merges results with [repo] tags for each."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        payments_path = Path("/ws/payments")
        auth_path = Path("/ws/auth")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, payments_path, auth_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        payments_results = [
            {
                "filepath": "billing.py",
                "function": "charge_card",
                "line_start": 20,
                "code": "def charge_card(): pass",
                "rerank_score": 0.9,
            }
        ]
        router._assemblers = {
            web_path: FakeAssembler(web_results),
            payments_path: FakeAssembler(payments_results),
            auth_path: FakeAssembler([]),
        }

        result = router.search_all("payment", top_k=5)

        # Should say "Searched 3 repos" (total number of repos)
        assert "Searched 3 repos:" in result
        # Should list the repos that have results (payments and web, not empty auth)
        assert "payments" in result
        assert "web" in result
        # Results should include [repo] tags
        assert "[payments]" in result
        assert "[web]" in result
        # charge_card should come before renderCheckout (0.9 > 0.6)
        payments_idx = result.find("charge_card")
        web_idx = result.find("renderCheckout")
        assert payments_idx < web_idx

    def test_search_all_no_matches(self):
        """search_all with no matches returns fail-closed message."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        router._assemblers = {repo_path: FakeAssembler([])}

        result = router.search_all("zzzzunusual", top_k=5)

        assert "Could not confidently determine" in result

    def test_search_all_formats_code_preview(self):
        """search_all includes code preview (first 200 chars)."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        code = "def render():\n    x = 1\n    y = 2\n    return x + y\n"
        results = [
            {
                "filepath": "checkout.py",
                "function": "render",
                "line_start": 10,
                "code": code,
                "rerank_score": 0.6,
            }
        ]
        router._assemblers = {repo_path: FakeAssembler(results)}

        result = router.search_all("render", top_k=5)

        assert "Code:" in result
        assert "def render()" in result


class TestAssembleAll:
    """Test the assemble_all method."""

    def test_assemble_all_single_repo(self):
        """assemble_all with one repo includes ## Repo: header."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        router._assemblers = {repo_path: FakeAssembler(results)}

        result = router.assemble_all("checkout", top_k=5)

        # Should have ## Repo: header
        assert "## Repo: web" in result
        # Should have ### for function
        assert "### checkout.py:renderCheckout" in result
        # Should include code
        assert "def renderCheckout()" in result

    def test_assemble_all_multiple_repos_grouped(self):
        """assemble_all groups results by repo with headers."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        payments_path = Path("/ws/payments")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, payments_path]

        web_results = [
            {
                "filepath": "checkout.py",
                "function": "renderCheckout",
                "line_start": 10,
                "code": "def renderCheckout(): pass",
                "rerank_score": 0.6,
            }
        ]
        payments_results = [
            {
                "filepath": "billing.py",
                "function": "charge_card",
                "line_start": 20,
                "code": "def charge_card(): pass",
                "rerank_score": 0.9,
            }
        ]
        router._assemblers = {
            web_path: FakeAssembler(web_results),
            payments_path: FakeAssembler(payments_results),
        }

        result = router.assemble_all("payment", top_k=5)

        # Should have both repo headers
        assert "## Repo: web" in result
        assert "## Repo: payments" in result
        # Should have function headers
        assert "### checkout.py:renderCheckout" in result
        assert "### billing.py:charge_card" in result

    def test_assemble_all_no_matches(self):
        """assemble_all with no matches returns fail-closed message."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        router._assemblers = {repo_path: FakeAssembler([])}

        result = router.assemble_all("zzzzunusual", top_k=5)

        assert "Could not confidently determine" in result


class TestOverview:
    """Test the overview method."""

    @patch("core.config.load_config")
    def test_overview_single_repo(self, mock_load_config):
        """overview() returns list with one repo's info."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        repo_path = Path("/ws/web")
        router.workspace_root = Path("/ws")
        router.repo_paths = [repo_path]

        # Mock config with profile
        mock_cfg = MagicMock()
        mock_cfg.profile = "python"
        mock_load_config.return_value = mock_cfg

        router._assemblers = {repo_path: FakeAssembler([], block_count=150)}

        overview = router.overview()

        assert len(overview) == 1
        assert overview[0]["name"] == "web"
        assert overview[0]["path"] == str(repo_path)
        assert overview[0]["profile"] == "python"
        assert overview[0]["blocks"] == 150

    @patch("core.config.load_config")
    def test_overview_multiple_repos(self, mock_load_config):
        """overview() returns list for multiple repos."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        payments_path = Path("/ws/payments")
        auth_path = Path("/ws/auth")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, payments_path, auth_path]

        # Mock config with profile
        mock_cfg = MagicMock()
        mock_cfg.profile = "python"
        mock_load_config.return_value = mock_cfg

        router._assemblers = {
            web_path: FakeAssembler([], block_count=100),
            payments_path: FakeAssembler([], block_count=250),
            auth_path: FakeAssembler([], block_count=75),
        }

        overview = router.overview()

        assert len(overview) == 3
        assert overview[0]["name"] == "web"
        assert overview[0]["blocks"] == 100
        assert overview[1]["name"] == "payments"
        assert overview[1]["blocks"] == 250
        assert overview[2]["name"] == "auth"
        assert overview[2]["blocks"] == 75

    @patch("core.config.load_config")
    def test_overview_error_handling(self, mock_load_config):
        """overview() gracefully handles errors per repo, sets blocks=0."""
        router = WorkspaceRouter.__new__(WorkspaceRouter)
        web_path = Path("/ws/web")
        auth_path = Path("/ws/auth")
        router.workspace_root = Path("/ws")
        router.repo_paths = [web_path, auth_path]

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.profile = "python"

        # Make load_config fail for auth repo
        def side_effect(path):
            if path == web_path:
                return mock_cfg
            raise Exception("Config load failed")

        mock_load_config.side_effect = side_effect

        router._assemblers = {
            web_path: FakeAssembler([], block_count=100),
            auth_path: FakeAssembler([], block_count=50),
        }

        overview = router.overview()

        assert len(overview) == 2
        # web should be fine
        assert overview[0]["name"] == "web"
        assert overview[0]["blocks"] == 100
        # auth should have blocks=0 due to error
        assert overview[1]["name"] == "auth"
        assert overview[1]["blocks"] == 0
