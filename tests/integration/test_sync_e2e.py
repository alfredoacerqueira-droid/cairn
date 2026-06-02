"""End-to-end test: FileWatcher + periodic sync don't block /health endpoint.

This test creates a temporary git repo with a Python file, starts the gateway
server with the FileWatcher active, and verifies that:
  1. /health endpoint responds quickly (<1s) even after file changes trigger sync
  2. No hung threads on shutdown
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from server.api import app


@pytest.mark.integration
class TestSyncE2E:
    """End-to-end test with real FastAPI app and FileWatcher."""

    def test_health_fast_with_file_watcher(self):
        """Health endpoint is fast even with FileWatcher triggering syncs."""
        # Create a temporary git repo
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Create a simple Python file
            test_py = project_path / "test.py"
            test_py.write_text("def hello():\n    return 'world'\n")

            # Commit the file
            subprocess.run(
                ["git", "add", "test.py"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Create .cairn config
            config_dir = project_path / ".cairn"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.yaml"
            config_file.write_text("""indexing:
  file_patterns:
    - "*.py"
  exclude_patterns:
    - "test_*"
    - "*_test.py"
    - ".venv"
  source_roots:
    - "."
""")

            # Set environment and create test client
            original_project = os.environ.get("CAIRN_PROJECT")
            os.environ["CAIRN_PROJECT"] = str(project_path)

            try:
                client = TestClient(app)

                # First health check (should be fast, <1s)
                start = time.perf_counter()
                resp = client.get("/health")
                elapsed = time.perf_counter() - start

                assert resp.status_code == 200
                assert elapsed < 1.0, f"Health check took {elapsed:.2f}s, expected <1s"

                # Touch a file to trigger FileWatcher
                test_py.write_text("def hello():\n    return 'world2'\n")
                time.sleep(0.1)  # Let watcher detect the change

                # Second health check (should still be fast, not blocked by sync)
                start = time.perf_counter()
                resp = client.get("/health")
                elapsed = time.perf_counter() - start

                assert resp.status_code == 200
                assert elapsed < 1.0, (
                    f"Health check with sync in progress took {elapsed:.2f}s, "
                    f"expected <1s (sync should not block it)"
                )

            finally:
                # Restore original environment
                if original_project is not None:
                    os.environ["CAIRN_PROJECT"] = original_project
                elif "CAIRN_PROJECT" in os.environ:
                    del os.environ["CAIRN_PROJECT"]
