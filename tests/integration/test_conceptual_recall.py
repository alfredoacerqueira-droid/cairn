"""Regression tests for conceptual-query recall with the confidence guard.

Verifies that conceptual/natural-language queries (e.g. "persist tasks",
"validate title") survive the guard when retrieval is lexical+structural-only
(no embeddings), while genuinely off-topic queries remain fail-closed.
"""

import subprocess
from pathlib import Path

from server.context_assembler import ContextAssembler
from tests.fixtures.harness import fresh_index


def _make_conceptual_python_repo(base: Path) -> Path:
    """Create a Python repo whose code can be found by conceptual queries."""
    repo = base / "conceptual-repo"
    repo.mkdir(exist_ok=True)

    (repo / "services").mkdir(exist_ok=True)
    (repo / "services" / "__init__.py").touch()

    (repo / "services" / "storage.py").write_text('''"""Task persistence layer."""


def save_tasks(tasks: list[dict]) -> None:
    """Persist tasks to disk storage."""
    import json
    with open("/tmp/tasks.json", "w") as f:
        json.dump(tasks, f)


def load_tasks() -> list[dict]:
    """Load persisted tasks from disk."""
    import json
    try:
        with open("/tmp/tasks.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
''')

    (repo / "services" / "tasks.py").write_text('''"""Task creation and validation."""


def create_task(title: str) -> dict:
    """Create a task and validate its title."""
    if not title or not title.strip():
        raise ValueError("Title must not be empty for task creation")
    if len(title) > 200:
        raise ValueError("Task title too long")
    return {"title": title.strip(), "done": False}
''')

    (repo / "models.py").write_text('''"""Domain models."""


class Task:
    """A task entity."""

    def __init__(self, title: str, done: bool = False):
        self.title = title
        self.done = done
''')

    # Initialize git
    subprocess.run(
        ["git", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


class TestConceptualRecall:
    """Conceptual queries return real code; off-topic stays fail-closed."""

    def test_persist_tasks_returns_storage(self, tmp_path):
        """'persist tasks' finds save_tasks or load_tasks."""
        repo = _make_conceptual_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("persist tasks", apply_guard=True)
        assert (
            len(results) > 0
        ), f"Expected non-empty results for 'persist tasks', got {len(results)}"
        functions = [r.get("function", "") for r in results]
        filepaths = [r.get("filepath", "") for r in results]
        found = any(
            "storage.py" in fp and fn in ("save_tasks", "load_tasks")
            for fp, fn in zip(filepaths, functions)
        )
        assert found, (
            f"No storage.py result for 'persist tasks': "
            f"functions={functions}, filepaths={filepaths}"
        )

    def test_validate_title_returns_create_task(self, tmp_path):
        """'validate title' finds create_task."""
        repo = _make_conceptual_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("validate title", apply_guard=True)
        assert (
            len(results) > 0
        ), f"Expected non-empty results for 'validate title', got {len(results)}"
        functions = [r.get("function", "") for r in results]
        assert (
            "create_task" in functions
        ), f"create_task not found for 'validate title': functions={functions}"

    def test_task_model_finds_task_class(self, tmp_path):
        """'Task model' finds the Task class."""
        repo = _make_conceptual_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("Task model", apply_guard=True)
        assert len(results) > 0, f"Expected non-empty results for 'Task model', got {len(results)}"
        functions = [r.get("function", "") for r in results]
        filepaths = [r.get("filepath", "") for r in results]
        found_task = any("Task" in fn and "models.py" in fp for fn, fp in zip(functions, filepaths))
        assert found_task, (
            f"Task class not found for 'Task model': "
            f"functions={functions}, filepaths={filepaths}"
        )

    def test_nonsense_fail_closed(self, tmp_path):
        """Off-topic queries still return [] (fail-closed preserved)."""
        repo = _make_conceptual_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        results = assembler.semantic_search("zzqwerty_nonsense_xyz", apply_guard=True)
        assert results == [], (
            f"Expected [] for nonsense query, got {len(results)} results: "
            f"{[r.get('function') for r in results]}"
        )

    def test_assemble_context_conceptual(self, tmp_path):
        """assemble_context('persist tasks') does NOT return 'No confident matches'."""
        repo = _make_conceptual_python_repo(tmp_path)
        fresh_index(repo, embeddings=False)
        assembler = ContextAssembler(project_path=repo)

        ctx = assembler.assemble_context("persist tasks")
        assert (
            "No confident matches found" not in ctx
        ), f"assemble_context returned guard-rejection string: {ctx[:300]}"
        assert (
            "save_tasks" in ctx or "load_tasks" in ctx
        ), f"assemble_context missing storage functions: {ctx[:300]}"
