"""Unit tests for core/repo.py — RepoManager."""

from pathlib import Path

from core.repo import RepoManager


class TestRepoManager:
    def test_paths(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        assert repo.get_chroma_path() == tmp_path / ".cairn" / "chroma"
        assert repo.get_repo_map_path() == tmp_path / ".cairn" / "repo_map.json"
        assert repo.get_memory_path() == tmp_path / ".cairn" / "memory.md"
        assert repo.get_learning_db_path() == tmp_path / ".cairn" / "learning.db"

    def test_ensure_directories(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        repo.ensure_directories()
        assert (tmp_path / ".cairn").is_dir()
        assert (tmp_path / ".cairn" / "chroma").is_dir()

    def test_save_and_load_repo_map(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        repo.ensure_directories()

        data = {"app/main.py": {"functions": [{"name": "main"}], "classes": []}}
        repo.save_repo_map(data)

        loaded = repo.load_repo_map()
        assert loaded == data

    def test_load_repo_map_missing(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        loaded = repo.load_repo_map()
        assert loaded == {}

    def test_load_repo_map_corrupt(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        repo.ensure_directories()
        repo.get_repo_map_path().write_text("not json")
        loaded = repo.load_repo_map()
        assert loaded == {}

    def test_append_and_load_memory(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        repo.append_memory("Fixed bug in auth handler")
        repo.append_memory("Added rate limiting")

        content = repo.load_memory(last_n=10)
        assert "Fixed bug" in content
        assert "rate limiting" in content

    def test_load_memory_missing(self, tmp_path):
        repo = RepoManager(project_path=tmp_path)
        content = repo.load_memory()
        assert content == ""

    def test_default_project_path(self):
        repo = RepoManager()
        assert repo.data_dir == Path.cwd() / ".cairn"
