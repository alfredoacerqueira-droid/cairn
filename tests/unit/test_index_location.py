"""Tests for index_location config knob."""

from pathlib import Path
from unittest.mock import patch

from core.config import Config, IndexingConfig, save_config
from core.repo import RepoManager, project_id


class TestIndexBaseDir:
    """Test the index_base_dir resolver."""

    def test_in_project_location(self, tmp_path):
        """Test 'in_project' forces index in .cairn/."""
        repo = RepoManager(tmp_path)
        base = repo.index_base_dir("in_project")
        assert base == repo.data_dir
        assert base.exists()

    def test_native_location(self, tmp_path, monkeypatch):
        """Test 'native' forces index under ~/.cache/cairn/<project-id>/."""
        # Use tmp_path as HOME to avoid polluting real ~/.cache
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        repo = RepoManager(tmp_path / "project")
        base = repo.index_base_dir("native")

        pid = project_id(str(repo.project_path.resolve()))
        expected = fake_home / ".cache" / "cairn" / pid
        assert base == expected
        assert base.exists()

    def test_in_project_default_for_tmp_path(self, tmp_path):
        """Test 'auto' on /tmp -> in_project (not on /mnt/*)."""
        repo = RepoManager(tmp_path)
        base = repo.index_base_dir("auto")
        assert base == repo.data_dir
        assert base.exists()

    def test_native_default_for_mnt_path(self, tmp_path, monkeypatch):
        """Test 'auto' on /mnt/* path -> native (simulated via mount detection)."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Create repo with path starting with /mnt/
        # We can't actually create a /mnt dir, so we monkeypatch _on_windows_mount
        repo = RepoManager(tmp_path / "project")
        with patch.object(repo, "_on_windows_mount", return_value=True):
            base = repo.index_base_dir("auto")

            pid = project_id(str(repo.project_path.resolve()))
            expected = fake_home / ".cache" / "cairn" / pid
            assert base == expected
            assert base.exists()

    def test_none_index_location_reads_from_config(self, tmp_path, monkeypatch):
        """Test None index_location reads from config."""
        # Create config with index_location="native"
        repo = RepoManager(tmp_path)
        repo.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config()
        cfg.indexing.index_location = "native"
        save_config(cfg, tmp_path)

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        base = repo.index_base_dir(None)  # Should read from config
        pid = project_id(str(repo.project_path.resolve()))
        expected = fake_home / ".cache" / "cairn" / pid
        assert base == expected

    def test_default_index_location_is_auto(self, tmp_path):
        """Test that default IndexingConfig has index_location='auto'."""
        cfg = IndexingConfig()
        assert cfg.index_location == "auto"


class TestGetChromaPath:
    """Test get_chroma_path with index_location parameter."""

    def test_chroma_path_in_project(self, tmp_path):
        """Test get_chroma_path with 'in_project'."""
        repo = RepoManager(tmp_path)
        path = repo.get_chroma_path("in_project")
        assert path == repo.data_dir / "chroma"
        assert path.parent.exists()  # Parent should be created

    def test_chroma_path_native(self, tmp_path, monkeypatch):
        """Test get_chroma_path with 'native'."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        repo = RepoManager(tmp_path / "project")
        path = repo.get_chroma_path("native")

        pid = project_id(str(repo.project_path.resolve()))
        expected = fake_home / ".cache" / "cairn" / pid / "chroma"
        assert path == expected
        assert path.parent.exists()

    def test_chroma_path_from_config(self, tmp_path, monkeypatch):
        """Test get_chroma_path with None reads from config."""
        repo = RepoManager(tmp_path)
        repo.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config()
        cfg.indexing.index_location = "in_project"
        save_config(cfg, tmp_path)

        path = repo.get_chroma_path(None)
        assert path == repo.data_dir / "chroma"


class TestGetLancePath:
    """Test get_lance_path with index_location parameter."""

    def test_lance_path_in_project(self, tmp_path):
        """Test get_lance_path with 'in_project'."""
        repo = RepoManager(tmp_path)
        path = repo.get_lance_path("in_project")
        assert path == repo.data_dir / "lance"
        assert path.parent.exists()

    def test_lance_path_native(self, tmp_path, monkeypatch):
        """Test get_lance_path with 'native'."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        repo = RepoManager(tmp_path / "project")
        path = repo.get_lance_path("native")

        pid = project_id(str(repo.project_path.resolve()))
        expected = fake_home / ".cache" / "cairn" / pid / "lance"
        assert path == expected
        assert path.parent.exists()


class TestMakeStoreParity:
    """Test that make_store and ContextAssembler resolve the same chroma dir."""

    def test_make_store_chroma_parity_in_project(self, tmp_path):
        """Test make_store and VectorIndexer use same chroma path (in_project)."""
        from pipeline.indexer import VectorIndexer
        from pipeline.store import make_store

        # Set up repo with in_project config
        repo = RepoManager(tmp_path)
        repo.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config()
        cfg.indexing.index_location = "in_project"
        cfg.local_llm.enabled = False  # Disable LLM to simplify
        save_config(cfg, tmp_path)

        # Build via make_store
        store = make_store(cfg, repo, project_root=tmp_path)
        store_chroma = store.indexer.chroma_path

        # Build via VectorIndexer directly (as ContextAssembler would)
        direct_indexer = VectorIndexer(chroma_path=repo.get_chroma_path())
        direct_chroma = direct_indexer.chroma_path

        assert store_chroma == direct_chroma, (
            f"Split-brain: make_store chroma={store_chroma}, "
            f"direct chroma={direct_chroma}"
        )

    def test_make_store_chroma_parity_native(self, tmp_path, monkeypatch):
        """Test make_store and VectorIndexer use same chroma path (native)."""
        from pipeline.indexer import VectorIndexer
        from pipeline.store import make_store

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Set up repo with native config
        repo = RepoManager(tmp_path / "project")
        repo.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config()
        cfg.indexing.index_location = "native"
        cfg.local_llm.enabled = False
        save_config(cfg, tmp_path / "project")

        # Build via make_store
        store = make_store(cfg, repo, project_root=tmp_path / "project")
        store_chroma = store.indexer.chroma_path

        # Build via VectorIndexer directly
        direct_indexer = VectorIndexer(chroma_path=repo.get_chroma_path())
        direct_chroma = direct_indexer.chroma_path

        assert store_chroma == direct_chroma, (
            f"Split-brain: make_store chroma={store_chroma}, "
            f"direct chroma={direct_chroma}"
        )
        # Both should be under ~/.cache
        assert ".cache" in str(store_chroma)


class TestWindowsMountDetection:
    """Test Windows mount detection logic."""

    def test_not_on_windows_mount_for_tmp(self, tmp_path):
        """Test /tmp path is not detected as Windows mount."""
        repo = RepoManager(tmp_path)
        assert not repo._on_windows_mount()

    def test_on_windows_mount_detection(self, monkeypatch):
        """Test /mnt/* path is detected as Windows mount."""
        # Create a mock path starting with /mnt/
        repo = RepoManager(Path("/mnt/c/Users/test/project"))
        assert repo._on_windows_mount()
