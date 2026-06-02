"""Unit tests for core/freshness.py — DBFreshness."""

from unittest.mock import patch

from core.freshness import DBFreshness, changed_files_since_index


class TestDBFreshness:
    def test_init_defaults(self, tmp_path):
        f = DBFreshness(tmp_path)
        assert f.quick_threshold == 1000
        assert f.full_threshold == 10000
        assert f._last_indexed_commit is None

    def test_custom_thresholds(self, tmp_path):
        f = DBFreshness(tmp_path, quick_threshold=50, full_threshold=500)
        assert f.quick_threshold == 50
        assert f.full_threshold == 500

    def test_get_current_commit_no_git(self, tmp_path):
        f = DBFreshness(tmp_path)
        result = f.get_current_commit()
        assert isinstance(result, str)

    def test_count_commits_behind_no_git(self, tmp_path):
        f = DBFreshness(tmp_path)
        result = f.count_commits_behind("nonexistent")
        assert result == -1

    def test_get_changed_files_no_git(self, tmp_path):
        f = DBFreshness(tmp_path)
        files = f.get_changed_files("nonexistent")
        assert isinstance(files, list)

    def test_check_freshness_no_git(self, tmp_path):
        f = DBFreshness(tmp_path)
        info = f.check_freshness()
        assert "current_commit" in info
        assert "last_indexed_commit" in info
        assert "commits_behind" in info

    def test_mark_indexed(self, tmp_path):
        f = DBFreshness(tmp_path)
        f.mark_indexed("abc123")
        assert f._last_indexed_commit == "abc123"
        info = f.check_freshness()
        assert info["last_indexed_commit"] == "abc123"

    def test_mark_indexed_persists(self, tmp_path):
        """Test that mark_indexed persists to disk."""
        f = DBFreshness(tmp_path)
        f.mark_indexed("abc123")

        # Verify file was written
        last_indexed_path = tmp_path / ".cairn" / "last_indexed.txt"
        assert last_indexed_path.exists()
        assert last_indexed_path.read_text().strip() == "abc123"

    def test_mark_indexed_survives_restart(self, tmp_path):
        """Test that persisted last_indexed_commit is loaded on init."""
        # First instance: mark indexed
        f1 = DBFreshness(tmp_path)
        f1.mark_indexed("abc123")

        # Second instance: should load persisted value
        f2 = DBFreshness(tmp_path)
        assert f2._last_indexed_commit == "abc123"

    def test_load_last_indexed_missing_file(self, tmp_path):
        """Test graceful handling when last_indexed.txt does not exist."""
        f = DBFreshness(tmp_path)
        assert f._last_indexed_commit is None

    def test_load_last_indexed_corrupt_file(self, tmp_path):
        """Test graceful handling when last_indexed.txt is corrupt."""
        last_indexed_path = tmp_path / ".cairn" / "last_indexed.txt"
        last_indexed_path.parent.mkdir(parents=True, exist_ok=True)
        # Write empty file
        last_indexed_path.write_text("")

        f = DBFreshness(tmp_path)
        assert f._last_indexed_commit is None

    def test_needs_reindex_below_threshold(self, tmp_path):
        f = DBFreshness(tmp_path, quick_threshold=100)
        f._last_indexed_commit = "old"
        with (
            patch.object(f, "get_current_commit", return_value="old"),
            patch.object(f, "count_commits_behind", return_value=0),
        ):
            assert f.needs_quick_reindex() is False
            assert f.needs_full_reindex() is False

    def test_needs_quick_reindex_mid_threshold(self, tmp_path):
        f = DBFreshness(tmp_path, quick_threshold=100, full_threshold=1000)
        f._last_indexed_commit = "old"
        with (
            patch.object(f, "get_current_commit", return_value="new"),
            patch.object(f, "count_commits_behind", return_value=150),
        ):
            assert f.needs_quick_reindex() is True
            assert f.needs_full_reindex() is False

    def test_needs_full_reindex_above_threshold(self, tmp_path):
        f = DBFreshness(tmp_path, quick_threshold=100, full_threshold=1000)
        f._last_indexed_commit = "old"
        with (
            patch.object(f, "get_current_commit", return_value="new"),
            patch.object(f, "count_commits_behind", return_value=1500),
        ):
            assert f.needs_quick_reindex() is False
            assert f.needs_full_reindex() is True


class TestChangedFilesSinceIndex:
    def test_changed_files_no_prior_index(self, tmp_path):
        """Return empty lists when no prior index exists."""
        modified, deleted = changed_files_since_index(tmp_path)
        assert modified == []
        assert deleted == []

    def test_changed_files_no_git(self, tmp_path):
        """Return empty lists when not a git repo."""
        # Mark as indexed
        f = DBFreshness(tmp_path)
        f.mark_indexed("abc123")

        # But directory is not a git repo, so git diff fails gracefully
        modified, deleted = changed_files_since_index(tmp_path)
        assert modified == []
        assert deleted == []

    def test_changed_files_parses_diff_output(self, tmp_path):
        """Test parsing of git diff --name-status output."""
        from core.freshness import DBFreshness

        # Mark as indexed
        f = DBFreshness(tmp_path)
        f.mark_indexed("old_commit")

        # Mock the git diff output
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "M\tapp/main.py\nA\tapp/new.py\nD\tapp/old.py\n"

            modified, deleted = changed_files_since_index(tmp_path)

            assert "app/main.py" in modified
            assert "app/new.py" in modified
            assert "app/old.py" in deleted
            assert len(modified) == 2
            assert len(deleted) == 1

    def test_changed_files_handles_git_error(self, tmp_path):
        """Handle git errors gracefully."""
        from core.freshness import DBFreshness

        # Mark as indexed
        f = DBFreshness(tmp_path)
        f.mark_indexed("old_commit")

        # Mock a git error
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128  # Fatal error

            modified, deleted = changed_files_since_index(tmp_path)
            assert modified == []
            assert deleted == []

    def test_changed_files_handles_timeout(self, tmp_path):
        """Handle git timeout gracefully."""
        from core.freshness import DBFreshness

        # Mark as indexed
        f = DBFreshness(tmp_path)
        f.mark_indexed("old_commit")

        # Mock a timeout
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = TimeoutError()

            modified, deleted = changed_files_since_index(tmp_path)
            assert modified == []
            assert deleted == []
