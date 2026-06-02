"""Unit tests for collect_source_files helper."""

from core.repo import collect_source_files


class TestCollectSourceFiles:
    def test_collect_default_all_files(self, tmp_path):
        """Collect all .py files with default exclusions."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")
        (tmp_path / "app" / "sub").mkdir()
        (tmp_path / "app" / "sub" / "util.py").write_text("def util(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x(): pass")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "lib").mkdir()
        (tmp_path / ".venv" / "lib" / "foo.py").write_text("def foo(): pass")
        (tmp_path / "benchmarks").mkdir()
        (tmp_path / "benchmarks" / "bench.py").write_text("def bench(): pass")

        # Collect with default exclusions
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[
                "**/node_modules/**",
                "**/.git/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/venv/**",
                "**/.cairn/**",
                "**/tests/**",
                "**/test/**",
                "**/benchmarks/**",
                "**/build/**",
                "**/dist/**",
                "**/*.egg-info/**",
                "**/.mypy_cache/**",
                "**/.pytest_cache/**",
                "**/.ruff_cache/**",
            ],
        )

        # Should include app files only
        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        assert "app/sub/util.py" in file_paths
        assert "tests/test_x.py" not in file_paths
        assert ".venv/lib/foo.py" not in file_paths
        assert "benchmarks/bench.py" not in file_paths
        assert len(files) == 2

    def test_collect_restricted_source_roots(self, tmp_path):
        """Collect files only from specified source_roots."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")
        (tmp_path / "app" / "sub").mkdir()
        (tmp_path / "app" / "sub" / "util.py").write_text("def util(): pass")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "helper.py").write_text("def helper(): pass")

        # Collect from app only
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[],
            source_roots=["app"],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        assert "app/sub/util.py" in file_paths
        assert "lib/helper.py" not in file_paths
        assert len(files) == 2

    def test_collect_multiple_source_roots(self, tmp_path):
        """Collect files from multiple source_roots."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "helper.py").write_text("def helper(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").write_text("def test(): pass")

        # Collect from app and lib
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[],
            source_roots=["app", "lib"],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        assert "lib/helper.py" in file_paths
        assert "tests/test.py" not in file_paths
        assert len(files) == 2

    def test_collect_multiple_patterns(self, tmp_path):
        """Collect files matching multiple patterns."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")
        (tmp_path / "app" / "config.json").write_text("{}")
        (tmp_path / "app" / "readme.md").write_text("# Readme")

        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py", "*.json"],
            exclude_patterns=[],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        assert "app/config.json" in file_paths
        assert "app/readme.md" not in file_paths
        assert len(files) == 2

    def test_collect_empty_when_no_matches(self, tmp_path):
        """Return empty list when no files match."""
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.txt").write_text("not python")

        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[],
        )

        assert len(files) == 0

    def test_collect_deduplication(self, tmp_path):
        """Files collected multiple times are deduplicated."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")

        # Collect with overlapping patterns (though rglob should prevent this)
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py", "main.py"],  # main.py matches both
            exclude_patterns=[],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        # Deduplication ensures only one entry
        assert len(files) == 1

    def test_collect_sorted_output(self, tmp_path):
        """Files are returned in sorted order."""
        # Create test structure
        (tmp_path / "z_file.py").write_text("z")
        (tmp_path / "a_file.py").write_text("a")
        (tmp_path / "m_file.py").write_text("m")

        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert file_paths == sorted(file_paths)
        assert file_paths == ["a_file.py", "m_file.py", "z_file.py"]

    def test_collect_nonexistent_source_root(self, tmp_path):
        """Handle missing source_roots gracefully."""
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=[],
            source_roots=["nonexistent"],
        )

        assert len(files) == 0

    def test_collect_exclude_by_posix_path(self, tmp_path):
        """Exclusion patterns match against POSIX-style relative paths."""
        # Create test structure
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")
        (tmp_path / "app" / "secret").mkdir()
        (tmp_path / "app" / "secret" / "token.py").write_text("TOKEN = 'xxx'")

        # Exclude using POSIX path pattern
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=["*.py"],
            exclude_patterns=["app/secret/**"],
        )

        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "app/main.py" in file_paths
        assert "app/secret/token.py" not in file_paths
        assert len(files) == 1

    def test_collect_top_level_exclusions(self, tmp_path):
        """Regression test: top-level excluded dirs must be excluded.

        The bug was that patterns like **/tests/** used fnmatch which doesn't
        support recursive ** properly, so top-level paths like tests/x.py
        weren't being excluded.
        """
        from core.config import IndexingConfig

        # Create test structure with excluded dirs at TOP LEVEL
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("def main(): pass")
        (tmp_path / "app" / "models.py").write_text("class Model: pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("def test_x(): pass")
        (tmp_path / "benchmarks").mkdir()
        (tmp_path / "benchmarks" / "bench.py").write_text("def bench(): pass")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "lib").mkdir()
        (tmp_path / ".venv" / "lib" / "foo.py").write_text("def foo(): pass")
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "api.py").write_text("def api(): pass")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.py").write_text("def out(): pass")

        # Use real default exclude patterns from IndexingConfig
        default_config = IndexingConfig()
        files = collect_source_files(
            project_path=tmp_path,
            file_patterns=default_config.file_patterns,
            exclude_patterns=default_config.exclude_patterns,
            source_roots=["."],
        )

        # Should include app files only
        file_paths = [f.relative_to(tmp_path).as_posix() for f in files]

        # These MUST be in the result
        assert "app/main.py" in file_paths, f"app/main.py not found in {file_paths}"
        assert "app/models.py" in file_paths, f"app/models.py not found in {file_paths}"

        # These MUST be excluded
        assert (
            "tests/test_x.py" not in file_paths
        ), f"tests/test_x.py should be excluded but found in {file_paths}"
        assert (
            "benchmarks/bench.py" not in file_paths
        ), f"benchmarks/bench.py should be excluded but found in {file_paths}"
        assert (
            ".venv/lib/foo.py" not in file_paths
        ), f".venv/lib/foo.py should be excluded but found in {file_paths}"
        assert (
            "build/out.py" not in file_paths
        ), f"build/out.py should be excluded but found in {file_paths}"

        # server/ is NOT in default excludes, so it's OK if it appears
        # (the bug was about tests/benchmarks/.venv/build)
        # So we expect 3 files: app/main.py, app/models.py, server/api.py

        # Should have exactly 3 files
        assert len(files) == 3, f"Expected 3 files, got {len(files)}: {file_paths}"
