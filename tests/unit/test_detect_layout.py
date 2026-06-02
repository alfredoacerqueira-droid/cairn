"""Unit tests for source layout detection."""

from pathlib import Path

from core.repo import detect_source_layout


class TestDetectSourceLayout:
    def test_single_source_dir_with_venv_and_tests(self, tmp_path):
        """Test that app/ is detected as source root, tests/ and .venv/ ignored."""
        # Create structure:
        # app/
        #   __init__.py
        #   module.py
        # tests/
        #   test_module.py
        # .venv/
        #   lib/...

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("")
        (app_dir / "module.py").write_text("def hello(): pass")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_module.py").write_text("def test_hello(): pass")

        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("pass")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should detect app as the source root
        assert "app" in source_roots or source_roots == ["."]
        # Should include Python pattern
        assert "*.py" in file_patterns
        # Should NOT include .venv or tests files when counting
        # (they are skipped in detection)

    def test_multilanguage_detection(self, tmp_path):
        """Test detection of multiple language files."""
        # Create structure:
        # src/
        #   main.py
        #   utils.js
        #   lib.ts

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main(): pass")
        (src_dir / "utils.js").write_text("function utils() {}")
        (src_dir / "lib.ts").write_text("export function lib() {}")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should detect src as source root
        assert "src" in source_roots or source_roots == ["."]
        # Should include all detected patterns
        assert "*.py" in file_patterns
        assert "*.js" in file_patterns
        assert "*.ts" in file_patterns

    def test_root_level_python_package(self, tmp_path):
        """Test detection when code is at repo root (no clear source dir)."""
        # Create structure:
        # main.py
        # utils.py
        # module.py

        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def util(): pass")
        (tmp_path / "module.py").write_text("def mod(): pass")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should fall back to root
        assert source_roots == ["."]
        assert "*.py" in file_patterns

    def test_multiple_candidate_dirs(self, tmp_path):
        """Test that largest source dir is preferred."""
        # Create structure:
        # src/
        #   file1.py
        #   file2.py
        #   file3.py
        # lib/
        #   helper.py

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "file1.py").write_text("pass")
        (src_dir / "file2.py").write_text("pass")
        (src_dir / "file3.py").write_text("pass")

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "helper.py").write_text("pass")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should prefer src (3 files) over lib (1 file)
        assert "src" in source_roots or source_roots == ["."]
        assert "*.py" in file_patterns

    def test_terraform_module_repo_keeps_root(self, tmp_path):
        """Root .tf module must not be dropped in favor of a modules/ subdir.

        Regression: a TF module repo (root *.tf as the primary module + modules/
        of submodules) collapsed to ["modules"], silently dropping the root
        module (where cluster encryption / IAM live). The root holds files of
        the dominant extension, so it must stay a source root.
        """
        (tmp_path / "main.tf").write_text('resource "aws_eks_cluster" "this" {}')
        (tmp_path / "variables.tf").write_text('variable "create_kms_key" {}')
        (tmp_path / "outputs.tf").write_text('output "cluster_arn" {}')

        modules = tmp_path / "modules" / "karpenter"
        modules.mkdir(parents=True)
        for i in range(6):
            (modules / f"m{i}.tf").write_text(f'resource "aws_sqs_queue" "q{i}" {{}}')

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Must include the root so the primary module is indexed.
        assert source_roots == ["."], f"root module dropped: {source_roots}"
        assert "*.tf" in file_patterns

    def test_src_layout_not_regressed_by_root_config(self, tmp_path):
        """A src/app package must stay isolated even with config files at root.

        Root-level config (.toml/.json/.yaml) is NOT the dominant code type, so
        it must not flip detection to ["."] (which would pull in the whole tree).
        """
        app = tmp_path / "django"
        app.mkdir()
        (app / "__init__.py").write_text("")
        for i in range(5):
            (app / f"mod{i}.py").write_text("def f(): pass")
        # Config files at root — must not count as primary source.
        (tmp_path / "pyproject.toml").write_text("[tool.x]")
        (tmp_path / "setup.json").write_text("{}")

        source_roots, _ = detect_source_layout(tmp_path)
        assert source_roots == ["django"], f"src layout regressed: {source_roots}"

    def test_empty_project(self, tmp_path):
        """Test handling of empty project."""
        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should fall back to root with default pattern
        assert source_roots == ["."]
        assert file_patterns == ["*.py"]

    def test_nocode_in_source_dirs(self, tmp_path):
        """Test when source dirs exist but contain no code."""
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "README.txt").write_text("This is not code")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should fall back to root
        assert source_roots == ["."]
        assert file_patterns == ["*.py"]

    def test_excludes_non_source_dirs(self, tmp_path):
        """Test that non-source dirs are properly excluded."""
        # Create structure:
        # src/
        #   main.py
        # build/
        #   output.py
        # dist/
        #   pkg.py
        # benchmarks/
        #   bench.py

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("pass")

        for non_src in ["build", "dist", "benchmarks"]:
            non_dir = tmp_path / non_src
            non_dir.mkdir()
            (non_dir / "file.py").write_text("pass")

        source_roots, file_patterns = detect_source_layout(tmp_path)

        # Should prefer src (which is real source)
        assert "src" in source_roots or source_roots == ["."]
        assert "*.py" in file_patterns
