"""Tests for the repository profile system."""

import tempfile
from pathlib import Path

import pytest

from core.profiles import PROFILES, detect_profile, get_profile
from core.repo import census_extensions


class TestDetectProfile:
    """Test profile detection from extension census."""

    def test_detect_iac_from_terraform(self):
        """IaC profile detected when .tf files present."""
        census = {".tf": 50, ".yaml": 10, ".sh": 5}
        profile = detect_profile(census)
        assert profile == "iac"

    def test_detect_iac_from_hcl(self):
        """IaC profile detected when .hcl files present."""
        census = {".hcl": 30, ".yaml": 15}
        profile = detect_profile(census)
        assert profile == "iac"

    def test_detect_iac_from_yaml_dominant(self):
        """IaC profile detected when yaml/yml dominant (no other code)."""
        census = {".yaml": 50, ".yml": 20}
        profile = detect_profile(census)
        assert profile == "iac"

    def test_detect_dotnet_from_csharp(self):
        """DotNet profile detected when .cs files present."""
        census = {".cs": 80, ".csproj": 5}
        profile = detect_profile(census)
        assert profile == "dotnet"

    def test_detect_python_dominant(self):
        """Python profile detected when Python dominant."""
        census = {".py": 200, ".txt": 50}
        profile = detect_profile(census)
        assert profile == "python"

    def test_detect_shell_dominant(self):
        """Shell profile detected when shell scripts dominant."""
        census = {".sh": 30, ".bash": 10, ".yaml": 5}
        profile = detect_profile(census)
        assert profile == "shell"

    def test_detect_code_default(self):
        """Code profile (default) for generic languages."""
        census = {".js": 40, ".ts": 60}
        profile = detect_profile(census)
        assert profile == "code"

    def test_detect_code_empty(self):
        """Code profile for empty census."""
        census = {}
        profile = detect_profile(census)
        assert profile == "code"

    def test_detect_cs_beats_python(self):
        """C# takes precedence over Python if both present."""
        census = {".cs": 10, ".py": 100}
        profile = detect_profile(census)
        # C# check comes before Python check
        assert profile == "dotnet"

    def test_detect_tf_beats_everything(self):
        """Terraform takes precedence."""
        census = {".tf": 5, ".py": 200, ".yaml": 100}
        profile = detect_profile(census)
        assert profile == "iac"


class TestGetProfile:
    """Test profile retrieval."""

    def test_get_known_profile(self):
        """get_profile returns spec for known profile."""
        spec = get_profile("iac")
        assert spec.name == "iac"
        assert spec.embedding_enabled is False
        assert spec.retrieval_mode == "hybrid"
        assert "structural" in spec.legs
        assert "lexical" in spec.legs

    def test_get_dotnet_profile(self):
        """dotnet profile has embeddings on."""
        spec = get_profile("dotnet")
        assert spec.name == "dotnet"
        assert spec.embedding_enabled is True
        assert spec.embedding_model == "qwen3-embedding:0.6b"

    def test_get_python_profile(self):
        """python profile has embeddings on."""
        spec = get_profile("python")
        assert spec.name == "python"
        assert spec.embedding_enabled is True
        assert spec.embedding_model == "nomic-embed-text"

    def test_get_unknown_fallback(self):
        """get_profile falls back to 'code' for unknown."""
        spec = get_profile("unknown_lang")
        assert spec.name == "code"

    def test_all_profiles_have_legs(self):
        """All profiles have non-empty legs list."""
        for name, spec in PROFILES.items():
            assert len(spec.legs) > 0, f"Profile {name} has no legs"
            assert isinstance(spec.legs, list)

    def test_iac_profile_embeddings_off(self):
        """IaC profile explicitly disables embeddings."""
        spec = get_profile("iac")
        assert spec.embedding_enabled is False
        assert spec.embedding_model == ""

    def test_shell_profile_embeddings_off(self):
        """Shell profile disables embeddings."""
        spec = get_profile("shell")
        assert spec.embedding_enabled is False


class TestCensusExtensions:
    """Test extension census collection."""

    def test_census_empty_dir(self):
        """census_extensions returns empty dict for empty dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            census = census_extensions(tmppath)
            assert census == {}

    def test_census_python_files(self):
        """census_extensions counts .py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "a.py").write_text("# a")
            (tmppath / "b.py").write_text("# b")
            (tmppath / "c.txt").write_text("text")
            census = census_extensions(tmppath)
            assert census[".py"] == 2
            assert ".txt" not in census  # Not in EXTENSION_MAP

    def test_census_respects_source_roots(self):
        """census_extensions only scans specified source roots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "a.py").write_text("# a")
            (tmppath / "test").mkdir()
            (tmppath / "test" / "b.py").write_text("# b")

            # Only scan src/
            census = census_extensions(tmppath, source_roots=["src"])
            assert census[".py"] == 1

            # Scan both
            census = census_extensions(tmppath, source_roots=["src", "test"])
            assert census[".py"] == 2

    def test_census_tf_and_yaml(self):
        """census_extensions counts .tf and .yaml together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "main.tf").write_text("resource {}")
            (tmppath / "vars.tfvars").write_text("var {}")
            (tmppath / "config.yaml").write_text("key: val")
            census = census_extensions(tmppath)
            assert census[".tf"] == 1
            assert census[".tfvars"] == 1
            assert census[".yaml"] == 1


class TestProfileIntegration:
    """Integration tests for profile detection + specs."""

    def test_iac_workflow(self):
        """IaC workflow: detect -> get -> check settings."""
        census = {".tf": 50, ".yaml": 20, ".sh": 10}
        profile_name = detect_profile(census)
        assert profile_name == "iac"

        spec = get_profile(profile_name)
        assert spec.embedding_enabled is False
        assert "structural" in spec.legs
        assert "lexical" in spec.legs

    def test_dotnet_workflow(self):
        """DotNet workflow: detect -> get -> check settings."""
        census = {".cs": 100, ".csproj": 10}
        profile_name = detect_profile(census)
        assert profile_name == "dotnet"

        spec = get_profile(profile_name)
        assert spec.embedding_enabled is True
        assert spec.embedding_model == "qwen3-embedding:0.6b"
        assert "embeddings" in spec.legs

    def test_python_workflow(self):
        """Python workflow: detect -> get -> check settings."""
        census = {".py": 500}
        profile_name = detect_profile(census)
        assert profile_name == "python"

        spec = get_profile(profile_name)
        assert spec.embedding_enabled is True
        assert "embeddings" in spec.legs
