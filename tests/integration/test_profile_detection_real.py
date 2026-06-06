"""Integration tests for profile detection using real fixture repos.

These tests drive the EXACT path the CLI uses:
  detect_source_layout -> census_extensions -> detect_infra_markers -> detect_profile
"""

import tempfile
from pathlib import Path

from core.profiles import detect_profile, get_profile
from core.repo import census_extensions, detect_infra_markers, detect_source_layout
from tests.fixtures.builders import (
    make_helm_repo,
    make_k8s_repo,
    make_python_repo,
    make_terraform_repo,
)


class TestProfileDetectionWithRealRepos:
    """Test profile detection using realistic fixture repos."""

    def test_helm_repo_detected_as_iac(self):
        """Helm repo (YAML + .sh + .json mix) -> 'iac' with embeddings OFF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_helm_repo(Path(tmpdir))

            # Simulate the CLI path
            detected_roots, _ = detect_source_layout(repo_root)
            ext_census = census_extensions(repo_root, source_roots=detected_roots)
            has_infra_markers = detect_infra_markers(repo_root, source_roots=detected_roots)

            # Detect profile
            profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
            profile = get_profile(profile_name)

            # Assertions
            assert profile_name == "iac", f"Expected 'iac', got '{profile_name}'"
            assert profile.embedding_enabled is False
            assert profile.retrieval_mode == "hybrid"
            assert "structural" in profile.legs
            assert "lexical" in profile.legs

    def test_terraform_repo_detected_as_iac(self):
        """Terraform repo (.tf files present) -> 'iac' with embeddings OFF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_terraform_repo(Path(tmpdir))

            detected_roots, _ = detect_source_layout(repo_root)
            ext_census = census_extensions(repo_root, source_roots=detected_roots)
            has_infra_markers = detect_infra_markers(repo_root, source_roots=detected_roots)

            profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
            profile = get_profile(profile_name)

            assert profile_name == "iac"
            assert profile.embedding_enabled is False
            assert "structural" in profile.legs
            assert "lexical" in profile.legs

    def test_k8s_repo_with_manifests_detected_as_iac(self):
        """K8s repo (manifests with kind: + apiVersion:) -> 'iac'.

        This tests that the detect_infra_markers helper correctly finds
        Kubernetes manifest markers and triggers 'iac' profile.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_k8s_repo(Path(tmpdir), with_pathological=False)

            detected_roots, _ = detect_source_layout(repo_root)
            ext_census = census_extensions(repo_root, source_roots=detected_roots)
            has_infra_markers = detect_infra_markers(repo_root, source_roots=detected_roots)

            # Verify the marker detection worked
            assert has_infra_markers is True, "K8s repo should have infrastructure markers"

            profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
            profile = get_profile(profile_name)

            assert profile_name == "iac"
            assert profile.embedding_enabled is False

    def test_python_repo_not_misclassified_as_iac(self):
        """Python repo -> 'python' (regression: don't over-trigger IaC).

        Ensure that the new IaC heuristic does NOT misclassify
        a Python repo as IaC, even if it has a few YAML files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_python_repo(Path(tmpdir))

            detected_roots, _ = detect_source_layout(repo_root)
            ext_census = census_extensions(repo_root, source_roots=detected_roots)
            has_infra_markers = detect_infra_markers(repo_root, source_roots=detected_roots)

            profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
            profile = get_profile(profile_name)

            assert profile_name == "python"
            assert profile.embedding_enabled is True
            assert "embeddings" in profile.legs

    def test_helm_repo_census_shows_yaml_dominant_without_markers_override(self):
        """Helm repo census shows YAML dominant; without markers, heuristic checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_helm_repo(Path(tmpdir))

            # Verify YAML is dominant in the census
            ext_census = census_extensions(repo_root)
            yaml_count = ext_census.get(".yaml", 0) + ext_census.get(".yml", 0)
            total = sum(ext_census.values())

            assert yaml_count > 0, "Helm repo should have YAML files"
            assert yaml_count > total * 0.4, "YAML should be a substantial portion"

    def test_detect_infra_markers_finds_chart_yaml(self):
        """detect_infra_markers should find Chart.yaml in Helm repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_helm_repo(Path(tmpdir))

            has_markers = detect_infra_markers(repo_root)
            assert has_markers is True

    def test_detect_infra_markers_finds_kustomization_yaml(self):
        """detect_infra_markers should find kustomization.yaml in K8s repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_k8s_repo(Path(tmpdir))

            has_markers = detect_infra_markers(repo_root)
            assert has_markers is True

    def test_detect_infra_markers_finds_kubernetes_manifest_markers(self):
        """detect_infra_markers should find kind:/apiVersion: in manifest files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_k8s_repo(Path(tmpdir))

            # Remove kustomization to test marker-based detection
            kust_path = repo_root / "kustomize" / "kustomization.yaml"
            if kust_path.exists():
                kust_path.unlink()

            # Should still find markers in deployment.yaml, etc.
            has_markers = detect_infra_markers(repo_root)
            assert has_markers is True

    def test_python_repo_no_infra_markers(self):
        """Python repo should NOT have infrastructure markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_python_repo(Path(tmpdir))

            has_markers = detect_infra_markers(repo_root)
            assert has_markers is False


class TestProfileDetectionEdgeCases:
    """Test edge cases in profile detection."""

    def test_yaml_dominant_without_markers_and_no_code(self):
        """Pure YAML repo (>50%, no programming languages) -> 'iac'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "yaml-only-repo"
            repo_root.mkdir()

            # Create only YAML files (no programming languages, no markers)
            (repo_root / "config1.yaml").write_text("key: value1\n")
            (repo_root / "config2.yaml").write_text("key: value2\n")
            (repo_root / "data.json").write_text('{"key": "value"}\n')

            ext_census = census_extensions(repo_root)
            has_markers = detect_infra_markers(repo_root)

            profile_name = detect_profile(ext_census, has_infra_markers=has_markers)
            profile = get_profile(profile_name)

            # YAML dominant + no programming languages + no markers -> still iac
            assert profile_name == "iac"
            assert profile.embedding_enabled is False

    def test_mixed_lang_python_dominant_beats_yaml(self):
        """Python dominant (>30%) beats YAML even if YAML is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "mixed-repo"
            repo_root.mkdir()

            # Create files
            for i in range(10):
                (repo_root / f"module{i}.py").write_text(f"# Python {i}\npass\n")
            for i in range(3):
                (repo_root / f"config{i}.yaml").write_text("key: value\n")
            for i in range(2):
                (repo_root / f"helper{i}.sh").write_text("#!/bin/bash\n")

            ext_census = census_extensions(repo_root)
            has_markers = detect_infra_markers(repo_root)

            profile_name = detect_profile(ext_census, has_infra_markers=has_markers)
            profile = get_profile(profile_name)

            # Python is dominant, should be 'python'
            assert profile_name == "python"
            assert profile.embedding_enabled is True
