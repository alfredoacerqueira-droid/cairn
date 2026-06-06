"""Regression test: hybrid/polyglot repos must index all source types.

Tests that a mixed-language repo (Python + Go + Terraform + YAML) doesn't
silently drop non-dominant file types when cairn init detects the profile.
Each file type must be indexed and searchable.
"""

import subprocess
import tempfile
from pathlib import Path


def test_hybrid_repo_indexes_all_file_types():
    """Test that a mixed Python/Go/Terraform/YAML repo indexes all types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create mixed-language structure
        (repo_path / "app").mkdir()
        (repo_path / "app" / "service.py").write_text(
            'def process_payment(amount: float, currency: str) -> dict:\n'
            '    """Process a payment transaction."""\n'
            '    return {"amount": amount, "currency": currency, "status": "success"}\n'
        )

        (repo_path / "cmd").mkdir()
        (repo_path / "cmd" / "main.go").write_text(
            'package main\n'
            '\n'
            'import "fmt"\n'
            '\n'
            'func RunServer() {\n'
            '    fmt.Println("Server started on :8000")\n'
            '}\n'
            '\n'
            'func main() {\n'
            '    RunServer()\n'
            '}\n'
        )

        (repo_path / "infra").mkdir()
        (repo_path / "infra" / "main.tf").write_text(
            'resource "aws_db_instance" "ledger" {\n'
            '  allocated_storage    = 20\n'
            '  storage_type         = "gp2"\n'
            '  engine               = "postgres"\n'
            '  engine_version       = "14.7"\n'
            '  instance_class       = "db.t3.micro"\n'
            '  identifier           = "ledger-db"\n'
            '}\n'
        )

        (repo_path / "deploy").mkdir()
        (repo_path / "deploy" / "app.yaml").write_text(
            'apiVersion: apps/v1\n'
            'kind: Deployment\n'
            'metadata:\n'
            '  name: payment-service\n'
            'spec:\n'
            '  replicas: 3\n'
            '  selector:\n'
            '    matchLabels:\n'
            '      app: payment-service\n'
        )

        # Commit all files
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Run cairn init (config only, no indexing)
        import sys

        sys.path.insert(0, str(repo_path))

        # Create .cairn directory and write config
        cairn_dir = repo_path / ".cairn"
        cairn_dir.mkdir(exist_ok=True)

        from core.config import Config, save_config
        from core.profiles import detect_profile, get_profile
        from core.repo import (
            census_extensions,
            collect_source_files,
            detect_infra_markers,
            detect_source_layout,
        )
        from pipeline.ast_parser import EXTENSION_MAP

        detected_roots, detected_patterns = detect_source_layout(repo_path)
        entire_repo_census = census_extensions(repo_path, source_roots=None)

        # For mixed repos, expand source_roots to ["."]
        detected_extensions = set(ext for ext in entire_repo_census if ext in EXTENSION_MAP)
        is_mixed_repo = len(detected_extensions) > 1
        if is_mixed_repo and detected_roots != ["."]:
            detected_roots = ["."]
        ext_census = census_extensions(repo_path, source_roots=detected_roots)
        has_infra_markers = detect_infra_markers(repo_path, source_roots=detected_roots)
        detected_profile_name = detect_profile(ext_census, has_infra_markers=has_infra_markers)
        detected_profile = get_profile(detected_profile_name)

        # Build final patterns (same as init fix)
        default_config = Config()
        default_patterns = set(default_config.indexing.file_patterns)
        detected_patterns_set = set(detected_patterns)

        for ext in entire_repo_census:
            if ext in EXTENSION_MAP and ext not in {".json", ".toml"}:
                pattern = f"*{ext}"
                detected_patterns_set.add(pattern)

        final_patterns = sorted(list(detected_patterns_set | default_patterns))

        # Write config
        cfg = Config()
        cfg.indexing.source_roots = detected_roots
        cfg.indexing.file_patterns = final_patterns
        cfg.profile = detected_profile_name
        cfg.embeddings_enabled = detected_profile.embedding_enabled
        if detected_profile.embedding_enabled:
            cfg.indexing.embedding_model = detected_profile.embedding_model
        save_config(cfg, repo_path)

        # Verify collected files
        collected = collect_source_files(
            repo_path,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            cfg.indexing.source_roots,
        )

        collected_names = {f.name for f in collected if f.is_file()}

        # Assert all 4 file types are present
        assert "service.py" in collected_names, f"Python file not indexed. Found: {collected_names}"
        assert "main.go" in collected_names, f"Go file not indexed. Found: {collected_names}"
        assert "main.tf" in collected_names, f"Terraform file not indexed. Found: {collected_names}"
        assert "app.yaml" in collected_names, f"YAML file not indexed. Found: {collected_names}"

        # Verify patterns include all types
        assert "*.py" in final_patterns, f"*.py not in patterns: {final_patterns}"
        assert "*.go" in final_patterns, f"*.go not in patterns: {final_patterns}"
        assert "*.tf" in final_patterns, f"*.tf not in patterns: {final_patterns}"
        assert "*.yaml" in final_patterns, f"*.yaml not in patterns: {final_patterns}"
