"""Unit tests for config loader."""

from pathlib import Path

from core.config import Config, EnabledConfig, ResourceConfig, load_config, save_config


class TestConfigLoader:
    def test_default_config(self):
        config = Config()
        assert config.enabled.file_watcher is True
        assert config.resources.max_cpu_percent == 50
        assert config.resources.max_memory_mb == 4096

    def test_load_missing_config(self):
        config = load_config(Path("/nonexistent"))
        assert config.enabled.file_watcher is True

    def test_save_and_load(self, tmp_path):
        config = Config(
            enabled=EnabledConfig(file_watcher=False),
            resources=ResourceConfig(max_cpu_percent=30),
        )

        save_config(config, tmp_path)

        loaded = load_config(tmp_path)
        assert loaded.enabled.file_watcher is False
        assert loaded.resources.max_cpu_percent == 30

    def test_partial_override(self, tmp_path):
        # Save with partial config
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"resources": {"max_cpu_percent": 75}}))

        loaded = load_config(tmp_path)
        assert loaded.resources.max_cpu_percent == 75
        assert loaded.enabled.file_watcher is True  # Default

    def test_stale_config_migration(self, tmp_path):
        """Test that old config without new excludes gets them merged in."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Old config with only 3 excludes (no **/.venv/**)
        old_config = {
            "indexing": {
                "exclude_patterns": [
                    "**/node_modules/**",
                    "**/.git/**",
                    "**/__pycache__/**",
                ]
            }
        }
        config_file.write_text(yaml.dump(old_config))

        loaded = load_config(tmp_path)

        # After migration, should have both old patterns and new ones
        assert "**/node_modules/**" in loaded.indexing.exclude_patterns
        assert "**/.git/**" in loaded.indexing.exclude_patterns
        assert "**/__pycache__/**" in loaded.indexing.exclude_patterns
        # New sentinel and others should be added
        assert "**/.venv/**" in loaded.indexing.exclude_patterns
        assert "**/tests/**" in loaded.indexing.exclude_patterns
        assert "**/benchmarks/**" in loaded.indexing.exclude_patterns

    def test_embedding_model_config_field_exists(self):
        """Test that IndexingConfig has embedding_model field."""
        from core.config import IndexingConfig

        config = IndexingConfig()
        assert "embedding_model" in config.model_dump()
        assert config.embedding_model == "nomic-embed-text"

    def test_embedding_model_roundtrip(self, tmp_path):
        """Test that embedding_model survives save/load cycle."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Create config with custom embedding model
        custom_config = {"indexing": {"embedding_model": "code-embedding-model-v1"}}
        config_file.write_text(yaml.dump(custom_config))

        loaded = load_config(tmp_path)
        assert loaded.indexing.embedding_model == "code-embedding-model-v1"

    def test_embedding_model_defaults_to_nomic(self, tmp_path):
        """Test that embedding_model defaults to nomic-embed-text."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Config without embedding_model
        minimal_config = {"indexing": {}}
        config_file.write_text(yaml.dump(minimal_config))

        loaded = load_config(tmp_path)
        assert loaded.indexing.embedding_model == "nomic-embed-text"
