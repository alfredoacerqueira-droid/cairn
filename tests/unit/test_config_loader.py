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

    def test_budget_config_defaults(self):
        """Test BudgetConfig defaults."""
        from core.config import BudgetConfig

        config = BudgetConfig()
        assert config.session_window == 200_000
        assert config.session_pct == 0.18
        assert config.tool_max_tokens == 8_000
        assert config.tokenizer_model == "claude"

    def test_budget_config_in_root_config(self):
        """Test that Config has budget field with BudgetConfig defaults."""
        config = Config()
        assert hasattr(config, "budget")
        assert config.budget.session_window == 200_000
        assert config.budget.session_pct == 0.18
        assert config.budget.tool_max_tokens == 8_000
        assert config.budget.tokenizer_model == "claude"

    def test_budget_computed_cap_sanity(self):
        """Test computed cap sanity: session_window * session_pct == 36000."""
        config = Config()
        computed_cap = config.budget.session_window * config.budget.session_pct
        assert computed_cap == 36_000.0

    def test_indexing_store_backend_default(self):
        """Test that IndexingConfig has store_backend defaulting to chroma."""
        from core.config import IndexingConfig

        config = IndexingConfig()
        assert config.store_backend == "chroma"

    def test_local_llm_config_new_fields_defaults(self):
        """Test LocalLLMConfig new fields defaults."""
        from core.config import LocalLLMConfig

        config = LocalLLMConfig()
        assert config.context_window == 8192
        assert config.max_local_tokens == 6000
        assert config.reduce_reserve_tokens == 1024
        assert config.chunk_overlap_pct == 0.12
        assert config.one_shot_threshold == 0.75
        assert config.embedder == "ollama"
        assert config.fastembed_model == "BAAI/bge-small-en-v1.5"
        assert config.map_concurrency == 1

    def test_local_llm_config_backward_compat(self):
        """Test LocalLLMConfig backward compat: old fields still work."""
        from core.config import LocalLLMConfig

        config = LocalLLMConfig(
            enabled=True,
            backend="openai_compatible",
            base_url="http://127.0.0.1:8000",
            model="llama2",
            embed_model="bge-small",
        )
        # Old fields preserved
        assert config.enabled is True
        assert config.backend == "openai_compatible"
        assert config.base_url == "http://127.0.0.1:8000"
        assert config.model == "llama2"
        assert config.embed_model == "bge-small"
        # New fields get defaults
        assert config.context_window == 8192
        assert config.embedder == "ollama"

    def test_budget_config_roundtrip(self, tmp_path):
        """Test that budget config survives save/load cycle."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Config with custom budget values
        custom_config = {
            "budget": {
                "session_window": 100_000,
                "session_pct": 0.25,
                "tool_max_tokens": 10_000,
                "tokenizer_model": "gpt2",
            }
        }
        config_file.write_text(yaml.dump(custom_config))

        loaded = load_config(tmp_path)
        assert loaded.budget.session_window == 100_000
        assert loaded.budget.session_pct == 0.25
        assert loaded.budget.tool_max_tokens == 10_000
        assert loaded.budget.tokenizer_model == "gpt2"

    def test_old_config_without_new_fields(self, tmp_path):
        """Backward compat: old config without budget/store_backend loads with
        new defaults."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Minimal old-style config with no new fields
        old_config = {
            "profile": "python",
            "local_llm": {"enabled": False},
            "indexing": {},
        }
        config_file.write_text(yaml.dump(old_config))

        loaded = load_config(tmp_path)
        # Old fields
        assert loaded.profile == "python"
        assert loaded.local_llm.enabled is False
        # New fields get defaults
        assert loaded.budget.session_window == 200_000
        assert loaded.indexing.store_backend == "chroma"
        assert loaded.local_llm.context_window == 8192

    def test_local_llm_new_fields_in_yaml(self, tmp_path):
        """Test LocalLLMConfig new fields survive YAML load."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Config with custom LocalLLM fields
        custom_config = {
            "local_llm": {
                "enabled": True,
                "backend": "ollama",
                "context_window": 4096,
                "max_local_tokens": 3000,
                "embedder": "fastembed",
                "fastembed_model": "intfloat/e5-small-v2",
                "map_concurrency": 2,
            }
        }
        config_file.write_text(yaml.dump(custom_config))

        loaded = load_config(tmp_path)
        assert loaded.local_llm.enabled is True
        assert loaded.local_llm.context_window == 4096
        assert loaded.local_llm.max_local_tokens == 3000
        assert loaded.local_llm.embedder == "fastembed"
        assert loaded.local_llm.fastembed_model == "intfloat/e5-small-v2"
        assert loaded.local_llm.map_concurrency == 2

    def test_cache_config_semantic_ttl_default(self):
        """Test that CacheConfig has semantic_ttl_seconds with 1800 default."""
        from core.config import CacheConfig

        config = CacheConfig()
        assert config.semantic_ttl_seconds == 1800
        assert config.ttl_seconds == 300

    def test_cache_semantic_ttl_in_root_config(self):
        """Test that Config has cache.semantic_ttl_seconds with correct default."""
        config = Config()
        assert hasattr(config.cache, "semantic_ttl_seconds")
        assert config.cache.semantic_ttl_seconds == 1800
        assert config.cache.ttl_seconds == 300

    def test_cache_semantic_ttl_roundtrip(self, tmp_path):
        """Test that cache.semantic_ttl_seconds survives save/load cycle."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Config with custom semantic_ttl_seconds
        custom_config = {
            "cache": {
                "enabled": True,
                "ttl_seconds": 300,
                "semantic_ttl_seconds": 3600,
                "max_entries": 100,
            }
        }
        config_file.write_text(yaml.dump(custom_config))

        loaded = load_config(tmp_path)
        assert loaded.cache.ttl_seconds == 300
        assert loaded.cache.semantic_ttl_seconds == 3600

    def test_old_config_without_semantic_ttl(self, tmp_path):
        """Test backward compat: old config without semantic_ttl_seconds loads with default."""
        import yaml

        config_dir = tmp_path / ".cairn"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Old-style config with no semantic_ttl_seconds
        old_config = {
            "cache": {
                "enabled": True,
                "ttl_seconds": 300,
                "max_entries": 100,
            }
        }
        config_file.write_text(yaml.dump(old_config))

        loaded = load_config(tmp_path)
        assert loaded.cache.ttl_seconds == 300
        assert loaded.cache.semantic_ttl_seconds == 1800  # default
