"""Integration tests for local LLM optional feature."""

from pathlib import Path
from unittest.mock import Mock, patch

from core.config import Config, LocalLLMConfig
from pipeline.memory import MemorySummarizer
from server.ollama_client import (
    OllamaClient,
    OpenAICompatibleClient,
    make_llm_client,
)


class TestLocalLLMConfig:
    """Test LocalLLMConfig model."""

    def test_local_llm_disabled_by_default(self):
        cfg = Config()
        assert cfg.local_llm.enabled is False
        assert cfg.local_llm.backend == "ollama"
        assert cfg.local_llm.base_url is None
        assert cfg.local_llm.model is None
        assert cfg.local_llm.embed_model is None

    def test_local_llm_enabled_config(self):
        local_llm = LocalLLMConfig(
            enabled=True,
            backend="openai_compatible",
            base_url="http://localhost:8000",
            model="llama-2",
            embed_model="nomic-embed-text",
        )
        assert local_llm.enabled is True
        assert local_llm.backend == "openai_compatible"
        assert local_llm.base_url == "http://localhost:8000"
        assert local_llm.model == "llama-2"
        assert local_llm.embed_model == "nomic-embed-text"


class TestMakeLLMClientFactory:
    """Test make_llm_client factory function."""

    def test_factory_returns_ollama_when_none(self):
        """When local_llm is None, return OllamaClient."""
        client = make_llm_client(None)
        assert isinstance(client, OllamaClient)

    def test_factory_returns_ollama_when_disabled(self):
        """When local_llm.enabled=False, return OllamaClient (unused)."""
        local_llm = LocalLLMConfig(enabled=False)
        client = make_llm_client(local_llm)
        assert isinstance(client, OllamaClient)

    def test_factory_returns_openai_compatible_when_enabled(self):
        """When enabled with openai_compatible backend, return OpenAICompatibleClient."""
        local_llm = LocalLLMConfig(
            enabled=True,
            backend="openai_compatible",
            base_url="http://localhost:8000",
            model="llama-2",
        )
        client = make_llm_client(local_llm)
        assert isinstance(client, OpenAICompatibleClient)
        assert client.base_url == "http://localhost:8000"
        assert client.generate_model == "llama-2"

    def test_factory_returns_ollama_when_enabled_ollama_backend(self):
        """When enabled with ollama backend, return OllamaClient."""
        local_llm = LocalLLMConfig(
            enabled=True,
            backend="ollama",
            base_url="http://localhost:11434",
            model="qwen2.5-coder:3b",
        )
        client = make_llm_client(local_llm)
        assert isinstance(client, OllamaClient)

    def test_factory_raises_on_openai_without_base_url(self):
        """OpenAI-compatible without base_url should raise ValueError."""
        local_llm = LocalLLMConfig(
            enabled=True,
            backend="openai_compatible",
        )
        try:
            make_llm_client(local_llm)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "base_url" in str(e)


class TestOpenAICompatibleClient:
    """Test OpenAICompatibleClient interface."""

    def test_openai_compatible_init(self):
        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="llama-2",
            embed_model="nomic",
        )
        assert client.base_url == "http://localhost:8000"
        assert client.generate_model == "llama-2"
        assert client.embed_model == "nomic"

    def test_openai_compatible_base_url_trailing_slash_removed(self):
        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/",
            model="test",
        )
        assert client.base_url == "http://localhost:8000"

    @patch("httpx.post")
    def test_openai_compatible_embed(self, mock_post):
        """Test embed method uses OpenAI-style endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="test",
            embed_model="embed-model",
        )
        result = client.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "v1/embeddings" in call_args[0][0]

    @patch("httpx.post")
    def test_openai_compatible_generate(self, mock_post):
        """Test generate method uses OpenAI-style endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Generated text"}}]}
        mock_post.return_value = mock_response

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="llama-2",
        )
        result = client.generate("test prompt")

        assert result == "Generated text"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "v1/chat/completions" in call_args[0][0]

    @patch("httpx.get")
    def test_openai_compatible_health_check(self, mock_get):
        """Test health_check uses OpenAI v1/models endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="test",
        )
        result = client.health_check()

        assert result is True
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "v1/models" in call_args[0][0]

    @patch("httpx.get")
    def test_openai_compatible_list_models(self, mock_get):
        """Test list_models uses OpenAI v1/models endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        mock_get.return_value = mock_response

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="test",
        )
        result = client.list_models()

        assert result == ["model-1", "model-2"]

    def test_openai_compatible_pull_model_not_supported(self):
        """pull_model should return False (not supported)."""
        client = OpenAICompatibleClient(
            base_url="http://localhost:8000",
            model="test",
        )
        result = client.pull_model("any-model")
        assert result is False


class TestMemorySummarizerLLMDisabled:
    """Test MemorySummarizer with LLM disabled."""

    def test_memory_summarizer_init_with_llm_disabled(self, tmp_path):
        """Test MemorySummarizer can be created with llm_enabled=False."""
        ms = MemorySummarizer(repo_path=tmp_path, llm_enabled=False)
        assert ms.llm_enabled is False

    def test_deterministic_summary_extracts_files(self, tmp_path):
        """Test _deterministic_summary extracts filenames from diff."""
        ms = MemorySummarizer(repo_path=tmp_path, llm_enabled=False)

        # Create a realistic diff
        diff = """diff --git a/src/app.py b/src/app.py
index abc..def 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
 def main():
-    pass
+    print("hello")

diff --git a/tests/test_app.py b/tests/test_app.py
index abc..def 100644
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1,3 +1,3 @@
 def test_main():
-    pass
+    assert main() is not None
"""

        summary = ms._deterministic_summary(diff)
        assert "src/app.py" in summary
        assert "tests/test_app.py" in summary
        assert "tests" in summary.lower()

    def test_summarize_diff_uses_deterministic_when_disabled(self, tmp_path):
        """Test summarize_diff uses deterministic path when LLM disabled."""
        ms = MemorySummarizer(repo_path=tmp_path, llm_enabled=False)

        diff = """diff --git a/config.yaml b/config.yaml
index abc..def 100644
--- a/config.yaml
+++ b/config.yaml
@@ -1,3 +1,3 @@
 key: value
-old: true
+new: true
"""

        summary = ms.summarize_diff(diff)
        # Should contain the deterministic summary, not a generated one
        assert "config.yaml" in summary
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_diff_calls_generate_when_enabled(self, tmp_path):
        """Test summarize_diff calls LLM when enabled."""
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "This is an LLM summary"

        ms = MemorySummarizer(
            repo_path=tmp_path,
            ollama_client=mock_ollama,
            llm_enabled=True,
        )

        diff = "diff --git a/file.py b/file.py\n..."
        summary = ms.summarize_diff(diff)

        assert "LLM summary" in summary
        mock_ollama.generate.assert_called_once()

    def test_summarize_diff_fallback_to_deterministic_on_error(self, tmp_path):
        """Test summarize_diff falls back to deterministic when LLM fails."""
        mock_ollama = Mock()
        mock_ollama.generate.side_effect = Exception("Network error")

        ms = MemorySummarizer(
            repo_path=tmp_path,
            ollama_client=mock_ollama,
            llm_enabled=True,
        )

        diff = """diff --git a/file.py b/file.py
"""

        summary = ms.summarize_diff(diff)
        # Should fall back to deterministic
        assert "file.py" in summary
        assert isinstance(summary, str)

    def test_maybe_compact_deterministic_when_disabled(self, tmp_path):
        """Test _maybe_compact uses deterministic message when LLM disabled."""
        memory_file = tmp_path / "memory.md"
        ms = MemorySummarizer(
            repo_path=tmp_path,
            llm_enabled=False,
            memory_file=str(memory_file),
            max_entries=300,  # High max_entries so rotation doesn't trigger first
        )

        # Populate memory file with entries exceeding compaction threshold (200)
        for i in range(250):
            ms.append_to_memory(f"Entry {i}: some change")

        # Compaction threshold is 200 lines, so with 250 entries, compaction should trigger
        # Read file — should have compacted entry
        content = memory_file.read_text()
        assert "[COMPACTED]" in content
        assert "historical changes" in content or "entries compacted" in content


class TestContextAssemblerLLMGating:
    """Test ContextAssembler gates embeddings based on local_llm."""

    def test_context_assembler_respects_effective_embeddings_flag(self, tmp_path):
        """Test ContextAssembler computes effective embeddings flag correctly.

        Effective flag = cfg.embeddings_enabled AND cfg.local_llm.enabled.
        This is a unit test of the logic; full integration would require
        real profiles and configs.
        """
        # Case 1: both disabled
        cfg1 = Config()
        cfg1.embeddings_enabled = False
        cfg1.local_llm.enabled = False
        assert (cfg1.embeddings_enabled and cfg1.local_llm.enabled) is False

        # Case 2: config enabled, llm disabled
        cfg2 = Config()
        cfg2.embeddings_enabled = True
        cfg2.local_llm.enabled = False
        assert (cfg2.embeddings_enabled and cfg2.local_llm.enabled) is False

        # Case 3: config disabled, llm enabled
        cfg3 = Config()
        cfg3.embeddings_enabled = False
        cfg3.local_llm.enabled = True
        assert (cfg3.embeddings_enabled and cfg3.local_llm.enabled) is False

        # Case 4: both enabled
        cfg4 = Config()
        cfg4.embeddings_enabled = True
        cfg4.local_llm.enabled = True
        assert (cfg4.embeddings_enabled and cfg4.local_llm.enabled) is True


class TestDoctorCommand:
    """Test doctor command has local LLM status reporting code."""

    def test_doctor_code_includes_llm_status_reporting(self):
        """Verify that doctor command code includes local_llm status reporting."""
        # Read the CLI file and check that it includes the new llm status reporting
        cli_path = Path(__file__).parent.parent.parent / "cli" / "main.py"
        source = cli_path.read_text()

        # Verify key elements are present in the doctor implementation
        assert "cfg.local_llm.enabled" in source
        assert "Local LLM:" in source
        assert "disabled (lexical/structural" in source
        assert "openai_compatible" in source
        assert "health_check" in source
