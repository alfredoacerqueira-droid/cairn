"""Unit tests for Ollama llama.cpp options passthrough."""

from unittest.mock import patch

from core.config import Config, LocalLLMConfig
from server.ollama_client import OllamaClient, _worker_num_ctx, make_llm_client


class DummyLocalLLM:
    """A minimal object that looks like LocalLLMConfig enough for make_llm_client."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestLocalLLMConfigOllamaOptions:
    def test_default_is_empty_dict(self):
        cfg = LocalLLMConfig()
        assert cfg.ollama_options == {}

    def test_old_config_without_field_loads(self):
        cfg = Config(**{"local_llm": {"enabled": False}})
        assert cfg.local_llm.ollama_options == {}

    def test_set_and_roundtrip(self):
        cfg = LocalLLMConfig(ollama_options={"num_gpu": 10, "low_vram": True})
        assert cfg.ollama_options == {"num_gpu": 10, "low_vram": True}
        # round-trip through dump/load
        data = cfg.model_dump()
        cfg2 = LocalLLMConfig(**data)
        assert cfg2.ollama_options == {"num_gpu": 10, "low_vram": True}


class TestOllamaClientGenOptions:
    def test_gen_options_includes_num_ctx_default(self):
        client = OllamaClient()
        gen_opts = client._gen_options()
        assert gen_opts["num_ctx"] == _worker_num_ctx()

    def test_gen_options_merges_user_overrides(self):
        client = OllamaClient(options={"num_gpu": 5, "low_vram": True})
        gen_opts = client._gen_options()
        assert gen_opts["num_gpu"] == 5
        assert gen_opts["low_vram"] is True
        assert gen_opts["num_ctx"] == _worker_num_ctx()  # default still present

    def test_gen_options_user_can_override_num_ctx(self):
        client = OllamaClient(options={"num_ctx": 2048})
        gen_opts = client._gen_options()
        assert gen_opts["num_ctx"] == 2048

    def test_embed_options_none_when_no_options(self):
        client = OllamaClient()
        assert client._embed_options() is None

    def test_embed_options_returns_copy_when_options_set(self):
        client = OllamaClient(options={"num_gpu": 3, "num_thread": 4})
        embed_opts = client._embed_options()
        assert embed_opts == {"num_gpu": 3, "num_thread": 4}
        # should be a copy, not the same dict
        embed_opts["extra"] = 99
        assert client._options == {"num_gpu": 3, "num_thread": 4}


class TestMakeLLmClientWithOptions:
    def test_ollama_client_receives_options(self):
        llm = DummyLocalLLM(
            enabled=True,
            backend="ollama",
            ollama_options={"num_gpu": 10, "low_vram": True},
        )
        client = make_llm_client(llm)
        assert isinstance(client, OllamaClient)
        gen_opts = client._gen_options()
        assert gen_opts["num_gpu"] == 10
        assert gen_opts["low_vram"] is True
        assert gen_opts["num_ctx"] == _worker_num_ctx()
        assert client._embed_options() == {"num_gpu": 10, "low_vram": True}

    def test_openai_client_ignores_options(self):
        llm = DummyLocalLLM(
            enabled=True,
            backend="openai_compatible",
            base_url="http://127.0.0.1:8000",
            ollama_options={"num_gpu": 10},
        )
        client = make_llm_client(llm)
        from server.ollama_client import OpenAICompatibleClient

        assert isinstance(client, OpenAICompatibleClient)
        # Should not crash — options are silently ignored

    def test_no_options_default_is_backward_compatible(self):
        llm = DummyLocalLLM(enabled=True, backend="ollama")
        client = make_llm_client(llm)
        assert isinstance(client, OllamaClient)
        assert client._gen_options() == {"num_ctx": _worker_num_ctx()}
        assert client._embed_options() is None

    def test_none_local_llm_default_client(self):
        client = make_llm_client(None)
        assert isinstance(client, OllamaClient)
        assert client._embed_options() is None


class TestEmbedMonkeypatch:
    """Optional: monkeypatch httpx to assert the options key in embed payload."""

    def test_embed_payload_has_no_options_key_when_none_set(self):
        client = OllamaClient()
        with patch("server.ollama_client.httpx.post") as mock_post:
            mock_post.return_value.raise_for_status = lambda: None
            mock_post.return_value.json.return_value = {"embedding": [0.1, 0.2]}
            client.embed("hello")
            call_kwargs = mock_post.call_args.kwargs
            assert "options" not in call_kwargs["json"]

    def test_embed_payload_has_options_key_when_options_set(self):
        client = OllamaClient(options={"num_gpu": 5})
        with patch("server.ollama_client.httpx.post") as mock_post:
            mock_post.return_value.raise_for_status = lambda: None
            mock_post.return_value.json.return_value = {"embedding": [0.1, 0.2]}
            client.embed("hello")
            call_kwargs = mock_post.call_args.kwargs
            assert "options" in call_kwargs["json"]
            assert call_kwargs["json"]["options"] == {"num_gpu": 5}
