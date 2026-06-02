"""Integration tests for the gateway server using FastAPI TestClient."""

from fastapi.testclient import TestClient

from core.version import CAIRN_VERSION
from server.api import app

client = TestClient(app)


class TestServerIntegration:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        # Source-of-truth constant, so this can't drift from pyproject again.
        assert data["version"] == CAIRN_VERSION

    def test_list_models(self):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "smart-context"

    def test_chat_completions_no_cloud_key(self, monkeypatch):
        """Without cloud API key returns local context or rejection (preserves model).

        Note: Context is now token-compressed by default. If confidence guard
        rejects the query, we get a rejection message instead of context.
        The important thing is that we get a response without cloud key.
        """

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart-context",
                "messages": [{"role": "user", "content": "How does auth work?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "smart-context"
        assert "choices" in data
        assert len(data["choices"]) > 0
        # Response is returned with some content
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_chat_completions_no_user_message_allowed(self):
        """Conversations without a user message proceed (e.g. tool-result only)."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart-context",
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completions_preserves_params(self):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart-context",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completions_invalid_json(self):
        response = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_chat_completions_with_tools(self):
        """Tools pass through correctly."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "smart-context",
                "messages": [{"role": "user", "content": "test"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "description": "Search the codebase",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        assert response.status_code == 200

    def test_messages_endpoint(self):
        """Anthropic /v1/messages endpoint returns correct response."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "What does main do?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "assistant"
        assert "content" in data

    def test_messages_endpoint_with_tools(self):
        """Anthropic format with tools passes through."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "test"}],
                "tools": [
                    {
                        "name": "search_code",
                        "description": "Search code",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )
        assert response.status_code == 200

    def test_messages_count_tokens(self):
        """Count tokens endpoint returns input_tokens."""
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] > 0
