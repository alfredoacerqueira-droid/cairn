"""Unit tests for Ollama client."""

from unittest.mock import MagicMock, patch

from server.ollama_client import OllamaClient


class TestOllamaClient:
    def test_health_check_success(self):
        client = OllamaClient()
        with patch.object(client, "health_check", return_value=True):
            assert client.health_check() is True

    def test_health_check_failure(self):
        client = OllamaClient()
        with patch.object(client, "health_check", return_value=False):
            assert client.health_check() is False

    def test_base_url_normalization(self):
        client = OllamaClient(base_url="http://127.0.0.1:11434/")
        assert client.base_url == "http://127.0.0.1:11434"

    def test_default_base_url(self):
        client = OllamaClient()
        assert client.base_url == "http://127.0.0.1:11434"

    def test_pull_model_success(self):
        """Test successful model pull (HTTP 200)."""
        client = OllamaClient()
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            result = client.pull_model("nomic-embed-text")
            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "api/pull" in call_args[0][0]
            assert call_args[1]["json"]["name"] == "nomic-embed-text"
            assert call_args[1]["json"]["stream"] is False

    def test_pull_model_failure_non_200(self):
        """Test model pull failure (non-200 status)."""
        client = OllamaClient()
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_post.return_value = mock_response

            result = client.pull_model("nonexistent-model")
            assert result is False

    def test_pull_model_exception(self):
        """Test model pull with network exception."""
        client = OllamaClient()
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("Network error")

            result = client.pull_model("test-model")
            assert result is False

    def test_pull_model_timeout(self):
        """Test that pull_model uses appropriate timeout."""
        client = OllamaClient()
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            client.pull_model("test-model")
            # Check timeout is 600s (10 minutes)
            assert mock_post.call_args[1]["timeout"] == 600.0
