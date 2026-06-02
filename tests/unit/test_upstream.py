"""Unit tests for upstream provider module (mostly with no cloud key)."""

import pytest

from server.canonical import CanonicalRequest, ContentBlock, Message
from server.upstream import call_upstream, is_cloud_configured, stream_upstream


class TestIsCloudConfigured:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)
        assert is_cloud_configured() is False

    def test_with_key(self, monkeypatch):
        monkeypatch.setenv("CLOUD_API_KEY", "sk-test")
        assert is_cloud_configured() is True


@pytest.mark.asyncio
class TestLocalNoopResponse:
    async def test_returns_canonical_response(self, monkeypatch):
        monkeypatch.delenv("CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)

        req = CanonicalRequest(
            messages=[
                Message(
                    role="system",
                    content=[ContentBlock(type="text", text="You are helpful")],
                ),
                Message(
                    role="user",
                    content=[ContentBlock(type="text", text="What is 2+2?")],
                ),
            ],
            model="smart-context",
        )

        response = await call_upstream(req)
        assert response.id.startswith("local-")
        assert response.model == "smart-context"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert "helpful" in response.content[0].text or "2+2" in response.content[0].text
        assert response.stop_reason == "stop"

    async def test_local_stream_response(self, monkeypatch):
        monkeypatch.delenv("CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)

        req = CanonicalRequest(
            messages=[
                Message(
                    role="user",
                    content=[ContentBlock(type="text", text="Test query")],
                ),
            ],
            model="smart-context",
        )

        events = []
        async for evt in stream_upstream(req):
            events.append(evt)

        assert len(events) > 0
        from server.canonical import MessageStartEvent, MessageStopEvent

        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[-1], MessageStopEvent)


@pytest.mark.asyncio
class TestUpstreamWithMock:
    """These tests verify the upstream module's behavior without actually
    calling external APIs. They just verify the local-only path works."""

    async def test_stream_yields_proper_sequence(self, monkeypatch):
        monkeypatch.delenv("CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)

        req = CanonicalRequest(
            messages=[
                Message(role="user", content=[ContentBlock(type="text", text="Q")]),
            ],
            stream=True,
        )

        from server.canonical import (
            ContentBlockDeltaEvent,
            ContentBlockStartEvent,
            ContentBlockStopEvent,
            MessageDeltaEvent,
            MessageStartEvent,
            MessageStopEvent,
        )

        types_seen = set()
        async for evt in stream_upstream(req):
            types_seen.add(type(evt))

        assert MessageStartEvent in types_seen
        assert MessageStopEvent in types_seen
        assert ContentBlockStartEvent in types_seen
        assert ContentBlockDeltaEvent in types_seen
        assert ContentBlockStopEvent in types_seen
        assert MessageDeltaEvent in types_seen
