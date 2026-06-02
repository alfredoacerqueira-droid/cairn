"""Unit tests for SSE streaming emitters (OpenAI and Anthropic dialects)."""

import pytest

from server.canonical import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    StreamEvent,
)
from server.streaming import emit_anthropic_sse, emit_openai_sse

pytestmark = pytest.mark.asyncio


async def _collect_bytes(generator):
    """Collect all bytes from an async generator."""
    result: list[bytes] = []
    async for chunk in generator:
        result.append(chunk)
    return result


async def _events_to_gen(events: list[StreamEvent]):
    """Create an async generator from a list of events."""
    for e in events:
        yield e


class TestOpenAISSE:
    async def test_text_stream(self):
        events = [
            MessageStartEvent(message_id="chatcmpl-001", model="gpt-4"),
            ContentBlockStartEvent(index=0, content_block_type="text"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="Hello"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text=" world"),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="stop"),
            MessageStopEvent(),
        ]
        chunks = await _collect_bytes(emit_openai_sse(_events_to_gen(events)))
        assert len(chunks) > 0

        decoded = [c.decode("utf-8") for c in chunks]
        combined = "".join(decoded)

        assert "data: [DONE]" in decoded[-1] or "data: [DONE]\n\n" in combined

    async def test_tool_call_stream(self):
        events = [
            MessageStartEvent(message_id="cmpl-002", model="gpt-4"),
            ContentBlockStartEvent(
                index=0,
                content_block_type="tool_use",
                tool_use_id="call_abc",
                tool_name="get_weather",
            ),
            ContentBlockDeltaEvent(
                index=0,
                delta_type="input_json_delta",
                partial_json='{"city":"London"}',
            ),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="tool_calls"),
            MessageStopEvent(),
        ]
        chunks = await _collect_bytes(emit_openai_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]
        combined = "".join(decoded)

        assert any("tool_calls" in d for d in decoded)
        assert "data: [DONE]" in combined

    async def test_ping_no_output(self):
        events = [PingEvent()]
        chunks = await _collect_bytes(emit_openai_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]
        assert "ping" in decoded[0].lower()

    async def test_emits_done_event(self):
        events = [MessageStopEvent()]
        chunks = await _collect_bytes(emit_openai_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]
        assert any("DONE" in d for d in decoded)


class TestAnthropicSSE:
    async def test_text_stream(self):
        events = [
            MessageStartEvent(message_id="msg_001", model="claude"),
            ContentBlockStartEvent(index=0, content_block_type="text"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="Hello"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text=" world"),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="end_turn", output_tokens=10),
            MessageStopEvent(),
        ]
        chunks = await _collect_bytes(emit_anthropic_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]

        assert any("message_start" in d for d in decoded)
        assert any("message_stop" in d for d in decoded)
        assert any("text_delta" in d for d in decoded)

    async def test_tool_call_stream(self):
        events = [
            MessageStartEvent(message_id="msg_002", model="claude"),
            ContentBlockStartEvent(
                index=0,
                content_block_type="tool_use",
                tool_use_id="toolu_001",
                tool_name="search",
            ),
            ContentBlockDeltaEvent(
                index=0,
                delta_type="input_json_delta",
                partial_json='{"query":"bugs"}',
            ),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="tool_use", output_tokens=25),
            MessageStopEvent(),
        ]
        chunks = await _collect_bytes(emit_anthropic_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]

        assert any("content_block_start" in d for d in decoded)
        assert any("input_json_delta" in d for d in decoded)
        assert any("tool_use" in d for d in decoded)

    async def test_ping_emitted(self):
        events = [PingEvent()]
        chunks = await _collect_bytes(emit_anthropic_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]
        assert any("ping" in d for d in decoded)

    async def test_stop_reason_normalization(self):
        events = [
            MessageStartEvent(message_id="m", model="c"),
            ContentBlockStartEvent(index=0, content_block_type="text"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="x"),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="tool_calls", output_tokens=1),
            MessageStopEvent(),
        ]
        chunks = await _collect_bytes(emit_anthropic_sse(_events_to_gen(events)))
        decoded = [c.decode("utf-8") for c in chunks]
        combined = "".join(decoded)
        assert "tool_use" in combined
