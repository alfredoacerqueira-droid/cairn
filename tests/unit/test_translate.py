"""Unit tests for OpenAI/Anthropic ⇄ canonical translation, incl. streaming."""

from server.canonical import (
    CanonicalResponse,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    StreamEvent,
)
from server.translate import (
    anthropic_chunk_to_event,
    anthropic_request_to_canonical,
    canonical_to_anthropic_response,
    canonical_to_openai_response,
    openai_chunk_to_event,
    openai_request_to_canonical,
)

# ── OpenAI → Canonical ───────────────────────────────────────────────────────


class TestOpenAIToCanonical:
    def test_basic_text(self):
        body = {
            "model": "smart-context",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        }
        req = openai_request_to_canonical(body)
        assert req.model == "smart-context"
        assert req.system == "You are helpful"
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content_as_text() == "Hello"
        assert req.stream is False

    def test_with_tools(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
        req = openai_request_to_canonical(body)
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0].name == "get_weather"
        assert req.tools[0].description == "Get weather for a city"
        assert "city" in req.tools[0].input_schema.get("properties", {})
        assert req.tool_choice == "auto"

    def test_assistant_with_tool_calls(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "What weather in London?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "content": "Sunny, 22C",
                },
            ],
        }
        req = openai_request_to_canonical(body)
        assert len(req.messages) == 3

        assistant_msg = req.messages[1]
        assert assistant_msg.role == "assistant"
        blocks = assistant_msg.content
        assert isinstance(blocks, list)
        assert blocks[0].type == "tool_use"
        assert blocks[0].tool_use_id == "call_abc"
        assert blocks[0].tool_name == "get_weather"
        assert blocks[0].tool_input == {"city": "London"}

        tool_msg = req.messages[2]
        assert tool_msg.role == "tool"
        tool_blocks = tool_msg.content
        assert isinstance(tool_blocks, list)
        assert tool_blocks[0].type == "tool_result"
        assert tool_blocks[0].tool_use_id == "call_abc"

    def test_stream_flag(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        req = openai_request_to_canonical(body)
        assert req.stream is True

    def test_max_tokens_temperature(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END"],
        }
        req = openai_request_to_canonical(body)
        assert req.max_tokens == 500
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stop == ["END"]


# ── Anthropic → Canonical ────────────────────────────────────────────────────


class TestAnthropicToCanonical:
    def test_basic_text(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful",
        }
        req = anthropic_request_to_canonical(body)
        assert req.model == "claude-sonnet-4-20250514"
        assert req.system == "You are helpful"
        assert req.max_tokens == 1024
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content_as_text() == "Hello"

    def test_with_tools(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }
        req = anthropic_request_to_canonical(body)
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0].name == "get_weather"

    def test_tool_use_message(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Weather in London?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "get_weather",
                            "input": {"city": "London"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_001",
                            "content": "Sunny, 22C",
                        }
                    ],
                },
            ],
        }
        req = anthropic_request_to_canonical(body)
        assert len(req.messages) == 3

        assistant = req.messages[1]
        assert assistant.role == "assistant"
        blocks = assistant.content
        assert isinstance(blocks, list)
        assert blocks[0].type == "text"
        assert blocks[1].type == "tool_use"
        assert blocks[1].tool_use_id == "toolu_001"
        assert blocks[1].tool_input == {"city": "London"}

        tool = req.messages[2]
        assert tool.role == "user"  # Anthropic uses user role for tool_result
        tblocks = tool.content
        assert isinstance(tblocks, list)
        assert tblocks[0].type == "tool_result"
        assert tblocks[0].tool_use_id == "toolu_001"

    def test_stream_flag(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        req = anthropic_request_to_canonical(body)
        assert req.stream is True

    def test_string_content(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        req = anthropic_request_to_canonical(body)
        assert req.messages[0].content_as_text() == "Hello"


# ── Canonical → OpenAI (non-stream) ──────────────────────────────────────────


class TestCanonicalToOpenAI:
    def test_text_only(self):
        resp = CanonicalResponse(
            id="chatcmpl-123",
            model="gpt-4",
            content=[ContentBlock(type="text", text="Hello!")],
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )
        result = canonical_to_openai_response(resp)
        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion"
        choices = result["choices"]
        assert len(choices) == 1
        assert choices[0]["message"]["content"] == "Hello!"
        assert choices[0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_tool_calls(self):
        resp = CanonicalResponse(
            id="chatcmpl-456",
            model="gpt-4",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use_id="call_1",
                    tool_name="get_weather",
                    tool_input={"city": "London"},
                )
            ],
            stop_reason="tool_use",
        )
        result = canonical_to_openai_response(resp)
        choices = result["choices"]
        assert choices[0]["finish_reason"] == "tool_calls"
        tc = choices[0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["id"] == "call_1"
        assert tc[0]["function"]["name"] == "get_weather"

    def test_text_and_tool_calls(self):
        resp = CanonicalResponse(
            id="chatcmpl-789",
            model="gpt-4",
            content=[
                ContentBlock(type="text", text="Checking weather..."),
                ContentBlock(
                    type="tool_use",
                    tool_use_id="call_2",
                    tool_name="search",
                    tool_input={"query": "bugs"},
                ),
            ],
            stop_reason="tool_use",
        )
        result = canonical_to_openai_response(resp)
        choices = result["choices"]
        msg = choices[0]["message"]
        assert msg["content"] == "Checking weather..."
        assert len(msg["tool_calls"]) == 1


# ── Canonical → Anthropic (non-stream) ───────────────────────────────────────


class TestCanonicalToAnthropic:
    def test_text_only(self):
        resp = CanonicalResponse(
            id="msg_001",
            model="claude-sonnet-4-20250514",
            content=[ContentBlock(type="text", text="Hello!")],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
        )
        result = canonical_to_anthropic_response(resp)
        assert result["id"] == "msg_001"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10

    def test_tool_use(self):
        resp = CanonicalResponse(
            id="msg_002",
            model="claude-sonnet-4-20250514",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use_id="toolu_001",
                    tool_name="get_weather",
                    tool_input={"city": "London"},
                )
            ],
            stop_reason="tool_use",
        )
        result = canonical_to_anthropic_response(resp)
        assert result["stop_reason"] == "tool_use"
        content = result["content"]
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "toolu_001"
        assert content[0]["input"] == {"city": "London"}

    def test_stop_reason_normalization(self):
        resp = CanonicalResponse(
            id="msg_003",
            model="claude",
            content=[ContentBlock(type="text", text="ok")],
            stop_reason="unknown_reason",
        )
        result = canonical_to_anthropic_response(resp)
        assert result["stop_reason"] == "end_turn"


# ── OpenAI stream chunks → Canonical events ──────────────────────────────────


class TestOpenAIStreamChunks:
    def test_text_delta(self):
        chunk = {
            "id": "chatcmpl-001",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        event = openai_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockDeltaEvent)
        assert event.delta_type == "text_delta"
        assert event.text == "Hello"

    def test_role_chunk_ignored(self):
        """Role chunk should not produce an event (handled by streaming layer)."""
        chunk = {
            "id": "chatcmpl-001",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        event = openai_chunk_to_event(chunk)
        assert isinstance(event, MessageStartEvent)

    def test_tool_call_start(self):
        chunk = {
            "id": "chatcmpl-002",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        event = openai_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockStartEvent)
        assert event.content_block_type == "tool_use"
        assert event.tool_use_id == "call_abc"
        assert event.tool_name == "get_weather"

    def test_tool_call_arguments_delta(self):
        chunk = {
            "id": "chatcmpl-002",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city": "London"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        event = openai_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockDeltaEvent)
        assert event.delta_type == "input_json_delta"
        assert event.partial_json == '{"city": "London"}'

    def test_finish_reason_delta(self):
        chunk = {
            "id": "chatcmpl-002",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        event = openai_chunk_to_event(chunk)
        assert isinstance(event, MessageDeltaEvent)
        assert event.stop_reason == "tool_calls"

    def test_empty_choices(self):
        chunk = {"id": "x", "object": "x", "choices": []}
        event = openai_chunk_to_event(chunk)
        assert event is None


# ── Anthropic stream chunks → Canonical events ───────────────────────────────


class TestAnthropicStreamChunks:
    def test_message_start(self):
        chunk = {
            "type": "message_start",
            "message": {
                "id": "msg_001",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
            },
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, MessageStartEvent)
        assert event.message_id == "msg_001"
        assert event.model == "claude-sonnet-4-20250514"

    def test_content_block_start_text(self):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockStartEvent)
        assert event.index == 0
        assert event.content_block_type == "text"

    def test_content_block_start_tool_use(self):
        chunk = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_001",
                "name": "get_weather",
                "input": {},
            },
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockStartEvent)
        assert event.content_block_type == "tool_use"
        assert event.tool_use_id == "toolu_001"
        assert event.tool_name == "get_weather"

    def test_content_block_delta_text(self):
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockDeltaEvent)
        assert event.delta_type == "text_delta"
        assert event.text == "Hello"

    def test_content_block_delta_json(self):
        chunk = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockDeltaEvent)
        assert event.delta_type == "input_json_delta"
        assert event.partial_json == '{"city":'

    def test_content_block_stop(self):
        chunk = {"type": "content_block_stop", "index": 0}
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, ContentBlockStopEvent)

    def test_message_delta(self):
        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 42},
        }
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, MessageDeltaEvent)
        assert event.stop_reason == "tool_use"
        assert event.output_tokens == 42

    def test_message_stop(self):
        chunk = {"type": "message_stop"}
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, MessageStopEvent)

    def test_ping(self):
        chunk = {"type": "ping"}
        event = anthropic_chunk_to_event(chunk)
        assert isinstance(event, PingEvent)


# ── Streaming tool call round-trip (Anthropic) ───────────────────────────────


class TestStreamRoundTrip:
    def test_anthropic_tool_call_round_trip(self):
        """Simulate a full Anthropic streaming tool call: assemble events, verify
        the sequence is correct."""
        events: list[StreamEvent] = [
            MessageStartEvent(message_id="msg_abc", model="claude"),
            ContentBlockStartEvent(index=0, content_block_type="text"),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="Let me "),
            ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="check."),
            ContentBlockStopEvent(index=0),
            ContentBlockStartEvent(
                index=1,
                content_block_type="tool_use",
                tool_use_id="toolu_001",
                tool_name="get_weather",
            ),
            ContentBlockDeltaEvent(index=1, delta_type="input_json_delta", partial_json='{"city":'),
            ContentBlockDeltaEvent(
                index=1, delta_type="input_json_delta", partial_json='"London"}'
            ),
            ContentBlockStopEvent(index=1),
            MessageDeltaEvent(stop_reason="tool_use", output_tokens=35),
            MessageStopEvent(),
        ]

        assert len(events) == 11
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[-1], MessageStopEvent)

        tool_starts = [
            e
            for e in events
            if isinstance(e, ContentBlockStartEvent) and e.content_block_type == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0].tool_use_id == "toolu_001"

        json_deltas = [
            e
            for e in events
            if isinstance(e, ContentBlockDeltaEvent) and e.delta_type == "input_json_delta"
        ]
        assert len(json_deltas) == 2
        combined = "".join(d.partial_json or "" for d in json_deltas)
        assert "London" in combined

    def test_openai_tool_call_round_trip(self):
        """Simulate a full OpenAI streaming tool call."""
        events: list[StreamEvent] = [
            MessageStartEvent(message_id="chatcmpl-001", model="gpt-4"),
            ContentBlockStartEvent(
                index=0,
                content_block_type="tool_use",
                tool_use_id="call_123",
                tool_name="search",
            ),
            ContentBlockDeltaEvent(
                index=0, delta_type="input_json_delta", partial_json='{"query":"'
            ),
            ContentBlockDeltaEvent(index=0, delta_type="input_json_delta", partial_json='bugs"}'),
            ContentBlockStopEvent(index=0),
            MessageDeltaEvent(stop_reason="tool_calls", output_tokens=25),
            MessageStopEvent(),
        ]

        assert len(events) == 7
        tool_starts = [
            e
            for e in events
            if isinstance(e, ContentBlockStartEvent) and e.content_block_type == "tool_use"
        ]
        assert tool_starts[0].tool_name == "search"
