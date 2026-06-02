"""Unit tests for the canonical request/response/stream model."""

from server.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    StreamEvent,
    Tool,
)


class TestContentBlock:
    def test_text_block(self):
        cb = ContentBlock(type="text", text="hello")
        assert cb.type == "text"
        assert cb.text == "hello"

    def test_tool_use_block(self):
        cb = ContentBlock(
            type="tool_use",
            tool_use_id="call_1",
            tool_name="get_weather",
            tool_input={"city": "London"},
        )
        assert cb.type == "tool_use"
        assert cb.tool_use_id == "call_1"
        assert cb.tool_name == "get_weather"
        assert cb.tool_input == {"city": "London"}

    def test_tool_result_block(self):
        cb = ContentBlock(
            type="tool_result",
            tool_use_id="call_1",
            tool_result_content="Sunny, 22C",
        )
        assert cb.type == "tool_result"
        assert cb.tool_result_content == "Sunny, 22C"


class TestMessage:
    def test_string_content(self):
        msg = Message(role="user", content="Hello")
        assert msg.content_as_text() == "Hello"

    def test_block_content(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type="text", text="I will help"),
                ContentBlock(type="text", text=" with that"),
            ],
        )
        assert "I will help" in msg.content_as_text()

    def test_tool_result_content(self):
        msg = Message(
            role="tool",
            content=[
                ContentBlock(
                    type="tool_result",
                    tool_use_id="c1",
                    tool_result_content="result",
                ),
            ],
        )
        assert msg.content_as_text() == "result"


class TestCanonicalRequest:
    def test_defaults(self):
        req = CanonicalRequest()
        assert req.messages == []
        assert req.stream is False

    def test_user_message_text(self):
        req = CanonicalRequest(
            messages=[
                Message(role="system", content="You are helpful"),
                Message(role="user", content="What is 2+2?"),
            ]
        )
        assert req.user_message_text() == "What is 2+2?"

    def test_user_message_text_blocks(self):
        req = CanonicalRequest(
            messages=[
                Message(
                    role="user",
                    content=[ContentBlock(type="text", text="Find the bug")],
                )
            ]
        )
        assert req.user_message_text() == "Find the bug"

    def test_no_user_message(self):
        req = CanonicalRequest(messages=[Message(role="system", content="sys")])
        assert req.user_message_text() is None

    def test_tools(self):
        req = CanonicalRequest(
            tools=[
                Tool(
                    name="search",
                    description="Search for code",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "search"
        assert req.tool_choice == "auto"


class TestCanonicalResponse:
    def test_text_content(self):
        resp = CanonicalResponse(
            id="r1",
            content=[ContentBlock(type="text", text="Hello world")],
        )
        assert resp.text_content() == "Hello world"

    def test_mixed_content(self):
        resp = CanonicalResponse(
            id="r1",
            content=[
                ContentBlock(type="text", text="Part 1"),
                ContentBlock(type="tool_use", tool_name="f"),
                ContentBlock(type="text", text="Part 2"),
            ],
        )
        assert "Part 1" in resp.text_content()
        assert "Part 2" in resp.text_content()


class TestStreamEvents:
    def test_message_start(self):
        e = MessageStartEvent(message_id="m1", model="claude")
        assert e.message_id == "m1"
        assert isinstance(e, StreamEvent)  # type: ignore[arg-type]

    def test_content_block_start(self):
        e = ContentBlockStartEvent(
            index=1, content_block_type="tool_use", tool_use_id="t1", tool_name="f"
        )
        assert e.tool_use_id == "t1"
        assert e.tool_name == "f"

    def test_content_block_delta_text(self):
        e = ContentBlockDeltaEvent(index=0, delta_type="text_delta", text="hello")
        assert e.text == "hello"

    def test_content_block_delta_json(self):
        e = ContentBlockDeltaEvent(index=1, delta_type="input_json_delta", partial_json='{"x":')
        assert e.partial_json == '{"x":'

    def test_content_block_stop(self):
        e = ContentBlockStopEvent(index=0)
        assert e.index == 0

    def test_message_delta(self):
        e = MessageDeltaEvent(stop_reason="tool_use", output_tokens=42)
        assert e.stop_reason == "tool_use"
        assert e.output_tokens == 42

    def test_message_stop(self):
        e = MessageStopEvent()
        assert isinstance(e, StreamEvent)  # type: ignore[arg-type]

    def test_ping(self):
        e = PingEvent()
        assert isinstance(e, StreamEvent)  # type: ignore[arg-type]
