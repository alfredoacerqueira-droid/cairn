"""Internal canonical request/response/stream-event model.

Both OpenAI /v1/chat/completions and Anthropic /v1/messages translate to/from this
shared representation so the gateway's core logic (context assembly, upstream
forwarding) operates on a single format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ContentBlock:
    """A single content block within a message (text, tool_use, or tool_result)."""

    type: Literal["text", "tool_use", "tool_result"] = "text"
    text: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result_content: str | None = None
    tool_result_is_error: bool = False


@dataclass
class Message:
    """A conversation message with either plain text or structured content blocks."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock] = ""

    def content_as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        texts: list[str] = []
        for b in self.content:
            if b.type == "text" and b.text:
                texts.append(b.text)
            elif b.type == "tool_result" and b.tool_result_content:
                texts.append(b.tool_result_content)
        return "\n".join(texts)


@dataclass
class Tool:
    """Tool / function definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRequest:
    """Internal request format for both agent-facing API dialects."""

    messages: list[Message] = field(default_factory=list)
    model: str = ""
    system: str | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False

    def user_message_text(self) -> str | None:
        """Return the text of the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content_as_text()
        return None


@dataclass
class CanonicalResponse:
    """Internal non-streaming response format."""

    id: str = ""
    model: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def text_content(self) -> str:
        texts: list[str] = []
        for b in self.content:
            if b.type == "text" and b.text:
                texts.append(b.text)
        return "\n".join(texts)


# ── Stream events ────────────────────────────────────────────────────────────


@dataclass
class StreamEvent:
    pass


@dataclass
class MessageStartEvent(StreamEvent):
    """Anthropic: message_start.  OpenAI: first chunk."""

    message_id: str = ""
    model: str = ""


@dataclass
class ContentBlockStartEvent(StreamEvent):
    """Anthropic: content_block_start.  OpenAI: tool_call chunk with id/name."""

    index: int = 0
    content_block_type: str = "text"  # text | tool_use
    tool_use_id: str | None = None
    tool_name: str | None = None


@dataclass
class ContentBlockDeltaEvent(StreamEvent):
    """Anthropic: content_block_delta.  OpenAI: delta.content / delta.tool_calls."""

    index: int = 0
    delta_type: str = "text_delta"  # text_delta | input_json_delta
    text: str | None = None
    partial_json: str | None = None


@dataclass
class ContentBlockStopEvent(StreamEvent):
    """Anthropic & OpenAI: block completed."""

    index: int = 0


@dataclass
class MessageDeltaEvent(StreamEvent):
    """Anthropic: message_delta carries stop_reason + usage.  OpenAI: final chunk."""

    stop_reason: str | None = None
    stop_sequence: str | None = None
    output_tokens: int = 0


@dataclass
class MessageStopEvent(StreamEvent):
    """Anthropic: message_stop.  OpenAI: final [DONE]."""

    pass


@dataclass
class PingEvent(StreamEvent):
    """Anthropic: keep-alive ping."""

    pass
