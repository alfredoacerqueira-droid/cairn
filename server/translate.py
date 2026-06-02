"""Translation between OpenAI, Anthropic, and the internal canonical model.

Covers request, non-stream response, and streaming chunk translation in both
directions for both agent-facing dialects.
"""

from __future__ import annotations

from typing import Any

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

# ── OpenAI → Canonical ───────────────────────────────────────────────────────


def openai_request_to_canonical(body: dict[str, Any]) -> CanonicalRequest:
    messages: list[Message] = []
    system: str | None = None

    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system = content if isinstance(content, str) else str(content)
            continue

        tool_calls = m.get("tool_calls")

        if role == "tool" and "tool_call_id" in m:
            blocks = [
                ContentBlock(
                    type="tool_result",
                    tool_use_id=m["tool_call_id"],
                    tool_result_content=content if isinstance(content, str) else str(content),
                )
            ]
            messages.append(Message(role="tool", content=blocks))
        elif isinstance(content, str) and content:
            messages.append(Message(role=role, content=[ContentBlock(type="text", text=content)]))
        elif isinstance(content, list):
            parts: list[ContentBlock] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(ContentBlock(type="text", text=part.get("text", "")))
            messages.append(Message(role=role, content=parts if parts else content))
        elif role == "assistant" and tool_calls:
            messages.append(Message(role=role, content=[]))

        if role == "assistant" and tool_calls:
            existing: list[ContentBlock] = (
                list(messages[-1].content) if isinstance(messages[-1].content, list) else []
            )
            for tc in tool_calls:
                fn = tc.get("function", {})
                existing.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tc.get("id", ""),
                        tool_name=fn.get("name", ""),
                        tool_input=_safe_json_parse(fn.get("arguments", "{}")),
                    )
                )
            messages[-1].content = existing

    tools: list[Tool] | None = None
    raw_tools = body.get("tools")
    if raw_tools:
        tools = []
        for t in raw_tools:
            if isinstance(t, dict):
                fn = t.get("function", t)
                tools.append(
                    Tool(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        input_schema=fn.get("parameters", fn.get("input_schema", {})),
                    )
                )

    tool_choice: str | dict[str, Any] | None = body.get("tool_choice")
    if not tool_choice and tools:
        tool_choice = "auto"

    return CanonicalRequest(
        messages=messages,
        model=body.get("model", ""),
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop"),
        stream=body.get("stream", False),
    )


# ── Anthropic → Canonical ────────────────────────────────────────────────────


def anthropic_request_to_canonical(body: dict[str, Any]) -> CanonicalRequest:
    messages: list[Message] = []

    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            messages.append(Message(role=role, content=[ContentBlock(type="text", text=content)]))
        elif isinstance(content, list):
            blocks: list[ContentBlock] = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype == "text":
                        blocks.append(ContentBlock(type="text", text=part.get("text", "")))
                    elif ptype == "tool_use":
                        blocks.append(
                            ContentBlock(
                                type="tool_use",
                                tool_use_id=part.get("id", ""),
                                tool_name=part.get("name", ""),
                                tool_input=part.get("input", {}),
                            )
                        )
                    elif ptype == "tool_result":
                        blocks.append(
                            ContentBlock(
                                type="tool_result",
                                tool_use_id=part.get("tool_use_id", ""),
                                tool_result_content=str(part.get("content", "")),
                                tool_result_is_error=part.get("is_error", False),
                            )
                        )
            messages.append(Message(role=role, content=blocks))

    tools: list[Tool] | None = None
    raw_tools = body.get("tools")
    if raw_tools:
        tools = []
        for t in raw_tools:
            if isinstance(t, dict):
                tools.append(
                    Tool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("input_schema", {}),
                    )
                )

    return CanonicalRequest(
        messages=messages,
        model=body.get("model", ""),
        system=body.get("system"),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        max_tokens=body.get("max_tokens", 1024),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop_sequences"),
        stream=body.get("stream", False),
    )


# ── Canonical → OpenAI (non-stream) ──────────────────────────────────────────


def canonical_to_openai_response(cr: CanonicalResponse) -> dict[str, Any]:
    import time

    response: dict[str, Any] = {
        "id": cr.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": cr.model,
    }

    message: dict[str, Any] = {"role": "assistant"}

    text_parts: list[str] = []
    tool_calls_list: list[dict[str, Any]] = []

    for b in cr.content:
        if b.type == "text" and b.text:
            text_parts.append(b.text)
        elif b.type == "tool_use":
            tool_calls_list.append(
                {
                    "id": b.tool_use_id or "",
                    "type": "function",
                    "function": {
                        "name": b.tool_name or "",
                        "arguments": _safe_json_dumps(b.tool_input),
                    },
                }
            )

    if text_parts:
        message["content"] = "\n".join(text_parts)
    if tool_calls_list:
        message["tool_calls"] = tool_calls_list

    finish_reason = "tool_calls" if tool_calls_list else (cr.stop_reason or "stop")
    if finish_reason == "end_turn":
        finish_reason = "stop"
    elif finish_reason == "tool_use":
        finish_reason = "tool_calls"

    response["choices"] = [{"index": 0, "message": message, "finish_reason": finish_reason}]
    response["usage"] = {
        "prompt_tokens": cr.input_tokens,
        "completion_tokens": cr.output_tokens,
        "total_tokens": cr.input_tokens + cr.output_tokens,
    }

    return response


# ── Canonical → Anthropic (non-stream) ───────────────────────────────────────


def canonical_to_anthropic_response(cr: CanonicalResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for b in cr.content:
        if b.type == "text" and b.text:
            content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": b.tool_use_id or "",
                    "name": b.tool_name or "",
                    "input": b.tool_input or {},
                }
            )

    stop_reason = cr.stop_reason or "end_turn"
    if stop_reason not in ("end_turn", "max_tokens", "stop_sequence", "tool_use"):
        stop_reason = "end_turn"

    return {
        "id": cr.id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": cr.model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": cr.input_tokens, "output_tokens": cr.output_tokens},
    }


# ── OpenAI stream → Canonical events ─────────────────────────────────────────


def openai_chunk_to_event(chunk: dict[str, Any]) -> StreamEvent | None:
    """Convert a single OpenAI SSE chunk to a canonical stream event."""
    choices = chunk.get("choices", [])
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")

    idx = choice.get("index", 0)

    if chunk.get("id") and delta.get("role"):
        return MessageStartEvent(message_id=chunk["id"], model=chunk.get("model", ""))

    if "tool_calls" in delta:
        # TODO(review): only first tool_call per chunk handled; parallel
        # tool calls in one chunk need multi-event support
        if delta["tool_calls"]:
            tc = delta["tool_calls"][0]
            tc_index = tc.get("index", 0)
            fn = tc.get("function", {})
            if "id" in tc:
                return ContentBlockStartEvent(
                    index=tc_index,
                    content_block_type="tool_use",
                    tool_use_id=tc["id"],
                    tool_name=fn.get("name", ""),
                )
            if "arguments" in fn:
                return ContentBlockDeltaEvent(
                    index=tc_index,
                    delta_type="input_json_delta",
                    partial_json=fn["arguments"],
                )
        return None

    if "content" in delta and delta["content"]:
        return ContentBlockDeltaEvent(index=idx, delta_type="text_delta", text=delta["content"])

    if finish:
        return MessageDeltaEvent(stop_reason=finish)

    return None


# ── Anthropic stream → Canonical events ──────────────────────────────────────


def anthropic_chunk_to_event(chunk: dict[str, Any]) -> StreamEvent | None:
    """Convert a single Anthropic SSE chunk to a canonical stream event."""
    etype = chunk.get("type", "")

    if etype == "message_start":
        msg = chunk.get("message", {})
        return MessageStartEvent(message_id=msg.get("id", ""), model=msg.get("model", ""))

    if etype == "content_block_start":
        idx = chunk.get("index", 0)
        cb = chunk.get("content_block", {})
        return ContentBlockStartEvent(
            index=idx,
            content_block_type=cb.get("type", "text"),
            tool_use_id=cb.get("id"),
            tool_name=cb.get("name"),
        )

    if etype == "content_block_delta":
        idx = chunk.get("index", 0)
        delta = chunk.get("delta", {})
        return ContentBlockDeltaEvent(
            index=idx,
            delta_type=delta.get("type", "text_delta"),
            text=delta.get("text"),
            partial_json=delta.get("partial_json"),
        )

    if etype == "content_block_stop":
        return ContentBlockStopEvent(index=chunk.get("index", 0))

    if etype == "message_delta":
        d = chunk.get("delta", {})
        u = chunk.get("usage", {})
        return MessageDeltaEvent(
            stop_reason=d.get("stop_reason"),
            stop_sequence=d.get("stop_sequence"),
            output_tokens=u.get("output_tokens", 0),
        )

    if etype == "message_stop":
        return MessageStopEvent()

    if etype == "ping":
        return PingEvent()

    return None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_json_parse(s: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_json_dumps(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"
