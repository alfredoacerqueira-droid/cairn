"""SSE streaming emitters for OpenAI and Anthropic dialects.

All emitters are async generators that accept canonical StreamEvent instances and
yield properly framed SSE byte strings.  The caller passes events in, receives SSE
bytes out, and sends those bytes over the wire.
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator

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


async def emit_openai_sse(events: AsyncGenerator[StreamEvent, None]) -> AsyncGenerator[bytes, None]:
    """Convert canonical stream events to OpenAI SSE chunks.

    Yields ``data: {json}\n\n`` chunks and a final ``data: [DONE]\n\n``.
    Accumulates tool-call fragments by index so the output is a well-formed
    sequence of chunks.
    """
    message_id = ""
    model = ""
    tool_call_id_by_index: dict[int, str] = {}
    tool_call_name_by_index: dict[int, str] = {}
    active_block_type: dict[int, str] = {}
    saw_start = False

    def _make_chunk(data: dict) -> bytes:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    async for event in events:
        if isinstance(event, PingEvent):
            yield b": ping\n\n"
            continue

        if isinstance(event, MessageStartEvent):
            message_id = event.message_id
            model = event.model
            saw_start = True
            yield _make_chunk(
                {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
            continue

        if isinstance(event, ContentBlockStartEvent):
            idx = event.index
            active_block_type[idx] = event.content_block_type
            if event.content_block_type == "tool_use" and event.tool_use_id:
                tool_call_id_by_index[idx] = event.tool_use_id or ""
                tool_call_name_by_index[idx] = event.tool_name or ""
                yield _make_chunk(
                    {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "id": event.tool_use_id,
                                            "type": "function",
                                            "function": {
                                                "name": event.tool_name or "",
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            continue

        if isinstance(event, ContentBlockDeltaEvent):
            idx = event.index
            if event.delta_type == "text_delta" and event.text:
                yield _make_chunk(
                    {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif event.delta_type == "input_json_delta" and event.partial_json:
                if idx not in active_block_type:
                    active_block_type[idx] = "tool_use"
                    if idx not in tool_call_id_by_index:
                        tool_call_id_by_index[idx] = ""
                        tool_call_name_by_index[idx] = ""
                    yield _make_chunk(
                        {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": idx,
                                                "id": tool_call_id_by_index.get(idx, ""),
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call_name_by_index.get(idx, ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                yield _make_chunk(
                    {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "function": {"arguments": event.partial_json},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            continue

        if isinstance(event, ContentBlockStopEvent):
            continue

        if isinstance(event, MessageDeltaEvent):
            finish = event.stop_reason or "stop"
            if finish == "end_turn":
                finish = "stop"
            elif finish == "tool_use":
                finish = "tool_calls"
            yield _make_chunk(
                {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
            )
            continue

        if isinstance(event, MessageStopEvent):
            if saw_start:
                yield b"data: [DONE]\n\n"
            continue

    if not saw_start:
        yield b"data: [DONE]\n\n"


async def emit_anthropic_sse(
    events: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[bytes, None]:
    """Convert canonical stream events to Anthropic SSE.

    Yields ``event: <type>\ndata: {json}\n\n`` lines with proper ping keep-alives.
    """
    message_id = ""
    model = ""
    started = False
    stopped = False

    def _make_sse(etype: str, data: dict) -> bytes:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {etype}\ndata: {payload}\n\n".encode("utf-8")

    async for event in events:
        if isinstance(event, PingEvent):
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
            continue

        if isinstance(event, MessageStartEvent):
            started = True
            message_id = event.message_id
            model = event.model
            yield _make_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                    },
                },
            )
            continue

        if isinstance(event, ContentBlockStartEvent):
            cb: dict = {"type": event.content_block_type}
            if event.content_block_type == "tool_use":
                cb["id"] = event.tool_use_id or ""
                cb["name"] = event.tool_name or ""
                cb["input"] = {}
            else:
                cb["text"] = ""
            yield _make_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": event.index,
                    "content_block": cb,
                },
            )
            continue

        if isinstance(event, ContentBlockDeltaEvent):
            delta: dict = {"type": event.delta_type}
            if event.delta_type == "text_delta":
                delta["text"] = event.text or ""
            elif event.delta_type == "input_json_delta":
                delta["partial_json"] = event.partial_json or ""
            yield _make_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": event.index,
                    "delta": delta,
                },
            )
            continue

        if isinstance(event, ContentBlockStopEvent):
            yield _make_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": event.index,
                },
            )
            continue

        if isinstance(event, MessageDeltaEvent):
            stop_reason = event.stop_reason or "end_turn"
            if stop_reason == "tool_calls":
                stop_reason = "tool_use"
            yield _make_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": event.stop_sequence,
                    },
                    "usage": {"output_tokens": event.output_tokens},
                },
            )
            continue

        if isinstance(event, MessageStopEvent):
            stopped = True
            yield _make_sse("message_stop", {"type": "message_stop"})
            continue

    if started and not stopped:
        yield _make_sse("message_stop", {"type": "message_stop"})
