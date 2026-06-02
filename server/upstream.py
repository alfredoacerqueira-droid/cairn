"""Upstream provider calls — stream and non-stream, provider-agnostic.

Supports OpenAI-compatible, Anthropic, local Ollama, and local enrichment mode.
Features RTK-style token compression (see server/token_compressor.py) and
clean proxy API key handling (no keys stored on gateway).

TODO(review): The plan calls for LiteLLM integration for broader provider support
and automatic format translation.  This httpx implementation provides the same
interface; swapping is isolated to this module.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from typing import Any, AsyncGenerator

import httpx

from server.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    ContentBlock,
    StreamEvent,
)
from server.translate import (
    anthropic_chunk_to_event,
    openai_chunk_to_event,
)

logger = logging.getLogger(__name__)

# Per-request API key via context variable (safe for async/concurrent requests)
_request_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_api_key", default=None
)


def set_request_api_key(key: str | None) -> None:
    """Set API key from request header for this async context (RTK-style clean proxy)."""
    _request_api_key.set(key)


def _api_key() -> str:
    """Get API key: header first (clean proxy), env fallback."""
    key = _request_api_key.get(None)
    if key:
        return key
    return os.environ.get("CLOUD_API_KEY", os.environ.get("UPSTREAM_API_KEY", ""))


def is_cloud_configured() -> bool:
    return bool(_api_key())


def is_local_enrich_enabled() -> bool:
    """Check if local enrichment mode is enabled."""
    return os.environ.get("LOCAL_ENRICH_MODE", "").lower() == "true"


def _openai_compatible_base() -> str:
    return os.environ.get(
        "CLOUD_API_BASE", os.environ.get("UPSTREAM_BASE_URL", "https://api.deepseek.com/v1")
    )


def _model_name(default: str = "deepseek-chat") -> str:
    return os.environ.get("CLOUD_MODEL_NAME", os.environ.get("UPSTREAM_MODEL", default))


# ── Context compression ──────────────────────────────────────────────────────


def _compress_context(context: str) -> str:
    """Compress context using RTK-style token compressor (pass-through message compression).

    Skips re-compression if context was already compressed by the gateway's
    assemble_context (marked with gateway compression header). This ensures
    gateway-assembled context (CLI, MCP) is NOT double-compressed by the proxy.
    Arbitrary message content from agents (not from our assembler) is still
    compressed for token efficiency.
    """
    # Skip if already compressed by gateway's assemble_context
    if "[already-compressed-by-gateway]" in context[:100]:
        logger.debug("Skipping re-compression of gateway-assembled context")
        return context

    try:
        from server.token_compressor import FilterLevel, Language, TokenCompressor

        level_str = os.environ.get("COMPRESSION_LEVEL", "minimal")
        try:
            level = FilterLevel(level_str)
        except ValueError:
            level = FilterLevel.MINIMAL

        compressor = TokenCompressor(level=level)
        result = compressor.compress(context, Language.PYTHON)
        stats = compressor.get_stats()
        logger.info(
            "Proxy pass-through compression: %s → %s tokens (%.1f%%)",
            stats["original_tokens"],
            stats["compressed_tokens"],
            stats["reduction_pct"],
        )
        return result
    except Exception:
        return context


# ── Upstream calls ───────────────────────────────────────────────────────────


async def call_upstream(request: CanonicalRequest) -> CanonicalResponse:
    """Non-streaming upstream call. Supports local enrich mode with compression."""
    if not is_cloud_configured():
        if is_local_enrich_enabled():
            return await _local_enrich_and_hint(request)
        return _local_noop_response(request)

    provider = _detect_provider()
    if provider == "anthropic":
        return await _call_anthropic(request)
    return await _call_openai_compatible(request)


async def stream_upstream(
    request: CanonicalRequest,
) -> AsyncGenerator[StreamEvent, None]:
    """Streaming upstream call. Yields canonical StreamEvent instances."""
    if not is_cloud_configured():
        async for event in _local_stream_response(request):
            yield event
        return

    provider = _detect_provider()
    if provider == "anthropic":
        async for event in _stream_anthropic(request):
            yield event
        return
    async for event in _stream_openai_compatible(request):
        yield event


async def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in text using the upstream provider."""
    if not is_cloud_configured():
        return len(text) // 4

    provider = _detect_provider()
    if provider == "anthropic":
        return await _count_tokens_anthropic(text, model)
    return len(text) // 4


async def count_tokens_messages(
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[dict[str, Any]] | None,
    model: str | None = None,
) -> int:
    """Count tokens in messages (Anthropic format)."""
    if not is_cloud_configured():
        total = len(json.dumps(messages) if messages else "")
        if system:
            total += len(system)
        return total // 4

    provider = _detect_provider()
    if provider == "anthropic":
        return await _count_tokens_anthropic_messages(messages, system, tools, model)
    total = len(json.dumps(messages) if messages else "")
    if system:
        total += len(system)
    return total // 4


# ── Provider detection ───────────────────────────────────────────────────────


def _detect_provider() -> str:
    base = _openai_compatible_base()
    if "anthropic.com" in base:
        return "anthropic"
    return "openai"


# ── OpenAI-compatible upstream ────────────────────────────────────────────────


def _canonical_to_openai_payload(request: CanonicalRequest) -> dict[str, Any]:
    model = request.model or _model_name()
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for msg in request.messages:
        messages.append(_canonical_message_to_openai(msg))

    payload: dict[str, Any] = {"model": model, "messages": messages}

    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop"] = request.stop

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

    return payload


def _canonical_message_to_openai(msg) -> dict[str, Any]:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for b in msg.content:
        if b.type == "text" and b.text:
            text_parts.append(b.text)
        elif b.type == "tool_use":
            import json as _json

            tool_calls.append(
                {
                    "id": b.tool_use_id or "",
                    "type": "function",
                    "function": {
                        "name": b.tool_name or "",
                        "arguments": _json.dumps(b.tool_input or {}, ensure_ascii=False),
                    },
                }
            )
        elif b.type == "tool_result":
            text_parts.append(b.tool_result_content or "")

    result: dict[str, Any] = {"role": msg.role}
    if msg.role == "tool":
        result["content"] = "\n".join(text_parts)
        result["tool_call_id"] = msg.content[0].tool_use_id if msg.content else ""
    elif tool_calls:
        result["content"] = "\n".join(text_parts) if text_parts else None
        result["tool_calls"] = tool_calls
    else:
        result["content"] = "\n".join(text_parts) if text_parts else ""

    return result


async def _call_openai_compatible(request: CanonicalRequest) -> CanonicalResponse:
    payload = _canonical_to_openai_payload(request)

    # Compress system messages (RTK-style) before sending to cloud
    if "messages" in payload:
        for msg in payload["messages"]:
            if msg.get("role") == "system" and msg.get("content"):
                msg["content"] = _compress_context(msg["content"])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key()}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{_openai_compatible_base()}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content: list[ContentBlock] = []

    if msg.get("content"):
        content.append(ContentBlock(type="text", text=msg["content"]))

    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        import json as _json

        try:
            tool_input = _json.loads(fn.get("arguments", "{}"))
        except (_json.JSONDecodeError, TypeError):
            tool_input = {}
        content.append(
            ContentBlock(
                type="tool_use",
                tool_use_id=tc.get("id", ""),
                tool_name=fn.get("name", ""),
                tool_input=tool_input if isinstance(tool_input, dict) else {},
            )
        )

    usage = data.get("usage", {})
    return CanonicalResponse(
        id=data.get("id", ""),
        model=data.get("model", request.model),
        content=content,
        stop_reason=choice.get("finish_reason"),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )


async def _stream_openai_compatible(
    request: CanonicalRequest,
) -> AsyncGenerator[StreamEvent, None]:
    payload = _canonical_to_openai_payload(request)

    # Compress system messages (RTK-style) before streaming to cloud
    if "messages" in payload:
        for msg in payload["messages"]:
            if msg.get("role") == "system" and msg.get("content"):
                msg["content"] = _compress_context(msg["content"])

    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key()}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{_openai_compatible_base()}/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                event = openai_chunk_to_event(chunk)
                if event is not None:
                    yield event


# ── Anthropic upstream ───────────────────────────────────────────────────────


def _canonical_to_anthropic_payload(request: CanonicalRequest) -> dict[str, Any]:

    model = request.model or _model_name("claude-sonnet-4-20250514")
    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    for msg in request.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        system_parts.append(block.text)
        else:
            messages.append(_canonical_message_to_anthropic(msg))

    system_text = ""
    if system_parts or request.system:
        combined = system_parts.copy()
        if request.system:
            combined.append(request.system)
        system_text = "\n\n".join(combined)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
    }

    if system_text:
        payload["system"] = system_text
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop_sequences"] = request.stop

    if request.tools:
        payload["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in request.tools
        ]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

    return payload


def _canonical_message_to_anthropic(msg) -> dict[str, Any]:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}

    content: list[dict[str, Any]] = []
    for b in msg.content:
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
        elif b.type == "tool_result":
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id or "",
                    "content": b.tool_result_content or "",
                    "is_error": b.tool_result_is_error,
                }
            )

    return {"role": msg.role, "content": content}


async def _call_anthropic(request: CanonicalRequest) -> CanonicalResponse:
    payload = _canonical_to_anthropic_payload(request)

    # Compress system message (RTK-style) before sending to cloud
    if payload.get("system"):
        payload["system"] = _compress_context(str(payload["system"]))

    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "anthropic-version": "2023-06-01",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    content: list[ContentBlock] = []
    for part in data.get("content", []):
        if part.get("type") == "text":
            content.append(ContentBlock(type="text", text=part.get("text", "")))
        elif part.get("type") == "tool_use":
            content.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=part.get("id", ""),
                    tool_name=part.get("name", ""),
                    tool_input=part.get("input", {}),
                )
            )

    usage = data.get("usage", {})
    return CanonicalResponse(
        id=data.get("id", ""),
        model=data.get("model", request.model),
        content=content,
        stop_reason=data.get("stop_reason"),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


async def _stream_anthropic(
    request: CanonicalRequest,
) -> AsyncGenerator[StreamEvent, None]:
    payload = _canonical_to_anthropic_payload(request)

    # Compress system message (RTK-style) before streaming to cloud
    if payload.get("system"):
        payload["system"] = _compress_context(str(payload["system"]))

    payload["stream"] = True

    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "anthropic-version": "2023-06-01",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for line in resp.aiter_lines():
                if not line:
                    if buffer:
                        try:
                            data = json.loads(buffer.removeprefix("data: "))
                        except json.JSONDecodeError:
                            buffer = ""
                            continue
                        event = anthropic_chunk_to_event(data)
                        if event is not None:
                            yield event
                        buffer = ""
                    continue
                if line.startswith("event: "):
                    if buffer:
                        try:
                            data = json.loads(buffer.removeprefix("data: "))
                        except json.JSONDecodeError:
                            buffer = ""
                            continue
                        event = anthropic_chunk_to_event(data)
                        if event is not None:
                            yield event
                        buffer = ""
                if line.startswith("data: "):
                    buffer = line


async def _count_tokens_anthropic(text: str, model: str | None = None) -> int:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "anthropic-version": "2023-06-01",
    }
    payload: dict[str, Any] = {
        "model": model or _model_name("claude-sonnet-4-20250514"),
        "messages": [{"role": "user", "content": text}],
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages/count_tokens",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json().get("input_tokens", len(text) // 4)


async def _count_tokens_anthropic_messages(
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[dict[str, Any]] | None,
    model: str | None = None,
) -> int:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "anthropic-version": "2023-06-01",
    }
    payload: dict[str, Any] = {
        "model": model or _model_name("claude-sonnet-4-20250514"),
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages/count_tokens",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json().get("input_tokens", 0)


# ── Local enrichment + compression + hint ────────────────────────────────────


async def _local_enrich_and_hint(request: CanonicalRequest) -> CanonicalResponse:
    """Enrich with context, compress (RTK-style), ask Ollama for validation/hint.

    Returns enriched + compressed + validated response for OpenCode to use.
    Tracks compression metrics for analytics.
    """
    import time as _time

    _start = _time.perf_counter()

    # 1. Assemble context from DB/cache/memory
    from server.context_assembler import ContextAssembler

    assembler = ContextAssembler(top_k=3)
    user_query = request.user_message_text() or ""
    context = assembler.assemble_context(user_query)

    # 2. Compress context (RTK-style) — skip if already compressed by assembler
    from server.token_compressor import FilterLevel, Language, TokenCompressor

    # Check if context is already compressed (marked by assemble_context)
    if "[already-compressed-by-gateway]" in context[:100]:
        # Already compressed, just use as-is and estimate stats
        compressed_context = context
        original_len = len(context) * 2  # Rough estimate
        compressed_len = len(context)
        stats: dict[str, Any] = {
            "original_tokens": max(1, original_len // 4),
            "compressed_tokens": max(1, compressed_len // 4),
            "reduction_pct": 0.0,  # Already compressed
            "strategies_applied": ["pre-compressed-by-assembler"],
        }
        logger.debug("Context already compressed by assembler, skipping re-compression")
    else:
        # Fresh context, compress it
        level_str = os.environ.get("COMPRESSION_LEVEL", "minimal")
        try:
            level = FilterLevel(level_str)
        except ValueError:
            level = FilterLevel.MINIMAL

        compressor = TokenCompressor(level=level)
        compressed_context = compressor.compress(context, Language.PYTHON)
        stats = compressor.get_stats()

    logger.info(
        "Compression: %s → %s tokens (%.1f%%) [%s]",
        stats["original_tokens"],
        stats["compressed_tokens"],
        stats["reduction_pct"],
        ", ".join(str(s) for s in stats["strategies_applied"]),
    )

    # 3. Track metrics
    exec_time = int((_time.perf_counter() - _start) * 1000)
    try:
        from server.token_tracking import TokenTracker

        tracker = TokenTracker()
        tracker.track(
            query=user_query,
            original_tokens=int(stats["original_tokens"]),
            compressed_tokens=int(stats["compressed_tokens"]),
            strategies=list(stats["strategies_applied"]),
            exec_time_ms=exec_time,
        )
    except Exception as e:
        logger.debug("Token tracking error: %s", e)

    # 4. Ask Ollama for validation/hint (opt-in via LOCAL_HINT_MODE)
    hint_section = ""
    if os.environ.get("LOCAL_HINT_MODE", "").lower() == "true":
        try:
            from server.ollama_client import OllamaClient

            ollama = OllamaClient()
            if ollama.health_check():
                validation_model = os.environ.get("OLLAMA_VALIDATION_MODEL", "qwen2.5-coder:1.5b")
                hint_prompt = (
                    f"Given this codebase context and question, provide:\n"
                    f"1. Validation: Is this question answerable with the given context? (yes/no)\n"
                    f"2. Hint: Brief hint (1-2 sentences) about how to answer\n"
                    f"3. Confidence: 0.0-1.0\n\n"
                    f"Context:\n{compressed_context[:3000]}\n\n"
                    f"Question:\n{user_query}\n\n"
                    f"Response format:\n"
                    f"Validation: yes/no\n"
                    f"Hint: [hint]\n"
                    f"Confidence: [0.0-1.0]"
                )
                hint_response = ollama.generate(hint_prompt, model=validation_model)

                hint_section = f"""## Gateway Validation & Hint

{hint_response}
"""
                logger.info("Ollama validation completed")
        except Exception as e:
            logger.debug("Ollama validation skipped: %s", e)

    # 5. Return enriched + compressed + validated response
    enriched_content = f"""# Codebase Context (Compressed)

{compressed_context}

---

{hint_section}## Compression Stats
- Original: {stats['original_tokens']} tokens
- Compressed: {stats['compressed_tokens']} tokens
- Reduction: {stats['reduction_pct']}%
- Strategies: {', '.join(str(s) for s in stats['strategies_applied'])}
"""

    return CanonicalResponse(
        id="gateway-enriched-" + str(int(_time.time() * 1000)),
        model="gateway-enriched",
        content=[ContentBlock(type="text", text=enriched_content)],
        stop_reason="stop",
        input_tokens=int(stats["compressed_tokens"]),
        output_tokens=len(hint_section) // 4,
    )


# ── Local-only (no cloud key) ────────────────────────────────────────────────


def _local_noop_response(request: CanonicalRequest) -> CanonicalResponse:
    user_text = request.user_message_text() or ""
    ctx_parts: list[str] = []
    for msg in request.messages:
        if msg.role == "system":
            ctx_parts.append(msg.content_as_text())
    return CanonicalResponse(
        id="local-" + str(int(time.time() * 1000)),
        model=request.model or "local",
        content=[
            ContentBlock(
                type="text",
                text="\n\n".join(ctx_parts) if ctx_parts else user_text,
            )
        ],
        stop_reason="stop",
        input_tokens=len(user_text) // 4,
        output_tokens=0,
    )


async def _local_stream_response(
    request: CanonicalRequest,
) -> AsyncGenerator[StreamEvent, None]:
    context = "\n\n".join(msg.content_as_text() for msg in request.messages if msg.role == "system")
    if not context:
        context = request.user_message_text() or ""

    msg_id = "local-" + str(int(time.time() * 1000))

    from server.canonical import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStartEvent,
        MessageStopEvent,
    )

    yield MessageStartEvent(message_id=msg_id, model=request.model or "local")
    yield ContentBlockStartEvent(index=0, content_block_type="text")
    yield ContentBlockDeltaEvent(index=0, delta_type="text_delta", text=context)
    yield ContentBlockStopEvent(index=0)
    yield MessageDeltaEvent(stop_reason="stop")
    yield MessageStopEvent()
