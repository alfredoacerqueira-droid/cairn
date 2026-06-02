"""FastAPI gateway — dual agent-facing API on a shared canonical core.

Supports:
  POST /v1/chat/completions   (OpenAI format, streaming + tool-calling)
  POST /v1/messages           (Anthropic format, streaming + tool-calling)
  POST /v1/messages/count_tokens  (token counting)
  GET  /health  /v1/models

Both endpoints preserve the FULL conversation (messages, system, tools,
tool_choice, stop params).  Assembled context is injected as an *additional*
system block — nothing is discarded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.version import CAIRN_VERSION
from server.canonical import CanonicalRequest, ContentBlock, Message, StreamEvent
from server.context_assembler import ContextAssembler
from server.streaming import emit_anthropic_sse, emit_openai_sse
from server.sync_engine import run_sync, should_sync
from server.translate import (
    anthropic_request_to_canonical,
    canonical_to_anthropic_response,
    canonical_to_openai_response,
    openai_request_to_canonical,
)
from server.upstream import (
    call_upstream,
    count_tokens_messages,
    set_request_api_key,
    stream_upstream,
)
from throttle.vram import VRAMPriority

logger = logging.getLogger(__name__)

vram = VRAMPriority()
_assembler: ContextAssembler | None = None
_PROJECT_PATH: Path | None = None

# Background sync state
_file_watcher = None
_sync_task: asyncio.Task | None = None
_last_periodic_check = 0.0
_SYNC_PERIODIC_INTERVAL = 60.0  # Seconds between periodic sync checks


def _ensure_index_exists(project_path: Path) -> None:
    """Auto-create index if missing or empty.

    Runs a one-time build on startup if the ChromaDB collection is not
    initialized. Wraps in try/except to log but not crash startup.
    """
    try:
        from core.config import load_config
        from core.repo import RepoManager, collect_source_files
        from pipeline.ast_parser import ASTParser
        from pipeline.indexer import VectorIndexer

        repo = RepoManager(project_path)
        indexer = VectorIndexer(chroma_path=repo.get_chroma_path())

        # Check if index is empty
        if indexer.count() > 0:
            logger.info("Index already populated (%d functions)", indexer.count())
            return

        logger.info("Index is empty, triggering auto-build...")
        cfg = load_config(project_path)
        parser = ASTParser()

        filtered_files = collect_source_files(
            project_path,
            cfg.indexing.file_patterns,
            cfg.indexing.exclude_patterns,
            getattr(cfg.indexing, "source_roots", ["."]),
        )

        if not filtered_files:
            logger.warning("No source files found to index")
            return

        total = 0
        for filepath in filtered_files:
            try:
                ast = parser.parse_file(filepath)
                indexer.index_ast(ast)
                total += len(ast.functions)
                for cls in ast.classes:
                    total += len(cls.methods)
            except Exception as e:
                logger.debug("Failed to index %s: %s", filepath, e)

        repo.write_index_meta()
        logger.info("Auto-indexed %d functions from %d files", total, len(filtered_files))

    except Exception as e:
        logger.error("Failed to auto-create index: %s", e)


async def _run_periodic_sync(project_path: Path) -> None:
    """Periodic background task that checks if sync is needed."""
    global _last_periodic_check
    while True:
        try:
            await asyncio.sleep(_SYNC_PERIODIC_INTERVAL)
            if should_sync(project_path, _last_periodic_check, _SYNC_PERIODIC_INTERVAL):
                _last_periodic_check = time.time()
                # Run sync in a thread to avoid blocking the event loop
                await asyncio.to_thread(run_sync, project_path, vram)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Periodic sync task failed: %s", e)


def _on_file_change(filepath: str) -> None:
    """Callback triggered by FileWatcher on code changes.

    Schedules run_sync via a background thread to avoid blocking the watcher.
    """

    def _sync_thread():
        try:
            run_sync(_PROJECT_PATH, vram)
        except Exception as e:
            logger.error("File watcher sync failed: %s", e)

    thread = threading.Thread(target=_sync_thread, daemon=True)
    thread.start()


def _start_background_sync(project_path: Path) -> None:
    """Start the FileWatcher and periodic sync task."""
    global _file_watcher, _sync_task

    try:
        # Start FileWatcher
        from core.config import load_config

        cfg = load_config(project_path)
        file_patterns = cfg.indexing.file_patterns
        exclude_patterns = cfg.indexing.exclude_patterns

        from pipeline.watcher import FileWatcher

        _file_watcher = FileWatcher(
            project_path=project_path,
            on_change=_on_file_change,
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            debounce_s=0.5,
        )
        _file_watcher.start()
        logger.info("FileWatcher started for %s", project_path)

    except Exception as e:
        logger.error("Failed to start FileWatcher: %s", e)

    # Start periodic sync task (asyncio)
    try:
        _sync_task = asyncio.create_task(_run_periodic_sync(project_path))
        logger.info("Periodic sync task started")
    except Exception as e:
        logger.error("Failed to start periodic sync task: %s", e)


async def _stop_background_sync() -> None:
    """Stop FileWatcher and periodic sync task cleanly."""
    global _file_watcher, _sync_task

    if _file_watcher is not None:
        try:
            _file_watcher.stop()
            logger.info("FileWatcher stopped")
        except Exception as e:
            logger.error("Failed to stop FileWatcher: %s", e)

    if _sync_task is not None:
        try:
            _sync_task.cancel()
            await _sync_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Failed to stop periodic sync task: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _assembler, _PROJECT_PATH
    # Fail-closed: strict project resolution without fallback to cwd.
    env_path = os.getenv("CAIRN_PROJECT") or os.getenv("GATEWAY_PROJECT")
    if not env_path:
        raise RuntimeError(
            "Gateway cannot start: CAIRN_PROJECT or GATEWAY_PROJECT not set. "
            "Set CAIRN_PROJECT to an indexed repo (a dir containing .cairn/)."
        )
    _PROJECT_PATH = Path(env_path).resolve()
    if not _PROJECT_PATH.exists():
        raise RuntimeError(f"Gateway cannot start: CAIRN_PROJECT path does not exist: {env_path}")
    cairn_dir = _PROJECT_PATH / ".cairn"
    if not cairn_dir.exists():
        raise RuntimeError(f"Gateway cannot start: no .cairn/ directory (not indexed): {env_path}")
    _assembler = ContextAssembler(project_path=_PROJECT_PATH)
    from core.repo import project_id

    pid = project_id(_PROJECT_PATH)
    logger.info("Semantic Gateway bound to %s (id=%s)", _PROJECT_PATH, pid)

    # Auto-create index if missing
    _ensure_index_exists(_PROJECT_PATH)

    # Start background sync (FileWatcher + periodic task)
    _start_background_sync(_PROJECT_PATH)

    yield
    _assembler = None
    _PROJECT_PATH = None

    # Stop background sync on shutdown
    await _stop_background_sync()


app = FastAPI(title="Cairn", version=CAIRN_VERSION, lifespan=lifespan)

# Freshness middleware removed: syncing is now done by background threads
# triggered by FileWatcher and periodic checks, not on every request.


# ── Endpoints ────────────────────────────────────────────────────────────


def get_assembler() -> ContextAssembler:
    global _assembler, _PROJECT_PATH
    if _assembler is None:
        if _PROJECT_PATH is None:
            _PROJECT_PATH = Path(
                os.getenv("CAIRN_PROJECT") or os.getenv("GATEWAY_PROJECT") or "."
            ).resolve()
        _assembler = ContextAssembler(project_path=_PROJECT_PATH)
    return _assembler


@app.get("/health")
async def health():
    global _PROJECT_PATH
    if _PROJECT_PATH is None:
        _PROJECT_PATH = Path(
            os.getenv("CAIRN_PROJECT") or os.getenv("GATEWAY_PROJECT") or "."
        ).resolve()
    return {
        "status": "ok",
        "version": CAIRN_VERSION,
        "project": str(_PROJECT_PATH),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "smart-context", "object": "model", "owned_by": "cairn"}],
    }


# ── /v1/chat/completions (OpenAI) ────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions with streaming + tool-calling."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    canonical = openai_request_to_canonical(body)
    stream = body.get("stream", False)

    # RTK-style: extract API key from request headers
    api_key = request.headers.get("X-Cloud-API-Key") or request.headers.get("api-key")
    set_request_api_key(api_key)

    vram.request("gateway")
    start_time = time.perf_counter()

    try:
        a = get_assembler()
        user_text = canonical.user_message_text()
        if user_text:
            context = a.assemble_context(user_text)
        else:
            context = a.assemble_context("Review and fix this code")

        enriched = _inject_context(canonical, context)

        if stream:

            async def sse_stream() -> AsyncGenerator[bytes, None]:
                async def event_gen() -> AsyncGenerator[StreamEvent, None]:
                    async for evt in stream_upstream(enriched):
                        yield evt

                async for chunk in emit_openai_sse(event_gen()):
                    yield chunk

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await call_upstream(enriched)
        openai_resp = canonical_to_openai_response(response)
        assembly_ms = int((time.perf_counter() - start_time) * 1000)
        _record_metrics(assembly_ms, error=False)
        openai_resp.setdefault("choices", [{}])[0].setdefault("metadata", {})[
            "assembly_ms"
        ] = assembly_ms
        return openai_resp

    except HTTPException:
        _record_metrics(int((time.perf_counter() - start_time) * 1000), error=True)
        raise
    except Exception as exc:
        _record_metrics(int((time.perf_counter() - start_time) * 1000), error=True)
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        vram.release("gateway")


# ── /v1/messages (Anthropic) ─────────────────────────────────────────────────


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    """Anthropic Messages API with streaming SSE + tool-calling."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    canonical = anthropic_request_to_canonical(body)
    stream = body.get("stream", False)

    # RTK-style: extract API key from request headers
    api_key = request.headers.get("X-Cloud-API-Key") or request.headers.get("api-key")
    set_request_api_key(api_key)

    vram.request("gateway")
    start_time = time.perf_counter()

    try:
        a = get_assembler()
        user_text = canonical.user_message_text()
        if user_text:
            context = a.assemble_context(user_text)
        else:
            context = a.assemble_context("Review and fix this code")

        enriched = _inject_context(canonical, context)

        if stream:

            async def sse_stream() -> AsyncGenerator[bytes, None]:
                async def event_gen() -> AsyncGenerator[StreamEvent, None]:
                    async for evt in stream_upstream(enriched):
                        yield evt

                async for chunk in emit_anthropic_sse(event_gen()):
                    yield chunk

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await call_upstream(enriched)
        anthropic_resp = canonical_to_anthropic_response(response)
        assembly_ms = int((time.perf_counter() - start_time) * 1000)
        _record_metrics(assembly_ms, error=False)
        return anthropic_resp

    except HTTPException:
        _record_metrics(int((time.perf_counter() - start_time) * 1000), error=True)
        raise
    except Exception as exc:
        _record_metrics(int((time.perf_counter() - start_time) * 1000), error=True)
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        vram.release("gateway")


# ── /v1/messages/count_tokens (Anthropic) ────────────────────────────────────


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request):
    """Count tokens in messages (Anthropic count_tokens API)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        input_tokens = await count_tokens_messages(
            messages=body.get("messages", []),
            system=body.get("system"),
            tools=body.get("tools"),
            model=body.get("model"),
        )
        return {"input_tokens": input_tokens}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _inject_context(req: CanonicalRequest, context: str) -> CanonicalRequest:
    """Return a new CanonicalRequest with context injected as an additional
    system block.  Preserves all existing messages, tools, and params."""
    context_messages: list[Message] = []
    context_messages.append(
        Message(role="system", content=[ContentBlock(type="text", text=context)])
    )

    new_system = None
    if req.system:
        context_messages.insert(
            0,
            Message(role="system", content=[ContentBlock(type="text", text=req.system)]),
        )
    else:
        for msg in req.messages:
            if msg.role == "system":
                context_messages.insert(0, msg)

    result_messages: list[Message] = []
    existing_systems_consumed = False
    for msg in req.messages:
        if msg.role == "system":
            if not existing_systems_consumed:
                result_messages.extend(context_messages)
                existing_systems_consumed = True
            continue
        if not existing_systems_consumed:
            result_messages.extend(context_messages)
            existing_systems_consumed = True
        result_messages.append(msg)

    if not existing_systems_consumed:
        result_messages.extend(context_messages)

    return CanonicalRequest(
        messages=result_messages,
        model=req.model,
        system=new_system,
        tools=req.tools,
        tool_choice=req.tool_choice,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        stream=req.stream,
    )


def _record_metrics(assembly_ms: int, error: bool = False) -> None:
    try:
        from core.metrics import Metrics

        m = Metrics()
        m.record_request(assembly_ms, error=error)
    except Exception:
        pass
