"""Per-WebSocket session state.

A session multiplexes any number of *contexts* (Cartesia-speak: a
``context_id`` is the bot's name for one bot turn). For each context we
maintain a :class:`TextCoalescer` and an emit task that ships PCM frames
back to the client as they're synthesized.

Lifecycle of a context within one WS session:

    1. First :class:`IncomingText` with a new ``context_id`` arrives —
       the session lazily creates a :class:`_Context`.
    2. Each subsequent text frame pushes into the coalescer; any chunks
       that pop out are queued for synthesis on the pool.
    3. A flush (empty transcript, ``continue=False``) or the end of the
       LLM stream drains the coalescer and emits a ``done`` frame once
       all queued chunks have been synthesized and shipped.
    4. A cancel frame marks the context cancelled; in-flight syntheses
       discard their result, queued chunks are dropped, no further
       audio is shipped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import audio as audio_mod
from . import wire
from .coalescer import TextCoalescer
from .pool import WorkerPool
from .worker import SynthResult

logger = logging.getLogger("supertonic_server.session")

# Type of the sender callable: takes a serialized JSON frame, ships it.
Sender = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class _Context:
    context_id: str
    voice_id: str
    output_sr: int
    coalescer: TextCoalescer
    # FIFO of chunks waiting to be synthesized + shipped. The emit task
    # awaits this; the receive loop pushes into it.
    queue: asyncio.Queue[str | None]  # `None` is the EOF sentinel
    emit_task: asyncio.Task[None] | None = None
    # First-byte tracking: wall-clock when the first text frame for this
    # context arrived, vs when the first PCM frame was shipped.
    first_text_at: float = 0.0
    first_audio_at: float = 0.0
    finished: bool = False


class Session:
    """One client WebSocket. Holds state for every concurrent context the
    client opens on this connection."""

    def __init__(self, pool: WorkerPool, send_text: Sender) -> None:
        self._pool = pool
        self._send_text = send_text
        self._contexts: dict[str, _Context] = {}
        self._closed = False

    async def close(self) -> None:
        """Cancel everything and wait for emit tasks to finish."""
        self._closed = True
        for ctx in list(self._contexts.values()):
            self._pool.cancel(ctx.context_id)
            await ctx.queue.put(None)
            if ctx.emit_task is not None:
                try:
                    await asyncio.wait_for(ctx.emit_task, timeout=2.0)
                except asyncio.TimeoutError:
                    ctx.emit_task.cancel()
        self._contexts.clear()

    async def handle(self, msg: wire.IncomingMessage) -> None:
        """Dispatch one decoded incoming frame."""
        if self._closed:
            return

        if isinstance(msg, wire.ParseError):
            logger.warning("parse error: %s | raw=%s", msg.reason, msg.raw)
            await self._send_text(
                wire.encode_error(context_id=None, message=msg.reason)
            )
            return

        if isinstance(msg, wire.IncomingCancel):
            await self._cancel(msg.context_id)
            return

        if isinstance(msg, wire.IncomingText):
            await self._on_text(msg)
            return

        # Should be exhaustive; defensive log.
        logger.error("unhandled incoming message type: %r", type(msg).__name__)

    async def _on_text(self, msg: wire.IncomingText) -> None:
        ctx = self._contexts.get(msg.context_id)
        if ctx is None:
            ctx = self._make_context(msg)
            self._contexts[msg.context_id] = ctx
        else:
            # Sanity: voice/format must not change mid-context.
            if ctx.voice_id != msg.voice.id:
                logger.warning(
                    "voice changed mid-context %s: %s -> %s (ignored)",
                    msg.context_id, ctx.voice_id, msg.voice.id,
                )
            if ctx.output_sr != msg.output_format.sample_rate:
                logger.warning(
                    "sample rate changed mid-context %s: %d -> %d (ignored)",
                    msg.context_id, ctx.output_sr, msg.output_format.sample_rate,
                )

        if ctx.first_text_at == 0.0:
            ctx.first_text_at = time.perf_counter()

        if msg.is_flush:
            # Drain coalescer, then signal EOF to the emit task.
            for chunk in ctx.coalescer.flush():
                await ctx.queue.put(chunk)
            await ctx.queue.put(None)
            return

        # Normal incremental text: push to coalescer, queue any popped chunks.
        for chunk in ctx.coalescer.push(msg.transcript):
            await ctx.queue.put(chunk)

        if not msg.continue_:
            # `continue=False` with non-empty transcript: this is the last
            # text frame for the context. Drain + EOF.
            for chunk in ctx.coalescer.flush():
                await ctx.queue.put(chunk)
            await ctx.queue.put(None)

    async def _cancel(self, context_id: str) -> None:
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return
        self._pool.cancel(context_id)
        # Drain the queue and inject EOF so the emit task wakes up and exits.
        while not ctx.queue.empty():
            try:
                ctx.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await ctx.queue.put(None)

    def _make_context(self, first_msg: wire.IncomingText) -> _Context:
        ctx = _Context(
            context_id=first_msg.context_id,
            voice_id=first_msg.voice.id,
            output_sr=first_msg.output_format.sample_rate,
            coalescer=TextCoalescer(),
            queue=asyncio.Queue(),
        )
        ctx.emit_task = asyncio.create_task(self._emit_loop(ctx))
        return ctx

    async def _emit_loop(self, ctx: _Context) -> None:
        """For one context: pop chunks off the queue, synthesize, ship
        the resulting PCM frames over the WebSocket, then send ``done``."""
        try:
            while True:
                chunk = await ctx.queue.get()
                if chunk is None:
                    break

                if self._pool.is_cancelled(ctx.context_id):
                    # Drop without synthesizing.
                    continue

                result = await self._pool.synthesize(
                    ctx.context_id, chunk, ctx.voice_id
                )
                if result is None:
                    # Cancelled while in flight.
                    continue

                await self._ship(ctx, result)

            if not self._pool.is_cancelled(ctx.context_id):
                await self._send_text(wire.encode_done(ctx.context_id))
        except Exception as exc:  # noqa: BLE001
            logger.exception("emit loop crashed for ctx %s", ctx.context_id)
            try:
                await self._send_text(
                    wire.encode_error(ctx.context_id, f"server: {exc}")
                )
            except Exception:
                pass
        finally:
            ctx.finished = True
            self._pool.forget(ctx.context_id)
            self._log_context_summary(ctx)

    async def _ship(self, ctx: _Context, result: SynthResult) -> None:
        """Convert one synth result into PCM frames and push them to the
        client. Sets ``first_audio_at`` on the first frame."""
        frames = audio_mod.pipeline(
            result.wav_f32,
            src_sr=result.sample_rate,
            dst_sr=ctx.output_sr,
        )
        for frame in frames:
            if self._pool.is_cancelled(ctx.context_id):
                return
            await self._send_text(wire.encode_chunk(ctx.context_id, frame))
            if ctx.first_audio_at == 0.0:
                ctx.first_audio_at = time.perf_counter()

    def _log_context_summary(self, ctx: _Context) -> None:
        if ctx.first_audio_at == 0.0 or ctx.first_text_at == 0.0:
            return
        ttfb_ms = (ctx.first_audio_at - ctx.first_text_at) * 1000.0
        logger.info(
            "ctx %s done: TTFB=%.0fms cancelled=%s",
            ctx.context_id, ttfb_ms, self._pool.is_cancelled(ctx.context_id),
        )

    # ----- introspection (for /metrics later) ----------------------------

    def context_stats(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ctx in self._contexts.values():
            ttfb_ms = (
                (ctx.first_audio_at - ctx.first_text_at) * 1000.0
                if ctx.first_audio_at and ctx.first_text_at
                else None
            )
            out.append({
                "context_id": ctx.context_id,
                "voice_id": ctx.voice_id,
                "ttfb_ms": ttfb_ms,
                "pending_chars": ctx.coalescer.pending_chars,
                "queue_depth": ctx.queue.qsize(),
                "finished": ctx.finished,
            })
        return out
