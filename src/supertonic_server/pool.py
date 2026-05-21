"""Synthesis-worker pool with per-context cancellation.

Wraps one or more :class:`SupertonicWorker` instances behind an asyncio API
so :mod:`session` can ``await pool.synthesize(...)`` without blocking the
event loop. ONNX Runtime inference releases the GIL in chunks, so a
``ThreadPoolExecutor`` is sufficient — no need for process isolation in P0.

Cancellation is cooperative: callers pass a context_id, and the pool tracks
a per-context "cancelled" flag. Each chunk's synthesize() runs to completion
(supertonic's ``synthesize()`` is not preemptible), but any further chunks
queued for a cancelled context are dropped.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .worker import SupertonicWorker, SynthResult

logger = logging.getLogger("supertonic_server.pool")

_DEFAULT_WORKERS: Final[int] = 2


@dataclass(slots=True)
class _PoolState:
    cancelled: set[str]


class WorkerPool:
    """Asyncio facade over a thread pool of :class:`SupertonicWorker`s.

    Per-context cancellation: call :meth:`cancel(context_id)` and any future
    ``synthesize(context_id, ...)`` calls return ``None`` immediately. An
    in-flight call for that context still completes (supertonic's
    ``synthesize`` isn't preemptible), but its result is discarded.
    """

    def __init__(
        self,
        n_workers: int = _DEFAULT_WORKERS,
        cache_dir: Path | None = None,
        total_steps: int = 8,
        speed: float = 1.0,
        lang: str = "hi",
        device: str = "auto",
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be ≥ 1, got {n_workers}")
        self._n_workers = n_workers
        self._cache_dir = cache_dir
        self._total_steps = total_steps
        self._speed = speed
        self._lang = lang
        self._device = device

        self._workers: list[SupertonicWorker] = []
        self._executor: ThreadPoolExecutor | None = None
        self._state = _PoolState(cancelled=set())
        self._next_worker_idx = 0
        self._ready = False

    async def start(self) -> None:
        """Load all worker models and run warmup. Idempotent.

        Blocks the event loop for the duration of model load (~0.3 s per
        worker) and warmup (~0.5 s per worker) — call this from a startup
        hook, not from a request path.
        """
        if self._ready:
            return
        loop = asyncio.get_running_loop()

        def _build_and_warm() -> list[SupertonicWorker]:
            built: list[SupertonicWorker] = []
            for i in range(self._n_workers):
                logger.info("loading worker %d/%d", i + 1, self._n_workers)
                w = SupertonicWorker(
                    cache_dir=self._cache_dir,
                    total_steps=self._total_steps,
                    speed=self._speed,
                    lang=self._lang,
                    device=self._device,
                )
                w.warmup()
                built.append(w)
            return built

        # Load workers on a tiny private executor so we don't block when
        # the main executor is also us.
        bootstrap = ThreadPoolExecutor(max_workers=1, thread_name_prefix="st-boot")
        try:
            self._workers = await loop.run_in_executor(bootstrap, _build_and_warm)
        finally:
            bootstrap.shutdown(wait=True)

        self._executor = ThreadPoolExecutor(
            max_workers=self._n_workers, thread_name_prefix="st-synth"
        )
        self._ready = True
        logger.info("pool ready: %d workers warmed", self._n_workers)

    async def stop(self) -> None:
        """Shut down the executor. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def cancel(self, context_id: str) -> None:
        """Mark a context cancelled. All subsequent synthesize() calls for
        this context return None. Idempotent. Safe to call from any task."""
        self._state.cancelled.add(context_id)
        logger.debug("context cancelled: %s", context_id)

    def forget(self, context_id: str) -> None:
        """Drop bookkeeping for a finished context. Call when the session
        is done with this context_id so memory doesn't grow."""
        self._state.cancelled.discard(context_id)

    def is_cancelled(self, context_id: str) -> bool:
        return context_id in self._state.cancelled

    async def synthesize(
        self,
        context_id: str,
        text: str,
        voice_name: str,
    ) -> SynthResult | None:
        """Synthesize one chunk on a free worker. Returns None if the
        context has been cancelled (either before submission or while
        synthesis was in flight)."""
        if not self._ready or self._executor is None:
            raise RuntimeError("pool not started — call start() first")

        if self.is_cancelled(context_id):
            return None

        # Round-robin worker assignment. With ONNX-on-CPU each worker
        # is independent, so any unused worker will do.
        worker = self._workers[self._next_worker_idx % self._n_workers]
        self._next_worker_idx += 1

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, worker.synthesize, text, voice_name
        )

        # Re-check after the (possibly slow) synth completed — caller may
        # have cancelled while we were synthesizing.
        if self.is_cancelled(context_id):
            return None
        return result
