"""FastAPI server: WebSocket TTS endpoint + REST voice catalog.

Endpoints
---------

* ``GET  /voices``                     — list available voice IDs.
* ``GET  /health``                     — process alive (always 200).
* ``GET  /ready``                      — 200 only after the worker pool has
                                         finished cold-start + warmup.
* ``WS   /tts/websocket``              — Cartesia-shaped streaming TTS.

The WebSocket path mirrors Cartesia's so the Pipecat
:class:`pipecat.services.cartesia.tts.CartesiaTTSService` is a drop-in
client (point its URL at us).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

from . import wire
from .pool import WorkerPool
from .session import Session
from .voices import list_voices

logger = logging.getLogger("supertonic_server.server")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-int %s=%r, using default %d", key, raw, default)
        return default


N_WORKERS = _env_int("SUPERTONIC_WORKERS", 2)
TOTAL_STEPS = _env_int("SUPERTONIC_STEPS", 8)
# Inference device: "cuda" (GPU, CPU fallback), "cpu", or "auto". Default is
# "cuda" — the worker logs an error and falls back to CPU if no GPU is usable.
DEVICE = os.environ.get("SUPERTONIC_DEVICE", "cuda").strip() or "cuda"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bring the worker pool up before serving traffic; tear down on stop."""
    logging.basicConfig(
        level=os.environ.get("SUPERTONIC_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    pool = WorkerPool(n_workers=N_WORKERS, total_steps=TOTAL_STEPS, device=DEVICE)
    logger.info(
        "starting pool: workers=%d steps=%d device=%s",
        N_WORKERS, TOTAL_STEPS, DEVICE,
    )
    await pool.start()
    app.state.pool = pool
    try:
        yield
    finally:
        logger.info("shutting pool down")
        await pool.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Supertonic Streaming TTS",
        version="0.0.1",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/ready")
    async def ready() -> JSONResponse:
        pool: WorkerPool = app.state.pool
        if pool.is_ready:
            return JSONResponse({"ready": True, "workers": pool.n_workers})
        return JSONResponse({"ready": False}, status_code=503)

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse({
            "voices": [
                {"name": v.name, "gender": v.gender}
                for v in list_voices()
            ]
        })

    @app.websocket("/tts/websocket")
    async def tts_ws(ws: WebSocket) -> None:
        await ws.accept()
        pool: WorkerPool = app.state.pool

        async def send(text: str) -> None:
            await ws.send_text(text)

        session = Session(pool=pool, send_text=send)
        try:
            while True:
                raw = await ws.receive_text()
                msg = wire.decode(raw)
                await session.handle(msg)
        except WebSocketDisconnect:
            logger.info("client disconnected")
        except Exception:
            logger.exception("ws handler crashed")
        finally:
            await session.close()

    return app


# Module-level instance for `uvicorn supertonic_server.server:app`.
app = create_app()


def main() -> None:
    """Entry point: ``python -m supertonic_server.server``."""
    import uvicorn

    host = os.environ.get("SUPERTONIC_HOST", "127.0.0.1")
    port = _env_int("SUPERTONIC_PORT", 7799)
    uvicorn.run(
        "supertonic_server.server:app",
        host=host,
        port=port,
        log_config=None,  # we configure root logging in _lifespan
    )


if __name__ == "__main__":
    main()
