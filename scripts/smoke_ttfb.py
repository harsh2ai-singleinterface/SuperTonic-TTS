"""End-to-end TTFB smoke test.

Connects to a running supertonic_server, sends a Cartesia-shaped text
frame, and measures the wall-clock between (a) the moment we shipped the
first text frame and (b) the moment the first ``chunk`` frame came back.

Run from a separate process while the server is up:

    # term 1:
    /Users/harsh/Desktop/audio_test/.venv/bin/python -m supertonic_server.server

    # term 2 (after ~5 s for warmup):
    /Users/harsh/Desktop/audio_test/.venv/bin/python scripts/smoke_ttfb.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from typing import Any

import websockets

URL = os.environ.get("SUPERTONIC_URL", "ws://127.0.0.1:7799/tts/websocket")
VOICE = os.environ.get("SUPERTONIC_VOICE", "F1")
SAMPLE_RATE = 24000

# 15-word Hindi sentence — same character as the IIFL prod bot would say.
TEXT = (
    "नमस्ते, मैं प्रिया बोल रही हूँ IIFL फाइनेंस से। बताइए, "
    "आपको गोल्ड लोन की जानकारी चाहिए?"
)


def _text_frame(ctx_id: str, text: str, continue_: bool) -> str:
    return json.dumps({
        "transcript": text,
        "continue": continue_,
        "context_id": ctx_id,
        "model_id": "sonic-3",
        "voice": {"mode": "id", "id": VOICE},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        },
        "add_timestamps": False,
    })


async def run() -> dict[str, Any]:
    ctx_id = f"smoke-{uuid.uuid4().hex[:8]}"
    print(f"connecting to {URL} (voice={VOICE}, ctx={ctx_id})")
    async with websockets.connect(URL, max_size=None) as ws:
        first_text_at = time.perf_counter()
        # Send the whole utterance in one frame, terminating (continue=False).
        await ws.send(_text_frame(ctx_id, TEXT, continue_=False))

        first_chunk_at: float | None = None
        chunks = 0
        total_bytes = 0
        done = False
        timeout_s = 30.0
        deadline = time.perf_counter() + timeout_s

        while not done:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"no `done` within {timeout_s} s")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "chunk":
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                pcm = base64.b64decode(msg["data"])
                chunks += 1
                total_bytes += len(pcm)
            elif kind == "done":
                done = True
            elif kind == "error":
                raise RuntimeError(f"server error: {msg}")
            else:
                print(f"  ?unhandled frame: {msg}")

        end_at = time.perf_counter()

    if first_chunk_at is None:
        raise RuntimeError("never received a chunk frame")

    ttfb_ms = (first_chunk_at - first_text_at) * 1000.0
    total_audio_s = total_bytes / 2 / SAMPLE_RATE  # 2 bytes per int16 sample
    wall_ms = (end_at - first_text_at) * 1000.0
    rtf = (wall_ms / 1000.0) / total_audio_s if total_audio_s else float("inf")

    return {
        "voice": VOICE,
        "text_chars": len(TEXT),
        "chunks": chunks,
        "total_pcm_bytes": total_bytes,
        "audio_seconds": round(total_audio_s, 3),
        "ttfb_ms": round(ttfb_ms, 1),
        "wall_total_ms": round(wall_ms, 1),
        "rtf": round(rtf, 4),
    }


def main() -> None:
    report = asyncio.run(run())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
