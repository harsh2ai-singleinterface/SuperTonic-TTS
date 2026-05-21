"""End-to-end TTFB smoke test, with latency decomposition.

Connects to a running supertonic_server, sends a Cartesia-shaped text
frame, and measures the wall-clock between (a) the moment we shipped the
first text frame and (b) the moment the first ``chunk`` frame came back.

It also breaks the latency down so a high TTFB can be attributed:

  * ``connect_ms``   — opening handshake (TCP + TLS + WS upgrade). Paid
                       once per connection; a real voice agent keeps the
                       socket open, so this should NOT count per turn.
  * ``ws_rtt_ms``    — application-level round-trip over the *established*
                       WebSocket (ping/pong). This is the true network
                       cost of one request/response and should be close
                       to ICMP ping. If it is much higher, something in
                       the data path (proxy buffering, Nagle/delayed-ACK)
                       is adding round-trips.
  * ``ttfb_ms``      — request frame sent -> first ``chunk`` received.
  * ``server_est_ms``— ttfb_ms - ws_rtt_ms: an estimate of pure server
                       processing. Compare against the server log's own
                       TTFB; a large mismatch points at the data path.

Run from a separate process while the server is up:

    # term 1:
    PYTHONPATH=src python -m supertonic_server.server

    # term 2 (after ~5 s for warmup):
    SUPERTONIC_URL=ws://<host>:7799/tts/websocket python scripts/smoke_ttfb.py
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
N_PING = int(os.environ.get("SUPERTONIC_PING_SAMPLES", "8"))

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


async def _measure_ws_rtt(ws: Any) -> list[float]:
    """Round-trip times (ms) of WebSocket ping/pong over the live socket.

    This is the network cost the data path actually sees — unlike ICMP, it
    goes through the exact same TCP connection, TLS session, and any L7
    proxy that real traffic does.
    """
    samples: list[float] = []
    for _ in range(N_PING):
        t0 = time.perf_counter()
        pong_waiter = await ws.ping()
        await pong_waiter
        samples.append((time.perf_counter() - t0) * 1000.0)
        await asyncio.sleep(0.05)
    return samples


async def run() -> dict[str, Any]:
    ctx_id = f"smoke-{uuid.uuid4().hex[:8]}"
    print(f"connecting to {URL} (voice={VOICE}, ctx={ctx_id})")

    # --- handshake cost (TCP + TLS + WS upgrade) -------------------------
    t_pre_connect = time.perf_counter()
    ws = await websockets.connect(URL, max_size=None)
    connect_ms = (time.perf_counter() - t_pre_connect) * 1000.0

    try:
        # --- application-level RTT over the established socket -----------
        ping_samples = await _measure_ws_rtt(ws)
        ws_rtt_min = min(ping_samples)
        ws_rtt_avg = sum(ping_samples) / len(ping_samples)

        # --- request -> first chunk --------------------------------------
        first_text_at = time.perf_counter()
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
    finally:
        await ws.close()

    if first_chunk_at is None:
        raise RuntimeError("never received a chunk frame")

    ttfb_ms = (first_chunk_at - first_text_at) * 1000.0
    total_audio_s = total_bytes / 2 / SAMPLE_RATE  # 2 bytes per int16 sample
    wall_ms = (end_at - first_text_at) * 1000.0
    rtf = (wall_ms / 1000.0) / total_audio_s if total_audio_s else float("inf")

    # ttfb minus one network round-trip ~= server-side processing, IF the
    # data path adds no extra round-trips beyond ws_rtt.
    server_est_ms = ttfb_ms - ws_rtt_min

    return {
        "voice": VOICE,
        "text_chars": len(TEXT),
        "chunks": chunks,
        "total_pcm_bytes": total_bytes,
        "audio_seconds": round(total_audio_s, 3),
        "connect_ms": round(connect_ms, 1),
        "ws_rtt_ms": {
            "min": round(ws_rtt_min, 1),
            "avg": round(ws_rtt_avg, 1),
            "samples": [round(s, 1) for s in ping_samples],
        },
        "ttfb_ms": round(ttfb_ms, 1),
        "server_est_ms": round(server_est_ms, 1),
        "wall_total_ms": round(wall_ms, 1),
        "rtf": round(rtf, 4),
    }


def main() -> None:
    report = asyncio.run(run())
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # --- plain-language interpretation -----------------------------------
    ttfb = report["ttfb_ms"]
    rtt = report["ws_rtt_ms"]["min"]
    server_est = report["server_est_ms"]
    print("\n--- interpretation ---")
    print(f"  network round-trip (WS ping/pong):  {rtt:.0f} ms")
    print(f"  request -> first audio (TTFB):      {ttfb:.0f} ms")
    print(f"  TTFB minus one round-trip:          {server_est:.0f} ms")
    print(f"  handshake (one-time, per connect):  {report['connect_ms']:.0f} ms")
    print(
        "  -> 'TTFB minus one round-trip' should be close to the server\n"
        "     log's own TTFB. If it is much larger, the data path is\n"
        "     adding extra round-trips (proxy buffering / Nagle), not the\n"
        "     model. If it matches, the TTFB is just network distance."
    )


if __name__ == "__main__":
    main()
