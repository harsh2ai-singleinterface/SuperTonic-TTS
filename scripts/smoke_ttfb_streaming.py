"""TTFB smoke with simulated LLM token streaming.

Pushes the utterance into the server one ~5-char "token" at a time with
``continue=True``, then sends a final ``continue=False`` flush. Mirrors how
Pipecat will actually feed text in prod.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid

import websockets

URL = os.environ.get("SUPERTONIC_URL", "ws://127.0.0.1:7799/tts/websocket")
VOICE = os.environ.get("SUPERTONIC_VOICE", "F1")
SAMPLE_RATE = 24000
TOKEN_LEN = int(os.environ.get("SUPERTONIC_TOKEN_LEN", "5"))
INTER_TOKEN_DELAY_MS = float(os.environ.get("SUPERTONIC_TOKEN_DELAY_MS", "30"))


# Three variants — short first sentence vs long.
UTTERANCES = {
    "long_first_sentence": (
        "नमस्ते, मैं प्रिया बोल रही हूँ IIFL फाइनेंस से। "
        "बताइए, आपको गोल्ड लोन की जानकारी चाहिए?"
    ),
    "short_first_sentence": (
        "नमस्ते। मैं प्रिया बोल रही हूँ IIFL फाइनेंस से। "
        "बताइए, आपको गोल्ड लोन की जानकारी चाहिए?"
    ),
    "very_short_opener": (
        "हाँ बिल्कुल। मैं आपको गोल्ड लोन की पूरी जानकारी दे सकती हूँ।"
    ),
}


def _frame(ctx_id: str, text: str, continue_: bool) -> str:
    return json.dumps({
        "transcript": text,
        "continue": continue_,
        "context_id": ctx_id,
        "model_id": "sonic-3",
        "voice": {"mode": "id", "id": VOICE},
        "output_format": {
            "container": "raw", "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        },
        "add_timestamps": False,
    })


async def receive_until_done(ws, t0: float) -> dict:
    first_chunk_at = None
    chunks = 0
    total_bytes = 0
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "chunk":
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            total_bytes += len(base64.b64decode(msg["data"]))
            chunks += 1
        elif kind == "done":
            end_at = time.perf_counter()
            audio_s = total_bytes / 2 / SAMPLE_RATE
            return {
                "ttfb_ms": round((first_chunk_at - t0) * 1000.0, 1),
                "wall_total_ms": round((end_at - t0) * 1000.0, 1),
                "chunks": chunks,
                "audio_seconds": round(audio_s, 3),
                "rtf": round(((end_at - t0) / audio_s), 4) if audio_s else None,
            }
        elif kind == "error":
            raise RuntimeError(f"server error: {msg}")


async def run_one(text: str, label: str) -> dict:
    ctx_id = f"strm-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(URL, max_size=None) as ws:
        t0 = time.perf_counter()
        # Stream tokens.
        i = 0
        while i < len(text):
            tok = text[i:i + TOKEN_LEN]
            i += TOKEN_LEN
            await ws.send(_frame(ctx_id, tok, continue_=True))
            await asyncio.sleep(INTER_TOKEN_DELAY_MS / 1000.0)
        # Final flush.
        await ws.send(_frame(ctx_id, "", continue_=False))
        result = await receive_until_done(ws, t0)
        result["label"] = label
        result["text_chars"] = len(text)
        return result


async def main() -> None:
    print(f"streaming via {URL} voice={VOICE} "
          f"token_len={TOKEN_LEN} inter_token_delay={INTER_TOKEN_DELAY_MS}ms\n")
    print(f"{'label':<25}{'chars':>7}{'TTFB ms':>10}{'wall ms':>10}"
          f"{'audio s':>10}{'RTF':>8}")
    print("-" * 70)
    for label, text in UTTERANCES.items():
        r = await run_one(text, label)
        print(f"{label:<25}{r['text_chars']:>7}{r['ttfb_ms']:>10.0f}"
              f"{r['wall_total_ms']:>10.0f}{r['audio_seconds']:>10.2f}"
              f"{r['rtf']:>8.3f}")


if __name__ == "__main__":
    asyncio.run(main())
