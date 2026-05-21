# SuperTonic-TTS

A **Cartesia-shaped streaming TTS WebSocket server**, backed by the on-device
[Supertonic 3](https://huggingface.co/Supertone/supertonic-3) ONNX model.

The point: any client that already speaks Cartesia's `/tts/websocket` API
(e.g. [Pipecat](https://github.com/pipecat-ai/pipecat)'s `CartesiaTTSService`)
can be pointed at this server with a URL change — no client-side code change
required. Audio synthesis runs locally on CPU (Mac / Linux / Raspberry Pi
class hardware), with no API key, no cloud round-trip, and no per-call cost.

```
                                   wss:// Cartesia client (unchanged)
                                              │
                                              ▼
                       ┌──────────────────────────────────────────┐
                       │           SuperTonic-TTS                 │
                       │                                          │
                       │  WebSocket server (FastAPI / uvicorn)    │
                       │       │                                  │
                       │       ▼                                  │
                       │  Per-session text coalescer  ──► worker pool
                       │                                  │       │
                       │                                  ▼       │
                       │                          Supertonic 3 ONNX
                       │                                  │       │
                       │                                  ▼       │
                       │       ◄────  44.1 kHz f32  ◄────         │
                       │                  │                       │
                       │                  ▼                       │
                       │  Resample → int16 LE → 20 ms frames      │
                       │                  │                       │
                       │       ──── base64 in Cartesia JSON ────► │
                       └──────────────────────────────────────────┘
```

---

## Why this exists

The Supertonic Python SDK ships an excellent on-device TTS model but only
exposes a synchronous, one-shot `tts.synthesize(text) → wav` API. For a
real-time voice agent that's a non-starter — you can't stream audio to the
caller until the whole utterance is rendered (~2-3 s of dead air per turn).

This project wraps the model in a **WebSocket streaming server with the same
wire protocol as Cartesia's TTS WebSocket API**, so:

  * **Cartesia clients work as a drop-in.** Existing voice-agent stacks
    (Pipecat, custom WS clients) only need to change the URL.
  * **First audio in ~500 ms.** Synthesis happens per-chunk on a worker
    pool; the first 20 ms PCM frame ships as soon as the first chunk is
    ready, not after the full utterance is rendered.
  * **No API key, no cloud, no rate limits.** Runs on the same box as
    the voice agent, or any other reachable host.

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

The first run downloads the Supertonic 3 model weights from Hugging Face into
`~/.cache/supertonic3/`. Subsequent runs reuse the cache.

### 2. Run the server

```bash
PYTHONPATH=src python -m supertonic_server.server
```

Defaults:

```
host  = 127.0.0.1
port  = 7799
workers = 2     (concurrent synth slots; each loads its own ONNX session)
steps   = 8     (Supertonic diffusion steps; 5-12 supported)
```

All overridable via env vars (see [Configuration](#configuration)).

You'll see a short cold-start log:

```
loading worker 1/2
Supertonic TTS model loaded in 0.40s
warmup: synthesizing 'नमस्ते।' with voice=F1
warmup done in 0.44s
loading worker 2/2
…
pool ready: 2 workers warmed
Uvicorn running on http://127.0.0.1:7799 (Press CTRL+C to quit)
```

`/ready` returns 200 only after the cold start completes (~2 s on Apple
Silicon). Use this in load-balancer health probes.

### 3. Pick a voice

```bash
curl http://127.0.0.1:7799/voices
# {"voices":[{"name":"F1","gender":"F"}, … {"name":"M5","gender":"M"}]}
```

10 voices in the default catalog: `F1`–`F5` (female), `M1`–`M5` (male). Voice
IDs are short names, **not** Cartesia UUIDs — when sending the Cartesia-shaped
`{"voice": {"mode":"id", "id":"…"}}` frame, pass one of these.

### 4. Drive it from any Cartesia client

```python
# Example: Pipecat with the supertonic backend
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings

tts = CartesiaTTSService(
    api_key="noop",                                       # server ignores it
    url="ws://127.0.0.1:7799/tts/websocket",              # ← the only change
    settings=CartesiaTTSSettings(voice="F1", model="supertonic-3"),
    sample_rate=24000,
)
```

The wire protocol is documented in [`src/supertonic_server/wire.py`](src/supertonic_server/wire.py)
and is intentionally identical to Cartesia's — the same encode/decode logic
works against both.

### 5. End-to-end smoke

```bash
python scripts/smoke_ttfb.py            # one-shot client (text sent in one frame)
python scripts/smoke_ttfb_streaming.py  # streaming client (text sent in tokens)
```

Both connect to the server, send Hindi sample text, and print the
time-to-first-byte plus per-utterance RTF.

---

## Performance

Measured on Apple Silicon (M-series) CPU, 2-worker pool, `total_steps=8`:

| Test | Input | TTFB |
|---|---|---|
| One-shot (87-char Hindi, full sentence) | one frame | **~820 ms** |
| Streaming, long first sentence | LLM-style 5-char tokens | ~1,220 ms |
| Streaming, short first sentence (`नमस्ते।`) | LLM-style 5-char tokens | **~570 ms** |
| Streaming, very-short opener (`हाँ बिल्कुल।`) | LLM-style 5-char tokens | ~510 ms |

Sustained RTF: **~0.20** (~5× real-time). Per-chunk synthesis runs at the
[Supertonic-3](https://huggingface.co/Supertone/supertonic-3) baseline.

On a GPU box (NVIDIA T4 / A10 / H100), TTFB is expected to drop to the
150–300 ms range, putting it within rounding distance of cloud Cartesia. The
Supertonic Python SDK currently exposes only CPU execution providers; GPU
support is upstream-pending.

---

## API surface

### `WS /tts/websocket`

The streaming TTS endpoint. Wire protocol is **Cartesia-shaped**:

**Client → server** — text frame:
```json
{
  "transcript": "<text>",
  "continue": false,
  "context_id": "<your ctx id>",
  "voice": {"mode": "id", "id": "F1"},
  "output_format": {
    "container": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 24000
  },
  "add_timestamps": false,
  "language": "hi"
}
```

`continue: true` keeps the context open for additional text (LLM token
streaming). `continue: false` with non-empty `transcript` is a one-shot.
`continue: false` with empty `transcript` is a flush.

**Client → server** — cancel:
```json
{"context_id": "<ctx>", "cancel": true}
```

**Server → client** — audio chunk:
```json
{"type": "chunk", "context_id": "<ctx>", "data": "<base64 int16-LE PCM>"}
```

**Server → client** — end of context:
```json
{"type": "done", "context_id": "<ctx>"}
```

**Server → client** — error:
```json
{"type": "error", "context_id": "<ctx>", "error": "<message>"}
```

PCM is little-endian int16, sliced into ~20 ms frames at the requested
`output_format.sample_rate`. Default 24 kHz matches Pipecat and most
voice-agent stacks.

### `GET /voices`
Returns the catalog from `~/.cache/supertonic3/voice_styles/`:
```json
{"voices":[{"name":"F1","gender":"F"}, …]}
```

### `GET /health`
Always 200 `ok`. Liveness probe.

### `GET /ready`
200 once the worker pool has finished cold-start + warmup, else 503.
Readiness probe — point your load balancer here.

---

## Configuration

All env-driven; defaults are sensible for local dev.

| Env var | Default | Effect |
|---|---|---|
| `SUPERTONIC_HOST` | `127.0.0.1` | Bind host |
| `SUPERTONIC_PORT` | `7799` | Bind port |
| `SUPERTONIC_WORKERS` | `2` | Concurrent synth workers (each ≈ 150 MB resident) |
| `SUPERTONIC_STEPS` | `8` | Diffusion steps per synth (5 = fastest, 12 = best quality) |
| `SUPERTONIC_LOG_LEVEL` | `INFO` | `DEBUG` for per-synth wall-clock + RTF logs |

There is no API-key concept; the server doesn't authenticate. Put it behind
a private VPC or a reverse proxy if you need access control.

---

## Repository layout

```
supertonic_server/
├── plan.md                     ← design notes
├── pyproject.toml
├── requirements.txt            ← runtime
├── requirements-dev.txt        ← + pytest, httpx
├── src/
│   └── supertonic_server/
│       ├── wire.py             ← Cartesia-shaped JSON encode/decode
│       ├── audio.py            ← f32→int16 + 44.1k→24k resample + 20 ms frames
│       ├── coalescer.py        ← token-stream → clause-boundary chunks
│       ├── worker.py           ← Supertonic TTS instance + warmup
│       ├── voices.py           ← catalog from ~/.cache/supertonic3/voice_styles/
│       ├── pool.py             ← async worker pool + per-context cancellation
│       ├── session.py          ← per-WS session state, emit loop, TTFB tracking
│       └── server.py           ← FastAPI app, REST + WS routes
├── tests/                      ← 88 unit tests (pytest)
└── scripts/
    ├── smoke_ttfb.py                  ← one-shot TTFB smoke
    ├── smoke_ttfb_streaming.py        ← token-stream TTFB smoke
    └── run_eva_with_supertonic.sh     ← drive an EVA run through this server
```

---

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest -v
```

88 unit tests cover:
* `wire.py` — Cartesia-shape encode/decode round-trips, malformed-frame handling
* `audio.py` — clipping, resample fidelity, frame slicing, padding
* `coalescer.py` — clause/sentence boundaries (Hindi `।` + western `.!?`),
  decimal protection, phone-number protection, force-flush
* `worker.py` — model load + warmup, voice-style cache hits
* `voices.py` — catalog loading, KeyError on unknown voice

Worker tests load the real model and take a couple of seconds. Mark them
with `pytest -m "not slow"` to skip if iterating.

---

## Limitations

* **CPU-only.** The upstream `supertonic` Python SDK doesn't expose
  `providers=` yet. GPU acceleration would require either upstream support
  or running the ONNX files directly via `onnxruntime-gpu` (losing the
  text-preprocessing / voice-catalog parts of the SDK).
* **No interrupt mid-synth.** A `cancel` frame stops sending audio after
  the current in-flight synth chunk completes (Supertonic's `synthesize()`
  call isn't preemptible). Keep chunks short for snappy cancellation.
* **Hindi-first quality.** The model handles English / multilingual but
  was tuned for Indian voices. Other languages may need a different model
  checkpoint.
* **No authentication.** Local-network only by design. Front with a reverse
  proxy + auth if exposing publicly.

---

## License

Inherits Supertonic's license for the underlying model weights. Server code
in this repo is provided as-is.
