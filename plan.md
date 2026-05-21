# Supertonic Streaming-TTS Backend — Plan

## Goal

Wrap the on-device Supertonic TTS engine in a **streaming WebSocket server**
that is wire-compatible with Cartesia's `/tts/websocket` API, so the existing
Pipecat `CartesiaTTSService` becomes a drop-in client (just flip the URL).

**Target metric:** time-to-first-audio-byte (TTFB) under ~300 ms on a single
H100 or fast Mac (Apple-silicon).
For reference today:

| Stack | TTS TTFB (prod-measured) |
|---|---:|
| Cartesia | ~250 ms |
| Google Gemini TTS (current IIFL prod) | ~1100 ms |
| Supertonic full-utterance synth | ~3000 ms (15 s clip) |
| **Supertonic streaming (this project)** | **target ≤ 300 ms** |

Without streaming, Supertonic is unusable in a real-time voice bot — it
synthesizes a 15 s utterance in ~3 s and ships it all at once.

---

## Architecture

```
                 ┌────────────────────────────────────────────────┐
LLM (Simplismart)│  Pipecat                                       │
   ───tokens───▶ │  └─ CartesiaTTSService (unchanged, URL flipped)│
                 └──────────────┬─────────────────────────────────┘
                                │  ws:// (Cartesia-shaped frames)
                 ┌──────────────▼─────────────────────────────────┐
                 │  Supertonic-TTS server                         │
                 │  ┌────────────────────────────────────────┐    │
                 │  │  WebSocket session manager             │    │
                 │  │   • handshake: voice + sample rate     │    │
                 │  │   • text-stream coalescer (clause splits)   │
                 │  │   • interrupt handler                  │    │
                 │  └────────────┬──────────────┬────────────┘    │
                 │   text chunks │              │ control         │
                 │  ┌────────────▼───────┐ ┌────▼─────────────┐   │
                 │  │ Synthesis worker   │ │ Cancellation bus │   │
                 │  │ pool (N workers)   │ │ (per-session)    │   │
                 │  │  • Supertonic TTS  │ └──────────────────┘   │
                 │  │    instance pinned │                        │
                 │  │  • voice cache     │                        │
                 │  └────────────┬───────┘                        │
                 │   wav (f32, 44.1k)                             │
                 │  ┌────────────▼───────────────────────────┐    │
                 │  │ Audio post-proc                        │    │
                 │  │  • resample 44.1k → 24k (Cartesia parity)   │
                 │  │  • float32 → int16 PCM                 │    │
                 │  │  • 20ms-frame slicer                   │    │
                 │  └────────────┬───────────────────────────┘    │
                 │               │ raw PCM frames                 │
                 │  ┌────────────▼───────────────────────────┐    │
                 │  │ Wire encoder                           │    │
                 │  │  • base64 + JSON-wrap (Cartesia shape) │    │
                 │  │  • emit ASAP (no pacing)               │    │
                 │  └────────────────────────────────────────┘    │
                 └────────────────────────────────────────────────┘
```

---

## Components

### Must-have (MVP)

| # | Component | Why |
|---|---|---|
| 1 | **WebSocket server** (`aiohttp` or FastAPI) | One persistent connection per voice-bot turn. Handshake selects voice + output sample rate. |
| 2 | **Cartesia-shaped wire protocol** | JSON control + base64 PCM in JSON. Match Cartesia's `{"type":"chunk","data":"<b64>"}` and `{"type":"done"}` exactly so Pipecat's existing `CartesiaTTSService` consumes it unchanged. |
| 3 | **Text-stream coalescer** | LLM streams tokens; can't synthesize per-token. Buffer until clause boundary (`।`, `.`, `,`, `?`, `!` or N-char threshold), flush that chunk to a worker. Keep remainder for next message. Mirrors Cartesia's `continue=True` behavior. |
| 4 | **Synthesis worker pool** | Supertonic's `synthesize()` is synchronous and CPU/GPU-bound. Run in an asyncio executor or dedicated process pool. Each worker holds its own `TTS` instance + voice-style cache (~100 MB each → 2–4 workers per box). |
| 5 | **Audio post-processing** | Supertonic emits float32 @ 44.1 kHz. Pipecat consumers expect int16 @ 24 kHz. Resample (`scipy.signal.resample_poly` or `samplerate`), cast int16, slice into ~20 ms frames (≈ 960 samples @ 24 kHz). |
| 6 | **Emit-ASAP semantics** | The whole point. Push the first PCM frame to the WebSocket as soon as the *first* chunk's `synthesize()` returns — not the whole utterance. Drops TTFB from ~3 s to one-chunk synth time (~300 ms). |
| 7 | **Voice catalog endpoint** | `GET /voices` returns F1–F5 / M1–M5 with metadata. Lets the bot pick at session-start without hard-coding. |

### Important (before prod bot use)

| # | Component | Why |
|---|---|---|
| 8 | **Interruption handling** | User starts speaking → bot must stop mid-utterance. WebSocket `{"type":"cancel"}`. Can't preempt a running `synthesize()` call, so keep chunks short (1–2 sentences) and drop any queued chunks for that session on cancel. |
| 9 | **Model pre-warming** | `TTS(auto_download=True)` + voice-style load is the ~45 s cold-start we measured. Load once at boot, hold in memory, never reload per request. Run a 1-token dummy synth at startup before serving traffic. |
| 10 | **Health + readiness probes** | `/health` (process alive) vs `/ready` (model loaded + warm-up done). Without `/ready`, a load balancer routes traffic to a process that still owes 45 s to its first caller. |
| 11 | **Metrics** | Per-request TTFB, RTF per chunk, queue depth, worker utilization. Expose `/metrics` (Prometheus). TTFB is *the* number we optimize against. |

### Nice-to-have (deferrable)

| # | Component | Why |
|---|---|---|
| 12 | HTTP one-shot endpoint | Cartesia has REST `/v1/tts` for non-streaming callers. Supertonic already ships `supertonic serve` for this; keep it for tests / dashboards / debug. |
| 13 | Auth + per-key rate limiting | Only if exposed beyond own VPC. |
| 14 | Custom Pipecat adapter class | Skip if MVP keeps Cartesia-shaped protocol. Need it only if we invent our own protocol. |
| 15 | GPU dispatch | Supertonic docs say GPU mode isn't supported yet; CPU only today. Re-check before promising GPU scaling. |
| 16 | Voice mixing / runtime style controls | Wire Supertonic's `total_steps` and `speed` into the WS handshake; surface emotion if/when supported. |

### Explicitly out-of-scope

- Neural vocoder — Supertonic outputs PCM directly.
- Separate text-normalization service — Supertonic does this in-process.
- LLM-side changes — if we keep Cartesia's wire protocol, the upstream pipeline does not notice the swap.

---

## Three sub-problems harder than they look

1. **First-chunk latency vs prosody.** Aggressively flushing on every comma gives best TTFB but degrades prosody — the model can't shape intonation across a boundary it can't see. The coalescer needs tuning; Cartesia invested heavily here.
2. **Resampling cost.** Naive `resample_poly` 44.1 k → 24 k on a 15 s clip is ~80–150 ms — not free. Options: resample per-chunk inline (fast on small chunks), pre-bake the model to output 24 kHz, or pass 44.1 kHz forward and resample at the telephony egress.
3. **Worker exhaustion under bursty load.** With N workers and slow per-chunk synth, cross-session queueing spikes one session's TTFB when another is mid-utterance. Cartesia's answer is "more workers + smaller chunks"; Supertonic-on-CPU may need per-tenant queues with admission control.

---

## Phased build

| Phase | Deliverable | Effort |
|---|---|---|
| **P0 — MVP** | Cartesia-shaped WS server, single voice, no interruption, no metrics. Pipecat can connect and play audio. | ~1–2 days |
| **P1 — Bot-ready** | Interruption, worker pool (4 workers), `/metrics`, voice catalog, pre-warm + `/ready`. | ~1 week |
| **P2 — Prod-grade** | Auth, multi-tenant, autoscaling, full observability, deployment manifests. | ~3–4 weeks + SRE work |

P0 is the gate that proves TTFB hits target; P1 makes it usable in EVA smoke runs; P2 makes it usable in prod traffic.

---

## Repo layout (proposed)

```
supertonic_server/
├── plan.md                     ← this file
├── pyproject.toml
├── src/
│   └── supertonic_server/
│       ├── __init__.py
│       ├── server.py           ← aiohttp/FastAPI app, WS handler
│       ├── session.py          ← per-WS session state
│       ├── coalescer.py        ← LLM-token → synthesis-chunk buffering
│       ├── worker.py           ← Supertonic TTS instance + synthesize()
│       ├── pool.py             ← worker pool, queueing, cancellation
│       ├── audio.py            ← resample, f32→int16, 20ms slicer
│       ├── wire.py             ← Cartesia-shaped JSON encoder/decoder
│       ├── voices.py           ← catalog endpoint + voice-style cache
│       └── metrics.py          ← TTFB / RTF / pool stats
├── tests/
│   ├── test_coalescer.py
│   ├── test_audio.py
│   ├── test_wire_compat.py     ← assert wire frames match Cartesia spec
│   └── test_ttfb_smoke.py      ← end-to-end TTFB measurement
└── scripts/
    └── bench.py                ← TTFB / RTF benchmark across loads
```

---

## Verification

| # | Check | Pass criterion |
|---|---|---|
| 1 | TTFB single-shot | First PCM byte ≤ 300 ms after first text-frame for a 15-word Hindi utterance. |
| 2 | RTF sustained | Mean RTF across 100-utterance benchmark ≤ 0.25 (4× real-time). |
| 3 | Wire compat | Pipecat `CartesiaTTSService` pointed at our URL produces audible audio in the EVA harness with no code change. |
| 4 | Interrupt latency | After `{"type":"cancel"}` arrives, no further PCM frames sent within 100 ms. |
| 5 | Bursty load | 10 concurrent sessions, each emitting 5 utterances, no session's TTFB > 1 s. |
| 6 | Cold start | `/ready` returns 200 within 60 s of process start. |
| 7 | Memory cap | Resident memory of 4-worker server stable under 2 GB after 1 hr soak. |

---

## Open questions to resolve before P0

1. **Sample-rate strategy.** Resample to 24 kHz on server (Cartesia parity, +CPU) or pass 44.1 kHz forward and resample downstream? Affects Pipecat config + telephony egress.
2. **WebSocket framework.** `aiohttp` (lighter, pure-asyncio) vs FastAPI (more batteries, but Starlette WS handling is heavier). MVP could go either way; pick by what the team already operates.
3. **Worker isolation.** Threadpool (fast IPC, GIL-contended) vs processpool (clean isolation, heavier startup). Decide based on whether ONNX Runtime releases the GIL during inference.
4. **Voice-style hot-swap.** Allowed mid-session (cheap re-`get_voice_style()` call) or fixed at handshake (simpler)? Cartesia allows mid-session voice changes; MVP can punt to fixed.
5. **Chunk boundary policy.** Hard char threshold vs sentence-final-punctuation only vs both. Affects (1) TTFB, (2) prosody, (3) cancellation latency. Run a small ablation in P1.
