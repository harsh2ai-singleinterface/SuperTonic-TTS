"""Audio post-processing for the Supertonic streaming TTS server.

Converts Supertonic's native output (float32 mono @ 44.1 kHz) into the wire
format Cartesia clients expect (int16 little-endian PCM mono @ 24 kHz) and
slices it into ~20 ms frames suitable for emit-ASAP WebSocket streaming.

Pure functions only — no I/O, no asyncio, no logging. Only `numpy` and
`scipy` are used.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "f32_to_int16",
    "resample",
    "slice_into_frames",
    "pipeline",
    "DEFAULT_FRAME_MS",
]


# 20 ms is the de-facto frame size Cartesia/Pipecat consumers expect.
# At 24 kHz that is 480 samples == 960 bytes per frame.
DEFAULT_FRAME_MS: float = 20.0

# Full-scale int16 magnitudes; pulled out so the clipping behaviour is
# obvious and trivially auditable.
_INT16_MAX: int = 32767
_INT16_MIN: int = -32768


def f32_to_int16(wav_f32: np.ndarray) -> np.ndarray:
    """Convert float32 mono audio in [-1, 1] to int16.

    Out-of-range values are clipped (saturated) so we never silently wrap
    around. Accepts shape ``(N,)`` or ``(1, N)``; always returns shape
    ``(N,)`` with ``dtype=int16``.
    """
    if not isinstance(wav_f32, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(wav_f32).__name__}")

    arr = wav_f32
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(
                f"f32_to_int16 is mono only; got shape {arr.shape}"
            )
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(
            f"f32_to_int16 expects 1-D or (1, N) input; got shape {arr.shape}"
        )

    # Promote to float32 for the multiply; keep behaviour identical when
    # already float32.
    arr = arr.astype(np.float32, copy=False)

    # Scale to int16 range, then clip. Multiplying by 32767 keeps positive
    # full-scale at +32767; we still clip the negative tail to -32768 for
    # symmetry with int16's asymmetric range.
    scaled = arr * float(_INT16_MAX)
    np.clip(scaled, _INT16_MIN, _INT16_MAX, out=scaled)
    return scaled.astype(np.int16, copy=False)


def resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample mono audio from ``src_sr`` to ``dst_sr``.

    Uses ``scipy.signal.resample_poly`` with up/down factors derived from
    ``gcd(src_sr, dst_sr)``. Preserves dtype (int16 stays int16,
    float32 stays float32). For int16 input we resample in float32 then
    clip+cast back so we don't lose headroom.
    """
    if not isinstance(wav, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(wav).__name__}")
    if wav.ndim != 1:
        raise ValueError(f"resample is mono only; got shape {wav.shape}")
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"sample rates must be positive; got {src_sr=} {dst_sr=}")

    if src_sr == dst_sr:
        # No-op but return a fresh array so callers can't mutate ours.
        return wav.copy()

    g = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // g
    down = int(src_sr) // g

    in_dtype = wav.dtype

    # resample_poly works on float internally anyway; do the cast explicitly
    # so the int16 branch round-trips cleanly.
    work = wav.astype(np.float32, copy=False)
    out = resample_poly(work, up, down).astype(np.float32, copy=False)

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        np.clip(out, info.min, info.max, out=out)
        return out.astype(in_dtype, copy=False)
    return out.astype(in_dtype, copy=False)


def slice_into_frames(
    pcm_int16: np.ndarray,
    sample_rate: int,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> list[bytes]:
    """Slice int16 mono PCM into fixed-length little-endian byte frames.

    Each frame is exactly ``int(sample_rate * frame_ms / 1000) * 2`` bytes.
    The final frame is zero-padded if the input doesn't divide evenly, so
    downstream encoders never see a runt frame.
    """
    if not isinstance(pcm_int16, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(pcm_int16).__name__}")
    if pcm_int16.dtype != np.int16:
        raise TypeError(f"expected int16 input, got dtype={pcm_int16.dtype}")
    if pcm_int16.ndim != 1:
        raise ValueError(
            f"slice_into_frames is mono only; got shape {pcm_int16.shape}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if frame_ms <= 0:
        raise ValueError(f"frame_ms must be positive; got {frame_ms}")

    samples_per_frame = int(sample_rate * frame_ms / 1000.0)
    if samples_per_frame <= 0:
        raise ValueError(
            f"frame_ms={frame_ms} at sample_rate={sample_rate} yields 0 samples"
        )

    n = pcm_int16.shape[0]
    if n == 0:
        return []

    # Ceil-div, then pad with silence so every emitted frame is exactly
    # samples_per_frame samples wide.
    n_frames = (n + samples_per_frame - 1) // samples_per_frame
    padded_len = n_frames * samples_per_frame
    if padded_len != n:
        padded = np.zeros(padded_len, dtype=np.int16)
        padded[:n] = pcm_int16
    else:
        padded = pcm_int16

    # numpy is little-endian on every platform we care about, but be
    # explicit so big-endian boxes don't silently mis-encode.
    le = padded.astype("<i2", copy=False)
    raw = le.tobytes()
    bytes_per_frame = samples_per_frame * 2
    return [
        raw[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        for i in range(n_frames)
    ]


def pipeline(
    wav_f32: np.ndarray,
    src_sr: int,
    dst_sr: int,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> list[bytes]:
    """End-to-end: resample (in float) → int16 → byte frames.

    Order matters: resample in float space *then* quantise to int16, so
    polyphase filter ringing doesn't compound with int16 quantisation noise.
    """
    if not isinstance(wav_f32, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(wav_f32).__name__}")

    arr = wav_f32
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"pipeline is mono only; got shape {wav_f32.shape}")

    arr = arr.astype(np.float32, copy=False)
    if src_sr != dst_sr:
        arr = resample(arr, src_sr, dst_sr)
    pcm = f32_to_int16(arr)
    return slice_into_frames(pcm, dst_sr, frame_ms=frame_ms)
