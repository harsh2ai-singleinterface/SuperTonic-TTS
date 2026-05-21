"""Tests for `supertonic_server.audio`.

These exercise the pure-DSP path Supertonic-output → Cartesia-wire-bytes:
  * float32 → int16 clipping & round-trip
  * 44.1 kHz → 24 kHz polyphase resampling (length + spectral fidelity)
  * 20 ms frame slicer (exact length + zero padding)
  * full pipeline byte-count sanity check
  * informational wall-clock benchmark on a 15 s clip
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from supertonic_server.audio import (
    DEFAULT_FRAME_MS,
    f32_to_int16,
    pipeline,
    resample,
    slice_into_frames,
)


# ---------- f32_to_int16 ----------

def test_f32_to_int16_clips_positive_overflow() -> None:
    x = np.array([0.0, 0.5, 1.0, 1.5, 100.0], dtype=np.float32)
    out = f32_to_int16(x)
    assert out.dtype == np.int16
    assert out[3] == 32767, "values above +1.0 must clip to +32767"
    assert out[4] == 32767, "huge values must clip, not wrap"


def test_f32_to_int16_clips_negative_overflow() -> None:
    x = np.array([-0.5, -1.0, -1.5, -100.0], dtype=np.float32)
    out = f32_to_int16(x)
    assert out[2] == -32768, "values below -1.0 must clip to -32768"
    assert out[3] == -32768, "huge negative values must clip, not wrap"


def test_f32_to_int16_accepts_2d_mono() -> None:
    x = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)  # shape (1, 3)
    out = f32_to_int16(x)
    assert out.shape == (3,)
    assert out.dtype == np.int16


def test_f32_to_int16_roundtrip_random_int16() -> None:
    rng = np.random.default_rng(0xC0FFEE)
    src = rng.integers(low=-32768, high=32768, size=10_000, dtype=np.int16)
    # Standard PCM round-trip convention: divide by 32768.0 to get float.
    f = src.astype(np.float32) / 32768.0
    # Our int16 quantiser scales by 32767 then clips. For any int16 input
    # whose float reconstruction is in [-1, 1) the result should match
    # within 1 LSB. Verify exact equality except for the single edge case
    # of int16==-32768 (-> float == -1.0 -> stays at -32768).
    out = f32_to_int16(f)
    # Worst-case error is 1 LSB from the 32767-vs-32768 asymmetry.
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"round-trip error too large: max diff={diff.max()}"


def test_f32_to_int16_rejects_stereo() -> None:
    x = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        f32_to_int16(x)


# ---------- resample ----------

def test_resample_length_44k_to_24k() -> None:
    src_sr, dst_sr = 44_100, 24_000
    n_in = src_sr  # exactly 1 s
    sig = np.zeros(n_in, dtype=np.float32)
    out = resample(sig, src_sr, dst_sr)
    expected = round(n_in * dst_sr / src_sr)
    assert abs(out.shape[0] - expected) <= 1, (
        f"expected ~{expected} samples, got {out.shape[0]}"
    )


def test_resample_preserves_dtype_int16() -> None:
    sig = np.zeros(44_100, dtype=np.int16)
    out = resample(sig, 44_100, 24_000)
    assert out.dtype == np.int16


def test_resample_preserves_dtype_float32() -> None:
    sig = np.zeros(44_100, dtype=np.float32)
    out = resample(sig, 44_100, 24_000)
    assert out.dtype == np.float32


def test_resample_noop_when_rates_equal() -> None:
    sig = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    out = resample(sig, 24_000, 24_000)
    assert out.shape == sig.shape
    assert np.allclose(out, sig)


def test_resample_440hz_sine_peak_frequency_preserved() -> None:
    src_sr, dst_sr = 44_100, 24_000
    freq_hz = 440.0
    duration_s = 1.0
    t = np.arange(int(src_sr * duration_s), dtype=np.float32) / src_sr
    sine = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    out = resample(sine, src_sr, dst_sr)

    # rfft → find peak bin → convert back to Hz.
    spec = np.abs(np.fft.rfft(out))
    peak_bin = int(np.argmax(spec))
    bin_hz = dst_sr / out.shape[0]
    peak_hz = peak_bin * bin_hz

    # ±5 % tolerance is the spec.
    assert abs(peak_hz - freq_hz) / freq_hz < 0.05, (
        f"peak shifted from {freq_hz} Hz to {peak_hz:.1f} Hz"
    )


# ---------- slice_into_frames ----------

def test_slice_exact_one_second_at_24k() -> None:
    sr = 24_000
    pcm = np.zeros(sr, dtype=np.int16)  # 1.000 s
    frames = slice_into_frames(pcm, sr, frame_ms=20.0)
    assert len(frames) == 50
    for f in frames:
        assert len(f) == 960, "each frame must be 480 samples * 2 bytes"


def test_slice_pads_partial_final_frame() -> None:
    sr = 24_000
    # int(24_000 * 1.005) == 24_119 samples → 50.248 frames → 51 frames.
    # First 50 frames consume 24_000 samples; last frame carries 119 real
    # samples and 361 zero-padded samples.
    n_in = int(sr * 1.005)
    pcm = np.ones(n_in, dtype=np.int16) * 1234
    frames = slice_into_frames(pcm, sr, frame_ms=20.0)
    assert len(frames) == 51
    last = np.frombuffer(frames[-1], dtype="<i2")
    assert last.shape[0] == 480
    real_samples = n_in - 50 * 480
    assert 0 < real_samples < 480
    assert np.all(last[:real_samples] == 1234)
    assert np.all(last[real_samples:] == 0), "tail of final frame must be zero-padded"


def test_slice_empty_input() -> None:
    sr = 24_000
    out = slice_into_frames(np.zeros(0, dtype=np.int16), sr)
    assert out == []


def test_slice_little_endian_byte_order() -> None:
    sr = 24_000
    samples_per_frame = int(sr * DEFAULT_FRAME_MS / 1000.0)
    # Single non-zero sample = 0x0100. Little-endian bytes = 00 01.
    pcm = np.zeros(samples_per_frame, dtype=np.int16)
    pcm[0] = 0x0100
    frames = slice_into_frames(pcm, sr)
    assert frames[0][0] == 0x00
    assert frames[0][1] == 0x01


def test_slice_rejects_non_int16() -> None:
    with pytest.raises(TypeError):
        slice_into_frames(np.zeros(100, dtype=np.float32), 24_000)


# ---------- pipeline ----------

def test_pipeline_end_to_end_byte_count() -> None:
    src_sr, dst_sr = 44_100, 24_000
    duration_s = 0.5
    freq_hz = 440.0
    t = np.arange(int(src_sr * duration_s), dtype=np.float32) / src_sr
    sine = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    frames = pipeline(sine, src_sr, dst_sr, frame_ms=20.0)

    samples_per_frame = int(dst_sr * 20.0 / 1000.0)  # 480
    bytes_per_frame = samples_per_frame * 2  # 960

    # 0.5 s at 24 kHz int16 = 12_000 samples = 24_000 bytes. That's an exact
    # multiple of 960 → 25 frames → 24_000 bytes total. Polyphase resample
    # can land ±1 sample so allow one extra frame of padding.
    expected_min_bytes = int(duration_s * dst_sr) * 2  # 24_000
    total_bytes = sum(len(f) for f in frames)
    assert total_bytes >= expected_min_bytes
    assert total_bytes <= expected_min_bytes + bytes_per_frame, (
        f"frame padding ran away: {total_bytes} bytes for {duration_s}s"
    )
    for f in frames:
        assert len(f) == bytes_per_frame


def test_pipeline_accepts_2d_mono() -> None:
    src_sr, dst_sr = 44_100, 24_000
    sig = np.zeros((1, src_sr), dtype=np.float32)
    frames = pipeline(sig, src_sr, dst_sr)
    assert len(frames) > 0
    assert all(len(f) == 960 for f in frames)


# ---------- benchmark (informational, not an assertion) ----------

def test_pipeline_benchmark_15s(capsys: pytest.CaptureFixture[str]) -> None:
    src_sr, dst_sr = 44_100, 24_000
    duration_s = 15.0
    freq_hz = 440.0
    rng = np.random.default_rng(42)
    t = np.arange(int(src_sr * duration_s), dtype=np.float32) / src_sr
    sine = (0.4 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    # Mix in a little noise so the signal isn't trivially compressible.
    sine = sine + 0.01 * rng.standard_normal(sine.shape).astype(np.float32)

    # One warm-up pass so JIT'd scipy paths are cached.
    _ = pipeline(sine, src_sr, dst_sr)

    t0 = time.perf_counter()
    frames = pipeline(sine, src_sr, dst_sr)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Sanity: still produced ~15 s worth of frames.
    assert len(frames) >= int(duration_s * 1000.0 / 20.0)

    # Print so `pytest -v -s` (or our captured stdout) shows the cost.
    msg = f"[bench] pipeline(15s @ 44.1k → 24k) took {dt_ms:.2f} ms"
    with capsys.disabled():
        print("\n" + msg)
