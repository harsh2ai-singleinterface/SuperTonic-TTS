"""Single Supertonic TTS worker.

One `SupertonicWorker` owns one `supertonic.TTS` instance plus a per-voice
Style cache. It is intentionally **synchronous** — the asyncio layer
(`pool.py`) is responsible for offloading `synthesize()` to a threadpool or
processpool. Keeping this module sync makes it trivially testable in
isolation and avoids accidentally serializing inference on the event loop.

Threading: a single worker is **not** thread-safe. Callers must serialize
calls into a given worker (the pool gives each worker its own queue).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from supertonic import TTS

from . import voices as _voices
from .cuda import cuda_is_usable

logger = logging.getLogger("supertonic_server.worker")

# Supertonic-3 emits float32 PCM at 44.1 kHz natively. Resampling to a
# downstream sample rate (e.g. 24 kHz for Cartesia parity) is `audio.py`'s
# job, not ours.
_NATIVE_SR: int = 44100

# Tiny prompt used for warmup — just enough to push the model through one
# full synth path so the first real call is hot.
_WARMUP_TEXT: str = "नमस्ते।"

# ONNX Runtime execution-provider lists by device. CUDA keeps CPU as a
# fallback so any op without a CUDA kernel still runs.
_CUDA_PROVIDERS: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_CPU_PROVIDERS: list[str] = ["CPUExecutionProvider"]

# Default device when none is given. "auto" => GPU if usable, else CPU.
_DEFAULT_DEVICE: str = "auto"


def _apply_device(device: str) -> list[str]:
    """Resolve ``device`` to an ONNX provider list and pin supertonic to it.

    ``device`` is one of:

    * ``"cuda"`` / ``"gpu"`` — run on the GPU; if the CUDA provider can't be
      loaded this logs an error and falls back to CPU (synthesis still works,
      just slower).
    * ``"cpu"``              — force CPU.
    * ``"auto"``             — GPU when usable, else CPU, without complaint.

    supertonic 1.3.1 hard-codes ``DEFAULT_ONNX_PROVIDERS`` and exposes no
    ``providers=`` knob on :class:`~supertonic.TTS`. Its loader reads
    ``supertonic.loader.DEFAULT_ONNX_PROVIDERS`` at model-load time, so we
    overwrite that module attribute here. Must be called *before* ``TTS()``.

    Returns the provider list that was applied.
    """
    device = device.strip().lower()
    if device not in ("cuda", "gpu", "auto", "cpu"):
        raise ValueError(
            f"unknown device {device!r}; expected 'cuda', 'cpu', or 'auto'"
        )

    if device == "cpu":
        providers = _CPU_PROVIDERS
    elif cuda_is_usable():
        providers = _CUDA_PROVIDERS
    else:
        if device in ("cuda", "gpu"):
            logger.error(
                "device=%s requested but the CUDA execution provider could "
                "not be loaded — falling back to CPU. Install 'onnxruntime-gpu' "
                "plus the nvidia-*-cu12 wheels (see requirements.txt).",
                device,
            )
        providers = _CPU_PROVIDERS

    # `supertonic.loader` looks up DEFAULT_ONNX_PROVIDERS as a module global at
    # session-creation time, so patching the attribute here takes effect for
    # the TTS() constructed next.
    import supertonic.loader as _st_loader

    _st_loader.DEFAULT_ONNX_PROVIDERS = providers
    return providers


@dataclass(slots=True)
class SynthResult:
    """Result of a single ``synthesize()`` call.

    Attributes:
        wav_f32:     Mono float32 audio, shape ``(N,)``.
        sample_rate: Always 44100 (supertonic's native rate).
        duration_s:  Audio length in seconds, as reported by the model
                     (not re-derived from ``len(wav)/sr``).
    """

    wav_f32: np.ndarray
    sample_rate: int
    duration_s: float


class SupertonicWorker:
    """Single Supertonic TTS instance pinned to this object.

    Caller is responsible for serializing calls (run in a threadpool or
    process — see ``pool.py``, not this file's concern).

    Voice-style objects are cached: ``TTS.get_voice_style()`` is called
    once per voice name on first use and reused for every subsequent
    ``synthesize()`` for that voice.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        total_steps: int = 8,
        speed: float = 1.0,
        lang: str = "hi",
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        """Load the TTS model immediately.

        Args:
            cache_dir:   Override for the voice-style cache directory. Default
                         is ``~/.cache/supertonic3/voice_styles/`` (resolved
                         lazily via :mod:`voices`).
            total_steps: Diffusion steps per synth. Lower = faster, less crisp.
            speed:       Playback speed multiplier (1.0 = natural).
            lang:        Language hint passed to ``synthesize``.
            device:      Inference device — ``"cuda"`` / ``"gpu"``, ``"cpu"``,
                         or ``"auto"`` (GPU when usable, else CPU). See
                         :func:`_apply_device`.
        """
        self._cache_dir = cache_dir
        self._total_steps = total_steps
        self._speed = speed
        self._lang = lang
        self._device = device

        # voice_name -> Style object (opaque; supertonic-internal)
        self._style_cache: dict[str, Any] = {}
        # voice_name -> hit count, useful for tests / metrics
        self._style_hits: dict[str, int] = {}
        self._warm: bool = False

        # Pin ONNX Runtime to the requested device *before* TTS() builds its
        # inference sessions.
        providers = _apply_device(device)

        logger.info("loading Supertonic TTS model (device=%s, providers=%s)",
                    device, providers)
        t0 = time.perf_counter()
        self._tts: TTS = TTS(auto_download=True)
        logger.info("Supertonic TTS model loaded in %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        """Run a tiny synthesis to warm JIT-style paths.

        Call once at startup so the first real :meth:`synthesize` doesn't
        pay cold-start cost (ONNX kernel selection, page-in of large
        constants, etc.). Idempotent — subsequent calls are no-ops.
        """
        if self._warm:
            return

        # Pick the first available voice; warmup is voice-agnostic in
        # effect since the style is small relative to the model graph.
        catalog = _voices.list_voices(self._cache_dir)
        if not catalog:
            raise RuntimeError(
                "cannot warmup: no voice styles found "
                f"(cache_dir={self._cache_dir!r})"
            )
        warm_voice = catalog[0].name

        logger.info("warmup: synthesizing %r with voice=%s", _WARMUP_TEXT, warm_voice)
        t0 = time.perf_counter()
        self.synthesize(_WARMUP_TEXT, warm_voice)
        logger.info("warmup done in %.2fs", time.perf_counter() - t0)

        self._warm = True

    def synthesize(self, text: str, voice_name: str) -> SynthResult:
        """Synthesize ``text`` with ``voice_name``. Blocking.

        The voice-style object is fetched on first use and cached for
        subsequent calls with the same ``voice_name``.
        """
        style = self._get_style(voice_name)

        t0 = time.perf_counter()
        wav, duration = self._tts.synthesize(
            text=text,
            lang=self._lang,
            voice_style=style,
            total_steps=self._total_steps,
            speed=self._speed,
        )
        synth_s = time.perf_counter() - t0

        # supertonic returns wav as shape (1, N) and duration as shape (1,).
        # Flatten to mono float32 and pull duration out as a scalar.
        wav_f32 = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).squeeze())
        if wav_f32.ndim != 1:
            # Defensive: if multi-channel ever appears, mix to mono.
            wav_f32 = wav_f32.mean(axis=0).astype(np.float32, copy=False)

        duration_s = float(np.asarray(duration).squeeze())

        logger.debug(
            "synthesize voice=%s chars=%d audio=%.2fs synth=%.3fs (RTF=%.3f)",
            voice_name,
            len(text),
            duration_s,
            synth_s,
            (synth_s / duration_s) if duration_s > 0 else float("inf"),
        )

        return SynthResult(wav_f32=wav_f32, sample_rate=_NATIVE_SR, duration_s=duration_s)

    @property
    def is_ready(self) -> bool:
        """True once the model has loaded *and* :meth:`warmup` has run."""
        return self._warm

    # ------------------------------------------------------------------ #
    # Test/metrics hooks (intentionally underscored — not API)
    # ------------------------------------------------------------------ #

    def _style_hit_count(self, voice_name: str) -> int:
        """How many times the cached style for ``voice_name`` was reused.

        First fetch increments this to 1; each subsequent synthesize() with
        the same voice bumps it again. Used by tests to verify we don't
        re-fetch on every call.
        """
        return self._style_hits.get(voice_name, 0)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_style(self, voice_name: str) -> Any:
        cached = self._style_cache.get(voice_name)
        if cached is None:
            # Validate against our catalog first so the error message is
            # consistent with `voices.get_voice` rather than supertonic's
            # raw FileNotFoundError.
            _voices.get_voice(voice_name, self._cache_dir)
            logger.info("loading voice style: %s", voice_name)
            cached = self._tts.get_voice_style(voice_name=voice_name)
            self._style_cache[voice_name] = cached
        self._style_hits[voice_name] = self._style_hits.get(voice_name, 0) + 1
        return cached
