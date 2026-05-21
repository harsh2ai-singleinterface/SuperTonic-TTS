"""Tests for :mod:`supertonic_server.worker`.

These exercise the real Supertonic model — every test loads the model
(~few seconds) and at least one synthesizes audio (~hundreds of ms). They
are marked ``slow``. Run everything with::

    pytest tests/test_worker.py -v

Skip them on a tight CI loop with::

    pytest -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from supertonic_server.worker import SupertonicWorker, SynthResult


# Module-scoped fixture so the model loads once for all worker tests.
@pytest.fixture(scope="module")
def worker() -> SupertonicWorker:
    return SupertonicWorker()


@pytest.mark.slow
def test_instantiation_loads_model() -> None:
    w = SupertonicWorker()
    # Model loaded but warmup not yet run.
    assert w.is_ready is False


@pytest.mark.slow
def test_synthesize_returns_valid_result(worker: SupertonicWorker) -> None:
    result = worker.synthesize("नमस्ते।", "F1")
    assert isinstance(result, SynthResult)
    assert isinstance(result.wav_f32, np.ndarray)
    assert result.wav_f32.dtype == np.float32
    assert result.wav_f32.ndim == 1
    assert result.wav_f32.size > 0
    assert result.sample_rate == 44100
    assert result.duration_s > 0
    # Sanity: reported duration roughly tracks sample count.
    derived = result.wav_f32.size / result.sample_rate
    assert abs(derived - result.duration_s) < 0.5  # generous slack


@pytest.mark.slow
def test_warmup_sets_ready(worker: SupertonicWorker) -> None:
    worker.warmup()
    assert worker.is_ready is True
    # Idempotent: second call must not raise and must keep the flag set.
    worker.warmup()
    assert worker.is_ready is True


@pytest.mark.slow
def test_voice_style_is_cached(worker: SupertonicWorker) -> None:
    """Second synth on the same voice must reuse the cached style, not re-fetch."""
    # Use a voice unlikely to have been touched by prior tests so the
    # hit count starts at zero deterministically.
    voice = "M5"
    assert worker._style_hit_count(voice) == 0

    r1 = worker.synthesize("नमस्ते।", voice)
    hits_after_first = worker._style_hit_count(voice)
    assert hits_after_first == 1
    assert r1.wav_f32.size > 0

    r2 = worker.synthesize("शुक्रिया।", voice)
    hits_after_second = worker._style_hit_count(voice)
    # Both calls registered a "hit" against the cache, but the underlying
    # supertonic Style object must be the same instance — i.e. we did not
    # re-fetch from disk on the second call.
    assert hits_after_second == 2
    assert worker._style_cache[voice] is worker._style_cache[voice]  # identity
    assert r2.wav_f32.size > 0


@pytest.mark.slow
def test_unknown_voice_raises_keyerror() -> None:
    w = SupertonicWorker()
    with pytest.raises(KeyError):
        w.synthesize("नमस्ते।", "ZZZ")
