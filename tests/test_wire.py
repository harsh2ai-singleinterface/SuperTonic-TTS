"""Tests for the Cartesia-shaped wire protocol module.

The "golden" fixtures below were built by reading Pipecat 0.0.x's
``pipecat.services.cartesia.tts.CartesiaTTSService._build_msg`` and
``on_audio_context_interrupted`` directly out of the venv at
``.venv/lib/python3.12/site-packages/pipecat/services/cartesia/tts.py``.
That code IS the source-of-truth for the protocol on the client side.
"""

from __future__ import annotations

import base64
import json
import random
import struct

import pytest

from supertonic_server.wire import (
    IncomingCancel,
    IncomingText,
    OutputFormat,
    ParseError,
    VoiceConfig,
    _decode_chunk_data,
    decode,
    encode_chunk,
    encode_done,
    encode_error,
    encode_timestamps,
)


# ---------------------------------------------------------------------------
# Golden fixtures — what Pipecat actually puts on the wire.
# ---------------------------------------------------------------------------


# Verbatim shape from Pipecat's _build_msg(text, continue_transcript=True,
# add_timestamps=True, context_id=...). Voice id and model swapped for a
# realistic-looking placeholder; structure is exact.
GOLDEN_TEXT_MSG = json.dumps(
    {
        "transcript": "Hello, how are you today?",
        "continue": True,
        "context_id": "ctx-abc-123",
        "model_id": "sonic-3",
        "voice": {"mode": "id", "id": "a0e99841-438c-4a64-b679-ae501e7d6091"},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "add_timestamps": True,
        "use_original_timestamps": True,
        "language": "en",
    }
)


# From CartesiaTTSService.flush_audio: same shape, empty transcript,
# continue=False.
GOLDEN_FLUSH_MSG = json.dumps(
    {
        "transcript": "",
        "continue": False,
        "context_id": "ctx-abc-123",
        "model_id": "sonic-3",
        "voice": {"mode": "id", "id": "a0e99841-438c-4a64-b679-ae501e7d6091"},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "add_timestamps": True,
        "use_original_timestamps": True,
    }
)


# Exactly the cancel_msg from on_audio_context_interrupted:
#   json.dumps({"context_id": context_id, "cancel": True})
GOLDEN_CANCEL_MSG = json.dumps({"context_id": "ctx-abc-123", "cancel": True})


# ---------------------------------------------------------------------------
# Decoder: golden fixtures.
# ---------------------------------------------------------------------------


def test_decode_golden_text_message() -> None:
    msg = decode(GOLDEN_TEXT_MSG)
    assert isinstance(msg, IncomingText)
    assert msg.context_id == "ctx-abc-123"
    assert msg.transcript == "Hello, how are you today?"
    assert msg.continue_ is True
    assert msg.voice == VoiceConfig(
        mode="id", id="a0e99841-438c-4a64-b679-ae501e7d6091"
    )
    assert msg.output_format == OutputFormat(
        container="raw", encoding="pcm_s16le", sample_rate=24000
    )
    assert msg.model_id == "sonic-3"
    assert msg.language == "en"
    assert msg.add_timestamps is True
    assert msg.use_original_timestamps is True
    assert msg.generation_config is None
    assert msg.pronunciation_dict_id is None
    assert msg.is_flush is False
    assert msg.extra == {}


def test_decode_golden_flush_message() -> None:
    msg = decode(GOLDEN_FLUSH_MSG)
    assert isinstance(msg, IncomingText)
    assert msg.transcript == ""
    assert msg.continue_ is False
    assert msg.is_flush is True


def test_decode_golden_cancel_message() -> None:
    msg = decode(GOLDEN_CANCEL_MSG)
    assert isinstance(msg, IncomingCancel)
    assert msg.context_id == "ctx-abc-123"


def test_decode_accepts_bytes() -> None:
    msg = decode(GOLDEN_TEXT_MSG.encode("utf-8"))
    assert isinstance(msg, IncomingText)
    assert msg.context_id == "ctx-abc-123"


# ---------------------------------------------------------------------------
# Decoder: optional fields.
# ---------------------------------------------------------------------------


def test_decode_with_generation_config_and_pron_dict() -> None:
    payload = json.loads(GOLDEN_TEXT_MSG)
    payload["generation_config"] = {"speed": 1.1, "emotion": "neutral"}
    payload["pronunciation_dict_id"] = "dict-42"
    msg = decode(json.dumps(payload))
    assert isinstance(msg, IncomingText)
    assert msg.generation_config == {"speed": 1.1, "emotion": "neutral"}
    assert msg.pronunciation_dict_id == "dict-42"


def test_decode_preserves_unknown_keys_in_extra() -> None:
    payload = json.loads(GOLDEN_TEXT_MSG)
    payload["future_field"] = "from-newer-pipecat"
    msg = decode(json.dumps(payload))
    assert isinstance(msg, IncomingText)
    assert msg.extra == {"future_field": "from-newer-pipecat"}


# ---------------------------------------------------------------------------
# Decoder: error paths (never raise; return ParseError).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("not json at all", "invalid json"),
        ("[]", "must be an object"),
        ("null", "must be an object"),
        ("123", "must be an object"),
        ('{"cancel": true}', "missing context_id"),
        ('{"cancel": true, "context_id": ""}', "missing context_id"),
        ('{"context_id": "x"}', "missing or non-string transcript"),
        ('{"context_id": "x", "transcript": "t"}', "non-bool 'continue'"),
    ],
)
def test_decode_returns_parse_error_on_malformed(raw: str, fragment: str) -> None:
    msg = decode(raw)
    assert isinstance(msg, ParseError)
    assert fragment in msg.reason


def test_decode_missing_voice() -> None:
    payload = json.loads(GOLDEN_TEXT_MSG)
    del payload["voice"]
    msg = decode(json.dumps(payload))
    assert isinstance(msg, ParseError)
    assert "voice" in msg.reason


def test_decode_missing_output_format() -> None:
    payload = json.loads(GOLDEN_TEXT_MSG)
    del payload["output_format"]
    msg = decode(json.dumps(payload))
    assert isinstance(msg, ParseError)
    assert "output_format" in msg.reason


def test_decode_bad_sample_rate() -> None:
    payload = json.loads(GOLDEN_TEXT_MSG)
    payload["output_format"]["sample_rate"] = 0
    msg = decode(json.dumps(payload))
    assert isinstance(msg, ParseError)
    assert "sample_rate" in msg.reason


def test_decode_truncates_long_raw_in_error() -> None:
    huge = "x" * 5000
    msg = decode(huge)
    assert isinstance(msg, ParseError)
    assert len(msg.raw) < len(huge)
    assert "truncated" in msg.raw


def test_decode_bad_utf8_bytes() -> None:
    msg = decode(b"\xff\xfe not utf-8")
    assert isinstance(msg, ParseError)


# ---------------------------------------------------------------------------
# Encoders: shape checks against what Pipecat's _process_messages reads.
# ---------------------------------------------------------------------------


def test_encode_chunk_shape() -> None:
    pcm = b"\x00\x01\x02\x03"
    frame = encode_chunk("ctx-1", pcm)
    obj = json.loads(frame)
    assert obj["type"] == "chunk"
    assert obj["context_id"] == "ctx-1"
    assert base64.b64decode(obj["data"]) == pcm
    # Pipecat reads exactly these keys; make sure we don't add weird ones.
    assert set(obj.keys()) == {"type", "context_id", "data"}


def test_encode_done_shape() -> None:
    obj = json.loads(encode_done("ctx-2"))
    assert obj == {"type": "done", "context_id": "ctx-2"}


def test_encode_error_with_context() -> None:
    obj = json.loads(encode_error("ctx-3", "synth failed"))
    assert obj["type"] == "error"
    assert obj["context_id"] == "ctx-3"
    assert obj["error"] == "synth failed"


def test_encode_error_without_context() -> None:
    obj = json.loads(encode_error(None, "connection-level boom"))
    assert obj["type"] == "error"
    assert obj["error"] == "connection-level boom"
    assert "context_id" not in obj


def test_encode_timestamps_shape() -> None:
    obj = json.loads(encode_timestamps("ctx-4", ["hi", "there"], [0.0, 0.25]))
    assert obj == {
        "type": "timestamps",
        "context_id": "ctx-4",
        "word_timestamps": {"words": ["hi", "there"], "start": [0.0, 0.25]},
    }


def test_encode_timestamps_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        encode_timestamps("ctx-4", ["a"], [0.0, 1.0])


def test_encode_chunk_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        encode_chunk("ctx", "not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trips.
# ---------------------------------------------------------------------------


def test_roundtrip_decode_then_reencode_preserves_structure() -> None:
    """Decode a text frame and re-encode the equivalent ``done`` for it."""
    msg = decode(GOLDEN_TEXT_MSG)
    assert isinstance(msg, IncomingText)
    done = json.loads(encode_done(msg.context_id))
    assert done["context_id"] == "ctx-abc-123"


def test_roundtrip_cancel() -> None:
    msg = decode(GOLDEN_CANCEL_MSG)
    assert isinstance(msg, IncomingCancel)
    # We don't expose an encoder for cancel because the server never sends
    # one — Pipecat is the only side that cancels. But we can hand-build
    # the equivalent and round-trip it through decode again.
    reraw = json.dumps({"context_id": msg.context_id, "cancel": True})
    again = decode(reraw)
    assert again == msg


def test_base64_pcm_roundtrip_random_int16_samples() -> None:
    """100 random int16 samples → encode_chunk → decode → bytes match."""
    rng = random.Random(0xC0FFEE)
    samples = [rng.randint(-32768, 32767) for _ in range(100)]
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    assert len(pcm) == 200  # int16 = 2 bytes each.

    frame = encode_chunk("ctx-pcm", pcm)
    obj = json.loads(frame)
    assert obj["type"] == "chunk"
    assert obj["context_id"] == "ctx-pcm"

    recovered = _decode_chunk_data(obj["data"])
    assert recovered == pcm

    recovered_samples = list(struct.unpack("<" + "h" * len(samples), recovered))
    assert recovered_samples == samples


def test_base64_pcm_roundtrip_empty() -> None:
    frame = encode_chunk("ctx", b"")
    obj = json.loads(frame)
    assert obj["data"] == ""
    assert _decode_chunk_data(obj["data"]) == b""


def test_incoming_text_is_frozen() -> None:
    """Sanity check — dataclasses must be immutable (frozen=True, slots=True)."""
    msg = decode(GOLDEN_TEXT_MSG)
    assert isinstance(msg, IncomingText)
    with pytest.raises((AttributeError, Exception)):
        msg.context_id = "other"  # type: ignore[misc]
