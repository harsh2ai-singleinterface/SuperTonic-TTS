"""Cartesia-shaped WebSocket wire protocol: encode/decode JSON frames.

This module is a pure leaf — no I/O, no asyncio, no logging, no imports from
other `supertonic_server.*` modules. The rest of the server passes typed
dataclasses; only this module knows the JSON keys.

The protocol mirrors Cartesia's `/tts/websocket` API exactly, because the
intended client is Pipecat's :class:`pipecat.services.cartesia.tts.CartesiaTTSService`,
which we want to use unchanged.

Outgoing-from-client (we decode) messages are observed in Pipecat's
``CartesiaTTSService._build_msg`` and ``on_audio_context_interrupted``:

* Text/flush::

    {
      "transcript": "<text or empty>",
      "continue": True/False,
      "context_id": "<ctx>",
      "model_id": "sonic-3",
      "voice": {"mode": "id", "id": "<voice_id>"},
      "output_format": {"container": "raw", "encoding": "pcm_s16le",
                        "sample_rate": 24000},
      "add_timestamps": True,
      "use_original_timestamps": True,
      "language": "en",                  # optional
      "generation_config": {...},        # optional
      "pronunciation_dict_id": "...",    # optional
    }

  An empty transcript with ``continue=False`` is Pipecat's flush signal
  (see ``CartesiaTTSService.flush_audio``).

* Cancel::

    {"context_id": "<ctx>", "cancel": True}

Outgoing-to-client (we encode) messages are consumed in
``CartesiaTTSService._process_messages``:

* Audio chunk: ``{"type": "chunk", "context_id": "<ctx>", "data": "<b64>"}``
  where ``data`` is base64-encoded int16-little-endian PCM at the
  ``output_format.sample_rate`` Pipecat requested.
* Done: ``{"type": "done", "context_id": "<ctx>"}``
* Word timestamps: ``{"type": "timestamps", "context_id": "<ctx>",
  "word_timestamps": {"words": [...], "start": [...]}}``
* Error: ``{"type": "error", "context_id": "<ctx>", "error": "<msg>"}``
  (Pipecat surfaces the whole message in the log, so any extra keys are OK;
  we keep ``context_id`` and ``error`` for clarity.)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Union

__all__ = [
    "OutputFormat",
    "VoiceConfig",
    "IncomingText",
    "IncomingCancel",
    "ParseError",
    "IncomingMessage",
    "decode",
    "encode_chunk",
    "encode_done",
    "encode_error",
    "encode_timestamps",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputFormat:
    """Audio output format requested by the client.

    Fields match Cartesia's ``output_format`` block exactly.
    """

    container: str  # e.g. "raw"
    encoding: str  # e.g. "pcm_s16le"
    sample_rate: int  # e.g. 24000


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Voice selection block.

    Cartesia uses ``{"mode": "id", "id": "<voice_id>"}``. We only support
    ``mode == "id"`` since that's all Pipecat ever sends.
    """

    mode: str
    id: str


@dataclass(frozen=True, slots=True)
class IncomingText:
    """A text-synthesis frame from the client.

    A frame with ``transcript == ""`` and ``continue_ is False`` is a flush
    request (see ``CartesiaTTSService.flush_audio``): finalize the context,
    do not synthesize new audio.
    """

    context_id: str
    transcript: str
    continue_: bool
    voice: VoiceConfig
    output_format: OutputFormat
    model_id: str | None = None
    language: str | None = None
    add_timestamps: bool = False
    use_original_timestamps: bool = False
    generation_config: Mapping[str, Any] | None = None
    pronunciation_dict_id: str | None = None
    # Any keys we didn't recognize, preserved for debugging.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_flush(self) -> bool:
        """True if this frame is a flush (empty transcript, continue=False)."""
        return self.transcript == "" and not self.continue_


@dataclass(frozen=True, slots=True)
class IncomingCancel:
    """A cancellation frame from the client.

    Pipecat sends ``{"context_id": "<ctx>", "cancel": True}`` from
    ``on_audio_context_interrupted``.
    """

    context_id: str


@dataclass(frozen=True, slots=True)
class ParseError:
    """A decoded message that couldn't be parsed.

    Returned (not raised) so callers can decide whether to send an error
    frame back, close the connection, or just log and continue.
    """

    reason: str
    raw: str  # truncated raw payload for diagnostics


IncomingMessage = Union[IncomingText, IncomingCancel, ParseError]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


_MAX_RAW_PREVIEW: Final[int] = 512


def _truncate(s: str) -> str:
    return s if len(s) <= _MAX_RAW_PREVIEW else s[:_MAX_RAW_PREVIEW] + "...<truncated>"


def _err(reason: str, raw: str) -> ParseError:
    return ParseError(reason=reason, raw=_truncate(raw))


def decode(raw: str | bytes) -> IncomingMessage:
    """Parse a single client frame.

    Returns a typed message on success or a :class:`ParseError` on any
    failure. Never raises (other than for genuinely catastrophic stdlib
    bugs).
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            return _err(f"invalid utf-8: {e}", raw.decode("utf-8", errors="replace"))
    else:
        text = raw

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return _err(f"invalid json: {e}", text)

    if not isinstance(obj, dict):
        return _err("top-level json must be an object", text)

    # Cancel takes priority — it's a small dedicated message.
    if obj.get("cancel") is True:
        ctx = obj.get("context_id")
        if not isinstance(ctx, str) or not ctx:
            return _err("cancel message missing context_id", text)
        return IncomingCancel(context_id=ctx)

    # Otherwise, treat it as a text/flush frame.
    return _decode_text(obj, text)


def _decode_text(obj: dict[str, Any], raw: str) -> IncomingMessage:
    # Required keys per Pipecat's _build_msg.
    ctx = obj.get("context_id")
    if not isinstance(ctx, str) or not ctx:
        return _err("missing or empty context_id", raw)

    transcript = obj.get("transcript")
    if not isinstance(transcript, str):
        return _err("missing or non-string transcript", raw)

    cont = obj.get("continue")
    if not isinstance(cont, bool):
        return _err("missing or non-bool 'continue'", raw)

    voice_raw = obj.get("voice")
    if not isinstance(voice_raw, dict):
        return _err("missing or non-object 'voice'", raw)
    voice_mode = voice_raw.get("mode")
    voice_id = voice_raw.get("id")
    if not isinstance(voice_mode, str) or not isinstance(voice_id, str):
        return _err("voice requires string 'mode' and 'id'", raw)
    voice = VoiceConfig(mode=voice_mode, id=voice_id)

    of_raw = obj.get("output_format")
    if not isinstance(of_raw, dict):
        return _err("missing or non-object 'output_format'", raw)
    container = of_raw.get("container")
    encoding = of_raw.get("encoding")
    sr = of_raw.get("sample_rate")
    if not isinstance(container, str) or not isinstance(encoding, str):
        return _err("output_format requires string 'container' and 'encoding'", raw)
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        return _err("output_format.sample_rate must be a positive int", raw)
    output_format = OutputFormat(container=container, encoding=encoding, sample_rate=sr)

    # Optional fields.
    model_id = obj.get("model_id")
    if model_id is not None and not isinstance(model_id, str):
        return _err("model_id must be string if present", raw)

    language = obj.get("language")
    if language is not None and not isinstance(language, str):
        return _err("language must be string if present", raw)

    add_ts = obj.get("add_timestamps", False)
    if not isinstance(add_ts, bool):
        return _err("add_timestamps must be bool if present", raw)

    use_orig_ts = obj.get("use_original_timestamps", False)
    if not isinstance(use_orig_ts, bool):
        return _err("use_original_timestamps must be bool if present", raw)

    gen_cfg = obj.get("generation_config")
    if gen_cfg is not None and not isinstance(gen_cfg, dict):
        return _err("generation_config must be object if present", raw)

    pron_dict = obj.get("pronunciation_dict_id")
    if pron_dict is not None and not isinstance(pron_dict, str):
        return _err("pronunciation_dict_id must be string if present", raw)

    known = {
        "context_id",
        "transcript",
        "continue",
        "voice",
        "output_format",
        "model_id",
        "language",
        "add_timestamps",
        "use_original_timestamps",
        "generation_config",
        "pronunciation_dict_id",
        "cancel",  # never set on a text frame, but ignore just in case
    }
    extra = {k: v for k, v in obj.items() if k not in known}

    return IncomingText(
        context_id=ctx,
        transcript=transcript,
        continue_=cont,
        voice=voice,
        output_format=output_format,
        model_id=model_id,
        language=language,
        add_timestamps=add_ts,
        use_original_timestamps=use_orig_ts,
        generation_config=gen_cfg,
        pronunciation_dict_id=pron_dict,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_chunk(context_id: str, pcm_int16_bytes: bytes) -> str:
    """Encode a chunk of raw int16-LE PCM as a Cartesia ``chunk`` frame.

    The caller is responsible for matching the sample rate the client asked
    for in its ``output_format``. We don't carry that here — chunks have no
    sample-rate field; the contract is "same rate the client requested".

    Args:
        context_id: Context this chunk belongs to.
        pcm_int16_bytes: Raw PCM bytes, int16 little-endian. Length must be
            a multiple of 2; we don't enforce that here because Pipecat
            doesn't either, but callers should respect it.

    Returns:
        JSON-encoded frame ready to send over the WebSocket.
    """
    if not isinstance(pcm_int16_bytes, (bytes, bytearray)):
        raise TypeError("pcm_int16_bytes must be bytes-like")
    payload = {
        "type": "chunk",
        "context_id": context_id,
        "data": base64.b64encode(bytes(pcm_int16_bytes)).decode("ascii"),
    }
    return json.dumps(payload)


def encode_done(context_id: str) -> str:
    """Encode the terminal ``done`` frame for a context.

    Pipecat treats ``done`` as the signal to drop the active context.
    """
    return json.dumps({"type": "done", "context_id": context_id})


def encode_error(context_id: str | None, message: str) -> str:
    """Encode an ``error`` frame.

    ``context_id`` may be ``None`` if the error is connection-level rather
    than tied to one context. Pipecat tolerates extra keys; we only set
    ``context_id`` when it's known to avoid spurious context lookups.
    """
    payload: dict[str, Any] = {"type": "error", "error": message}
    if context_id is not None:
        payload["context_id"] = context_id
    return json.dumps(payload)


def encode_timestamps(
    context_id: str, words: list[str], starts: list[float]
) -> str:
    """Encode a ``timestamps`` frame.

    Pipecat reads ``msg["word_timestamps"]["words"]`` and
    ``msg["word_timestamps"]["start"]``. We mirror that shape.
    """
    if len(words) != len(starts):
        raise ValueError("words and starts must have equal length")
    return json.dumps(
        {
            "type": "timestamps",
            "context_id": context_id,
            "word_timestamps": {"words": list(words), "start": list(starts)},
        }
    )


# Helper to decode the base64 PCM back out, exposed for tests and for any
# loopback / dev tools the server might grow.
def _decode_chunk_data(b64: str) -> bytes:
    """Reverse of :func:`encode_chunk` for the data field. Internal."""
    return base64.b64decode(b64)
