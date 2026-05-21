"""Voice catalog for the Supertonic streaming-TTS server.

Lists the voice-style JSON files supertonic ships with (F1..F5, M1..M5) from
its on-disk cache, so callers can enumerate or resolve them by name without
loading the heavy TTS model.

Stdlib-only on purpose — `worker.py` and the HTTP catalog endpoint both
import this and we don't want a numpy / supertonic dependency just to list
filenames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Default location supertonic-3 writes voice styles to. Mirrors the value
# of `TTS(auto_download=False).model_dir / "voice_styles"`.
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "supertonic3" / "voice_styles"

# Matches the shipped voice filenames: F1.json..F5.json, M1.json..M5.json,
# and any future F<n>/M<n> style names. Anything else in the dir is ignored.
_VOICE_FILE_RE = re.compile(r"^(?P<gender>[FM])(?P<num>\d+)\.json$")


@dataclass(frozen=True, slots=True)
class Voice:
    """A single Supertonic voice style on disk.

    Attributes:
        name:   Voice identifier (e.g. ``"F1"``, ``"M3"``).
        gender: ``"F"`` or ``"M"``.
        path:   Absolute path to the voice-style JSON file.
    """

    name: str
    gender: str
    path: Path


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    return DEFAULT_CACHE_DIR if cache_dir is None else cache_dir


def list_voices(cache_dir: Path | None = None) -> list[Voice]:
    """List voices from supertonic's voice-style cache.

    Default ``cache_dir`` is ``~/.cache/supertonic3/voice_styles/``.
    Only files named ``[FM][0-9]+.json`` are returned. Result is sorted by
    gender (F before M) then by numeric suffix. A missing cache directory
    yields an empty list — callers can treat that as "no voices available"
    rather than an error.
    """
    root = _resolve_cache_dir(cache_dir)
    if not root.is_dir():
        return []

    voices: list[Voice] = []
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        m = _VOICE_FILE_RE.match(entry.name)
        if m is None:
            continue
        voices.append(
            Voice(
                name=f"{m.group('gender')}{int(m.group('num'))}",
                gender=m.group("gender"),
                path=entry.resolve(),
            )
        )

    voices.sort(key=lambda v: (v.gender, int(v.name[1:])))
    return voices


def get_voice(name: str, cache_dir: Path | None = None) -> Voice:
    """Return the named voice or raise ``KeyError`` listing what's available."""
    available = list_voices(cache_dir)
    for v in available:
        if v.name == name:
            return v

    avail_names = ", ".join(v.name for v in available) or "<none>"
    raise KeyError(
        f"voice {name!r} not found in {_resolve_cache_dir(cache_dir)} "
        f"(available: {avail_names})"
    )
