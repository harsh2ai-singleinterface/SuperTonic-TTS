"""Tests for :mod:`supertonic_server.voices`.

Stdlib-only module so these tests are fast — no marker needed.
"""

from __future__ import annotations

import pytest

from supertonic_server.voices import Voice, get_voice, list_voices


def test_list_voices_default_cache_has_all_ten() -> None:
    """Dev box ships F1..F5 + M1..M5 under ~/.cache/supertonic3/voice_styles/."""
    voices = list_voices()
    assert len(voices) >= 10, f"expected >=10 voices, got {len(voices)}: {voices}"
    names = {v.name for v in voices}
    for expected in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
        assert expected in names, f"missing voice {expected!r}; got {sorted(names)}"


def test_list_voices_sorted_gender_then_number() -> None:
    voices = list_voices()
    # All F voices come before any M voice; within a gender, ascending number.
    f_names = [v.name for v in voices if v.gender == "F"]
    m_names = [v.name for v in voices if v.gender == "M"]
    assert f_names == sorted(f_names, key=lambda n: int(n[1:]))
    assert m_names == sorted(m_names, key=lambda n: int(n[1:]))
    # And concatenation order matches the returned list.
    assert [v.name for v in voices][: len(f_names)] == f_names


def test_get_voice_f1() -> None:
    v = get_voice("F1")
    assert isinstance(v, Voice)
    assert v.name == "F1"
    assert v.gender == "F"
    assert v.path.is_file()
    assert v.path.name == "F1.json"


def test_get_voice_m3() -> None:
    v = get_voice("M3")
    assert v.name == "M3"
    assert v.gender == "M"
    assert v.path.is_file()


def test_get_voice_unknown_raises_with_available_names() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_voice("ZZZ")
    msg = str(excinfo.value)
    # Must surface the bad name and at least one real voice for the user.
    assert "ZZZ" in msg
    assert "F1" in msg or "M1" in msg, f"error should list available names, got: {msg}"


def test_list_voices_empty_dir(tmp_path) -> None:
    assert list_voices(tmp_path) == []


def test_get_voice_empty_dir(tmp_path) -> None:
    with pytest.raises(KeyError) as excinfo:
        get_voice("F1", tmp_path)
    # Empty-cache message should still mention what's available ("<none>").
    assert "F1" in str(excinfo.value)


def test_list_voices_ignores_non_voice_files(tmp_path) -> None:
    (tmp_path / "F1.json").write_text("{}")
    (tmp_path / "M2.json").write_text("{}")
    (tmp_path / "README.md").write_text("not a voice")
    (tmp_path / "garbage.json").write_text("{}")
    voices = list_voices(tmp_path)
    assert [v.name for v in voices] == ["F1", "M2"]
