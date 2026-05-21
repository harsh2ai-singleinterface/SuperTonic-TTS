"""Tests for the streaming text coalescer."""

from __future__ import annotations

import pytest

from supertonic_server.coalescer import TextCoalescer


# ---------------------------------------------------------------- basic cases


def test_single_sentence_one_push() -> None:
    c = TextCoalescer()
    assert c.push("Hello world.") == ["Hello world."]
    assert c.pending_chars == 0
    assert c.flush() == []


def test_no_boundary_then_flush() -> None:
    c = TextCoalescer()
    assert c.push("partial") == []
    assert c.pending_chars == len("partial")
    assert c.flush() == ["partial"]
    assert c.pending_chars == 0


def test_empty_flush() -> None:
    c = TextCoalescer()
    assert c.flush() == []


def test_empty_push() -> None:
    c = TextCoalescer()
    assert c.push("") == []
    assert c.pending_chars == 0


# ---------------------------------------------------- token-by-token streaming


def test_token_by_token_preserves_text() -> None:
    src = "Hello, world. How are you?"
    c = TextCoalescer()
    out: list[str] = []
    for ch in src:
        out.extend(c.push(ch))
    out.extend(c.flush())
    assert "".join(out) == src
    # Should have produced at least two chunks (period then question mark).
    assert len(out) >= 2


def test_token_by_token_longer_paragraph() -> None:
    src = (
        "This is the first sentence. And here comes the second one, "
        "with a clause boundary too! Finally, a question? Yes."
    )
    c = TextCoalescer()
    out: list[str] = []
    for ch in src:
        out.extend(c.push(ch))
    out.extend(c.flush())
    assert "".join(out) == src


# ----------------------------------------------------------- multi-sentence


def test_multiple_sentences_one_push() -> None:
    c = TextCoalescer()
    chunks = c.push("One. Two. Three.")
    assert len(chunks) == 3
    assert "".join(chunks) == "One. Two. Three."
    # First two chunks should carry the trailing space; last one shouldn't.
    assert chunks[0].rstrip().endswith(".")
    assert chunks[-1].endswith(".")
    assert c.flush() == []


def test_question_and_exclamation_are_sentence_final() -> None:
    c = TextCoalescer()
    chunks = c.push("Really? Yes! Done.")
    assert len(chunks) == 3
    assert "".join(chunks) == "Really? Yes! Done."


# ------------------------------------------------------------ soft-min logic


def test_soft_min_blocks_short_comma_clause() -> None:
    """`"Hi,"` is way under 40 chars; comma must NOT fire."""
    c = TextCoalescer()
    chunks = c.push("Hi, there.")
    assert chunks == ["Hi, there."]


def test_soft_min_allows_long_comma_clause() -> None:
    """Once buffer length passes soft_min, a comma should fire."""
    c = TextCoalescer(soft_min_chars=20)
    src = "This is long enough indeed, and continues on."
    # 27 chars up to and including the comma — clears the 20-char soft min.
    chunks = c.push(src)
    # Expect two chunks: one ending at the comma, one ending at the period.
    assert len(chunks) == 2
    assert chunks[0].rstrip().endswith(",")
    assert chunks[1].rstrip().endswith(".")
    assert "".join(chunks) == src


def test_semicolon_and_colon_behave_like_comma() -> None:
    c = TextCoalescer(soft_min_chars=10)
    chunks = c.push("Listen carefully; this matters: indeed.")
    # `;` after 17 chars (>10) fires; `:` after that fires too; `.` at end fires.
    assert len(chunks) == 3
    assert "".join(chunks) == "Listen carefully; this matters: indeed."


# --------------------------------------------------------------- hard-max


def test_hard_max_force_flush_at_whitespace() -> None:
    """A long run of words with no punctuation force-flushes at whitespace."""
    # 250 chars: 50 repeats of "word " is 250 chars exactly.
    src = "word " * 50
    assert len(src) == 250
    c = TextCoalescer(soft_min_chars=40, hard_max_chars=200)
    chunks = c.push(src)
    leftover = c.flush()
    all_out = chunks + leftover
    # Should have at least one chunk emitted before flush.
    assert len(chunks) >= 1
    # First emitted chunk must fit inside the hard cap.
    assert len(chunks[0]) <= 200
    # And must have broken at a whitespace (so it doesn't slice a word).
    assert chunks[0].endswith(" ")
    # Concatenation preserves the original text.
    assert "".join(all_out) == src


def test_hard_max_force_flush_no_whitespace() -> None:
    """If a single huge word exceeds hard_max, we cut at the cap itself."""
    src = "a" * 250
    c = TextCoalescer(soft_min_chars=40, hard_max_chars=200)
    chunks = c.push(src)
    leftover = c.flush()
    assert chunks == ["a" * 200]
    assert leftover == ["a" * 50]


# --------------------------------------------------------- decimal protection


def test_decimal_dot_does_not_fire() -> None:
    c = TextCoalescer()
    chunks = c.push("Rate is 9.24 percent today.")
    assert chunks == ["Rate is 9.24 percent today."]


def test_decimal_dot_with_hindi_does_not_fire() -> None:
    c = TextCoalescer()
    chunks = c.push("रेट 9.24 प्रतिशत है।")
    assert len(chunks) == 1
    assert chunks[0].endswith("।")


# ---------------------------------------------------- phone-number protection


def test_phone_number_comma_does_not_split_inside_digits() -> None:
    c = TextCoalescer(soft_min_chars=10)
    chunks = c.push("Call me at 9876543210, please.")
    leftover = c.flush()
    all_out = chunks + leftover
    # No chunk should end on the comma that's glued to the phone number.
    for ch in all_out:
        stripped = ch.rstrip()
        if stripped.endswith(","):
            # If a comma chunk exists, the token before it must not be all digits.
            token = stripped[:-1].rsplit(" ", 1)[-1]
            assert not (token.isdigit() and len(token) >= 6), (
                f"Comma fired inside phone number: {ch!r}"
            )
    # We should eventually emit on the period.
    assert any(ch.rstrip().endswith(".") for ch in all_out)
    # And we should preserve the original text.
    assert "".join(all_out) == "Call me at 9876543210, please."


def test_postal_code_comma_protection() -> None:
    c = TextCoalescer(soft_min_chars=10)
    # 6-digit postal code followed by a comma deep into the buffer.
    src = "Please send it to 400053, Mumbai shortly."
    chunks = c.push(src)
    leftover = c.flush()
    all_out = chunks + leftover
    assert "".join(all_out) == src
    # Comma straight after a 6-digit run shouldn't split.
    for ch in all_out:
        s = ch.rstrip()
        if s.endswith(","):
            tok = s[:-1].rsplit(" ", 1)[-1]
            assert not tok.isdigit() or len(tok) < 6


# ---------------------------------------------------------------- multi-script


def test_hindi_sentence_one_push() -> None:
    c = TextCoalescer()
    chunks = c.push("नमस्ते, मैं प्रिया हूँ।")
    assert len(chunks) == 1
    assert chunks[0].endswith("।")
    assert "".join(chunks) == "नमस्ते, मैं प्रिया हूँ।"


def test_mixed_script_two_sentences() -> None:
    c = TextCoalescer()
    src = "नंबर 9876543210 है। धन्यवाद।"
    chunks = c.push(src)
    assert len(chunks) == 2
    assert chunks[0].rstrip().endswith("।")
    assert chunks[1].rstrip().endswith("।")
    assert "".join(chunks) == src


def test_hindi_question_and_exclamation() -> None:
    c = TextCoalescer()
    chunks = c.push("क्या यह सही है? हाँ! बिल्कुल।")
    assert len(chunks) == 3
    assert "".join(chunks) == "क्या यह सही है? हाँ! बिल्कुल।"


# ----------------------------------------------------------------- pending_chars


def test_pending_chars_tracks_buffer() -> None:
    c = TextCoalescer()
    assert c.pending_chars == 0
    c.push("abc")
    assert c.pending_chars == 3
    c.push("def")
    assert c.pending_chars == 6
    # Boundary fires, buffer drains.
    c.push(".")
    assert c.pending_chars == 0
    # Re-fill.
    c.push("xyz")
    assert c.pending_chars == 3
    c.flush()
    assert c.pending_chars == 0


def test_pending_chars_after_partial_emit() -> None:
    c = TextCoalescer()
    c.push("First. Second")  # `First.` emits + " " absorbed; "Second" stays.
    assert c.pending_chars == len("Second")


# ----------------------------------------------------------------- misc / API


def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError):
        TextCoalescer(soft_min_chars=-1)
    with pytest.raises(ValueError):
        TextCoalescer(hard_max_chars=0)
    with pytest.raises(ValueError):
        TextCoalescer(soft_min_chars=100, hard_max_chars=50)


def test_default_thresholds() -> None:
    c = TextCoalescer()
    assert c.soft_min_chars == 40
    assert c.hard_max_chars == 200


def test_trailing_whitespace_attaches_to_chunk() -> None:
    """After a sentence-final period followed by spaces, the spaces ride along
    with the emitted chunk so the next chunk doesn't start with a stray space.
    """
    c = TextCoalescer()
    chunks = c.push("Done.   And then more")
    assert len(chunks) == 1
    assert chunks[0] == "Done.   "
    assert c.pending_chars == len("And then more")


def test_state_resets_after_flush() -> None:
    c = TextCoalescer()
    c.push("buffered without boundary")
    leftover = c.flush()
    assert leftover == ["buffered without boundary"]
    # After flush we should be fresh again.
    assert c.pending_chars == 0
    assert c.push("Hello.") == ["Hello."]
