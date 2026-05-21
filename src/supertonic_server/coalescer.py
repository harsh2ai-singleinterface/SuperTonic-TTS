"""Buffer streamed LLM text and emit synthesis-ready chunks at clause boundaries.

LLMs stream text token-by-token, but Supertonic has fixed per-synth overhead
and shapes prosody across whatever text it sees in one call. So we buffer
incoming tokens and only release a chunk to the TTS worker when we hit a
clause boundary: a sentence-final mark, a "long enough" clause-final mark,
or a hard character cap.

This module is pure: no I/O, no asyncio, no logging, stdlib only.
"""

from __future__ import annotations

# Sentence-final punctuation across the scripts we care about.
# `।` (U+0964, Devanagari danda) is the Hindi full stop.
_SENTENCE_FINAL: frozenset[str] = frozenset({".", "?", "!", "।"})

# Clause-final punctuation: triggers an emit *only* once the buffer is long
# enough that the resulting chunk is worth synthesizing on its own.
_CLAUSE_FINAL: frozenset[str] = frozenset({",", ";", ":"})

_BOUNDARY: frozenset[str] = _SENTENCE_FINAL | _CLAUSE_FINAL


class TextCoalescer:
    """Buffer streamed text; emit chunks at clause boundaries.

    Boundary policy: emit when ONE of:
      - sentence-final punctuation (`.`, `?`, `!`, `।`)
      - clause-final punctuation (`,`, `;`, `:`) AND buffer length
        (up to and including the punctuation) > ``soft_min_chars``
      - hard limit: buffer length > ``hard_max_chars`` (force-flush at the
        last whitespace if any, otherwise at the limit itself)

    After a boundary fires the chunk text (including the terminating
    punctuation plus any trailing whitespace) is emitted; everything
    after it stays buffered for the next call.

    On :meth:`flush` (LLM signaled done), whatever remains is emitted,
    even if it doesn't end at a boundary.

    Decimal protection: a punctuation mark with a digit on *both* sides
    (e.g. the dot in ``9.24``) is not treated as a boundary.

    Phone-number protection: if the chunk that would be emitted ends inside
    a "mostly-digit" trailing token (>=60% digits in the last 8 chars),
    we wait for at least one more push before honouring that boundary.
    """

    __slots__ = ("_buf", "soft_min_chars", "hard_max_chars")

    def __init__(
        self,
        soft_min_chars: int = 40,
        hard_max_chars: int = 200,
    ) -> None:
        if soft_min_chars < 0:
            raise ValueError("soft_min_chars must be >= 0")
        if hard_max_chars <= 0:
            raise ValueError("hard_max_chars must be > 0")
        if hard_max_chars < soft_min_chars:
            raise ValueError("hard_max_chars must be >= soft_min_chars")
        self.soft_min_chars: int = soft_min_chars
        self.hard_max_chars: int = hard_max_chars
        self._buf: str = ""

    # ------------------------------------------------------------------ API

    def push(self, text: str) -> list[str]:
        """Append ``text`` to the buffer; return any chunks now ready.

        May return zero, one, or many chunks depending on what's already in
        the buffer and how many boundaries the new text crosses.
        """
        if text:
            self._buf += text
        chunks: list[str] = []
        while True:
            cut = self._find_boundary(self._buf)
            if cut is None:
                break
            chunk, rest = self._buf[:cut], self._buf[cut:]
            # Carry over trailing whitespace with the emitted chunk so the
            # next chunk doesn't start with a stray space.
            i = 0
            while i < len(rest) and rest[i].isspace():
                i += 1
            if i:
                chunk += rest[:i]
                rest = rest[i:]
            self._buf = rest
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        """Drain whatever is left in the buffer (the LLM stream has ended)."""
        if not self._buf:
            return []
        leftover = self._buf
        self._buf = ""
        return [leftover]

    @property
    def pending_chars(self) -> int:
        """Length of the in-flight buffer (for metrics / backpressure)."""
        return len(self._buf)

    # ------------------------------------------------------- internal helpers

    def _find_boundary(self, buf: str) -> int | None:
        """Return the cut index (exclusive end of the chunk to emit), or None.

        Scans ``buf`` left-to-right looking for the first qualifying boundary.
        If none is found but the buffer exceeds ``hard_max_chars``, picks a
        force-flush point at the last whitespace within the cap (or at the
        cap itself).
        """
        n = len(buf)
        for i, ch in enumerate(buf):
            if ch not in _BOUNDARY:
                continue

            # Decimal / inside-number protection: digit on both sides.
            if (
                0 < i < n - 1
                and buf[i - 1].isdigit()
                and buf[i + 1].isdigit()
            ):
                continue

            cut = i + 1  # include the punctuation in the emitted chunk

            if ch in _SENTENCE_FINAL:
                if self._ends_in_digit_run(buf[:cut]):
                    # Wait — we're still inside a mostly-numeric token.
                    # Skip this boundary; maybe the next char in a later push
                    # will be something we can safely emit on.
                    continue
                return cut

            # Otherwise it's clause-final: only fire if the prospective
            # chunk is long enough to be worth synthesizing.
            if cut <= self.soft_min_chars:
                continue
            if self._ends_in_digit_run(buf[:cut]):
                continue
            return cut

        # No qualifying boundary found. Force-flush only if we've blown
        # past the hard cap.
        if n > self.hard_max_chars:
            # Prefer a whitespace split inside the cap so we don't slice
            # a word in half.
            window = buf[: self.hard_max_chars]
            ws = window.rfind(" ")
            if ws > 0:
                return ws + 1  # include the space with the emitted chunk
            return self.hard_max_chars

        return None

    @staticmethod
    def _ends_in_digit_run(s: str) -> bool:
        """Heuristic: does ``s`` end inside a mostly-numeric token?

        We look at the trailing token (chars back to the last whitespace)
        and check its final up-to-8 chars. If >=60% are digits, we treat it
        as a number-in-progress and suppress the boundary.
        """
        if not s:
            return False
        # Trailing token = chars after the last whitespace.
        tail_start = max(s.rfind(" "), s.rfind("\t"), s.rfind("\n")) + 1
        tail = s[tail_start:]
        if not tail:
            return False
        window = tail[-8:] if len(tail) > 8 else tail
        digits = sum(1 for c in window if c.isdigit())
        return (digits / len(window)) >= 0.6
