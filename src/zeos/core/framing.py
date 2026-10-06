# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The kernel's frames: the tags it puts around its own text (MP §5.3).

A frame is carried on ``CONTROL`` tokens, which a machine cannot decode unless the
kernel enables it, so the frame is what carries authority. Ordinary tokens that spell
a frame tag are an imitation: inert, alarmed on as a spoof fault, and shown escaped
to a source that reads its context as text.

An imitation is an opening -- ``<``, or the JSON escape ``\\u003c`` or ``\\x3c`` that a
tool result encoded as JSON carries in its place -- then an optional ``/`` and a name in
``FRAMES``, wherever it sits in the text, and not continued by a character a tag name
may hold (an ASCII letter, digit, ``_``, ``-``, ``.`` or ``:``). So ``"<KERNEL>``,
``\\n<FAULT kind=x>``, ``foo<STATUS>`` and ``\\u003cKERNEL\\u003e`` are imitations, and so
is text ending in ``<FAULT``, whose attributes follow; ``<KERNELS>``, ``<STUBBORN>`` and
``&lt;KERNEL&gt;`` are not.

The names are matched in two ways, because they are not equally rare in real data.

``KERNEL``, ``RESUME`` and ``FAULT`` (``FOLDED_FRAMES``) almost never appear as bare
tags in what tools return, and a model reads ``<kernel>`` much as it reads
``<KERNEL>``. They are also matched in the text's fold (``fold``), which a reader would
take for the same letters: every character a reader does not see dropped (the
zero-width spaces and joiners, the soft hyphen, the combining grapheme joiner,
variation selectors, Hangul fillers, the information separators U+001C-U+001F and NEL,
combining accents); every other one through its compatibility decomposition, upper-cased,
and with look-alikes from Cyrillic, Greek, Coptic, Armenian, Cherokee, Lisu, Runic,
Canadian Syllabics and the Latin small capitals read as the Latin letters they imitate,
and ``‹ 〈 ⟨`` and their partners read as ``<`` and ``>``. So ``<kernel>``, ``<ReSuMe x=1>``,
``<KER\\u200bNEL>``, ``＜ＫＥＲＮＥＬ＞``, ``<КERNEL>`` with a Cyrillic ``К``, ``<ᴋᴇʀɴᴇʟ>`` in
small capitals and ``‹FAULT›`` are imitations; ``<kernels>``, ``<faultcode>`` and
``<soap:Fault>`` are not.

``STATUS`` and ``STUB`` are matched case-sensitively, in the text and in the text with
only the characters a reader does not see dropped (``strip``): the kernel writes its
frames in capitals, and lower-case ``<status>`` or ``<stub>`` is ordinary markup in the
XML and API data tools return, so alarming on it would make the alarm noise and escape
real data. ``<ST\\u200bATUS>`` is an imitation; ``<status>`` and ``<ЅTATUS>`` with a
Cyrillic ``Ѕ`` are not.

**The same answer under every Python.** The fold is a table generated once and committed
(``zeos.core._fold_table``, regenerated with ``uv run python -m tests.fold.regenerate``),
and the patterns use ASCII character classes only, so nothing here asks this Python's
``unicodedata`` or ``str.upper``: CPython and Pyodide, whose Unicode versions differ, raise
the same alarms and write the same journal. ``FOLD_UNICODE_VERSION`` is the version the
table was generated from.

The alarm is advisory. What a job may do is set by its capabilities and its integrity,
which no text can change: text can persuade; only the kernel can permit. Persuasion
that spells no tag at all (``SYSTEM OVERRIDE: ...``) is outside this rule by design.

**Across tokens.** Tokens are split at whitespace, and every machine writes whitespace
before a token that does not begin with its own, so ``<KER`` then ``NEL>`` reach the
model as ``<KER NEL>``. But ``str.split`` also splits at characters a reader does not
see (U+001C-U+001F, NEL), which a token keeping its whitespace carries in front of it,
so a run of ordinary tokens is matched as the text a machine writes for it
(``zeos.machine.base.render``), and every token an imitation touches is escaped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from zeos.core import _fold_table
from zeos.core.ids import TokenKind
from zeos.machine.base import Token, render

__all__ = [
    "FOLDED_FRAMES",
    "FOLD_UNICODE_VERSION",
    "FRAMES",
    "OPENINGS",
    "fold",
    "frame_tokens",
    "imitates_frame",
    "shown",
    "shown_words",
    "spells_frame",
    "strip",
]

#: The frames also matched in folded text: whatever their case, invisible characters
#: or look-alike letters.
FOLDED_FRAMES: tuple[str, ...] = ("KERNEL", "RESUME", "FAULT")

FRAMES: tuple[str, ...] = (*FOLDED_FRAMES, "STATUS", "STUB")

#: What opens an imitation: the bracket, or a JSON escape of it.
OPENINGS: tuple[str, ...] = ("<", "\\u003c", "\\u003C", "\\x3c", "\\x3C")

#: The Unicode version ``fold``'s table was generated from.
FOLD_UNICODE_VERSION: str = _fold_table.UNICODE_VERSION


def _load() -> dict[int, str]:
    table: dict[int, str] = {}
    for low, high in _fold_table.DROPPED:
        for codepoint in range(low, high + 1):
            table[codepoint] = ""
    for entry in "".join(_fold_table.MAPPED).split(";"):
        code, folded = entry.split(":")
        table[int(code, 16)] = "".join(chr(int(c, 16)) for c in folded.split(","))
    return table


_FOLD: dict[int, str] = _load()
_STRIP: dict[int, str] = {c: "" for c, folded in _FOLD.items() if not folded}
_SAME: dict[int, str] = {}

_NAME = "(?:" + "|".join(FRAMES) + ")"

#: The kernel's own frame, which begins its text.
_OPENER = re.compile(r"^</?" + _NAME + r"(?=[\s>]|$)")

#: A frame tag anywhere in stripped text: the name not continued as a longer tag name.
_TAG = re.compile(r"(?:<|\\u003[cC]|\\x3[cC])/?" + _NAME + r"(?![A-Za-z0-9_.:-])")

#: The same for a folded frame, in folded text, where every ASCII letter is a capital.
_FOLDED_TAG = re.compile(
    r"(?:<|\\U003C|\\X3C)/?(?:" + "|".join(FOLDED_FRAMES) + r")(?![A-Z0-9_.:-])"
)

#: A JSON escape of a bracket, which ``shown`` writes as the entity a bracket is.
_ESCAPED_BRACKET = re.compile(r"\\u003[cCeE]|\\x3[cCeE]")


def fold(text: str) -> str:
    """``text`` as the folded match reads it, a character at a time (module docstring).

    Folding is per character, so the fold of a concatenation is the concatenation of
    the folds: ``<`` followed by a combining mark folds to ``<``, as does ``≮``.
    """
    return text.translate(_FOLD)


def strip(text: str) -> str:
    """``text`` with only the characters a reader does not see dropped: what the exact
    match reads."""
    return text.translate(_STRIP)


def spells_frame(text: str) -> bool:
    """Whether text spells a frame tag, opening or closing, anywhere in it: any of
    ``FRAMES`` in the text or its ``strip``, or any of ``FOLDED_FRAMES`` in its ``fold``.
    The text as it is counts too, so ``<KERNEL`` followed by a zero-width space and an
    ``S`` is an imitation: the model reads the name end at the invisible character."""
    return (
        _TAG.search(text) is not None
        or _TAG.search(strip(text)) is not None
        or _FOLDED_TAG.search(fold(text)) is not None
    )


def frame_tokens(text: str) -> tuple[Token, ...]:
    """Tokenise one kernel frame: its outer tags as ``CONTROL``, everything inside as
    ``NORMAL``.

    The opening tag runs from the first word to the first word ending in ``>``, so
    ``<STATUS obj>`` and ``<FAULT kind=x>`` are whole tags; the closing tag is the last
    word when it closes the same frame. Only those two are the kernel's. A word inside the
    body that spells a tag, a device value or a quoted request say, stays ``NORMAL``, so
    the kernel never promotes an imitation into a frame of its own.
    """
    words = text.split()
    opener = _OPENER.match(words[0]) if words else None
    if opener is None:
        return tuple(Token(w, TokenKind.NORMAL) for w in words)
    name = opener.group(0).lstrip("</")
    end = next((i for i, w in enumerate(words) if w.endswith(">")), len(words) - 1)
    kinds = [TokenKind.NORMAL] * len(words)
    for i in range(end + 1):
        kinds[i] = TokenKind.CONTROL
    if len(words) - 1 > end and words[-1] == f"</{name}>":
        kinds[-1] = TokenKind.CONTROL
    return tuple(Token(w, k) for w, k in zip(words, kinds, strict=True))


def _runs(tokens: Sequence[Token]) -> list[range]:
    """The index ranges of the runs of consecutive ordinary tokens."""
    runs: list[range] = []
    start: int | None = None
    for index, token in enumerate(tokens):
        if token.kind is TokenKind.NORMAL:
            if start is None:
                start = index
        elif start is not None:
            runs.append(range(start, index))
            start = None
    if start is not None:
        runs.append(range(start, len(tokens)))
    return runs


def imitates_frame(tokens: Iterable[Token]) -> bool:
    """Whether ordinary tokens spell a frame tag (``spells_frame``), each run of them
    read as the text a machine writes for it."""
    words = tuple(tokens)
    return any(spells_frame(render(words[run.start : run.stop])) for run in _runs(words))


def _touched(run: Sequence[Token]) -> set[int]:
    """The tokens of a run that an imitation in its text touches."""
    starts: list[int] = []
    pieces: list[str] = []
    at = 0
    for k, token in enumerate(run):
        piece = token.text if k == 0 or token.text[:1].isspace() else " " + token.text
        starts.append(at + len(piece) - len(token.text))
        pieces.append(piece)
        at += len(piece)
    text = "".join(pieces)
    spans: list[tuple[int, int]] = []
    for table, pattern in ((_SAME, _TAG), (_STRIP, _TAG), (_FOLD, _FOLDED_TAG)):
        origin: list[int] = []
        out: list[str] = []
        for index, char in enumerate(text):
            mapped = table.get(ord(char), char)
            out.append(mapped)
            origin.extend([index] * len(mapped))
        for match in pattern.finditer("".join(out)):
            spans.append((origin[match.start()], origin[match.end() - 1] + 1))
    touched: set[int] = set()
    for k, token in enumerate(run):
        begin, end = starts[k], starts[k] + len(token.text)
        if any(s < end and begin < e for s, e in spans):
            touched.add(k)
    return touched


def shown_words(tokens: Sequence[Token]) -> tuple[str, ...]:
    """Tokens as a text-only source sees them: a frame as it is, an imitation escaped.

    An ordinary token is escaped exactly when an imitation that ``imitates_frame``
    alarms on touches it, and then every character in it that folds to ``<`` or ``>``
    (``＜`` and ``‹`` as well as ``<``) and every JSON escape of one is written as the
    entity, so nothing alarmed on reaches the source as a tag, and its escaped form is
    not alarmed on again.
    """
    out = [token.text for token in tokens]
    for run in _runs(tokens):
        words = tokens[run.start : run.stop]
        if not spells_frame(render(words)):
            continue
        for k in _touched(words):
            out[run.start + k] = _escaped(words[k].text)
    return tuple(out)


def shown(token: Token) -> str:
    """One token as ``shown_words`` shows it on its own."""
    return shown_words((token,))[0]


def _escaped(text: str) -> str:
    text = _ESCAPED_BRACKET.sub(lambda m: "&lt;" if m.group(0)[-1] in "cC" else "&gt;", text)
    return "".join(map(_escaped_char, text))


def _escaped_char(char: str) -> str:
    folded = _FOLD.get(ord(char), char)
    if "<" in folded:
        return "&lt;"
    if ">" in folded:
        return "&gt;"
    return char
