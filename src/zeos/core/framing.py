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

**Where a name ends** is where a reader sees it end, which is not always where the
``strip`` or the ``fold`` would put it. Both matches below match a name across the
characters they drop, but the name's follower is read from the text as it is. A
combining accent that a Latin letter decomposes to (U+0301 and the like, the generated
table's ``ACCENTS``) belongs to the letter it sits on, so the follower is the first
character after the last letter's accents, and then:

- a plain ASCII name character continues any name (``<kernels>``, ``<КERNELS>`` with a
  Cyrillic ``К``, ``<kerne\\u013as>``);
- so does a Latin letter with diacritics, one that decomposes to an ASCII letter and
  accents (``ACCENTED``: ``é``, ``ü``, ``ĺ``), precomposed or decomposed:
  ``<Kernelübersicht>``, ``<kernelé>`` and ``<kernel\\u0301s>`` are no imitations;
- a look-alike of a name character from another script or block (Cyrillic, Greek,
  fullwidth, the Kelvin sign) continues a name whose last letter is a look-alike too
  (``<ｋｅｒｎｅｌｓ>``), and ends one whose last letter is ASCII or accented
  (``<fault\\u2010x>`` with a hyphen look-alike, ``<kernel\\uff11>`` with a fullwidth
  digit);
- anything else ends the name, including every character a reader does not see (a
  zero-width space, a soft hyphen, a variation selector, a direction mark, a tag
  character) and every combining mark outside ``ACCENTS``.

So ``<KER\\u200bNEL\\u200bS>``, ``<STA\\u200bTUS\\u200bx>``, ``<fault\\u2010x>``,
``<kernel\\uff11>`` and ``<kernel\\u0301\\u200bs>`` are imitations, although they strip or
fold to ``<KERNELS>``, ``<STATUSx>``, ``<FAULT-X>``, ``<KERNEL1>`` and ``<KERNELS>``.

The names are matched in two ways, because they are not equally rare in real data.

``KERNEL``, ``RESUME`` and ``FAULT`` (``FOLDED_FRAMES``) almost never appear as bare
tags in what tools return, and a model reads ``<kernel>`` much as it reads
``<KERNEL>``. They are also matched in the text's fold (``fold``), which a reader would
take for the same letters: every character a reader does not see dropped (the
zero-width spaces and joiners, the soft hyphen, the combining grapheme joiner,
variation selectors, Hangul fillers, the information separators U+001C-U+001F and NEL,
combining accents); every other one through its compatibility decomposition, read as the
ASCII its prototype in Unicode's confusables (UTS #39, version 15.0.0) is, and upper-cased,
with a small supplement for what confusables does not map to ASCII (the Latin small
capitals, the angle-bracket ornaments that are the prototypes of ``〈`` and ``⟨``). ASCII
itself is only upper-cased. So ``<kernel>``, ``<ReSuMe x=1>``, ``<KER\\u200bNEL>``,
``＜ＫＥＲＮＥＬ＞``, ``<КERNEL>`` with a Cyrillic ``К``, ``<FAUłT>`` with a Polish ``ł``,
``<RESUɱE>``, ``<ᴋᴇʀɴᴇʟ>`` in small capitals and ``‹FAULT›`` are imitations;
``<kernels>``, ``<faultcode>``, ``<soap:Fault>`` and the ASCII ``<kerne1>`` are not.

``STATUS`` and ``STUB`` are matched case-sensitively, in the text with only the
characters a reader does not see dropped (``strip``): the kernel writes its
frames in capitals, and lower-case ``<status>`` or ``<stub>`` is ordinary markup in the
XML and API data tools return, so alarming on it would make the alarm noise and escape
real data. ``<ST\\u200bATUS>`` is an imitation; ``<status>`` and ``<ЅTATUS>`` with a
Cyrillic ``Ѕ`` are not.

**The same answer under every Python.** The fold is a table generated once and committed
(``zeos.core._fold_table``, regenerated with ``uv run --frozen python -m tests.fold.regenerate``),
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
(``zeos.machine.base.render``), and every token an imitation touches is escaped
(``shown_words``). The escape rewrites the opening as the match read it, gaps and
look-alikes and all (``\\u00`` then a zero-width space then ``3c`` becomes ``&lt;``), so
what is escaped is never alarmed on again.
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


def _codepoints(ranges: Iterable[tuple[int, int]]) -> frozenset[int]:
    return frozenset(c for low, high in ranges for c in range(low, high + 1))


_FOLD: dict[int, str] = _load()
_STRIP: dict[int, str] = {c: "" for c, folded in _FOLD.items() if not folded}

#: The Latin letters with diacritics (``é``, ``ĺ``), and the combining marks they
#: decompose to (U+0301 and the like): what a reader sees continue a name.
_ACCENTED: frozenset[int] = _codepoints(_fold_table.ACCENTED)
_ACCENTS: frozenset[int] = _codepoints(_fold_table.ACCENTS)

#: What a match reads for a character it drops, and in front of what other characters
#: map to. All four are C0 controls the fold drops (as ``_DROPPED``), so none survives
#: from the text itself.
#:
#: - ``_DROPPED``: a character a reader does not see.
#: - ``_ACCENT``: a combining mark of ``_ACCENTS``, which the fold drops but a reader
#:   sees as part of the letter it sits on.
#: - ``_LATIN``: in front of what a Latin letter with diacritics maps to.
#: - ``_LOOKALIKE``: in front of each character any other non-ASCII character maps to.
_DROPPED = "\x01"
_LOOKALIKE = "\x02"
_LATIN = "\x03"
_ACCENT = "\x04"


def _marked(table: dict[int, str]) -> dict[int, str]:
    """``table`` as a match reads it, with the marks above, so a match can tell where a
    name ends in the text a reader sees (``_tag``). A Latin letter with diacritics is
    marked even where ``table`` leaves it as it is."""
    marked: dict[int, str] = {c: _LATIN + table.get(c, chr(c)) for c in _ACCENTED}
    for code, mapped in table.items():
        if code in _ACCENTED:
            continue
        if not mapped:
            marked[code] = _ACCENT if code in _ACCENTS else _DROPPED
        elif code < 0x80:
            marked[code] = mapped
        else:
            marked[code] = "".join(_LOOKALIKE + c for c in mapped)
    return marked


_STRIP_MARKED: dict[int, str] = _marked(_STRIP)
_FOLD_MARKED: dict[int, str] = _marked(_FOLD)

_NAME = "(?:" + "|".join(FRAMES) + ")"

#: The kernel's own frame, which begins its text.
_OPENER = re.compile(r"^</?" + _NAME + r"(?=[\s>]|$)")

#: What a match skips: characters dropped and accents.
_SKIPPED = f"[{_DROPPED}{_ACCENT}]*"

#: Before each character of an opening or a name after its first: anything skipped,
#: then the mark of a non-ASCII character, if it is one.
_GAP = f"{_SKIPPED}[{_LOOKALIKE}{_LATIN}]?"


def _tag(openings: Sequence[str], names: Sequence[str], continues: str) -> re.Pattern[str]:
    """A frame tag anywhere in marked text: an opening (group ``opening``), an optional
    ``/`` and a name, matched across what is skipped, and not continued as a longer tag
    name.

    ``continues`` is the class of an ASCII character a tag name may hold. What follows
    the name, after any accents on its last letter, continues it when a reader would see
    the name run on: a plain ASCII character of ``continues`` or a Latin letter with
    diacritics after any name (``<kernels>``, ``<Kernelübersicht>``), and a look-alike of
    a character of ``continues`` after a name whose last letter is a look-alike too
    (``<ｋｅｒｎｅｌｓ>``). Anything else ends the name: a character dropped
    (``<KER\\u200bNEL\\u200bS>``), and a look-alike after an ASCII or accented last letter
    (``<fault\\u2010x>``, ``<kernel\\uff11>``).
    Each gap is a run of one class of character then an optional other, and each
    lookahead follows a distinct opening, so the scan stays linear.
    """

    def spelled(literal: str) -> str:
        return re.escape(literal[0]) + "".join(_GAP + re.escape(c) for c in literal[1:])

    def name(text: str) -> str:
        last = re.escape(text[-1])
        latin = f"(?!{_ACCENT}*(?:{_LATIN}|{continues}))"
        lookalike = f"(?!{_ACCENT}*(?:{_LATIN}|{_LOOKALIKE}?{continues}))"
        return (
            "".join(_GAP + re.escape(c) for c in text[:-1])
            + f"{_SKIPPED}(?:{_LOOKALIKE}{last}{lookalike}|{_LATIN}?{last}{latin})"
        )

    opening = "(?P<opening>" + "|".join(spelled(o) for o in openings) + ")"
    return re.compile(opening + f"(?:{_GAP}/)?" + "(?:" + "|".join(name(n) for n in names) + ")")


#: A frame tag in marked stripped text.
_TAG = _tag(OPENINGS, FRAMES, "[A-Za-z0-9_.:-]")

#: The same for a folded frame, in marked folded text, where every ASCII letter is a
#: capital.
_FOLDED_TAG = _tag(("<", "\\U003C", "\\X3C"), FOLDED_FRAMES, "[A-Z0-9_.:-]")

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
    ``FRAMES`` in its ``strip``, or any of ``FOLDED_FRAMES`` in its ``fold``, the name
    ending where a reader sees it end (module docstring). So ``<KERNEL`` followed by a
    zero-width space and an ``S`` is an imitation: the model reads the name end at the
    invisible character."""
    return (
        _TAG.search(text.translate(_STRIP_MARKED)) is not None
        or _FOLDED_TAG.search(text.translate(_FOLD_MARKED)) is not None
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


def _imitations(text: str) -> list[tuple[int, int, int]]:
    """Every imitation ``spells_frame`` finds in ``text``, as offsets into it: where it
    begins, where its opening ends and where it ends."""
    found: list[tuple[int, int, int]] = []
    for table, pattern in ((_STRIP_MARKED, _TAG), (_FOLD_MARKED, _FOLDED_TAG)):
        origin: list[int] = []
        out: list[str] = []
        for index, char in enumerate(text):
            mapped = table.get(ord(char), char)
            out.append(mapped)
            origin.extend([index] * len(mapped))
        for match in pattern.finditer("".join(out)):
            opened = origin[match.end("opening") - 1] + 1
            found.append((origin[match.start()], opened, origin[match.end() - 1] + 1))
    return found


def shown_words(tokens: Sequence[Token]) -> tuple[str, ...]:
    """Tokens as a text-only source sees them: a frame as it is, an imitation escaped.

    An ordinary token is escaped exactly when an imitation that ``imitates_frame``
    alarms on touches it. The imitation's opening, however it is spelled (``<``, ``＜``,
    ``\\u003c``, ``\\u00`` and a zero-width space then ``3c``), is written as the entity
    ``&lt;``, keeping only the characters in it a reader does not see; and every other
    character in the token that folds to ``<`` or ``>`` and every JSON escape of one is
    written as the entity too. So no alarmed imitation reaches the source as a tag, and
    its escaped form is not alarmed on again.
    """
    out = [token.text for token in tokens]
    for run in _runs(tokens):
        words = tokens[run.start : run.stop]
        if not spells_frame(render(words)):
            continue
        starts: list[int] = []
        pieces: list[str] = []
        at = 0
        for k, token in enumerate(words):
            piece = token.text if k == 0 or token.text[:1].isspace() else " " + token.text
            starts.append(at + len(piece) - len(token.text))
            pieces.append(piece)
            at += len(piece)
        found = _imitations("".join(pieces))
        for k, token in enumerate(words):
            begin, end = starts[k], starts[k] + len(token.text)
            if any(s < end and begin < e for s, _, e in found):
                openings = [(s - begin, o - begin) for s, o, _ in found if s < end and begin < o]
                out[run.start + k] = _escaped(token.text, openings)
    return tuple(out)


def shown(token: Token) -> str:
    """One token as ``shown_words`` shows it on its own."""
    return shown_words((token,))[0]


def _escaped(text: str, openings: Iterable[tuple[int, int]]) -> str:
    """A token escaped (``shown_words``), with the openings of the imitations in it at
    these offsets, which may run past either end of it."""
    pieces: list[str] = []
    at = 0
    for begin, end in sorted(openings):
        if begin >= at:
            pieces += (_escaped_brackets(text[at:begin]), "&lt;")
        hidden = text[max(begin, at) : end]
        pieces += (c for c in hidden if _STRIP_MARKED.get(ord(c)) == _DROPPED)
        at = max(at, end)
    pieces.append(_escaped_brackets(text[at:]))
    return "".join(pieces)


def _escaped_brackets(text: str) -> str:
    text = _ESCAPED_BRACKET.sub(lambda m: "&lt;" if m.group(0)[-1] in "cC" else "&gt;", text)
    return "".join(map(_escaped_char, text))


def _escaped_char(char: str) -> str:
    folded = _FOLD.get(ord(char), char)
    if "<" in folded:
        return "&lt;"
    if ">" in folded:
        return "&gt;"
    return char
