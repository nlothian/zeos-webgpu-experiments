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

An imitation is ``<NAME`` or ``</NAME``, for a name in ``FRAMES``, wherever it sits in a
token's text, and not continued by a character a tag name may hold (a letter, digit,
``_``, ``-``, ``.`` or ``:``). So ``"<KERNEL>``, ``\\n<FAULT kind=x>`` and ``foo<STATUS>``
are imitations, as a tool result delivered as JSON carries them, and so is a token ending
in ``<FAULT``, whose attributes follow in the next token; ``<KERNELS>``, ``<STUBBORN>``
and ``&lt;KERNEL&gt;`` are not.

The names are matched in two ways, because they are not equally rare in real data.

``KERNEL``, ``RESUME`` and ``FAULT`` (``FOLDED_FRAMES``) almost never appear as bare
tags in what tools return, and a model reads ``<kernel>`` much as it reads
``<KERNEL>``. They are also matched in the token's folded text (``fold``): invisible
format characters (Unicode category Cf: the zero-width space and joiners, the word
joiner, the byte-order mark, the soft hyphen) removed, each character
NFKC-normalised, so ``＜ＫＥＲＮＥＬ＞`` reads as ``<KERNEL>``, a few Cyrillic and Greek
look-alikes of the letters in those names mapped to Latin (``_HOMOGLYPHS``), and the
result upper-cased. So ``<kernel>``, ``<ReSuMe x=1>``, ``<KER\\u200bNEL>``,
``<\\u200bKERNEL>`` and ``<КERNEL>`` with a Cyrillic ``К`` are imitations; the run-on
rule applies to the folded text, so ``<kernels>``, ``<faultcode>`` and ``<soap:Fault>``
are not.

``STATUS`` and ``STUB`` are matched case-sensitively, in the text as it is: the kernel
writes its frames in capitals, and lower-case ``<status>`` or ``<stub>`` is ordinary
markup in the XML and API data tools return, so alarming on it would make the alarm
noise and escape real data.

The alarm is advisory. What a job may do is set by its capabilities and its integrity,
which no text can change: text can persuade; only the kernel can permit. Persuasion
that spells no tag at all (``SYSTEM OVERRIDE: ...``) is outside this rule by design.

A tag never spans two tokens. Tokens are split at whitespace, and every machine writes
whitespace before a token that does not begin with its own, so ``<KER`` then ``NEL>``
reach the model as ``<KER NEL>``. What follows a tag in the next token is therefore
whitespace in the model's view, which is why a tag ending its token counts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from zeos.core.ids import TokenKind
from zeos.machine.base import Token

__all__ = [
    "FOLDED_FRAMES",
    "FRAMES",
    "fold",
    "frame_tokens",
    "imitates_frame",
    "shown",
    "spells_frame",
]

#: The frames also matched in folded text: whatever their case, invisible characters
#: or look-alike letters.
FOLDED_FRAMES: tuple[str, ...] = ("KERNEL", "RESUME", "FAULT")

FRAMES: tuple[str, ...] = (*FOLDED_FRAMES, "STATUS", "STUB")

#: Cyrillic and Greek letters that look like a Latin letter of a ``FOLDED_FRAMES`` name,
#: in either case, as that Latin capital. Applied after NFKC, so the Kelvin sign and the
#: kappa symbol arrive here as ``K`` and ``κ``.
_HOMOGLYPHS: dict[str, str] = {
    "\u0410": "A", "\u0430": "A", "\u0391": "A", "\u03b1": "A",  # А а Α α
    "\u0415": "E", "\u0435": "E", "\u0395": "E", "\u03b5": "E",  # Е е Ε ε
    "\u03dc": "F",  # Ϝ
    "\u041a": "K", "\u043a": "K", "\u039a": "K", "\u03ba": "K",  # К к Κ κ
    "\u04c0": "L", "\u04cf": "L",  # Ӏ ӏ
    "\u041c": "M", "\u043c": "M", "\u039c": "M",  # М м Μ
    "\u039d": "N", "\u043f": "N", "\u03b7": "N",  # Ν п η
    "\u0433": "R",  # г
    "\u0405": "S", "\u0455": "S",  # Ѕ ѕ
    "\u0422": "T", "\u0442": "T", "\u03a4": "T", "\u03c4": "T",  # Т т Τ τ
    "\u03c5": "U",  # υ
}  # fmt: skip

_NAME = "(?:" + "|".join(FRAMES) + ")"

#: The kernel's own frame, which begins its text.
_OPENER = re.compile(r"^</?" + _NAME + r"(?=[\s>]|$)")

#: A frame tag anywhere in a token: the name not continued as a longer tag name.
_TAG = re.compile(r"</?" + _NAME + r"(?![\w.:-])")

#: The same for a folded frame, in folded text.
_FOLDED_TAG = re.compile(r"</?(?:" + "|".join(FOLDED_FRAMES) + r")(?![\w.:-])")


def _fold_char(char: str) -> str:
    if unicodedata.category(char) == "Cf":
        return ""
    return "".join(_HOMOGLYPHS.get(c, c) for c in unicodedata.normalize("NFKC", char)).upper()


class _FoldTable(dict[int, str]):
    """``_fold_char`` by code point, for ``str.translate``, filled as characters are met."""

    def __missing__(self, codepoint: int) -> str:
        folded = self[codepoint] = _fold_char(chr(codepoint))
        return folded


_FOLD = _FoldTable()


def fold(text: str) -> str:
    """``text`` as the folded match reads it, a character at a time: an invisible
    format character (category Cf) dropped, each other one NFKC-normalised, a
    ``_HOMOGLYPHS`` letter made Latin, and the whole upper-cased.

    Folding is per character, so the fold of a concatenation is the concatenation of
    the folds, and ``<`` followed by a combining mark is not recomposed into ``≮``.
    """
    if text.isascii():
        return text.upper()
    return text.translate(_FOLD)


def spells_frame(text: str) -> bool:
    """Whether a token's text spells a frame tag, opening or closing, anywhere in it:
    any of ``FRAMES`` in the text as it is, or any of ``FOLDED_FRAMES`` in its fold."""
    return _TAG.search(text) is not None or _FOLDED_TAG.search(fold(text)) is not None


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


def imitates_frame(tokens: Iterable[Token]) -> bool:
    """Whether ordinary tokens spell a frame tag (``spells_frame``)."""
    return any(t.kind is TokenKind.NORMAL and spells_frame(t.text) for t in tokens)


def shown(token: Token) -> str:
    """A token as a text-only source sees it: a frame as it is, an imitation escaped.

    An ordinary token is escaped exactly when ``imitates_frame`` alarms on it, and then
    every character that folds to ``<`` or ``>`` in it (``＜`` and ``﹤`` as well as
    ``<``), so nothing alarmed on reaches the source as a tag, and its escaped form is
    not alarmed on again.
    """
    if token.kind is not TokenKind.NORMAL or not spells_frame(token.text):
        return token.text
    return "".join(map(_escaped, token.text))


def _escaped(char: str) -> str:
    folded = _FOLD[ord(char)]
    if "<" in folded:
        return "&lt;"
    if ">" in folded:
        return "&gt;"
    return char
