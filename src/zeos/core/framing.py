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

Matching is case-sensitive. The kernel writes its frames in capitals, so ``<KERNEL>``
is what a model has seen carry authority in its own context, and lower-case
``<status>``, ``<fault>`` or ``<stub>`` are ordinary markup in the XML and HTML that
tools return; alarming on them would make the alarm noise and escape real data.

A tag never spans two tokens. Tokens are split at whitespace, and every machine writes
whitespace before a token that does not begin with its own, so ``<KER`` then ``NEL>``
reach the model as ``<KER NEL>``. What follows a tag in the next token is therefore
whitespace in the model's view, which is why a tag ending its token counts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from zeos.core.ids import TokenKind
from zeos.machine.base import Token

__all__ = ["FRAMES", "frame_tokens", "imitates_frame", "shown", "spells_frame"]

FRAMES: tuple[str, ...] = ("KERNEL", "RESUME", "FAULT", "STATUS", "STUB")

_NAME = "(?:" + "|".join(FRAMES) + ")"

#: The kernel's own frame, which begins its text.
_OPENER = re.compile(r"^</?" + _NAME + r"(?=[\s>]|$)")

#: A frame tag anywhere in a token: the name not continued as a longer tag name.
_TAG = re.compile(r"</?" + _NAME + r"(?![\w.:-])")


def spells_frame(text: str) -> bool:
    """Whether a token's text spells a frame tag, opening or closing, anywhere in it."""
    return _TAG.search(text) is not None


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
    every ``<`` and ``>`` in it, so nothing alarmed on reaches the source as a tag.
    """
    if token.kind is TokenKind.NORMAL and spells_frame(token.text):
        return token.text.replace("<", "&lt;").replace(">", "&gt;")
    return token.text
