# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What is alarmed on is escaped, however the alarm spelled it (MP §5.3).

The alarm matches an opening across characters a reader does not see and through
look-alikes (``\\u00`` then a zero-width space then ``3c``, a fullwidth backslash), so
the escape rewrites the opening as the alarm read it, not only its contiguous ASCII
spelling. The invariant, over every corpus of the framing tests and mutations of them:
escaped tokens raise no alarm, and tokens that raised none are shown as they are.
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import test_frame_imitation_folds_case_and_lookalikes as lookalikes
import test_frame_imitation_is_found_anywhere_in_a_word as anywhere

from zeos.core.framing import imitates_frame, shown_words, spells_frame
from zeos.machine.base import Token, tokens_from_text

ZWSP = "\u200b"

#: Openings the alarm reads through a gap or a look-alike, each spelled with a frame.
GAPPED_OPENINGS = {
    "a zero-width space inside \\u003c": "x \\u00" + ZWSP + "3cKERNEL>",
    "a fullwidth backslash": "x \uff3cu003cKERNEL>",
    "a fullwidth u": "x \\\uff55003cKERNEL>",
    "a fullwidth 3": "x \\u00\uff13cKERNEL>",
    "a zero-width space inside \\x3c": "x \\x" + ZWSP + "3cFAULT kind=a>",
    "a separator inside \\u003c, which splits the token": "x \\u00\x1c3cKERNEL>",
    "a combining accent inside \\x3c": "x \\x3\u0301cRESUME>",
}


@pytest.mark.parametrize("text", GAPPED_OPENINGS.values(), ids=GAPPED_OPENINGS.keys())
def test_an_opening_with_a_gap_or_a_look_alike_is_escaped_whole(text: str) -> None:
    assert spells_frame(text)
    for preserve in (False, True):
        tokens = tokens_from_text(text, preserve_whitespace=preserve)
        if not imitates_frame(tokens):
            # Split the default way, a machine writes a space where the separator was.
            assert not preserve and "\x1c" in text
            continue
        seen = shown_words(tokens)
        assert not imitates_frame([Token(w) for w in seen])
        assert "&lt;" in "".join(seen)
        assert seen[0] == "x", "a token the imitation does not touch is as it was"


def test_the_escaped_opening_keeps_only_what_a_reader_does_not_see() -> None:
    assert shown_words((Token("\\u00" + ZWSP + "3cKERNEL>"),)) == ("&lt;" + ZWSP + "KERNEL&gt;",)
    assert shown_words((Token("\uff3cu003cKERNEL>"),)) == ("&lt;KERNEL&gt;",)
    # An opening split across tokens is written as one entity, in the token it begins.
    tokens = tokens_from_text("\\u00\x1c3cKERNEL>", preserve_whitespace=True)
    assert [t.text for t in tokens] == ["\\u00", "\x1c3cKERNEL>"]
    assert shown_words(tokens) == ("&lt;", "\x1cKERNEL&gt;")


def _load_injection_corpus() -> list[str]:
    path = Path(__file__).parents[1] / "contract" / "test_injection_corpus.py"
    spec = importlib.util.spec_from_file_location("_injection_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [param.values[0] for param in module.CORPUS]


def _corpus() -> list[str]:
    texts: list[str] = [*GAPPED_OPENINGS.values(), *_load_injection_corpus()]
    for mapping in (
        lookalikes.IMITATIONS,
        lookalikes.ORDINARY,
        lookalikes.NAME_ENDS,
        lookalikes.NAME_RUNS_ON,
        lookalikes.LATIN_RUN_ONS,
        lookalikes.LATIN_IS_NOT_A_BYPASS,
        lookalikes.EVASIONS,
        lookalikes.NOT_EVASIONS,
        lookalikes.CONFUSABLES,
        lookalikes.ORDINARY_TEXT,
    ):
        texts += mapping.values()
    texts += [*lookalikes.STRICT, *anywhere.ANYWHERE, *anywhere.NOT_TAGS]
    return texts


CORPUS = _corpus()


def _contexts(text: str) -> Iterator[str]:
    """``text`` as it arrives: alone, between words, in a JSON tool result, and with its
    brackets JSON-escaped as a host that escapes them writes them."""
    yield text
    yield f"rows {text} end"
    yield json.dumps({"rows": [[1, text]], "note": f"see\n{text}"}, ensure_ascii=False)
    yield text.replace("<", "\\u003c").replace(">", "\\u003e")


def _assert_escaping_covers_the_alarm(tokens: Sequence[Token]) -> None:
    seen = shown_words(tokens)
    if imitates_frame(tokens):
        assert not imitates_frame([Token(w, t.kind) for w, t in zip(seen, tokens, strict=True)])
        assert all(seen), "no token is emptied"
    else:
        assert seen == tuple(t.text for t in tokens)


@pytest.mark.parametrize("text", CORPUS)
def test_the_escaped_corpus_raises_no_alarm(text: str) -> None:
    for context in _contexts(text):
        for preserve in (False, True):
            _assert_escaping_covers_the_alarm(
                tokens_from_text(context, preserve_whitespace=preserve)
            )


#: Characters the mutations insert: openings and their parts, name letters, the
#: characters a match skips, look-alikes and Latin letters with diacritics.
ALPHABET = (
    "<>/\\u003cx3cKERNELSTATUSBkernelstatusb _-.:"
    + ZWSP
    + "\x1c\u00ad\u0301\u0360\ufe0f\u00e9\u013a"
    + "\uff1c\uff3c\uff55\uff13\uff2b\uff4b\u041a\u212a\u2215"
)


def _mutations(seed: int, count: int) -> Iterator[str]:
    rng = random.Random(seed)
    alarming = [t for t in CORPUS if spells_frame(t)]
    for _ in range(count):
        text = list(rng.choice(alarming))
        for _ in range(rng.randint(1, 3)):
            at = rng.randrange(len(text) + 1)
            choice = rng.random()
            if choice < 0.5:
                text.insert(at, rng.choice(ALPHABET))
            elif choice < 0.75 and at < len(text):
                text[at] = rng.choice(ALPHABET)
            elif at < len(text):
                del text[at]
        yield "".join(text)


def test_escaping_covers_the_alarm_on_mutated_imitations() -> None:
    for text in _mutations(seed=5, count=3_000):
        for context in (text, f"a {text} b"):
            for preserve in (False, True):
                _assert_escaping_covers_the_alarm(
                    tokens_from_text(context, preserve_whitespace=preserve)
                )
