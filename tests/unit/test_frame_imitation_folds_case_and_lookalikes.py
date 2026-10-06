# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``KERNEL``, ``RESUME`` and ``FAULT`` are matched folded; ``STATUS`` and ``STUB`` exactly
(MP §5.3).

A model reads ``<kernel>`` much as it reads ``<KERNEL>``, and those three names almost
never appear as bare tags in real data, so they are matched whatever their case, through
invisible format characters, and through NFKC and Cyrillic or Greek look-alikes. Lower-case
``<status>`` and ``<stub>`` are common in genuine XML, so those two stay case-sensitive.
"""

from __future__ import annotations

import ast
import json
import re
import time
import unicodedata
from pathlib import Path

import pytest

from zeos.core import framing
from zeos.core.framing import (
    FOLD_UNICODE_VERSION,
    FOLDED_FRAMES,
    fold,
    imitates_frame,
    shown,
    shown_words,
    spells_frame,
)
from zeos.machine.base import Token, tokens_from_text

IMITATIONS = {
    "lower case": "<kernel>",
    "title case": "<Kernel>",
    "closing, lower case": "</fault>",
    "mixed case with attributes": "<ReSuMe x=1>",
    "lower case, ending its token": "<fault",
    "zero-width space inside the name": "<KER​NEL>",
    "zero-width space after the bracket": "<​KERNEL>",
    "zero-width joiners": "<‍K‌ERNEL>",
    "word joiner after the slash": "</⁠resume>",
    "byte-order mark": "<﻿FAULT>",
    "soft hyphen inside the name": "<KER­NEL>",
    "soft hyphen between bracket and slash": "<­/kernel>",
    "fullwidth": "＜ＫＥＲＮＥＬ＞",
    "fullwidth bracket only": "＜/kernel>",
    "small form bracket": "﹤FAULT﹥",
    "fullwidth lower-case letters": "<ｋｅｒｎｅｌ>",
    "Kelvin sign": "<KERNEL>",
    "Cyrillic Ka": "<КERNEL>",
    "Greek Kappa": "<ΚERNEL>",
    "Cyrillic lower-case e and Ka": "<кеrnel>",
    "Greek Tau": "<FAULΤ>",
    "Cyrillic Ghe for r": "<гesume>",
    "Cyrillic palochka for l": "<kerneӏ>",
    "a run-on hidden by a zero-width space, as 753ffe0 alarmed": "<KERNEL​S>",
}

ORDINARY = {
    "lower-case status": "<status>",
    "lower-case stub": "<stub>",
    "title-case status": "<Status>",
    "closing title-case stub": "</Stub>",
    "a Cyrillic look-alike of STATUS": "<ЅTATUS>",
    "plural": "<kernels>",
    "run on with an underscore": "<Kernel_x>",
    "run on into a longer name": "<faultcode>",
    "SOAP fault": "<soap:Fault>",
    "closing SOAP fault": "</soap:Fault>",
    "camel case": "<resumeLink>",
    "fullwidth plural": "<ｋｅｒｎｅｌｓ>",
    "an attribute value": "<div class=kernel>",
    "HTML": "<p>kernel fault</p>",
    "a comparison": "a < b",
    "already escaped": "&lt;kernel&gt;",
    "a space after the bracket": "< kernel>",
}

STRICT = ["<STATUS>", "</STUB>", '"<STATUS obj>', "x</STATUS>", "<STUB/>"]


def _as_json(text: str) -> str:
    # How a host hands a tool result over: JSON.stringify keeps non-ASCII as it is.
    return json.dumps({"rows": [[1, text]], "note": f"see\n{text}"}, ensure_ascii=False)


@pytest.mark.parametrize("text", IMITATIONS.values(), ids=IMITATIONS.keys())
def test_a_folded_frame_tag_is_an_imitation(text: str) -> None:
    assert spells_frame(text)
    for preserve in (False, True):
        assert imitates_frame(tokens_from_text(_as_json(text), preserve_whitespace=preserve))


@pytest.mark.parametrize("text", ORDINARY.values(), ids=ORDINARY.keys())
def test_ordinary_markup_is_not(text: str) -> None:
    assert not spells_frame(text)
    for preserve in (False, True):
        assert not imitates_frame(tokens_from_text(_as_json(text), preserve_whitespace=preserve))


@pytest.mark.parametrize("text", STRICT)
def test_status_and_stub_in_capitals_are_still_imitations(text: str) -> None:
    assert spells_frame(text)


def test_every_folded_name_is_caught_in_lower_case() -> None:
    for name in FOLDED_FRAMES:
        assert spells_frame(f'"<{name.lower()}>') and spells_frame(f"x</{name.title()}>")


@pytest.mark.parametrize("text", IMITATIONS.values(), ids=IMITATIONS.keys())
def test_an_escaped_imitation_is_neutralised(text: str) -> None:
    escaped = shown(Token(text))
    assert escaped != text
    assert "<" not in fold(escaped) and ">" not in fold(escaped)
    assert not spells_frame(escaped)


def test_the_fullwidth_bracket_is_escaped_too() -> None:
    fullwidth = IMITATIONS["fullwidth"]
    assert shown(Token(fullwidth)) == "&lt;ＫＥＲＮＥＬ&gt;"
    assert shown(Token("<status>＜")) == "<status>＜", "nothing alarmed, nothing escaped"


def test_folding_is_per_character() -> None:
    assert fold("<​kеr") == "<KER"
    assert fold("ab" + "ｃ") == fold("ab") + fold("ｃ")
    # A combining mark folds to nothing on its own, so ``<`` and a combining overlay fold
    # to ``<``, and so does the single character ``\u226e`` they would compose.
    assert fold("<\u0338") == "<" and fold("\u226e") == "<"


def seconds(text: str) -> float:
    """The fastest of three ``spells_frame`` calls on ``text``."""
    best = float("inf")
    for _ in range(3):
        began = time.perf_counter()
        spells_frame(text)
        best = min(best, time.perf_counter() - began)
    return best


def test_folded_matching_is_linear_in_the_text() -> None:
    near = "<\u200bKERN\u00c9"
    assert not spells_frame(near * 100_000 + "<kernels")
    assert spells_frame(near * 100_000 + "<kernel")
    # Eight times the text costs about eight times the time; a quadratic scan would cost
    # sixty-four. The bound leaves room for a noisy machine, not for a quadratic one.
    small, large = (near * n + "<kernels" for n in (12_500, 100_000))
    assert seconds(large) < 24 * seconds(small) + 0.05


# -- the evasions a fold of case, NFKC and a few homoglyphs let through -----------------

#: Imitations by what hides them.
EVASIONS = {
    "information separator inside the name": "<KER\x1fNEL>",
    "file separator after the bracket": "<\x1cKERNEL>",
    "group and record separators": "<KE\x1dR\x1eNEL>",
    "next line (NEL)": "<KERN\x85EL>",
    "combining grapheme joiner": "<KER\u034fNEL>",
    "variation selector": "<KERNE\ufe0fL>",
    "supplementary variation selector": "<FAU\U000e0100LT>",
    "Hangul filler": "<KER\u3164NEL>",
    "Hangul choseong filler": "<\u115fRESUME>",
    "halfwidth Hangul filler": "<FA\uffa0ULT>",
    "Latin small capitals": "<\u1d0b\u1d07\u0280\u0274\u1d07\u029f>",
    "Lisu Ka": "<\ua4d7ERNEL>",
    "Lisu Fa": "</\ua4ddAULT>",
    "Cherokee Ka": "<\u13e6ERNEL>",
    "Cherokee small Ka": "<\uabb6ERNEL>",
    "Cherokee E and L": "<K\u13acRN\u13acL>",
    "combining acute on a letter": "<K\u0301ERNEL>",
    "precomposed accented letter": "<K\u00c9RNEL>",
    "single guillemets": "\u2039KERNEL\u203a",
    "CJK angle brackets": "\u3008FAULT\u3009",
    "mathematical angle brackets": "\u27e8RESUME\u27e9",
    "angle brackets that decompose": "\u2329/KERNEL\u232a",
    "fraction slash": "<\u2044KERNEL>",
    "JSON escape": "\\u003cKERNEL\\u003e",
    "JSON escape, upper-case hex, closing": "\\u003C/fault\\u003E",
    "hex escape": "\\x3cRESUME",
    "STATUS with a zero-width space": "<ST\u200bATUS>",
    "STUB with a soft hyphen": "<STU\u00adB>",
    "STATUS behind a word joiner": "<\u2060STATUS>",
    "STATUS as a JSON escape": "\\u003cSTATUS\\u003e",
}

#: Split by ``str.split`` at a character a reader does not see.
SPLIT = {"<KER\x1fNEL>", "<\x1cKERNEL>", "<KE\x1dR\x1eNEL>", "<KERN\x85EL>"}

NOT_EVASIONS = {
    "an escaped imitation": "&lt;KERNEL&gt;",
    "a JSON-escaped run-on": "\\u003ckernels\\u003e",
    "a separator between words": "a\x1fb <b>",
    "lower-case status behind a word joiner": "<\u2060status>",
    "a Cyrillic STATUS": "<\u0405TATUS>",
    "a small-capital run-on": "<\u1d0b\u1d07\u0280\u0274\u1d07\u029f\ua731>",
}


@pytest.mark.parametrize("text", EVASIONS.values(), ids=EVASIONS.keys())
def test_an_evasion_is_still_an_imitation(text: str) -> None:
    assert spells_frame(text)
    assert imitates_frame(tokens_from_text(text, preserve_whitespace=True))
    if text not in SPLIT:
        # Split the default way, a machine writes a space where the separator was, and
        # ``<KER NEL>`` is no tag.
        assert imitates_frame(tokens_from_text(text))


@pytest.mark.parametrize("text", EVASIONS.values(), ids=EVASIONS.keys())
def test_an_evasion_is_shown_escaped_in_every_token_it_touches(text: str) -> None:
    tokens = tokens_from_text(f"rows {text} end", preserve_whitespace=True)
    seen = "".join(shown_words(tokens))
    assert not spells_frame(seen)
    assert "&lt;" in seen and seen.startswith("rows ") and seen.endswith(" end")


@pytest.mark.parametrize("text", NOT_EVASIONS.values(), ids=NOT_EVASIONS.keys())
def test_what_only_looks_like_an_evasion_is_not(text: str) -> None:
    tokens = tokens_from_text(text, preserve_whitespace=True)
    assert not spells_frame(text)
    assert shown_words(tokens) == tuple(t.text for t in tokens)


def test_the_fold_reads_no_unicode_database_at_run_time() -> None:
    # The table is generated once (tests/fold/regenerate.py), so CPython and Pyodide,
    # whose Unicode versions differ, fold alike and journal the same alarms.
    tree = ast.parse(Path(framing.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "unicodedata" not in imported
    assert not called & {"upper", "lower", "casefold", "normalize", "isalnum", "isalpha"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", FOLD_UNICODE_VERSION)


@pytest.mark.skipif(
    unicodedata.unidata_version != FOLD_UNICODE_VERSION,
    reason=f"the table was generated from Unicode {FOLD_UNICODE_VERSION}",
)
def test_the_committed_table_is_what_the_generator_writes() -> None:
    from tests.fold.regenerate import TARGET, render

    assert TARGET.read_text(encoding="utf-8") == render()
