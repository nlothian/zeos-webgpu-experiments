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

import json

import pytest

from zeos.core.framing import FOLDED_FRAMES, fold, imitates_frame, shown, spells_frame
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
    # ``<`` and a combining overlay are not recomposed into the single character ``≮``.
    assert fold("≮") == "≮"


def test_folded_matching_is_linear_in_the_text() -> None:
    assert not spells_frame("<​KERNÉ" * 100_000 + "<kernels")
    assert spells_frame("<​KERNÉ" * 100_000 + "<kernel")
