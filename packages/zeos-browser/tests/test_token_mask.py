# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The token mask: the language the llama seat's grammar admits, walked one piece at a time."""

from __future__ import annotations

import pytest
from zeos.machine.abi import DEFAULT, SyscallABI, Verb
from zeos.machine.base import OpKind

from zeos_browser.token_mask import CommandLanguage, TokenMask

#: counter-a's bindings: a stdin to sleep on, a stdout to wake its peer, an actuator.
COUNTER = CommandLanguage(DEFAULT, ("stdin", "stdout", "tools"), valued=("tools",))


def admits(text: str, language: CommandLanguage = COUNTER) -> bool:
    return bool(language.advance(language.start, text))


@pytest.mark.parametrize(
    "text",
    [
        "say 1; say 2; write tools 10;",
        " write stdout go;",
        "read stdin;",
        "exit;",
        "write tools 0;",
        "write tools 123456789;",
        "say " + "x" * 16 + ";",
        "say",  # a prefix is admitted: the round is still going
        "write st",
    ],
)
def test_rounds_the_grammar_admits(text: str) -> None:
    assert admits(text)


@pytest.mark.parametrize(
    "text",
    [
        "write tools 010;",  # no leading zeros, so a job cannot write 000000000
        "write tools 1234567890;",  # ten digits
        "write tools go;",  # an actuator takes a number
        "write stdout ;",  # a payload has at least one character
        "say " + "x" * 17 + ";",  # max_text
        "say <STATUS>;",  # a job cannot type an angle bracket
        "say 1;say 2;",  # the next command opens with a space
        "exit; say 1;",  # nothing follows the call that closes a round
        "  say 1;",  # one opening space, not two
        "shout 1;",  # an undeclared verb
        "write peer 1;",  # an alias this descriptor does not bind
    ],
)
def test_rounds_the_grammar_refuses(text: str) -> None:
    assert not admits(text)


def test_a_job_with_no_stdin_is_offered_no_read() -> None:
    language = CommandLanguage(DEFAULT, ("stdout",))
    assert not admits("read stdout;", language)
    assert admits("write stdout done;", language)


def test_a_language_with_no_call_is_refused() -> None:
    """As ``build_grammar`` refuses one: a round must be able to end."""
    abi = SyscallABI(verbs=(Verb("say", text=True), Verb("write", OpKind.WRITE, pipe=True)))
    with pytest.raises(ValueError, match="no call"):
        CommandLanguage(abi, ())


def test_the_mask_reserves_ids_and_admits_control_only_when_enabled() -> None:
    pieces = ["<pad>", "say", " 1;", "", "<ctl>", "exit;", "say 1;"]
    mask = TokenMask(pieces, reserved=[0], control=[4, 6])
    allowed = mask.allowed("d", COUNTER, COUNTER.start, allow_control=False)
    assert list(allowed) == [0, 1, 0, 0, 0, 1, 0]
    # A control id is admitted with control enabled, but only if the grammar admits its text.
    allowed = mask.allowed("d", COUNTER, COUNTER.start, allow_control=True)
    assert list(allowed) == [0, 1, 0, 0, 0, 1, 1]


def test_the_mask_is_computed_once_per_state() -> None:
    mask = TokenMask(["say", " 1;"], reserved=[], control=[])
    first = mask.allowed("d", COUNTER, COUNTER.start, allow_control=False)
    assert mask.allowed("d", COUNTER, COUNTER.start, allow_control=False) is first
    after_say = COUNTER.advance(COUNTER.start, "say")
    assert list(mask.allowed("d", COUNTER, after_say, allow_control=False)) == [0, 1]


def test_a_piece_closing_two_commands_is_refused() -> None:
    """The seat's parser splits a piece at its first terminator only, so a piece that also
    closes the next command would leave it unread and the round stuck."""
    pieces = ["say", " a; exit;", " a;", " a; say b"]
    mask = TokenMask(pieces, reserved=[], control=[])
    after_say = COUNTER.advance(COUNTER.start, "say")
    assert list(mask.allowed("d", COUNTER, after_say, allow_control=False)) == [0, 0, 1, 1]
    assert COUNTER.walk(after_say, " a; exit;")[1] == 2


@pytest.mark.parametrize(
    ("text", "admitted"),
    [
        ("exit%%", True),
        ("say hi%% write stdout go%%", True),
        ("say hi%%write stdout go%%", False),  # the next command opens with a space
        ("say h%", True),  # a prefix: the terminator has begun
        ("say h%%", True),
        ("say h%llo", False),  # a payload excludes the terminator's characters, as in GBNF
        ("write tools 12%%", True),
        ("read%%", False),  # read takes a pipe
    ],
)
def test_a_terminator_of_several_characters_is_matched_as_a_literal(
    text: str, admitted: bool
) -> None:
    abi = SyscallABI(verbs=DEFAULT.verbs, terminator="%%")
    language = CommandLanguage(abi, ("stdin", "stdout", "tools"), valued=("tools",))
    assert admits(text, language) is admitted


def test_a_long_terminator_split_across_pieces_closes_once() -> None:
    abi = SyscallABI(verbs=DEFAULT.verbs, terminator="%%")
    language = CommandLanguage(abi, ("stdin", "stdout"))
    state = language.advance(language.start, "exit%")
    assert language.walk(state, "%") == (((5,),), 1)
