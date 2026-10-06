# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The syscall ABI as a per-step token mask, so a model in the browser can only emit commands.

The llama seat compiles the ABI to a GBNF grammar (``zeos_coop_count.grammar``) and lets
llama.cpp's sampler walk it. A JavaScript worker has no grammar engine to hand, so the
browser seat does the walking itself: before every decode it works out which vocabulary
ids keep the text of the current round a prefix of some valid round, and sends that set
as ``allowedTokens``.

**The language** is the one ``build_grammar`` renders, verb for verb and pipe for pipe:
any number of the ABI's request-free verbs, then one call; a call that takes a pipe is
offered only the aliases the descriptor binds, an actuator alias takes a number with no
leading zeros, a payload is one to ``max_text`` characters that are not the terminator,
``<`` or a newline, and a job with no ``stdin`` is offered no read. Two spaces differ,
both because the seat speaks every word after a job's first with a leading space (see
``zeos.machine.seat.words_of``): a round may open with one space, and the space a GBNF
``end`` puts after a terminator is instead the one that opens the next command. After
the terminator that closes a call nothing more is accepted; the seat resets the round
there, as the llama seat rebuilds its grammar.

**The matcher** is a small nondeterministic automaton whose states are tuples of
integers, run one character at a time; a set of its states, as a sorted tuple, is the
state of a round. A token is allowed from a round state when every character of its
piece is accepted and the piece completes at most one terminator: the seat's parser
splits on the first terminator it sees, so a second command closed inside the same
piece would never reach the kernel and the round would never reset. A terminator of
several characters is matched as a literal, and a payload excludes each of its
characters, as the GBNF character class does. Empty pieces are refused, since a token
that adds no character cannot advance a command and a model could otherwise emit it
for ever.

**Cost.** A mask is computed once per (descriptor, round state, ``allow_control``) and
cached, which costs one pass over the vocabulary at roughly one automaton step per
character of each piece. Every later step from a cached state is a dictionary lookup.
Counting rounds revisit a few dozen states, so after the first few commands nearly
every step is a hit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from zeos.machine.abi import SyscallABI

__all__ = ["CommandLanguage", "RoundState", "TokenMask"]

#: A round state: the automaton states still alive, sorted, so it hashes and compares
#: the same way in every process.
RoundState = tuple[tuple[int, ...], ...]

# Automaton states. Each is a tuple whose first element says which kind it is.
_START = (0,)  # a round has begun and nothing has been said in it
_COMMAND = (1,)  # at the first character of a command
_LITERAL = 2  # (2, alt, p): p characters of alternative alt's literal head matched
_TAIL = 3  # (3, alt, k): the head is matched and k characters of the tail follow it
_SPACE = (4,)  # a request-free command closed; the space opening the next is due
_CLOSED = (5,)  # a call closed; the round is over
_END = 6  # (6, alt, p): p characters of a terminator longer than one matched

_NO_TAIL, _TEXT, _NUMBER = 0, 1, 2
#: Tail state of a number that began with ``0``, which can only be closed.
_ZERO = -1
#: Digits a number may carry, as ``[1-9] [0-9]{0,8}`` in the GBNF does.
_MAX_DIGITS = 9


@dataclass(frozen=True, slots=True)
class _Alternative:
    """One way a command can be spelled: a literal head, then a tail of one kind."""

    head: str
    tail: int
    call: bool


class CommandLanguage:
    """The rounds one descriptor may speak, as an automaton over characters."""

    def __init__(
        self, abi: SyscallABI, pipes: Sequence[str], *, valued: Sequence[str] = ()
    ) -> None:
        self.abi = abi
        valued_set = set(valued)
        plain = [p for p in pipes if p not in valued_set]
        valued_bound = [p for p in pipes if p in valued_set]
        readable = list(pipes) if "stdin" in pipes else []
        alternatives: list[_Alternative] = []
        for call, verbs in ((False, abi.lines), (True, abi.calls)):
            for verb in verbs:
                if not verb.pipe:
                    head = f"{verb.name} " if verb.text else verb.name
                    alternatives.append(_Alternative(head, _TEXT if verb.text else _NO_TAIL, call))
                elif verb.text:
                    alternatives += [_Alternative(f"{verb.name} {p} ", _TEXT, call) for p in plain]
                    alternatives += [
                        _Alternative(f"{verb.name} {p} ", _NUMBER, call) for p in valued_bound
                    ]
                else:
                    alternatives += [
                        _Alternative(f"{verb.name} {p}", _NO_TAIL, call) for p in readable
                    ]
        if not any(a.call for a in alternatives):
            raise ValueError(f"no call of the ABI can be made with pipes {list(pipes)}")
        self._alternatives = tuple(alternatives)
        # As the GBNF's ``[^...]`` class does, a payload excludes every character of the
        # terminator, so where a terminator begins is never in doubt.
        self._excluded = frozenset({*abi.terminator, "<", "\n"})

    @property
    def start(self) -> RoundState:
        return (_START,)

    def advance(self, state: RoundState, text: str) -> RoundState:
        """The state after ``text``, or an empty state if the text left the language."""
        return self.walk(state, text)[0]

    def walk(self, state: RoundState, text: str) -> tuple[RoundState, int]:
        """The state after ``text``, and how many commands the text closed on the way."""
        closed = 0
        for char in text:
            if not state:
                break
            state = tuple(sorted({n for s in state for n in self._step(s, char)}))
            # Both states exist only on the character that completes a terminator.
            closed += _SPACE in state or _CLOSED in state
        return state, closed

    def _step(self, state: tuple[int, ...], char: str) -> Iterable[tuple[int, ...]]:
        kind = state[0]
        if state == _START:
            if char == " ":
                return (_COMMAND,)
            return self._step(_COMMAND, char)
        if state == _COMMAND:
            return [
                (_TAIL, i, 0) if len(a.head) == 1 else (_LITERAL, i, 1)
                for i, a in enumerate(self._alternatives)
                if a.head[0] == char
            ]
        if kind == _LITERAL:
            _, i, p = state
            head = self._alternatives[i].head
            if head[p] != char:
                return ()
            return ((_TAIL, i, 0) if p + 1 == len(head) else (_LITERAL, i, p + 1),)
        if kind == _TAIL:
            return self._tail(state, char)
        if kind == _END:
            _, i, p = state
            terminator = self.abi.terminator
            if terminator[p] != char:
                return ()
            return self._closed(i) if p + 1 == len(terminator) else ((_END, i, p + 1),)
        if state == _SPACE:
            return (_COMMAND,) if char == " " else ()
        return ()  # _CLOSED accepts nothing

    def _closed(self, alt: int) -> tuple[tuple[int, ...], ...]:
        return (_CLOSED if self._alternatives[alt].call else _SPACE,)

    def _terminate(self, alt: int) -> tuple[tuple[int, ...], ...]:
        """The first character of the terminator, matched."""
        if len(self.abi.terminator) == 1:
            return self._closed(alt)
        return ((_END, alt, 1),)

    def _tail(self, state: tuple[int, ...], char: str) -> tuple[tuple[int, ...], ...]:
        _, i, k = state
        alt = self._alternatives[i]
        opens_end = char == self.abi.terminator[0]
        if alt.tail == _NO_TAIL:
            return self._terminate(i) if opens_end else ()
        if alt.tail == _TEXT:
            if opens_end:
                return self._terminate(i) if k >= 1 else ()
            if char in self._excluded:
                return ()
            limit = self.abi.max_text
            if limit is None:
                # Unbounded: all the state needs to know is whether one character is in.
                return ((_TAIL, i, 1),)
            return ((_TAIL, i, k + 1),) if k < limit else ()
        # A number: "0" | [1-9] [0-9]{0,8}
        if opens_end:
            return self._terminate(i) if k != 0 else ()
        if not ("0" <= char <= "9"):
            return ()
        if k == 0:
            return ((_TAIL, i, _ZERO if char == "0" else 1),)
        if k == _ZERO or k >= _MAX_DIGITS:
            return ()
        return ((_TAIL, i, k + 1),)


class TokenMask:
    """Which vocabulary ids a step may emit, given a language and a round state.

    ``reserved`` ids are never allowed (block padding, end of sequence). ``control``
    ids are allowed only when the kernel enables control tokens for the step, and then
    only if the grammar accepts their text too.
    """

    def __init__(
        self, pieces: Sequence[str], *, reserved: Iterable[int], control: Iterable[int]
    ) -> None:
        self._pieces = tuple(pieces)
        self._reserved = frozenset(reserved)
        self._control = frozenset(control)
        self._cache: dict[tuple[str, RoundState, bool], bytes] = {}

    @property
    def size(self) -> int:
        return len(self._pieces)

    def allowed(
        self, key: str, language: CommandLanguage, state: RoundState, *, allow_control: bool
    ) -> bytes:
        """One byte per vocabulary id, 1 where the id may be emitted. ``key`` names the
        language in the cache, so one mask serves every job running the same descriptor."""
        cached = self._cache.get((key, state, allow_control))
        if cached is not None:
            return cached
        flags = bytearray(len(self._pieces))
        for token_id, piece in enumerate(self._pieces):
            if not piece or token_id in self._reserved:
                continue
            if token_id in self._control and not allow_control:
                continue
            # The parser splits a piece at its first terminator only, so a piece that
            # closes two commands would leave the second unread and the round stuck.
            after, closed = language.walk(state, piece)
            if after and closed <= 1:
                flags[token_id] = 1
        mask = bytes(flags)
        self._cache[(key, state, allow_control)] = mask
        return mask
