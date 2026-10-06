# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""An ``AsyncModelWorker`` with no model, in Python, that takes real time to answer.

``FakePilotWorker`` stands in for the channel to the in-browser model when the port runs
under CPython: the machine and the prompt arm are driven against it by the tests and by
the CPython end-to-end run. ``web/stub/pilot_stub_worker.js`` is its twin on the model
side of the channel and is kept to the same behaviour (``tests/test_stub_parity.py``).
What it has:

* **a fixed vocabulary**: coop-count-web's five reserved tokens (``<pad>``, ``<eos>``,
  ``<unk>``, ``<|im_start|>``, ``<|im_end|>``), then every word of the pilot's commands,
  of the prompt arm's replies and of the ChatML headers, each with and without a leading
  space, sorted;
* **a whitespace tokenizer**: each run of non-space characters is one token, written
  with a single leading space if any space preceded it, ``<unk>`` if the vocabulary
  lacks it. A board or a descriptor body is mostly ``<unk>``, which costs nothing;
* **a script per kind of context**, read from the context id (``"<job>:<descriptor>"``):
  a ``prompt`` context answers each assistant turn it is given with the next reply (the
  replies are counted over all prompt contexts, so forks of one prefix take turns), a
  word per step, then ``<|im_end|>``, reading where it is from the context (so a step
  outside an assistant turn, such as the warm-up's, says ``<|im_end|>`` and moves
  nothing on); any other context is a pilot and says ``write stdout <move>;`` for the
  next of ``moves``, a word per step (``reads=True`` adds ``read stdin;`` after each).
  The script follows the grammar: a word the step's ``allowedTokens`` refuses means the
  machine reset the round (an ``invalidate``), so the half-said command is dropped and
  the next one begins, which must be allowed;
* **latency against the wall clock**: a step costs ``position_ms`` for each pending
  position, filled in runs of at most ``maxChunk``, then ``step_ms`` for the decode
  itself. The result is ready at a wall-clock time; ``pollDecode`` sleeps until then or
  for its timeout, whichever is first. A cancel is noticed before each run, so a step
  cancelled mid-fill stops at the next chunk boundary with the positions filled so far
  resident; a result that was ready before its cancel was polled is dropped (race rule 1).

A step chooses a token without changing the context, as the real worker does. The script
moves on only when a step's token is delivered, so a cancelled step's word is the next
step's word.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from zeos.machine.seat import words_of

from zeos_space_invaders_web.contracts import (
    CHANNEL_BUSY,
    DecodeCancelled,
    DecodeDone,
    DecodeStats,
)

__all__ = [
    "CONTROL_IDS",
    "EOS_ID",
    "FakePilotWorker",
    "IM_END_ID",
    "IM_START_ID",
    "ModelInfo",
    "PAD_ID",
    "PROMPT_DESCRIPTOR",
    "SPECIAL",
    "UNK_ID",
    "WORKER_CHUNK",
    "fill_chunks",
]

#: The reserved tokens, in id order, as coop-count-web's stub has them.
SPECIAL = ("<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>")
PAD_ID, EOS_ID, UNK_ID = 0, 1, 2
IM_START_ID, IM_END_ID = 3, 4
CONTROL_IDS = (IM_START_ID, IM_END_ID)
#: Words of the ChatML headers, so framing tokenizes to something other than unknown.
FRAMING_WORDS = ("system", "user", "assistant")
#: The descriptor part of a context id that the prompt arm uses.
PROMPT_DESCRIPTOR = "prompt"
#: The fill chunk when a step does not say ``maxChunk``: the OPT graph's own.
WORKER_CHUNK = 2048
DEFAULT_MOVES = ("left", "right", "shoot")

#: ASCII whitespace only, as in coop-count-web's twin workers.
_WORD = re.compile(r"[ \t\n\r\f\v]*[^ \t\n\r\f\v]+")
_SPACE = " \t\n\r\f\v"


def fill_chunks(positions: int, max_chunk: int) -> list[int]:
    """How ``positions`` pending positions are cut into prefill runs of ``max_chunk``."""
    if max_chunk < 1:
        raise ValueError(f"maxChunk is {max_chunk}; it must be >= 1")
    full, rest = divmod(positions, max_chunk)
    return [max_chunk] * full + ([rest] if rest else [])


@dataclass(frozen=True, slots=True)
class ModelInfo:
    blockSize: int
    padId: int
    controlIds: tuple[int, ...]
    eosId: int
    vocabSize: int


@dataclass
class _Context:
    descriptor: str
    ids: list[int] = field(default_factory=list[int])
    #: Positions whose KV the simulated model holds; ``ids[kv:]`` are pending.
    kv: int = 0
    #: Pilot: commands begun so far. Prompt: the reply the open turn is given.
    issued: int = 0
    #: Prompt only: where the assistant turn being answered opens, -1 for none.
    reply_at: int = -1
    #: Pieces of the current command or reply not yet delivered.
    pending: list[str] = field(default_factory=list[str])
    spoken: bool = False


@dataclass(frozen=True, slots=True)
class _Choice:
    """A step's token, and the script state that delivering it leaves."""

    token_id: int
    pending: tuple[str, ...]
    issued: int
    reply_at: int
    #: Prompt only: this token opens an assistant turn the worker has not answered.
    new_turn: bool = False


@dataclass
class _Flight:
    request_id: int
    key: str
    choice: _Choice
    kv_before: int
    chunks: list[int]
    begun: float
    position_s: float
    step_s: float
    cancel_at: float | None = None

    @property
    def positions(self) -> int:
        return sum(self.chunks)

    def boundary(self, k: int) -> float:
        """When the check before run ``k`` happens; run ``len(chunks)`` is the decode."""
        return self.begun + self.position_s * sum(self.chunks[:k])

    def stopped_before(self) -> int | None:
        """The run a cancel stopped the step in front of, or None if it came too late."""
        if self.cancel_at is None:
            return None
        for k in range(len(self.chunks) + 1):
            if self.cancel_at <= self.boundary(k):
                return k
        return None

    def ready_at(self) -> float:
        stopped = self.stopped_before()
        if stopped is not None:
            return self.boundary(stopped)
        return self.boundary(len(self.chunks)) + self.step_s


class FakePilotWorker:
    """``AsyncModelWorker`` in Python. Method names are the JavaScript interface's."""

    def __init__(
        self,
        moves: Sequence[str] = DEFAULT_MOVES,
        *,
        replies: Sequence[str] | None = None,
        step_ms: float = 0.0,
        position_ms: float = 0.0,
        block_size: int = 16,
        terminator: str = ";",
        reads: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not moves:
            raise ValueError("a pilot script needs at least one move")
        self.moves = tuple(moves)
        self.replies = tuple(replies) if replies is not None else self.moves
        if not self.replies:
            raise ValueError("a prompt script needs at least one reply")
        for text in (*self.moves, *self.replies):
            if not text.isascii():
                raise ValueError(f"script text {text!r} is not ASCII")
        self.step_ms = step_ms
        self.position_ms = position_ms
        self.reads = reads
        self._block_size = block_size
        self._terminator = terminator
        self._clock = clock
        self._sleep = sleep

        words: set[str] = set(FRAMING_WORDS)
        for n in range(len(self.moves) * (2 if reads else 1)):
            words.update(
                w.strip() for w in words_of(self._command(n), terminator=terminator, lead=False)
            )
        for reply in self.replies:
            words.update(reply.split())
        self._vocab = SPECIAL + tuple(sorted(words | {" " + w for w in words}))
        self._ids = {piece: i for i, piece in enumerate(self._vocab)}
        self._contexts: dict[str, _Context] = {}
        self._flight: _Flight | None = None
        self._next_request = 0
        #: Assistant turns answered over every prompt context: a decision forked from a
        #: shared prefix still gets the next reply.
        self._prompt_turns = 0

        #: What the steps did, for tests: delivered tokens, cancelled steps, the
        #: positions filled and the options the last step was begun with.
        self.delivered = 0
        self.cancelled = 0
        self.filled = 0
        self.last_options: Mapping[str, object] | None = None

    # -- the script ----------------------------------------------------------

    def _command(self, n: int) -> str:
        if self.reads:
            return (
                "read stdin" if n % 2 else f"write stdout {self.moves[(n // 2) % len(self.moves)]}"
            )
        return f"write stdout {self.moves[n % len(self.moves)]}"

    def _reply(self, n: int) -> list[str]:
        return [" " + w for w in self.replies[n % len(self.replies)].split()] + [SPECIAL[IM_END_ID]]

    def _choose(self, key: str, ctx: _Context, allowed: object) -> _Choice:
        def ok(piece: str) -> bool:
            return allowed is None or bool(allowed[self._ids[piece]])  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]

        pending, issued = list(ctx.pending), ctx.issued
        if ctx.descriptor == PROMPT_DESCRIPTOR:
            return self._choose_reply(key, ctx, ok)
        else:
            if pending and not ok(pending[0]):
                # The round was reset under the command: drop the rest of it.
                pending = []
            if not pending:
                pending = words_of(
                    self._command(issued), terminator=self._terminator, lead=ctx.spoken
                )
                issued += 1
        piece = pending[0]
        if not ok(piece):
            raise ValueError(f"{key}: the token mask refuses the script's next word {piece!r}")
        return _Choice(self._ids[piece], tuple(pending[1:]), issued, ctx.reply_at)

    def _choose_reply(self, key: str, ctx: _Context, ok: Callable[[str], bool]) -> _Choice:
        """A prompt context answers the open assistant turn, read off the context itself:
        a reply per turn, a word per token already in the turn, then ``<|im_end|>``.
        With no assistant turn open (the warm-up step) it says ``<|im_end|>``."""
        header = self._ids["assistant"]
        at = next(
            (
                i
                for i in range(len(ctx.ids) - 2, -1, -1)
                if ctx.ids[i] == IM_START_ID and ctx.ids[i + 1] == header
            ),
            -1,
        )
        new_turn = at >= 0 and at != ctx.reply_at
        if at < 0:
            piece, issued = SPECIAL[IM_END_ID], ctx.issued
        else:
            issued = self._prompt_turns if new_turn else ctx.issued
            words = self._reply(issued)
            piece = words[min(len(ctx.ids) - at - 2, len(words) - 1)]
        if not ok(piece):
            raise ValueError(f"{key}: the token mask refuses the script's next word {piece!r}")
        return _Choice(self._ids[piece], (), issued, at if at >= 0 else ctx.reply_at, new_turn)

    # -- identity ------------------------------------------------------------

    def info(self) -> ModelInfo:
        return ModelInfo(
            blockSize=self._block_size,
            padId=PAD_ID,
            controlIds=CONTROL_IDS,
            eosId=EOS_ID,
            vocabSize=len(self._vocab),
        )

    def _idle(self, method: str) -> None:
        if self._flight is not None:
            raise RuntimeError(f"{CHANNEL_BUSY} ({method})")

    def tokenize(self, text: str) -> list[int]:
        self._idle("tokenize")
        out: list[int] = []
        for match in _WORD.finditer(text):
            run = match.group(0)
            word = run.lstrip(_SPACE)
            piece = (" " + word) if len(word) < len(run) else word
            out.append(self._ids.get(piece, UNK_ID))
        return out

    def piece(self, tokenId: int) -> str:
        if not 0 <= tokenId < len(self._vocab):
            raise IndexError(f"token id {tokenId} is outside a vocabulary of {len(self._vocab)}")
        return self._vocab[tokenId]

    # -- contexts ------------------------------------------------------------

    def _ctx(self, job_id: str) -> _Context:
        ctx = self._contexts.get(job_id)
        if ctx is None:
            raise KeyError(f"no context {job_id!r}")
        return ctx

    def _fresh(self, job_id: str) -> _Context:
        if job_id in self._contexts:
            raise KeyError(f"context {job_id!r} already exists")
        return _Context(descriptor=job_id.partition(":")[2])

    def createContext(self, jobId: str) -> None:
        self._idle("createContext")
        self._contexts[jobId] = self._fresh(jobId)

    def destroyContext(self, jobId: str) -> None:
        self._idle("destroyContext")
        self._ctx(jobId)
        del self._contexts[jobId]

    def length(self, jobId: str) -> int:
        self._idle("length")
        return len(self._ctx(jobId).ids)

    def append(self, jobId: str, ids: object) -> None:
        self._idle("append")
        ctx = self._ctx(jobId)
        for token_id in cast("Iterable[int]", ids):
            self.piece(int(token_id))
            ctx.ids.append(int(token_id))

    def truncate(self, jobId: str, n: int) -> None:
        self._idle("truncate")
        ctx = self._ctx(jobId)
        if not 0 <= n <= len(ctx.ids):
            raise IndexError(f"truncate at {n} outside [0, {len(ctx.ids)}]")
        del ctx.ids[n:]
        ctx.kv = min(ctx.kv, n)
        if ctx.reply_at >= n:
            ctx.reply_at = -1

    def fork(self, parentId: str, childId: str) -> None:
        self._idle("fork")
        parent = self._ctx(parentId)
        child = self._fresh(childId)
        child.ids = list(parent.ids)
        child.kv = parent.kv
        child.spoken = parent.spoken
        self._contexts[childId] = child

    @property
    def context_ids(self) -> tuple[str, ...]:
        """The contexts that exist, for tests; not part of the worker interface."""
        return tuple(self._contexts)

    def resident(self, jobId: str) -> int:
        """Positions with KV behind them, for tests; not part of the worker interface."""
        return self._ctx(jobId).kv

    # -- decoding ------------------------------------------------------------

    @staticmethod
    def _opt(opts: object, name: str) -> object:
        if isinstance(opts, Mapping):
            return opts.get(name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return getattr(opts, name, None)

    def beginDecodeStep(self, jobId: str, opts: object) -> int:
        self._idle("beginDecodeStep")
        ctx = self._ctx(jobId)
        if not ctx.ids:
            raise ValueError(f"decodeStep on {jobId!r}, which holds no tokens")
        blocks = self._opt(opts, "allowedBlocks")
        expected = (len(ctx.ids) + self._block_size - 1) // self._block_size
        if blocks is not None and len(blocks) != expected:  # pyright: ignore[reportArgumentType]
            raise ValueError(f"allowedBlocks has {len(blocks)} entries for {expected} blocks")  # pyright: ignore[reportArgumentType]
        allowed = self._opt(opts, "allowedTokens")
        if allowed is not None and len(allowed) != len(self._vocab):  # pyright: ignore[reportArgumentType]
            raise ValueError(
                f"allowedTokens has {len(allowed)} entries for {len(self._vocab)} ids"  # pyright: ignore[reportArgumentType]
            )
        chunk = self._opt(opts, "maxChunk")
        max_chunk = WORKER_CHUNK if chunk is None else int(chunk)  # pyright: ignore[reportArgumentType]
        self.last_options = {
            "allowedBlocks": blocks,
            "allowedTokens": allowed,
            "maxChunk": chunk,
        }
        choice = self._choose(jobId, ctx, allowed)
        request_id = self._next_request
        self._next_request += 1
        self._flight = _Flight(
            request_id=request_id,
            key=jobId,
            choice=choice,
            kv_before=ctx.kv,
            chunks=fill_chunks(len(ctx.ids) - ctx.kv, max_chunk),
            begun=self._clock(),
            position_s=self.position_ms / 1000.0,
            step_s=self.step_ms / 1000.0,
        )
        return request_id

    def pollDecode(self, timeoutMs: float) -> DecodeDone | DecodeCancelled | None:
        flight = self._flight
        if flight is None:
            return None
        now = self._clock()
        deadline = now + max(timeoutMs, 0.0) / 1000.0
        while now < flight.ready_at():
            wait = min(deadline, flight.ready_at()) - now
            if wait <= 0:
                return None
            self._sleep(wait)
            now = self._clock()
        return self._land(flight)

    def cancelDecode(self) -> None:
        if self._flight is not None and self._flight.cancel_at is None:
            self._flight.cancel_at = self._clock()

    @property
    def inFlight(self) -> bool:
        return self._flight is not None

    def _land(self, flight: _Flight) -> DecodeDone | DecodeCancelled:
        self._flight = None
        ctx = self._contexts.get(flight.key)
        stopped = flight.stopped_before()
        runs = len(flight.chunks) if stopped is None else min(stopped, len(flight.chunks))
        filled = sum(flight.chunks[:runs])
        stats: DecodeStats = {
            "positions": filled,
            "chunks": runs,
            "fillMs": filled * self.position_ms,
        }
        self.filled += filled
        resident = flight.kv_before + filled
        if ctx is not None:
            ctx.kv = resident
        if flight.cancel_at is not None or ctx is None:
            # Race rule 1: a step its caller cancelled never delivers its token.
            self.cancelled += 1
            return {"cancelled": True, "resident": resident, "stats": stats}
        ctx.pending = list(flight.choice.pending)
        ctx.issued = flight.choice.issued
        ctx.reply_at = flight.choice.reply_at
        self._prompt_turns += flight.choice.new_turn
        ctx.spoken = True
        self.delivered += 1
        return {
            "tokenId": flight.choice.token_id,
            "attention": None,
            "cancelled": False,
            "resident": resident,
            "stats": stats,
        }

    def decodeStep(self, jobId: str, opts: object) -> DecodeDone:
        """The synchronous step: begin, then wait for it."""
        self.beginDecodeStep(jobId, opts)
        while True:
            result = self.pollDecode(math.inf)
            if result is not None:
                if result["cancelled"] is True:
                    raise RuntimeError("a synchronous step cannot be cancelled")
                return result
