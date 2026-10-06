# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The pilot's machine: ``JsMachine`` over a worker channel that never blocks the loop.

``JsMachine`` asks its worker for a token and waits for it. In the browser that wait is
an ``Atomics.wait`` the game clock cannot interrupt, so a world tick could not land while
the model thinks and the reflex could never preempt the pilot. ``PilotJsMachine`` keeps
``JsMachine``'s token bookkeeping, framing and grammar mask, and changes how a step is
taken:

* **A step is begun, then polled.** ``decode`` plans the step (``_plan_step``), posts it
  with ``beginDecodeStep`` and polls for at most ``stall_ms``. While the answer is not
  in, ``decode`` returns an empty step -- the native machine's stall -- so the job stays
  RUNNING and the kernel can take the machine away from it at any token boundary. When
  it is in, ``_accept_step`` joins it to the context as ``JsMachine`` would.
* **A step whose context changes is cancelled.** An inject, trunc, splice or fork, a
  narrowing ``set_mask``, block padding, ``invalidate``, and the decode of another job
  (the kernel descheduled this one) all cancel the step in flight; its token is never
  delivered (race rule 1 in ``CONTRACTS.md``). Every operation that has to call the
  worker synchronously first *settles*: cancels the step and waits for the channel to
  hand it back, at most one prefill chunk later. Operations that do not touch the
  worker only post the cancel, and the step drains on a later ``decode``.
* **Locally-served behaviours.** ``register_behaviour`` is the native machine's: a
  descriptor so registered is decoded by a Python function, and its jobs never reach
  the worker -- no context is created for them, so the reflex runs while a pilot step
  is still in flight.
* **One move, one turn.** With ``turn_ends_at_call`` a completed ``write`` is followed by
  a ``read stdin`` the machine issues itself, as the native machine ends every reply.
* **What a cancelled turn leaves.** ``invalidate`` is the native text format's
  ``partial="keep"``: the words decoded so far stay in the context, and the command
  parser and the grammar round start again, so the next step begins a fresh command.

The worker's KV survives a cancel: the positions a cancelled step filled stay resident
and the next step for the context resumes from there (race rule 3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import JobId, TokenKind
from zeos.machine.abi import SyscallABI
from zeos.machine.base import (
    AttentionHint,
    ContextStats,
    ControlTokenViolation,
    DecodeResult,
    OpKind,
    RawWindow,
    RawWord,
    SpliceResult,
    Token,
)
from zeos.machine.scripted import PAD_TOKEN
from zeos_coop_count_web.js_machine import (
    Bridge,
    JsMachine,
    _StepPlan,  # pyright: ignore[reportPrivateUsage]
)
from zeos_space_invaders.players.zeos.api_machine import ABI, Native

from zeos_space_invaders_web.contracts import (
    DEFAULT_MAX_CHUNK,
    PREWARM_COMMANDS,
    AsyncModelWorker,
    DecodeCancelled,
    DecodeDone,
    DecodeStats,
    StepLogEntry,
    StepOutcome,
)

__all__ = [
    "PILOT_DESCRIPTORS",
    "PilotJsMachine",
    "SETTLE_POLL_MS",
    "normalise_poll",
    "pilot_abi",
]

#: The pipe aliases the pilot binds (``goals/pilot.md``); the grammar offers no other.
PILOT_DESCRIPTORS: Mapping[str, Sequence[str]] = {"pilot": ("stdin", "stdout")}

#: How long one wait inside ``_settle`` lasts. A settle loops until the step is handed
#: back, so this bounds only how often the loop wakes, not how long it waits.
SETTLE_POLL_MS = 1000.0

#: Where a turn goes to sleep, as the native machine reads it.
_READ_STDIN = "read stdin"


def pilot_abi(forbid_verbs: Sequence[str] = ()) -> SyscallABI:
    """The native pilot ABI (``write``, ``read``, ``exit``) less ``forbid_verbs``."""
    unknown = sorted(set(forbid_verbs) - {v.name for v in ABI.verbs})
    if unknown:
        raise ValueError(f"cannot forbid {unknown}: the pilot ABI has no such verb")
    verbs = tuple(v for v in ABI.verbs if v.name not in forbid_verbs)
    return SyscallABI(
        verbs=verbs, aliases=ABI.aliases, terminator=ABI.terminator, max_text=ABI.max_text
    )


def _stats(raw: object) -> DecodeStats:
    if raw is None or type(raw).__name__ == "JsNull":
        return {"positions": 0, "chunks": 0, "fillMs": 0.0}
    return {
        "positions": int(_field(raw, "positions", 0)),  # pyright: ignore[reportArgumentType]
        "chunks": int(_field(raw, "chunks", 0)),  # pyright: ignore[reportArgumentType]
        "fillMs": float(_field(raw, "fillMs", 0.0)),  # pyright: ignore[reportArgumentType]
    }


def _field(raw: object, name: str, default: object = None) -> object:
    """A field of a poll result: a dict from a Python worker, a JsProxy under Pyodide."""
    if isinstance(raw, Mapping):
        return raw.get(name, default)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return getattr(raw, name, default)


def normalise_poll(raw: object) -> DecodeDone | DecodeCancelled | None:
    """``pollDecode``'s answer as the contract's TypedDicts; ``None`` while running."""
    if raw is None or type(raw).__name__ == "JsNull":
        return None
    stats = _stats(_field(raw, "stats"))
    if bool(_field(raw, "cancelled", False)):
        return {"cancelled": True, "resident": int(_field(raw, "resident", 0)), "stats": stats}  # pyright: ignore[reportArgumentType]
    return {
        "tokenId": int(_field(raw, "tokenId")),  # pyright: ignore[reportArgumentType]
        "attention": _field(raw, "attention"),
        "cancelled": False,
        "stats": stats,
    }


@dataclass
class _Native:
    """A locally-served job: the native machine's bookkeeping, and no worker context."""

    descriptor: str
    tokens: list[Token] = field(default_factory=list[Token])
    mask: frozenset[int] | None = None
    #: How many times the behaviour has been decoded; its program counter.
    step: int = 0
    #: What the kernel handed the job since its last step.
    arrived: list[Token] = field(default_factory=list[Token])


@dataclass
class _Round:
    """Per served job: where its current command stands, beyond what ``JsMachine`` keeps."""

    #: A ``write`` completed; the next decode puts the job to sleep on ``stdin``.
    auto_read: bool = False
    #: When the round's first step was begun, for ``roundtrips``.
    opened: float | None = None
    #: Kernel offset where the round's first word went, so a splice knows whether it
    #: reached the command being said.
    start: int = 0


@dataclass
class _Flight:
    """The one step the channel has in flight."""

    job: JobId
    plan: _StepPlan
    request_id: int
    begun: float
    cancelled_at: float | None = None


class PilotJsMachine(JsMachine):
    """``JsMachine`` whose decode never waits on the model; see the module docstring."""

    def __init__(
        self,
        worker: AsyncModelWorker,
        *,
        bridge: Bridge | None = None,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        block_size: int = 16,
        stall_ms: float = 1.0,
        max_chunk: int = DEFAULT_MAX_CHUNK,
        turn_ends_at_call: bool = True,
        forbid_verbs: Sequence[str] = (),
        tokenize_cache: int = 4096,
    ) -> None:
        if stall_ms < 0:
            raise ValueError("stall_ms must be >= 0")
        if max_chunk < 1:
            raise ValueError("max_chunk must be >= 1")
        super().__init__(
            worker,
            bridge=bridge,
            abi=pilot_abi(forbid_verbs),
            descriptors=PILOT_DESCRIPTORS if descriptors is None else descriptors,
            block_size=block_size,
        )
        self._async = worker
        self._stall_ms = stall_ms
        self._max_chunk = max_chunk
        self._turn_ends_at_call = turn_ends_at_call
        self.forbid_verbs = tuple(forbid_verbs)
        self._behaviours: dict[str, Callable[[Native], DecodeResult]] = {}
        self._native: dict[JobId, _Native] = {}
        self._rounds: dict[JobId, _Round] = {}
        self._flight: _Flight | None = None
        #: Word to ids. A board is a few hundred words, each tokenized on its own, and a
        #: tokenize is a round trip to the model thread; boards repeat their words.
        self._token_cache: dict[str, list[int]] = {}
        self._token_cache_size = tokenize_cache

        self.last_roundtrip: float | None = None
        self.roundtrips: list[float] = []
        self.cancellations = 0
        self.cancel_ms: list[float] = []
        self.step_log: list[StepLogEntry] = []
        #: Tokens the worker delivered, and empty steps handed out while one was awaited.
        self.steps = 0
        self.stalls = 0
        #: Kept apart from ``steps`` because a reflex reaches no model at all.
        self.native_words = 0
        #: Tokenize round trips the cache saved and made.
        self.tokenize_hits = 0
        self.tokenize_calls = 0

    # -- locally-served behaviours -------------------------------------------

    def register_behaviour(
        self, descriptor: str, behaviour: Callable[[Native], DecodeResult]
    ) -> None:
        """Serve ``descriptor`` with local code instead of the worker."""
        self._behaviours[descriptor] = behaviour

    def _decode_native(self, job: JobId, ctx: _Native) -> DecodeResult:
        arrived = " ".join(t.text for t in ctx.arrived if t.kind is not TokenKind.CONTROL)
        ctx.arrived.clear()
        result = self._behaviours[ctx.descriptor](
            Native(job=job, descriptor=ctx.descriptor, step=ctx.step, arrived=arrived)
        )
        ctx.step += 1
        if any(t.kind is TokenKind.CONTROL for t in result.tokens):
            raise ControlTokenViolation(
                f"native behaviour {ctx.descriptor!r} returned a CONTROL token "
                "while control tokens were disabled for this step (ZEOS-AM §9)"
            )
        ctx.tokens.extend(result.tokens)
        self.native_words += len(result.tokens)
        return result

    # -- the channel ---------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        cached = self._token_cache.get(text)
        if cached is not None:
            self.tokenize_hits += 1
            return list(cached)
        self.tokenize_calls += 1
        ids = super()._tokenize(text)
        if len(self._token_cache) >= self._token_cache_size:
            self._token_cache.clear()
        self._token_cache[text] = ids
        return list(ids)

    def _with_chunk(self, options: object) -> object:
        """The step's options with ``maxChunk``: a dict from ``PythonBridge``, a
        JavaScript object from ``PyodideBridge``."""
        if isinstance(options, dict):
            return {**options, "maxChunk": self._max_chunk}  # pyright: ignore[reportUnknownVariableType]
        setattr(options, "maxChunk", self._max_chunk)  # noqa: B010 - a JsProxy attribute
        return options

    def _poll(self, timeout_ms: float) -> DecodeDone | DecodeCancelled | None:
        return normalise_poll(self._async.pollDecode(timeout_ms))

    def _cancel(self) -> None:
        """Ask the step in flight to stop; never waits."""
        flight = self._flight
        if flight is None or flight.cancelled_at is not None:
            return
        self._async.cancelDecode()
        flight.cancelled_at = time.monotonic()
        self.cancellations += 1

    def _drained(self) -> None:
        """The cancelled step has been handed back; whatever it answered is dropped."""
        flight = self._flight
        assert flight is not None and flight.cancelled_at is not None
        self.cancel_ms.append((time.monotonic() - flight.cancelled_at) * 1000.0)
        self._flight = None

    def _settle(self) -> None:
        """Cancel the step in flight and wait for the channel to hand it back, so that a
        synchronous worker call can be made. At most one prefill chunk."""
        if self._flight is None:
            return
        self._cancel()
        while self._poll(SETTLE_POLL_MS) is None:
            pass
        self._drained()

    def _cancel_job(self, job: JobId) -> None:
        if self._flight is not None and self._flight.job == job:
            self._cancel()

    def _reset_round(self, job: JobId) -> None:
        """Start the job's next command afresh: the parser forgets the words it holds
        and the grammar returns to the start of a round."""
        ctx = self._ctx_of(job)
        ctx.parser.reset()
        ctx.round = self._language(ctx.descriptor).start
        rnd = self._rounds[job]
        rnd.opened = None
        rnd.start = len(ctx.tokens)

    def _log(
        self, job: JobId, outcome: StepOutcome, began: float, stats: DecodeStats | None = None
    ) -> None:
        self.step_log.append(
            {
                "job": str(job),
                "outcome": outcome,
                "positions": stats["positions"] if stats else 0,
                "chunks": stats["chunks"] if stats else 0,
                "fill_ms": stats["fillMs"] if stats else 0.0,
                "wall_ms": (time.monotonic() - began) * 1000.0,
            }
        )

    @staticmethod
    def _stall() -> DecodeResult:
        """A step that produced nothing (ZEOS-AM §6.1), leaving the job RUNNING."""
        return DecodeResult(tokens=(), attention=None, attention_hint=AttentionHint())

    # -- lifecycle -----------------------------------------------------------

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        if descriptor in self._behaviours:
            self._native[job] = _Native(descriptor=descriptor)
            return
        self._settle()
        super().create_context(job, descriptor)
        self._rounds[job] = _Round()

    def destroy_context(self, job: JobId) -> None:
        if self._native.pop(job, None) is not None:
            return
        if job in self._contexts:
            self._settle()
        super().destroy_context(job)
        self._rounds.pop(job, None)

    def stats(self, job: JobId) -> ContextStats:
        native = self._native.get(job)
        if native is None:
            return super().stats(job)
        n = len(native.tokens)
        return ContextStats(
            resident_tokens=n,
            blocks=self._block_count(n),
            open_segment_tokens=n % self.block_size,
        )

    def close(self) -> None:
        """Cancel and drain any step in flight and destroy every context."""
        self._settle()
        self._native.clear()
        super().close()
        self._rounds.clear()

    # -- the five ops --------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        began = time.monotonic()
        native = self._native.get(job)
        if native is not None:
            # The kernel took the machine away from whatever was in flight.
            self._cancel()
            result = self._decode_native(job, native)
            self._log(job, "native", began)
            return result

        flight = self._flight
        if flight is not None and flight.cancelled_at is None:
            if flight.job != job:
                self._cancel()  # descheduled
            elif flight.plan.allow_control != allow_control:
                self._cancel()  # planned under the other token mask
        if self._flight is not None and self._flight.cancelled_at is not None:
            if self._poll(self._stall_ms) is None:
                self.stalls += 1
                self._log(job, "stall", began)
                return self._stall()
            self._drained()
            self._log(job, "cancelled", began)

        rnd = self._rounds[job]
        if self._flight is None:
            if rnd.auto_read:
                rnd.auto_read = False
                rnd.start = len(self._ctx_of(job).tokens)
                self._log(job, "native", began)
                return DecodeResult(
                    tokens=(),
                    request=self.abi.parse(_READ_STDIN),
                    attention=None,
                    attention_hint=AttentionHint(tags=("self",)),
                )
            plan = self._plan_step(job, allow_control=allow_control)
            now = time.monotonic()
            request_id = self._async.beginDecodeStep(plan.ctx.key, self._with_chunk(plan.options))
            self._flight = _Flight(job=job, plan=plan, request_id=request_id, begun=now)
            if rnd.opened is None:
                rnd.opened = now

        flight = self._flight
        assert flight is not None
        answer = self._poll(self._stall_ms)
        if answer is None:
            self.stalls += 1
            self._log(job, "stall", began)
            return self._stall()
        if answer["cancelled"] is True:
            # Nobody here asked; the step is gone all the same, and so is its token.
            if flight.cancelled_at is None:
                flight.cancelled_at = time.monotonic()
                self.cancellations += 1
            self._drained()
            self._log(job, "cancelled", began, answer["stats"])
            return self._stall()
        self._flight = None
        result = self._accept_step(flight.plan, answer["tokenId"], answer["attention"])
        self.steps += 1
        self._log(job, "token", began, answer["stats"])
        if result.request.op is not OpKind.NONE:
            ctx = self._ctx_of(job)
            rnd.start = len(ctx.tokens)
            if rnd.opened is not None:
                self.last_roundtrip = round(time.monotonic() - rnd.opened, 4)
                self.roundtrips.append(self.last_roundtrip)
                rnd.opened = None
            if self._turn_ends_at_call and result.request.op is OpKind.WRITE:
                rnd.auto_read = True
        return result

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        native = self._native.get(job)
        if native is not None:
            start = len(native.tokens)
            native.tokens.extend(tokens)
            native.arrived.extend(tokens)
            return start, len(native.tokens)
        if not tokens:
            return super().inject(job, tokens)
        self._settle()
        span = super().inject(job, tokens)
        # The prefix moved under the command being said; it belongs to the completion
        # the arrival interrupted.
        self._reset_round(job)
        return span

    def trunc(self, job: JobId, at: int) -> int:
        native = self._native.get(job)
        if native is not None:
            if at < 0 or at > len(native.tokens):
                raise IndexError(f"trunc at {at} outside [0, {len(native.tokens)}]")
            dropped = len(native.tokens) - at
            del native.tokens[at:]
            return dropped
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        if at == len(ctx.tokens):
            return 0
        self._settle()
        dropped = super().trunc(job, at)
        self._reset_round(job)
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        native = self._native.get(parent)
        if native is not None:
            self._native[child] = _Native(
                descriptor=native.descriptor, tokens=list(native.tokens), mask=native.mask
            )
            return len(native.tokens)
        self._settle()
        shared = super().fork(parent, child)
        self._rounds.setdefault(child, _Round())
        self._rounds[child].auto_read = False
        self._reset_round(child)
        return shared

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        native = self._native.get(job)
        if native is not None:
            if not 0 <= start <= end <= len(native.tokens):
                raise IndexError(f"splice [{start}, {end}) outside {len(native.tokens)}")
            downstream = len(native.tokens) - end
            native.tokens[start:end] = list(tokens)
            return SpliceResult(tokens_in=len(tokens), invalidated_downstream=downstream)
        self._settle()
        result = super().splice(job, start, end, tokens)
        rnd = self._rounds[job]
        if end > rnd.start:
            self._reset_round(job)
        else:
            # Wholly upstream of the command being said: it keeps its words, renumbered.
            rnd.start += len(tokens) - (end - start)
        return result

    # -- the serving-stack contract ------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        native = self._native.get(job)
        if native is not None:
            native.mask = allowed_blocks
            return
        before = self.visible_blocks(job)
        super().set_mask(job, allowed_blocks)
        if not before <= self.visible_blocks(job):
            # Only a narrowing: a decoding job's mask widens at every block boundary.
            self._cancel_job(job)
            self._reset_round(job)

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        native = self._native.get(job)
        if native is None:
            return super().visible_blocks(job)
        every = frozenset(range(self._block_count(len(native.tokens))))
        return every if native.mask is None else every & native.mask

    def pad_to_block(self, job: JobId) -> int:
        native = self._native.get(job)
        if native is None:
            ctx = self._ctx_of(job)
            if len(ctx.tokens) % self.block_size:
                # The step was planned against the context without the padding.
                self._cancel_job(job)
            return super().pad_to_block(job)
        remainder = len(native.tokens) % self.block_size
        if remainder == 0:
            return 0
        padding = self.block_size - remainder
        native.tokens.extend([PAD_TOKEN] * padding)
        return padding

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        if job not in self._native:
            return super().blocks_for_range(job, start, end)
        if end <= start:
            return frozenset()
        return frozenset(range(start // self.block_size, (end - 1) // self.block_size + 1))

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        native = self._native.get(job)
        if native is None:
            return super().transcript(job)
        return tuple(native.tokens)

    def raw(self, job: JobId) -> RawWindow:
        native = self._native.get(job)
        if native is None:
            self._settle()
            return super().raw(job)
        return RawWindow(
            words=tuple(RawWord(pieces=(t.text,)) for t in native.tokens), kv_resident=0
        )

    def lines(self, job: JobId) -> tuple[str, ...]:
        if job in self._native:
            return ()
        return super().lines(job)

    # -- generations ---------------------------------------------------------

    def invalidate(self, job: JobId) -> None:
        """Drop whatever is being generated for ``job``: cancel its step in flight and
        start its command afresh, keeping the words already decoded (native
        ``partial="keep"``). A completed command's turn end still stands. A no-op for a
        locally-served job, or one with no context."""
        if job in self._native or job not in self._contexts:
            return
        self._cancel_job(job)
        self._reset_round(job)

    def prewarm(
        self, descriptor: str = "pilot", commands: Sequence[str] = PREWARM_COMMANDS
    ) -> float:
        """Walk ``commands`` through ``descriptor``'s grammar as the worker tokenizes
        them, with and without the leading space a job's later commands carry, filling
        the mask cache for every round state on the way. Returns the milliseconds it
        took. Synchronous; called before the clock starts."""
        began = time.monotonic()
        self._settle()
        language = self._language(descriptor)
        for command in commands:
            for text in (command, " " + command):
                state = language.start
                for token_id in self._tokenize(text):
                    # The state a step chooses from; the one after the terminator is
                    # never asked for, since the round restarts there.
                    self._mask.allowed(descriptor, language, state, allow_control=False)
                    state = language.advance(state, self._pieces[token_id])
                    if not state:
                        break
        return (time.monotonic() - began) * 1000.0
