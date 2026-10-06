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
  narrowing ``set_mask``, block padding and ``invalidate`` cancel the job's step in
  flight, and so does a decode of another served job that needs the channel for a step
  of its own. A natively-served decode and an automatic ``read stdin`` need no channel
  and cancel nothing: a preemption reaches the machine as ``invalidate``, from the
  driver that reads the journal; its token is never
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
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

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
from zeos_coop_count_web.token_mask import CommandLanguage
from zeos_space_invaders.game import ACTIONS
from zeos_space_invaders.players.zeos.api_machine import ABI, Native

from zeos_space_invaders_web.contracts import (
    DEFAULT_MAX_CHUNK,
    DEFAULT_STALL_MS,
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
    "PILOT_PAYLOADS",
    "PayloadLanguage",
    "PilotJsMachine",
    "SETTLE_POLL_MS",
    "normalise_poll",
    "pilot_abi",
]

#: The pipe aliases the pilot binds (``goals/pilot.md``); the grammar offers no other.
PILOT_DESCRIPTORS: Mapping[str, Sequence[str]] = {"pilot": ("stdin", "stdout")}

#: The words a payload may be, per pipe alias: the pilot's ``stdout`` takes one move.
#: With a free-text payload the 4B wrote ``write stdout left left left 1`` on every board
#: (``bench/RESULTS.md``); narrowed, every command was a valid move, in fewer tokens.
PILOT_PAYLOADS: Mapping[str, Sequence[str]] = {"stdout": tuple(ACTIONS)}

#: A threshold for ``uncapped_above``, off by default: a step with more positions than
#: this to fill -- the first step over a descriptor body, a replay after the pager
#: splices -- would run at the worker's own chunk, where the per-run overhead matters
#: least. The price is that such a step can be cancelled only between the worker's own
#: (2048-position) runs, so an inject, trunc or splice behind it waits seconds in
#: ``_settle``.
UNCAPPED_ABOVE = 1024

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


_WORD = 7  # (7, alt, word, p): p characters of the alternative's payload word matched


class PayloadLanguage(CommandLanguage):
    """``CommandLanguage`` with the text payload of some pipe aliases narrowed to a choice
    of literal words, the terminator straight after.

    Verbs, pipes and the rest of the syntax are the ABI's as before; only a text verb's
    payload on an alias in ``payloads`` changes, from any run of characters to exactly one
    of that alias's words. The automaton gains one kind of state, ``(7, alt, word, p)``:
    ``p`` characters of word ``word`` matched after alternative ``alt``'s head.
    """

    def __init__(
        self,
        abi: SyscallABI,
        pipes: Sequence[str],
        *,
        valued: Sequence[str] = (),
        payloads: Mapping[str, Sequence[str]],
    ) -> None:
        super().__init__(abi, pipes, valued=valued)
        self._words: dict[int, tuple[str, ...]] = {}
        for i, alt in enumerate(self._alternatives):
            parts = alt.head.split(" ")
            if alt.tail == 1 and len(parts) == 3 and parts[1] in payloads:  # a text payload
                words = tuple(payloads[parts[1]])
                if not words or any(not w or w[0] == " " for w in words):
                    raise ValueError(f"payload words for {parts[1]!r} must be non-empty words")
                self._words[i] = words

    def _tail(self, state: tuple[int, ...], char: str) -> tuple[tuple[int, ...], ...]:
        words = self._words.get(state[1])
        if words is None:
            return super()._tail(state, char)
        return tuple((_WORD, state[1], w, 1) for w, word in enumerate(words) if word[0] == char)

    def _step(self, state: tuple[int, ...], char: str) -> Iterable[tuple[int, ...]]:
        if state[0] != _WORD:
            return super()._step(state, char)
        _, i, w, p = state
        word = self._words[i][w]
        if p == len(word):
            return self._terminate(i) if char == self.abi.terminator[0] else ()
        return ((_WORD, i, w, p + 1),) if word[p] == char else ()


def normalise_poll(raw: object) -> DecodeDone | DecodeCancelled | None:
    """``pollDecode``'s answer as the contract's TypedDicts; ``None`` while running.

    A Python worker (and ``PyodideAsyncWorker``) already answers in that shape, and is
    taken at its word; a JsProxy is read field by field. A field the contract requires
    and the answer lacks is an error, not a zero.
    """
    if raw is None or type(raw).__name__ == "JsNull":
        return None
    if isinstance(raw, dict):
        return cast("DecodeDone | DecodeCancelled", raw)
    js: Any = raw
    stats: DecodeStats = {
        "positions": int(js.stats.positions),
        "chunks": int(js.stats.chunks),
        "fillMs": float(js.stats.fillMs),
    }
    if bool(js.cancelled):
        return {"cancelled": True, "resident": int(js.resident), "stats": stats}
    return {
        "tokenId": int(js.tokenId),
        "attention": js.attention,
        "cancelled": False,
        "resident": int(js.resident),
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
    #: Model positions the worker holds KV for, as far as the machine knows: set from
    #: each step's answer, lowered by a trunc or splice.
    filled: int = 0


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
        stall_ms: float = DEFAULT_STALL_MS,
        max_chunk: int = DEFAULT_MAX_CHUNK,
        turn_ends_at_call: bool = True,
        forbid_verbs: Sequence[str] = (),
        payloads: Mapping[str, Sequence[str]] = PILOT_PAYLOADS,
        uncapped_above: int | None = None,
        tokenize_cache: int = 4096,
        step_log_size: int = 4096,
    ) -> None:
        if stall_ms < 0:
            raise ValueError("stall_ms must be >= 0")
        if max_chunk < 1:
            raise ValueError("max_chunk must be >= 1")
        if turn_ends_at_call and "read" in forbid_verbs:
            raise ValueError(
                "turn_ends_at_call ends each turn with a read, so 'read' cannot be forbidden"
            )
        if tokenize_cache < 0:
            raise ValueError("tokenize_cache must be >= 0")
        super().__init__(
            worker,
            bridge=bridge,
            abi=pilot_abi(forbid_verbs),
            descriptors=PILOT_DESCRIPTORS if descriptors is None else descriptors,
            block_size=block_size,
        )
        self._async = worker
        self.payloads = {k: tuple(v) for k, v in payloads.items()}
        self._uncapped_above = uncapped_above
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
        self._token_cache: OrderedDict[str, list[int]] = OrderedDict()
        self._token_cache_size = tokenize_cache

        self.last_roundtrip: float | None = None
        self.roundtrips: list[float] = []
        self.cancellations = 0
        self.cancel_ms: list[float] = []
        #: The most recent decodes; ``step_totals`` counts every one.
        self.step_log: deque[StepLogEntry] = deque(maxlen=step_log_size)
        self.step_totals: dict[StepOutcome, int] = {
            "token": 0,
            "stall": 0,
            "cancelled": 0,
            "native": 0,
        }
        self.positions_total = 0
        self.fill_ms_total = 0.0
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
        if self._token_cache_size == 0:
            self.tokenize_calls += 1
            return super()._tokenize(text)
        cached = self._token_cache.get(text)
        if cached is not None:
            self._token_cache.move_to_end(text)
            self.tokenize_hits += 1
            return list(cached)
        self.tokenize_calls += 1
        ids = super()._tokenize(text)
        self._token_cache[text] = ids
        if len(self._token_cache) > self._token_cache_size:
            self._token_cache.popitem(last=False)
        return list(ids)

    def _rewind(self, ctx: Any, kv_at: int) -> None:
        super()._rewind(ctx, kv_at)
        for job, held in self._contexts.items():
            if held is ctx and job in self._rounds:
                self._rounds[job].filled = min(self._rounds[job].filled, kv_at)

    def _plan_step(self, job: JobId, *, allow_control: bool) -> _StepPlan:
        # Planning flushes and frames through synchronous worker calls.
        assert self._flight is None, "a step was planned with another in flight"
        return super()._plan_step(job, allow_control=allow_control)

    def _language(self, descriptor: str) -> CommandLanguage:
        language = self._languages.get(descriptor)
        if language is None:
            language = PayloadLanguage(
                self.abi,
                self.aliases(descriptor),
                valued=self.valued(descriptor),
                payloads=self.payloads,
            )
            self._languages[descriptor] = language
        return language

    def _with_chunk(self, options: object, pending: int) -> object:
        """The step's options with ``maxChunk`` (a dict from ``PythonBridge``, a
        JavaScript object from ``PyodideBridge``), left out for a long fill."""
        if self._uncapped_above is not None and pending > self._uncapped_above:
            return options
        if isinstance(options, dict):
            return {**options, "maxChunk": self._max_chunk}  # pyright: ignore[reportUnknownVariableType]
        setattr(options, "maxChunk", self._max_chunk)  # noqa: B010 - a JsProxy attribute
        return options

    def _poll(self, timeout_ms: float) -> DecodeDone | DecodeCancelled | None:
        """Poll the step in flight. A worker that answers with an error has ended the
        step, so the flight is dropped before the error goes on up."""
        try:
            return normalise_poll(self._async.pollDecode(timeout_ms))
        except BaseException:
            self._flight = None
            raise

    def _cancel(self) -> None:
        """Ask the step in flight to stop; never waits."""
        flight = self._flight
        if flight is None or flight.cancelled_at is not None:
            return
        self._async.cancelDecode()
        flight.cancelled_at = time.monotonic()
        self.cancellations += 1

    def _drained(self, answer: DecodeDone | DecodeCancelled) -> None:
        """The cancelled step has been handed back; whatever it answered is dropped."""
        flight = self._flight
        assert flight is not None and flight.cancelled_at is not None
        self.cancel_ms.append((time.monotonic() - flight.cancelled_at) * 1000.0)
        self._flight = None
        rnd = self._rounds.get(flight.job)
        if rnd is not None:
            # A step that finished regardless filled everything it was planned against.
            rnd.filled = answer["resident"]

    def _settle(self) -> None:
        """Cancel the step in flight and wait for the channel to hand it back, so that a
        synchronous worker call can be made. At most one prefill chunk."""
        if self._flight is None:
            return
        self._cancel()
        answer = self._poll(SETTLE_POLL_MS)
        while answer is None:
            answer = self._poll(SETTLE_POLL_MS)
        self._drained(answer)

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
        # As the native machine drops a finished reply's pending read with the reply.
        rnd.auto_read = False

    def _log(
        self, job: JobId, outcome: StepOutcome, began: float, stats: DecodeStats | None = None
    ) -> None:
        self.step_totals[outcome] += 1
        if stats:
            self.positions_total += stats["positions"]
            self.fill_ms_total += stats["fillMs"]
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
            # Served locally: the channel, and any step in flight on it, is untouched. A
            # preemption of the job that step is for reaches here as ``invalidate``.
            result = self._decode_native(job, native)
            self._log(job, "native", began)
            return result

        rnd = self._rounds[job]
        if rnd.auto_read:
            # Needs no worker either, so a step in flight for another job is left alone.
            rnd.auto_read = False
            rnd.start = len(self._ctx_of(job).tokens)
            self._log(job, "native", began)
            return DecodeResult(
                tokens=(),
                request=self.abi.parse(_READ_STDIN),
                attention=None,
                attention_hint=AttentionHint(tags=("self",)),
            )

        flight = self._flight
        if flight is not None and flight.cancelled_at is None:
            if flight.job != job:
                # This job's step needs the channel, which serves one step at a time.
                self._cancel()
            elif flight.plan.allow_control != allow_control:
                self._cancel()  # planned under the other token mask
        if self._flight is not None and self._flight.cancelled_at is not None:
            drained_job = self._flight.job
            drained = self._poll(self._stall_ms)
            if drained is None:
                self.stalls += 1
                self._log(job, "stall", began)
                return self._stall()
            self._drained(drained)
            self._log(drained_job, "cancelled", began, drained["stats"])

        if self._flight is None:
            plan = self._plan_step(job, allow_control=allow_control)
            now = time.monotonic()
            pending = len(plan.ctx.ids) - rnd.filled
            request_id = self._async.beginDecodeStep(
                plan.ctx.key, self._with_chunk(plan.options, pending)
            )
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
            self._flight = None
            raise RuntimeError(f"job {job}: the worker cancelled a step nobody cancelled")
        self._flight = None
        rnd.filled = answer["resident"]
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
        self._rounds[child].filled = self._rounds[parent].filled
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
        ``partial="keep"``). A pending automatic ``read stdin`` goes too, as the native
        machine drops a finished reply's read with it. A no-op for a locally-served job,
        or one with no context."""
        if job in self._native or job not in self._contexts:
            return
        self._cancel_job(job)
        self._reset_round(job)

    def prewarm(
        self, descriptor: str = "pilot", commands: Sequence[str] = PREWARM_COMMANDS
    ) -> float:
        """Walk ``commands`` through ``descriptor``'s grammar as the worker tokenizes
        them, with and without the leading space a job's later commands carry, filling
        the mask cache for every round state a step chooses from on the way, and for the
        state after each space. Returns the milliseconds it
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
                # And after each space, for a model that puts the space at the end of a
                # token where the worker's tokenizer puts it at the start of the next.
                for at, char in enumerate(text):
                    if char == " " and at:
                        spaced = language.advance(language.start, text[: at + 1])
                        if spaced:
                            self._mask.allowed(descriptor, language, spaced, allow_control=False)
        return (time.monotonic() - began) * 1000.0
