# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The seams between the pieces of the browser port, written down before the pieces.

Every class in the port is coded against the shapes here rather than against another
piece's implementation, so the channel, the machine, the runners and the page can be
built at the same time. The JavaScript half of the same contract -- the channel's slot
layout and its race rules -- is ``CONTRACTS.md`` beside this package.

What is a Protocol here is implemented elsewhere: ``AsyncModelWorker`` by the
coop-count-web channel (``SyncModelWorker`` in ``model_channel.js``, ``NodeModelWorker``
in ``node_worker.py``) and by ``FakePilotWorker``; ``PilotMachine`` by
``machine.PilotJsMachine``; ``PromptArm`` by ``prompt_player.BrowserPromptPlayer``;
``Runner`` by ``runner.WallClockZeosRunner`` and ``runner.WallClockPromptRunner``;
``OpenRun`` by ``page.open_run``. A constructor's signature is pinned by a ``*Factory``
Protocol whose ``__call__`` takes the constructor's arguments.

The plain data -- ``BoardSpec``, ``RunResult``, the poll results, frames and decision
records -- is concrete, and so are ``load_board``, ``MonotonicClock`` and ``LocalStop``,
which have one obvious implementation.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Final, Literal, NotRequired, Protocol, TypedDict

from zeos.core.ids import JobId
from zeos.machine.base import DecodeResult
from zeos_coop_count_web.js_machine import Bridge, ZeosModelWorker
from zeos_space_invaders.game import Rules

if TYPE_CHECKING:
    from zeos_space_invaders.players.zeos.api_machine import Native
    from zeos_space_invaders.players.zeos.player import ZeosDriver

# --- names ---------------------------------------------------------------------------

Arm = Literal["zeos", "prompt"]
"""Which player drives the game: the descriptor tree under the kernel, or the prompt loop."""

BoardName = Literal["default", "ablation"]
"""The two boards, by the settings file each is a copy of."""

MachineKind = Literal["model", "stub"]
"""What answers the worker channel: the real model thread, or the latency-simulating stub."""

ARMS: Final[tuple[Arm, ...]] = ("zeos", "prompt")
BOARDS: Final[tuple[BoardName, ...]] = ("default", "ablation")

# --- the worker channel (JavaScript side in CONTRACTS.md) ------------------------------

#: Int32 slots of the channel's SharedArrayBuffer. 0 and 1 are coop-count-web's as they
#: were; 2 and 3 are the additions this port makes. Each holds ``requestId + 1`` so 0
#: always means "none".
SLOT_STATE: Final = 0
SLOT_LENGTH: Final = 1
SLOT_ABORT: Final = 2
SLOT_IN_FLIGHT: Final = 3
#: Byte offset of the reply frame, unchanged.
FRAME_OFFSET: Final = 16

#: What a synchronous ``call()`` throws while a decode step is in flight. A caller that
#: wants the channel must ``cancelDecode()`` and drain ``pollDecode`` first.
CHANNEL_BUSY: Final = "channel busy: decode in flight"

#: Prefill chunk for a pilot step: a cancel lands within one chunk of a board being
#: read. 128, chosen end to end on WebGPU with a quiet GPU (README, *Measured*): against
#: 256 it lowered the ablation board's pilot lag p95 from 28-34 to 16-19 ticks and the
#: longest cancel, and was level on the default board. Chunk 64 read a board 1.65x
#: slower than 256 in the benchmark (``bench/RESULTS.md``), from each run's fixed cost.
DEFAULT_MAX_CHUNK: Final = 128

#: How long ``decode`` polls a step in flight before handing the kernel a stall.
DEFAULT_STALL_MS: Final = 5.0


class DecodeOptions(TypedDict):
    """The options ``beginDecodeStep`` takes; ``decodeStep``'s, plus ``maxChunk``.

    ``allowedBlocks`` and ``allowedTokens`` are 0/1 flag arrays (``None`` means all), in
    whatever form the ``Bridge`` makes of them. ``maxChunk`` caps the positions one
    prefill run may cover, which is the granularity at which a cancel is noticed.
    """

    allowedBlocks: object | None
    allowedTokens: object | None
    sample: NotRequired[Mapping[str, float] | None]
    maxChunk: NotRequired[int]


class DecodeStats(TypedDict):
    """What one step cost the worker, reported whether it finished or was cancelled.

    ``positions``: KV positions computed by this step (the replay and the new suffix).
    ``chunks``: prefill runs those positions took. ``fillMs``: wall time spent filling,
    excluding the final one-position decode.
    """

    positions: int
    chunks: int
    fillMs: float


class DecodeDone(TypedDict):
    """A step that ran to the end: the token chosen and the attention it measured.

    ``attention`` is the per-block mass ``decodeStep`` reports (``None`` for a backend that
    cannot measure it). ``resident`` is the context's positions in the KV cache after the
    step, as the channel's reply carries it. Never delivered for a step that was
    cancelled, even when the worker had finished it by the time the cancel arrived.
    """

    tokenId: int
    attention: object | None
    cancelled: Literal[False]
    resident: int
    stats: DecodeStats


class DecodeCancelled(TypedDict):
    """A step abandoned at a chunk boundary.

    ``resident`` is how many of the context's positions are in the KV cache and stay
    valid: the next step for the same context resumes filling from there rather than
    from the start.
    """

    cancelled: Literal[True]
    resident: int
    stats: DecodeStats


PollResult = DecodeDone | DecodeCancelled | None
"""``pollDecode``'s answer once normalised to Python: ``None`` means still running."""


class AsyncModelWorker(ZeosModelWorker, Protocol):
    """``ZeosModelWorker`` with a decode step that does not block the caller.

    At most one step is in flight per worker. While one is, the synchronous methods
    (``tokenize``, ``append``, ``decodeStep`` and the rest) raise ``CHANNEL_BUSY``;
    ``piece`` and a cached ``info`` do not touch the channel and keep working.
    """

    def beginDecodeStep(self, jobId: str, opts: object) -> int:
        """Post one decode step for ``jobId`` and return its request id; never waits.

        ``opts`` is a ``DecodeOptions`` as the ``Bridge`` renders it. Raises
        ``CHANNEL_BUSY`` if a step is already in flight.
        """
        ...

    def pollDecode(self, timeoutMs: float) -> object | None:
        """Wait up to ``timeoutMs`` for the step in flight; ``None`` if it is still running.

        Returns early the moment the result lands. The answer is a ``DecodeDone`` or a
        ``DecodeCancelled`` (as a JsProxy under Pyodide; the machine normalises it), and
        receiving it clears ``inFlight``. With nothing in flight it returns ``None`` at
        once. This is the run loop's only sleep while a step is in flight.
        """
        ...

    def cancelDecode(self) -> None:
        """Ask the step in flight to stop at its next chunk boundary; never waits.

        A no-op with nothing in flight. The step stays in flight until ``pollDecode``
        returns its ``DecodeCancelled``; a result that raced the cancel is dropped and
        reported as cancelled.
        """
        ...

    @property
    def inFlight(self) -> bool:
        """Whether a step has been begun and its result not yet returned by ``pollDecode``."""
        ...


# --- the machine ----------------------------------------------------------------------

#: Commands decoded once at load so the grammar's mask cache holds their round states
#: before the clock starts (a cold state costs 0.1-0.8 s).
PREWARM_COMMANDS: Final[tuple[str, ...]] = (
    "write stdout left;",
    "write stdout right;",
    "write stdout shoot;",
    "read stdin;",
)

StepOutcome = Literal["token", "stall", "cancelled", "native"]


class StepLogEntry(TypedDict):
    """One ``decode`` of the machine, for the benchmark and the integrator's tuning.

    ``outcome``: ``token`` (a step completed), ``stall`` (still in flight, the kernel got
    a stall), ``cancelled`` (a cancelled step was drained), ``native`` (a registered
    behaviour answered without the worker). ``wall_ms`` is the time ``decode`` took.
    """

    job: str
    outcome: StepOutcome
    positions: int
    chunks: int
    fill_ms: float
    wall_ms: float


class PilotMachine(Protocol):
    """What ``PilotJsMachine`` adds to ``JsMachine``, and what ``ZeosDriver`` reads off it.

    ``decode`` never blocks on the model: while a step is in flight it polls for at most
    ``stall_ms`` and returns a stall, so the job stays preemptible. Every synchronous
    worker call goes through ``_settle()``, which cancels and drains a step in flight
    first. A step in flight is cancelled when its context changes under it (inject,
    trunc, splice, fork, a narrowing ``set_mask``), when another job is decoded
    (deschedule), and on ``invalidate``.
    """

    #: Request-to-syscall seconds of the last completed pilot command; ``ZeosDriver``
    #: puts it on the decision.
    last_roundtrip: float | None
    roundtrips: list[float]
    #: Steps cancelled, for any reason.
    cancellations: int
    #: Cancel-to-drained wall time of each cancellation, in milliseconds.
    cancel_ms: list[float]
    #: The most recent decodes, bounded; running totals live beside it.
    step_log: deque[StepLogEntry]

    def register_behaviour(
        self, descriptor: str, behaviour: Callable[[Native], DecodeResult]
    ) -> None:
        """Serve ``descriptor``'s jobs locally: ``decode`` calls ``behaviour`` and never
        touches the worker. The native ``APIMachineBase.register_behaviour`` semantics."""
        ...

    def invalidate(self, job: JobId) -> None:
        """Drop whatever is being generated for ``job``: cancel its step in flight, reset
        its command parser and its grammar round. The text decoded so far is kept
        (native ``partial="keep"``). A no-op if nothing is in flight."""
        ...

    def prewarm(
        self, descriptor: str = "pilot", commands: Sequence[str] = PREWARM_COMMANDS
    ) -> float:
        """Fill the grammar's mask cache for ``descriptor`` with ``commands``, and return
        the milliseconds it took. Called before the clock starts; synchronous."""
        ...

    def close(self) -> None:
        """Cancel and drain any step in flight and destroy every context."""
        ...


class PilotMachineFactory(Protocol):
    """``PilotJsMachine.__init__``.

    ``stall_ms`` bounds each ``pollDecode`` inside ``decode``; ``max_chunk`` is passed as
    ``maxChunk``; ``turn_ends_at_call`` ends a completion at its first complete syscall;
    ``forbid_verbs`` removes verbs from the grammar (``("exit",)`` is the fallback for a
    model that exits).
    """

    def __call__(
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
    ) -> PilotMachine: ...


# --- the prompt arm -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptReply:
    """One finished prompt-loop request.

    ``action`` is ``PromptPlayer.parse(text)``: ``None`` when the reply named no action,
    which counts against the parse rate. ``latency`` is begin-to-reply seconds.
    """

    text: str
    action: str | None
    latency: float
    tokens: int

    @property
    def parsed(self) -> bool:
        return self.action is not None


class PromptArm(Protocol):
    """``BrowserPromptPlayer``: the prompt loop over the same worker, without blocking.

    Owns two worker contexts. ``warm`` prefills the system prompt into one before the
    clock starts; each ``begin`` forks the other from it, appends the history and the
    board, and starts an
    unconstrained reply of at most ``max_new`` tokens through ``beginDecodeStep``.
    """

    @property
    def busy(self) -> bool:
        """Whether a request is under way (begun, and its reply not yet returned)."""
        ...

    def warm(self) -> None:
        """Create the context and prefill the system prompt; synchronous."""
        ...

    def begin(self, obs: str, info: Mapping[str, object]) -> None:
        """Start a request for this board. Raises if one is already under way."""
        ...

    def poll(self, timeout_s: float) -> PromptReply | None:
        """Advance the request for at most ``timeout_s``: the reply once it is complete,
        else ``None``. With nothing under way, returns ``None`` at once."""
        ...

    def close(self) -> None:
        """Cancel any request under way and destroy the context."""
        ...


class PromptArmFactory(Protocol):
    """``BrowserPromptPlayer.__init__``; ``view`` is a ``zeos_space_invaders`` view."""

    def __call__(
        self,
        worker: AsyncModelWorker,
        *,
        view: object,
        rules: Rules,
        history: int = 5,
        max_new: int = 6,
    ) -> PromptArm: ...


# --- boards ---------------------------------------------------------------------------

#: Settings-file keys under ``board`` whose ``Rules`` field has another name.
_RULES_FIELD: Final[Mapping[str, str]] = {"width": "w", "height": "h"}


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """One board as a run needs it: the game's rules and the ``play`` settings.

    ``seed`` is the one asked for, else the file's (the ablation file has none, so
    ``None`` there means an unseeded game). ``actions_per_tick`` ``None`` is the game's
    default budget.
    """

    name: BoardName
    rules: Rules
    tick_seconds: float
    max_steps: int
    history: int
    view: str
    effort: str
    seed: int | None
    actions_per_tick: int | None


def board_text(name: BoardName) -> str:
    """The packaged settings file for ``name``, verbatim."""
    return (resources.files(__package__) / "boards" / f"{name}.json").read_text()


def load_board(name: BoardName, seed: int | None = None) -> BoardSpec:
    """The board ``name``, from the package's copy of its settings file.

    The same mapping ``agent --settings`` applies: ``width``/``height`` are ``Rules.w``/
    ``Rules.h`` and every other ``board`` key is a ``Rules`` field of the same name.
    """
    if name not in BOARDS:
        raise ValueError(f"unknown board {name!r}; expected one of {BOARDS}")
    settings = json.loads(board_text(name))
    board: dict[str, object] = settings["board"]
    play: dict[str, object] = settings["play"]
    rules = Rules(**{_RULES_FIELD.get(k, k): v for k, v in board.items()})  # pyright: ignore[reportArgumentType]
    file_seed = play.get("seed")
    per_tick = play.get("actions_per_tick")
    return BoardSpec(
        name=name,
        rules=rules,
        tick_seconds=float(str(play["tick"])),
        max_steps=int(str(play["max_steps"])),
        history=int(str(play["history"])),
        view=str(play["view"]),
        effort=str(play["effort"]),
        seed=seed if seed is not None else (None if file_seed is None else int(str(file_seed))),
        actions_per_tick=None if per_tick is None else int(str(per_tick)),
    )


# --- the clock and the stop flag ------------------------------------------------------


class Clock(Protocol):
    """Wall time for the runners, injectable so a test can drive it."""

    def now(self) -> float:
        """Monotonic seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` (at most), used only when nothing is in flight. Under
        Pyodide this is an ``Atomics.wait`` on the control buffer, so a stop wakes it."""
        ...


class MonotonicClock:
    """``Clock`` over ``time.monotonic`` and ``time.sleep``, for CPython."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class StopFlag(Protocol):
    """Asked once per loop iteration; set from outside the loop. In the page it reads
    ``control[0]`` of the control SharedArrayBuffer, because the loop never yields to the
    worker's event loop and a ``stop`` message would never be delivered."""

    def is_set(self) -> bool: ...


class LocalStop:
    """``StopFlag`` set by a call, for CPython and tests."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


# --- frames and decisions -------------------------------------------------------------

DecisionBy = Literal["pilot", "evade", "prompt"]


class DecisionRecord(TypedDict):
    """One move, as the page draws it and the ``decision`` message carries it.

    ``tick`` is the board the move was chosen against, ``tick_applied`` the tick it
    landed on; ``lag_ticks`` is their difference. ``latency`` is request-to-move seconds
    (``None`` for the reflex, which asks no model). ``preempted``: the kernel journalled a
    preemption while this move was being made.
    """

    by: DecisionBy
    action: str
    tick: int
    tick_applied: int
    lag_ticks: int
    latency: float | None
    preempted: bool
    applied: bool


class Frame(TypedDict):
    """The game after one tick, as the ``frame`` message carries it. JSON-able.

    The game fields are ``snapshot(game)``'s, with ``kills = score // 10`` and positions
    as ``[row, col]`` lists. ``decisions`` are the moves made since the previous frame;
    ``catchup`` is how many extra ticks this frame applied because the loop fell behind.
    The running totals are the ones ``RunResult`` ends with.
    """

    arm: Arm
    board: BoardName
    tick: int
    text: str
    lives: int
    kills: int
    score: int
    player: int
    monsters: list[list[int]]
    missile: list[int] | None
    dangers: list[list[int]]
    can_shoot: bool
    over: bool
    won: bool
    decisions: list[DecisionRecord]
    catchup: int
    preemptions: int
    cancellations: int
    reflexes: int


class FrameSink(Protocol):
    """Where a runner hands each frame; the page's sink posts a ``frame`` message."""

    def __call__(self, frame: Frame) -> None: ...


# --- results --------------------------------------------------------------------------


class LagStats(TypedDict):
    """``DecisionRecord.lag_ticks`` over a run's model-made moves (0.0 for none)."""

    mean: float
    p50: float
    p95: float
    max: float


@dataclass
class RunResult:
    """What one run ends with: the page's ``finished`` message and the handover table.

    ``decisions`` counts moves by any author; ``reflexes`` those made by ``evade``.
    ``overrun_ms`` is the total time the loop spent past a tick's due time, and
    ``catchup_ticks`` the extra ticks applied to catch up. ``verdicts`` are
    ``ZeosDriver.verdicts()`` and ``journal`` the encoded kernel records (both empty for
    the prompt arm); ``parse_rate`` is the prompt arm's share of parseable replies.
    """

    arm: Arm
    board: BoardName
    seed: int | None
    lives: int
    kills: int
    ticks: int
    decisions: int
    preemptions: int
    cancellations: int
    reflexes: int
    lag_ticks: LagStats
    overrun_ms: float
    catchup_ticks: int
    verdicts: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    journal: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    parse_rate: float | None = None
    #: Per-arm measurements beyond the table above, for the handover and tuning: warm-up
    #: and prewarm time, overrun percentiles, cancel latency, step totals, faults, the
    #: longest gap between pilot moves. Plain JSON values; ``{}`` when there are none.
    extras: dict[str, object] = field(default_factory=dict[str, object])

    def to_json(self) -> dict[str, object]:
        return asdict(self)


# --- runners --------------------------------------------------------------------------


class Runner(Protocol):
    """One episode against the wall clock, in a single thread.

    The zeos loop: ``driver.run_kernel(deadline=due)``; when ``now >= due`` tick the
    controls once per elapsed interval (extra ticks count as catch-up), emit a frame,
    ``driver.sense(...)``, advance ``due``. Ends when the game is over, ``max_steps`` is
    reached or the stop flag is set (checked every iteration, so within one tick).
    """

    def run(self) -> RunResult: ...


class ZeosRunnerFactory(Protocol):
    """``WallClockZeosRunner.__init__``: ``driver`` is the native ``ZeosDriver``, built
    over a ``PilotMachine``."""

    def __call__(
        self,
        driver: ZeosDriver,
        spec: BoardSpec,
        *,
        stop: StopFlag | None = None,
        on_frame: FrameSink | None = None,
        clock: Clock | None = None,
    ) -> Runner: ...


class PromptRunnerFactory(Protocol):
    """``WallClockPromptRunner.__init__``."""

    def __call__(
        self,
        player: PromptArm,
        spec: BoardSpec,
        *,
        stop: StopFlag | None = None,
        on_frame: FrameSink | None = None,
        clock: Clock | None = None,
    ) -> Runner: ...


# --- the page -------------------------------------------------------------------------


class Run(Protocol):
    """A run ``open_run`` has assembled: warm it (the ``warming`` phase: system prompt
    prefill and mask prewarm, before the clock starts), then run it."""

    def warm(self) -> None: ...

    def run(self) -> RunResult: ...

    def close(self) -> None: ...


class OpenRun(Protocol):
    """``page.open_run``: everything ``si_worker.js``'s ``start`` needs, in one call.

    ``stub`` says the worker is the stub (``machine: "stub"``), which changes nothing but
    what the result is labelled with and the default latency.
    """

    def __call__(
        self,
        arm: Arm,
        board: BoardName,
        seed: int | None,
        worker: AsyncModelWorker,
        *,
        stub: bool = False,
        on_frame: FrameSink | None = None,
        stop: StopFlag | None = None,
    ) -> Run: ...


#: Messages between the page and ``si_worker.js``, by their ``type`` field (the
#: coop-count-web convention: ``{type, ...body}``). Bodies are in ``CONTRACTS.md``.
class PageMessage:
    BOOT: Final = "boot"
    ATTACH_MODEL: Final = "attachModel"
    START: Final = "start"


class WorkerMessage:
    READY: Final = "ready"
    WARMING: Final = "warming"
    STARTED: Final = "started"
    FRAME: Final = "frame"
    DECISION: Final = "decision"
    FINISHED: Final = "finished"
    ERROR: Final = "error"


#: Int32 slots of the control SharedArrayBuffer the page writes and the loop reads.
CONTROL_STOP: Final = 0
CONTROL_BYTES: Final = 16
