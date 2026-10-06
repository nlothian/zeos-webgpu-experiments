# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The wall clock, in one thread: the game ticks on time while the model works.

The native runners pace the world on a clock thread and decide on another;
Pyodide cannot start a thread, so here one loop does both. For the zeos arm it
pumps the kernel up to the next tick's due time, applies every tick that has
fallen due (more than one when the loop overran, counted as catch-up), hands the
newest board to the driver and goes round again. The model never blocks it: a
decode step in flight costs the kernel a stall of ``stall_ms``, which is the
loop's only sleep while anything is in flight, and ``Clock.sleep`` is reached
only when every job is blocked on a pipe. The prompt arm is the same clock with
``PromptArm.poll`` in place of the kernel.

``ZeosDriver`` is the native one, unchanged: ``run_kernel``, ``sense``,
``collect`` (inside ``run_kernel``) and ``verdicts``. The driver compares
deadlines against ``time.monotonic()``, so the clock handed to the zeos runner
must tell the same time; ``MonotonicClock`` does, and so does a clock that only
changes how it sleeps. Under Pyodide the page injects such a clock: ``now()`` is
``time.monotonic()`` and ``sleep()`` an ``Atomics.wait`` on the control buffer, so a
stop wakes it. The zeos runner refuses a clock that disagrees at construction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from zeos.machine.base import DecodeResult
from zeos_space_invaders.game import (
    Controls,
    Game,
    snapshot,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.players.base import FALLBACK_ACTION
from zeos_space_invaders.players.zeos.api_machine import Native
from zeos_space_invaders.players.zeos.player import (
    GAME_STATE,
    REFLEX,
    DriverMachine,
    ZeosDriver,
    evade_behaviour,
    threat_reading,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.runlog import Decision
from zeos_space_invaders.utils import VIEWS

from .contracts import (
    Arm,
    BoardSpec,
    Clock,
    DecisionBy,
    DecisionRecord,
    Frame,
    FrameSink,
    MonotonicClock,
    PromptArm,
    PromptReply,
    RunResult,
    StopFlag,
)
from .metrics import (
    OverrunStats,
    due_ticks,
    lag_stats,
    model_lags,
    overrun_stats,
    parse_rate,
)

#: How long one warm-up batch may pump before the loop looks at the clock again.
WARM_BATCH_S = 0.05
#: Long enough for a ~3k-token system prompt at ~300 tok/s, with room to spare.
WARM_TIMEOUT_S = 120.0

#: How far the zeos runner's clock may disagree with ``time.monotonic()``.
CLOCK_TOLERANCE_S = 0.5

PREEMPTED = "job.preempted"
READ = "pipe.read"
FIRED = "vector.fired"


# --- what the runner reads off its collaborators --------------------------------


class KernelEvents(Protocol):
    """The kernel's event log, which is all the runner reads off the kernel."""

    @property
    def events(self) -> Sequence[object]: ...


class CountingMachine(Protocol):
    """What the runner reads off the machine: ``DriverMachine.cancellations``."""

    @property
    def cancellations(self) -> int: ...


class ZeosDriverLike(Protocol):
    """The part of ``ZeosDriver`` the runner drives; ``ZeosDriver`` is the one."""

    controls: Controls | None
    last_applied: bool

    @property
    def kernel(self) -> KernelEvents: ...

    @property
    def machine(self) -> CountingMachine: ...

    @property
    def started(self) -> bool: ...

    @property
    def preemptions(self) -> int: ...

    def start(self) -> None: ...

    def run_kernel(self, deadline: float | None = None) -> Decision | None: ...

    def sense(self, board: str, info: dict[str, Any], note: object = None) -> None: ...

    def verdicts(self) -> list[dict[str, Any]]: ...

    def journal(self) -> list[dict[str, Any]]: ...


class ReflexMachine(DriverMachine, Protocol):
    """A ``DriverMachine`` that can serve the reflex locally; ``PilotJsMachine`` and
    ``APIMachineBase`` are both one."""

    def register_behaviour(
        self, descriptor: str, behaviour: Callable[[Native], DecodeResult]
    ) -> None: ...


def build_driver(machine: ReflexMachine, spec: BoardSpec) -> ZeosDriver:
    """A ``ZeosDriver`` over ``machine`` for ``spec``'s board: the reflex registered,
    and the pilot told the rules and view of the game it will fly."""
    machine.register_behaviour(REFLEX, evade_behaviour)
    return ZeosDriver(machine=machine, view=VIEWS[spec.view](), rules=spec.rules)


# --- the parts both arms share ---------------------------------------------------


class _WallClock:
    """The game, its stick and the tick bookkeeping, for either arm."""

    arm: Arm

    def __init__(
        self,
        spec: BoardSpec,
        *,
        stop: StopFlag | None,
        on_frame: FrameSink | None,
        clock: Clock | None,
        max_ticks: int | None,
        max_seconds: float | None,
    ) -> None:
        self.spec = spec
        self.clock: Clock = clock if clock is not None else MonotonicClock()
        self.stop = stop
        self.on_frame = on_frame
        self.tick_s = spec.tick_seconds
        self.max_ticks = spec.max_steps if max_ticks is None else max_ticks
        self.max_seconds = max_seconds
        # `Any`: the game package is untyped, and its shapes are pinned by `Frame`.
        self.game: Any = Game(seed=spec.seed, rules=spec.rules)
        # The game's own budget, whoever is driving: one write, one move.
        self.controls: Any = Controls(self.game, per_tick=spec.actions_per_tick)
        self.view: Any = VIEWS[spec.view]()
        self.records: list[DecisionRecord] = []
        #: How late each batch of due ticks was noticed, in milliseconds.
        self.overruns_ms: list[float] = []
        self.catchup_ticks = 0
        self.frames = 0
        self._unframed: list[DecisionRecord] = []
        #: ``(board, info)`` of the tick just applied, rendered once for the frame
        #: and whoever is handed the board next.
        self._now: tuple[str, dict[str, Any]] | None = None
        self.reflexes = 0
        #: What the last frame carried, so a closing frame is sent only when the
        #: page has not seen the end of the run.
        self._framed: tuple[int, int, int, int] | None = None
        self._due = 0.0
        self._started_at = 0.0
        #: Clock time the loop exited, before the result is assembled (judging the
        #: criteria and encoding the journal take a while on a long run).
        self.ended_at = 0.0

    def _info(self) -> dict[str, Any]:
        """``snapshot(game)``: what a view, the driver and a player are handed."""
        return cast(dict[str, Any], snapshot(self.game))

    def _look(self) -> tuple[str, dict[str, Any]]:
        """The board and snapshot as they are now; the tick's own when nothing has
        moved since it was applied."""
        if self._now is None:
            self._now = (self.game.render(), self._info())
        return self._now

    # -- the clock ------------------------------------------------------------

    def _finished(self) -> bool:
        if self.stop is not None and self.stop.is_set():
            return True
        if self.game.over or self.game.ticks >= self.max_ticks:
            return True
        return (
            self.max_seconds is not None and self.clock.now() - self._started_at >= self.max_seconds
        )

    def _start_clock(self) -> None:
        self._started_at = self.clock.now()
        self._due = self._started_at + self.tick_s

    def _tick_due(self) -> int:
        """Apply every tick that has fallen due; how many were applied.

        One frame for the lot, carrying the extra ticks as ``catchup``: a loop that
        overran does not get the ticks it missed back, it pays for them at once,
        which is what the world would have done to a player that slow.
        """
        now = self.clock.now()
        due = due_ticks(now, self._due, self.tick_s)
        if not due:
            return 0
        self.overruns_ms.append(round((now - self._due) * 1000, 3))
        applied = 0
        for _ in range(due):
            self.controls.tick()
            applied += 1
            if self.game.over or self.game.ticks >= self.max_ticks:
                break
        self._due += due * self.tick_s
        self.catchup_ticks += applied - 1
        self._now = None
        self._emit(catchup=applied - 1)
        return applied

    # -- records and frames ---------------------------------------------------

    def _record(
        self,
        by: DecisionBy,
        action: str,
        tick: int,
        *,
        latency: float | None,
        preempted: bool,
        applied: bool,
    ) -> DecisionRecord:
        applied_at = self.game.ticks
        record: DecisionRecord = {
            "by": by,
            "action": action,
            "tick": tick,
            "tick_applied": applied_at,
            "lag_ticks": applied_at - tick,
            "latency": latency,
            "preempted": preempted,
            "applied": applied,
        }
        self.records.append(record)
        self._unframed.append(record)
        if by == "evade":
            self.reflexes += 1
        # A move changes the board, so the tick's rendering is stale.
        self._now = None
        return record

    def _counts(self) -> tuple[int, int, int]:
        """Running ``(preemptions, cancellations, reflexes)`` for a frame."""
        return 0, 0, 0

    def _emit(self, catchup: int) -> None:
        board, info = self._look()
        decisions, self._unframed = self._unframed, []
        preemptions, cancellations, reflexes = self._counts()
        self._framed = (int(self.game.ticks), preemptions, cancellations, reflexes)
        self.frames += 1
        if self.on_frame is None:
            return
        missile = info["missile"]
        frame: Frame = {
            "arm": self.arm,
            "board": self.spec.name,
            "tick": int(info["ticks"]),
            "text": board,
            "lives": int(info["lives"]),
            "kills": int(info["score"]) // 10,
            "score": int(info["score"]),
            "player": int(info["player"]),
            "monsters": [list(pos) for pos in cast(Mapping[int, Any], info["monsters"]).values()],
            "missile": None if missile is None else list(missile),
            "dangers": [list(d) for d in cast(list[Any], info["dangers"])],
            "can_shoot": bool(info["can_shoot"]),
            "over": bool(self.game.over),
            "won": bool(self.game.won),
            "decisions": decisions,
            "catchup": catchup,
            "preemptions": preemptions,
            "cancellations": cancellations,
            "reflexes": reflexes,
        }
        self.on_frame(frame)

    def _close(self) -> None:
        """End the loop: note when, and send a closing frame if the last one sent
        does not already carry every decision and the final totals."""
        self.ended_at = self.clock.now()
        if self._unframed or self._framed != (int(self.game.ticks), *self._counts()):
            self._emit(catchup=0)

    def overrun_stats(self) -> OverrunStats:
        return overrun_stats(self.overruns_ms)

    def _result(self, **arm_fields: Any) -> RunResult:
        return RunResult(
            arm=self.arm,
            board=self.spec.name,
            seed=self.spec.seed,
            lives=int(self.game.lives),
            kills=int(self.game.score) // 10,
            ticks=int(self.game.ticks),
            decisions=len(self.records),
            lag_ticks=lag_stats(model_lags(self.records)),
            overrun_ms=round(sum(self.overruns_ms), 3),
            catchup_ticks=self.catchup_ticks,
            **arm_fields,
        )


# --- the zeos arm ----------------------------------------------------------------


class WallClockZeosRunner(_WallClock):
    """One episode with the kernel deciding, against the wall clock, in one thread.

    ``warm()`` brings the kernel up and pumps it with the game paused until every
    job is blocked -- the pilot waiting on its first board -- so the system-prompt
    prefill is paid before the clock starts. ``run()`` warms if that has not
    happened, delivers the tick-0 board and plays until the game ends, ``max_ticks``
    (the board's ``max_steps`` by default) is reached, ``max_seconds`` pass or the
    stop flag is set.
    """

    arm: Arm = "zeos"

    def __init__(
        self,
        driver: ZeosDriverLike,
        spec: BoardSpec,
        *,
        stop: StopFlag | None = None,
        on_frame: FrameSink | None = None,
        clock: Clock | None = None,
        max_ticks: int | None = None,
        max_seconds: float | None = None,
        warm_timeout_s: float = WARM_TIMEOUT_S,
    ) -> None:
        super().__init__(
            spec,
            stop=stop,
            on_frame=on_frame,
            clock=clock,
            max_ticks=max_ticks,
            max_seconds=max_seconds,
        )
        skew = self.clock.now() - time.monotonic()
        if abs(skew) > CLOCK_TOLERANCE_S:
            raise ValueError(
                f"the clock is {skew:+.3f}s off time.monotonic(); ZeosDriver reads its "
                "deadlines on time.monotonic(), so the zeos runner needs a clock that "
                "tells the same time (MonotonicClock, or under Pyodide one whose now() "
                "is time.monotonic() and whose sleep() is an Atomics.wait)"
            )
        self.driver = driver
        self.warm_timeout_s = warm_timeout_s
        # The driver applies a write inside its own pump, so a job resumed in the
        # same breath sees the ship where the write put it.
        driver.controls = self.controls
        self.warmed = False
        #: Wall seconds `warm()` took.
        self.warm_s = 0.0
        self._event_cursor = 0
        self._preemptions = 0
        #: The tick of the newest board handed to the driver.
        self._delivered: int | None = None
        #: The tick of the board the pilot last read off `game.state`: what its next
        #: move answers. The driver stamps a move with the newest board delivered,
        #: which runs ahead of this whenever a board arrives mid-completion.
        self._answering: int | None = None
        #: The tick of the newest threat handed to the driver, and of the one the
        #: running reflex was dispatched for. The driver stamps a reflex move with
        #: the tick of the latest reading, threat or board, which runs ahead.
        self._threatened: int | None = None
        self._evading: int | None = None

    # -- warming --------------------------------------------------------------

    def warm(self) -> None:
        """Pump the paused game's kernel until it has nothing to run.

        A batch that ends before its deadline without a write means ``tick()`` found
        nothing runnable: the pilot has blocked on ``game.state``. A pilot that has
        not by ``warm_timeout_s`` is a machine that never yields, and is raised. A
        stop ends warm-up without marking it done.
        """
        if self.warmed:
            return
        began = self.clock.now()
        if not self.driver.started:
            self.driver.start()
        while not (self.stop is not None and self.stop.is_set()):
            deadline = self.clock.now() + WARM_BATCH_S
            decision = self._take(self.driver.run_kernel(deadline=deadline))
            if decision is None and self.clock.now() < deadline:
                self.warmed = True
                break
            if self.clock.now() - began > self.warm_timeout_s:
                raise TimeoutError(
                    f"the kernel was still busy {self.warm_timeout_s:g}s into warm-up"
                )
        self.warm_s = round(self.clock.now() - began, 3)

    # -- the loop -------------------------------------------------------------

    def run(self) -> RunResult:
        self.warm()
        self._start_clock()
        self._emit(catchup=0)
        if not self._finished():
            self._sense()
        while not self._finished():
            decision = self._take(self.driver.run_kernel(deadline=self._due))
            if self._tick_due():
                if not self._finished():
                    self._sense()
            elif decision is None:
                # Nothing ran to the deadline and nothing was written: every job is
                # blocked on a pipe, and only the next tick can feed one.
                self.clock.sleep(max(0.0, self._due - self.clock.now()))
        self._close()
        return self._result(
            preemptions=self.driver.preemptions,
            cancellations=self._cancellations(),
            reflexes=self.reflexes,
            verdicts=self.driver.verdicts(),
            journal=self.driver.journal(),
        )

    def _sense(self) -> None:
        board, info = self._look()
        # `sense` delivers the board unless the tick carries a threat; a read or a
        # dispatch it causes is journalled inside the call, so the stamp goes first.
        if threat_reading(info) is None:
            self._delivered = self.game.ticks
        else:
            self._threatened = self.game.ticks
        self.driver.sense(self.view.state(board, info), info, self.view.history(board, info))

    def _take(self, decision: Decision | None) -> Decision | None:
        """Record a move the driver collected; it landed on the tick now showing.

        Events are scanned first: a batch ends at the write it made, so a read in it
        came before the write and is the board the write answers.
        """
        self._scan()
        if decision is None:
            return None
        by = cast(DecisionBy, decision.by)
        tick = self.game.ticks if decision.tick is None else decision.tick
        if by == "pilot" and self._answering is not None:
            tick = self._answering
        elif by == "evade" and self._evading is not None:
            tick = self._evading
        self._record(
            by,
            decision.action,
            tick,
            latency=decision.latency,
            preempted=bool(decision.preempted),
            applied=self.driver.last_applied,
        )
        return decision

    def _scan(self) -> int:
        """Read the journal since the last scan: count preemptions, note which board
        a read of ``game.state`` took and which threat a reflex was dispatched for.
        Returns the preemptions so far."""
        events = self.driver.kernel.events
        for event in events[self._event_cursor :]:
            kind = getattr(type(event), "KIND", "")
            if kind == PREEMPTED:
                self._preemptions += 1
            elif kind == READ and str(getattr(event, "pipe", "")) == str(GAME_STATE):
                # `sense` keeps one board in the pipe, so a read takes the newest.
                self._answering = self._delivered
            elif kind == FIRED:
                # The only vector is the threat's; a coalesced firing reads the newest.
                self._evading = self._threatened
        self._event_cursor = len(events)
        return self._preemptions

    def _cancellations(self) -> int:
        """The machine's own count of steps it stopped wanting, not the kernel's."""
        return self.driver.machine.cancellations

    def _counts(self) -> tuple[int, int, int]:
        return self._scan(), self._cancellations(), self.reflexes


# --- the prompt arm --------------------------------------------------------------


class WallClockPromptRunner(_WallClock):
    """The prompt loop against the same clock: the game ticks while the model works.

    One request at a time, begun the moment the previous reply has been played,
    against the board showing then. ``PromptArm.poll`` bounded by the next due time
    is the loop's sleep. A reply that arrives with the tick's budget spent waits for
    the next tick rather than being thrown away, as ``RealtimeRunner`` does; one
    that names no action plays ``FALLBACK_ACTION`` and counts against the parse
    rate.
    """

    arm: Arm = "prompt"

    def __init__(
        self,
        player: PromptArm,
        spec: BoardSpec,
        *,
        stop: StopFlag | None = None,
        on_frame: FrameSink | None = None,
        clock: Clock | None = None,
        max_ticks: int | None = None,
        max_seconds: float | None = None,
    ) -> None:
        super().__init__(
            spec,
            stop=stop,
            on_frame=on_frame,
            clock=clock,
            max_ticks=max_ticks,
            max_seconds=max_seconds,
        )
        self.player = player
        self.warmed = False
        self.warm_s = 0.0
        self.replies: list[PromptReply] = []
        self._asked_tick = 0
        #: A reply waiting for the stick's budget to reopen.
        self._held: PromptReply | None = None

    def warm(self) -> None:
        """Prefill the system prompt before the clock starts."""
        if self.warmed:
            return
        began = self.clock.now()
        self.player.warm()
        self.warm_s = round(self.clock.now() - began, 3)
        self.warmed = True

    def run(self) -> RunResult:
        self.warm()
        self._start_clock()
        self._emit(catchup=0)
        self._ask()
        while not self._finished():
            if self._tick_due():
                if self._held is not None and not self.controls.full and not self._finished():
                    self._play(self._held)
                continue
            wait = max(0.0, self._due - self.clock.now())
            if self._held is not None:
                self.clock.sleep(wait)
            elif self.player.busy:
                reply = self.player.poll(wait)
                if reply is not None:
                    self.replies.append(reply)
                    if self.controls.full:
                        self._held = reply
                    else:
                        self._play(reply)
            else:
                self._ask()
        self._close()
        parsed = sum(1 for r in self.replies if r.parsed)
        return self._result(
            preemptions=0,
            cancellations=0,
            reflexes=0,
            parse_rate=parse_rate(parsed, len(self.replies)),
        )

    def _ask(self) -> None:
        # Never on the tick that ends the run: nothing would collect the reply.
        if self._finished():
            return
        self._asked_tick = self.game.ticks
        self.player.begin(*self._look())

    def _play(self, reply: PromptReply) -> None:
        self._held = None
        action = reply.action or FALLBACK_ACTION
        applied = self.controls.write(action)
        self._record(
            "prompt",
            action,
            self._asked_tick,
            latency=reply.latency,
            preempted=False,
            applied=applied,
        )
        self._ask()
