# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What ``si_worker.js`` calls into: open a run, and turn its end into JSON for the page.

Every function takes and returns plain values -- names, numbers, JSON text -- or a
``Run``, so the JavaScript side needs no knowledge of the kernel's types.

``open_run`` is the ``OpenRun`` of ``contracts.py``. It looks the arm up in
``RUN_BUILDERS``, the integration seam: a builder per arm assembles that arm's run
from a ``RunContext``, to the ``Run`` contract -- ``ZeosRun`` (``PilotJsMachine`` ->
``ZeosDriver`` -> ``WallClockZeosRunner``) and ``PromptRun`` (``BrowserPromptPlayer`` ->
``WallClockPromptRunner``). An arm with no builder, or a run with no worker, gets a
``FakeRun``: a real ``Game`` on the real board, against the
wall clock, whose moves are random and whose model latency is simulated, so the page
runs end to end. A ``FakeRun`` never touches its worker and judges no criteria; its
verdicts' detail says so.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from zeos.core.ids import JobId
from zeos.debugger.payload import build_payload
from zeos.descriptor.loader import load_case
from zeos.journal.codec import decode_record
from zeos.journal.writer import JournalRecord
from zeos.machine.base import SpliceResult, Token
from zeos_coop_count_web.js_machine import Bridge
from zeos_coop_count_web.page import findings
from zeos_space_invaders.game import ACTIONS, Controls, Game
from zeos_space_invaders.game import (
    snapshot as _snapshot,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.players.zeos.player import CASE_ROOT, dodge
from zeos_space_invaders.players.zeos.player import (
    load_criteria as _load_criteria,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.players.zeos.player import (
    threat_reading as _threat_reading,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.utils import VIEWS

from zeos_space_invaders_web.contracts import (
    ARMS,
    BOARDS,
    Arm,
    AsyncModelWorker,
    BoardName,
    BoardSpec,
    Clock,
    DecisionBy,
    DecisionRecord,
    Frame,
    FrameSink,
    LagStats,
    MonotonicClock,
    Run,
    RunResult,
    StopFlag,
    load_board,
)
from zeos_space_invaders_web.machine import PilotJsMachine
from zeos_space_invaders_web.prompt_player import BrowserPromptPlayer
from zeos_space_invaders_web.runner import (
    WallClockPromptRunner,
    WallClockZeosRunner,
    build_driver,
)

__all__ = [
    "PROMPT_OPTIONS",
    "RUNNER_OPTIONS",
    "RUN_BUILDERS",
    "ZEOS_KERNEL_OPTIONS",
    "ZEOS_MACHINE_OPTIONS",
    "FakeRun",
    "LoopClock",
    "PromptRun",
    "ZeosRun",
    "channel",
    "configure_json",
    "RunContext",
    "RunBuilder",
    "describe_json",
    "finished_json",
    "json_sink",
    "lag_stats",
    "make_frame",
    "open_run",
    "payload_json",
]


# --- the native game, typed at the edge (the game module carries no annotations) --------


def snapshot(game: Any) -> dict[str, Any]:
    """``zeos_space_invaders.game.snapshot``."""
    return _snapshot(game)


def threat_reading(info: dict[str, Any]) -> str | None:
    """``ZeosDriver``'s threat sensor, the reflex's trigger."""
    return _threat_reading(info)


def _criteria() -> list[dict[str, Any]]:
    """The case's criteria, raw, as ``load_criteria`` reads them."""
    raw = cast("tuple[dict[str, Any], ...]", _load_criteria())
    return [dict(c) for c in raw]


# --- the integration seam -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything ``open_run`` was given, resolved: the board loaded, the defaults filled.

    ``worker`` is the channel to the model thread, or to the stub thread for the stub
    machine; ``None`` when the build has no stub, which only a ``FakeRun`` accepts.
    """

    arm: Arm
    spec: BoardSpec
    worker: AsyncModelWorker | None
    stub: bool
    on_frame: FrameSink | None
    stop: StopFlag | None
    clock: Clock


RunBuilder = Callable[[RunContext], Run]
"""Assembles one arm's real run from a ``RunContext``."""

#: The integration seam: arm -> builder. Both arms' real runs (``ZeosRun``,
#: ``PromptRun``) are registered further down; ``open_run`` falls back to ``FakeRun``
#: for an arm missing here, and for a run with no worker (a build without the stub).
RUN_BUILDERS: dict[Arm, RunBuilder] = {}


def open_run(
    arm: Arm,
    board: BoardName,
    seed: int | None,
    worker: AsyncModelWorker | None,
    *,
    stub: bool = False,
    on_frame: FrameSink | None = None,
    stop: StopFlag | None = None,
    clock: Clock | None = None,
) -> Run:
    """The run ``si_worker.js``'s ``start`` asked for, assembled and not yet warmed.

    ``clock`` is an addition to the contract's ``OpenRun`` (optional, so the Protocol
    still holds): under Pyodide the worker passes one whose ``sleep`` is an
    ``Atomics.wait`` on the control buffer, so Stop wakes a sleeping loop.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    context = RunContext(
        arm=arm,
        spec=load_board(board, seed),
        worker=worker,
        stub=stub,
        on_frame=on_frame,
        stop=stop,
        clock=clock if clock is not None else MonotonicClock(),
    )
    builder = RUN_BUILDERS.get(arm)
    if builder is None or worker is None:
        return FakeRun(context)
    return builder(context)


# --- frames ---------------------------------------------------------------------------


def make_frame(
    game: Any,
    *,
    arm: Arm,
    board: BoardName,
    decisions: Sequence[DecisionRecord] = (),
    catchup: int = 0,
    preemptions: int = 0,
    cancellations: int = 0,
    reflexes: int = 0,
) -> Frame:
    """The ``Frame`` for ``game`` as it stands: ``snapshot`` plus the run's totals."""
    info = snapshot(game)
    missile = info["missile"]
    return {
        "arm": arm,
        "board": board,
        "tick": int(info["ticks"]),
        "text": game.render(),
        "lives": int(info["lives"]),
        "kills": int(info["score"]) // 10,
        "score": int(info["score"]),
        "player": int(info["player"]),
        "monsters": [[int(r), int(c)] for r, c in info["monsters"].values()],
        "missile": None if missile is None else [int(missile[0]), int(missile[1])],
        "dangers": [[int(r), int(c)] for r, c in info["dangers"]],
        "can_shoot": bool(info["can_shoot"]),
        "over": bool(game.over),
        "won": bool(info["won"]),
        "decisions": list(decisions),
        "catchup": catchup,
        "preemptions": preemptions,
        "cancellations": cancellations,
        "reflexes": reflexes,
    }


def lag_stats(lags: Sequence[float]) -> LagStats:
    """Mean, median, 95th percentile (nearest rank) and maximum; all 0.0 for none."""
    if not lags:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(x) for x in lags)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "p50": float(statistics.median(ordered)),
        "p95": ordered[rank],
        "max": ordered[-1],
    }


# --- the fake ---------------------------------------------------------------------------

#: Simulated request-to-move seconds, ``(low, high)``, by arm and machine. The model
#: figures are the plan's predictions (about 1.2 s a pilot move, 5 s a prompt reply).
FAKE_LATENCY: Final[Mapping[tuple[Arm, bool], tuple[float, float]]] = {
    ("zeos", False): (0.7, 1.8),
    ("zeos", True): (0.3, 0.9),
    ("prompt", False): (3.0, 7.0),
    ("prompt", True): (1.0, 2.5),
}
#: Simulated warm-up (system-prompt prefill) seconds, by machine (stub or not).
FAKE_WARM: Final[Mapping[bool, float]] = {False: 1.5, True: 0.4}
#: Share of prompt replies that name no action.
FAKE_UNPARSED: Final = 0.08
#: The longest the loop sleeps at once, so a clock that cannot be woken still sees Stop.
SLICE_S: Final = 0.05


@dataclass(slots=True)
class _Request:
    """A simulated model request in flight: the board it was asked against."""

    board_tick: int
    begun: float
    done_at: float


class FakeRun:
    """A ``Run`` with the real game and a simulated player, for the page before wiring.

    The ``zeos`` arm has a reflex: when ``threat_reading`` fires it dodges at once
    (``by: evade``) and a pilot request in flight is preempted (cancelled and begun
    again against the next board). The ``prompt`` arm has none, and some of its
    replies do not parse. Moves are random, biased to shoot under a monster.
    """

    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.spec = context.spec
        self.arm: Arm = context.arm
        self.clock = context.clock
        self.game: Any = Game(seed=self.spec.seed, rules=self.spec.rules)
        self.controls: Any = Controls(self.game, per_tick=self.spec.actions_per_tick)
        self.rng = random.Random(f"fake:{self.arm}:{self.spec.name}:{self.spec.seed}")
        self.latency = FAKE_LATENCY[(self.arm, context.stub)]
        self.warmed = False
        self.closed = False
        self.preemptions = 0
        self.cancellations = 0
        self.reflexes = 0
        self.decisions = 0
        self.replies = 0
        self.parsed = 0
        self.lags: list[float] = []
        self.overrun_ms = 0.0
        self.catchup_ticks = 0
        self._request: _Request | None = None
        self._preempted = False
        #: Replies that landed between two ticks, reported with the next frame.
        self._pending: list[DecisionRecord] = []

    # -- the Run protocol -----------------------------------------------------

    def warm(self) -> None:
        """Pretend to prefill the system prompt; Stop cuts it short."""
        self._sleep_until(self.clock.now() + FAKE_WARM[self.context.stub])
        self.warmed = True

    def run(self) -> RunResult:
        if not self.warmed:
            self.warm()
        tick_s = self.spec.tick_seconds
        due = self.clock.now() + tick_s
        self._emit([], 0)  # the board before the first tick
        while not self._stopped() and not self.game.over:
            if self.game.ticks >= self.spec.max_steps:
                break
            self._request_if_idle()
            wake = due if self._request is None else min(due, self._request.done_at)
            self._sleep_until(wake)
            if self._stopped():
                break
            pending = self._land_reply()
            now = self.clock.now()
            if now < due:
                if pending is not None:
                    self._pending.append(pending)
                continue
            late = now - due
            ticks = min(1 + int(late // tick_s), self.spec.max_steps - self.game.ticks)
            self.overrun_ms += late * 1000.0
            self.catchup_ticks += ticks - 1
            decisions = [*self._pending, *([pending] if pending is not None else [])]
            self._pending = []
            for _ in range(ticks):
                self.controls.tick()
                if self.game.over:
                    break
            decisions.extend(self._reflex())
            self._emit(decisions, ticks - 1)
            due += ticks * tick_s
        if self._pending:
            # Replies that landed after the last tick, e.g. just before Stop: they were
            # played and counted, so they are reported.
            self._emit(self._pending, 0)
            self._pending = []
        return self._result()

    def close(self) -> None:
        self.closed = True
        self._request = None

    # -- the simulated player -------------------------------------------------

    def _stopped(self) -> bool:
        stop = self.context.stop
        return stop is not None and bool(stop.is_set())

    def _sleep_until(self, when: float) -> None:
        while not self._stopped():
            left = when - self.clock.now()
            if left <= 0:
                return
            self.clock.sleep(min(left, SLICE_S))

    def _request_if_idle(self) -> None:
        if self._request is not None or self.game.over:
            return
        now = self.clock.now()
        low, high = self.latency
        self._request = _Request(self.game.ticks, now, now + self.rng.uniform(low, high))

    def _choose(self) -> str:
        """Random, biased: shoot when a monster is overhead and the gun is free."""
        overhead = any(c == self.game.player for _, c in self.game.monsters.values())
        if overhead and self.game.missile is None and self.rng.random() < 0.7:
            return "shoot"
        return self.rng.choice(ACTIONS)

    def _land_reply(self) -> DecisionRecord | None:
        """The model's move, if its simulated reply has arrived."""
        request = self._request
        if request is None or self.clock.now() < request.done_at:
            return None
        self._request = None
        latency = self.clock.now() - request.begun
        by: DecisionBy = "pilot" if self.arm == "zeos" else "prompt"
        if self.arm == "prompt":
            self.replies += 1
            if self.rng.random() < FAKE_UNPARSED:
                return None
            self.parsed += 1
        action = self._choose()
        applied = self.controls.write(action)
        lag = self.game.ticks - request.board_tick
        self.decisions += 1
        self.lags.append(float(lag))
        preempted, self._preempted = self._preempted, False
        return {
            "by": by,
            "action": action,
            "tick": request.board_tick,
            "tick_applied": self.game.ticks,
            "lag_ticks": lag,
            "latency": latency,
            "preempted": preempted,
            "applied": applied,
        }

    def _reflex(self) -> list[DecisionRecord]:
        """The ``zeos`` arm's evade: dodge now, preempting a pilot request in flight."""
        if self.arm != "zeos" or self.game.over:
            return []
        threat = threat_reading(snapshot(self.game))
        if threat is None:
            return []
        self.reflexes += 1
        preempted = self._request is not None
        if preempted:
            self.preemptions += 1
            self.cancellations += 1
            self._request = None
            self._preempted = True
        action = dodge(threat)
        applied = self.controls.write(action)
        self.decisions += 1
        return [
            {
                "by": "evade",
                "action": action,
                "tick": self.game.ticks,
                "tick_applied": self.game.ticks,
                "lag_ticks": 0,
                "latency": None,
                "preempted": preempted,
                "applied": applied,
            }
        ]

    def _emit(self, decisions: list[DecisionRecord], catchup: int) -> None:
        sink = self.context.on_frame
        if sink is None:
            return
        sink(
            make_frame(
                self.game,
                arm=self.arm,
                board=self.spec.name,
                decisions=decisions,
                catchup=catchup,
                preemptions=self.preemptions,
                cancellations=self.cancellations,
                reflexes=self.reflexes,
            )
        )

    def _result(self) -> RunResult:
        verdicts: list[dict[str, object]] = []
        if self.arm == "zeos":
            verdicts = [
                {
                    "id": str(c.get("id", c.get("kind", "?"))),
                    "kind": str(c.get("kind", "?")),
                    "passed": None,
                    "detail": "not judged: a FakeRun runs no kernel",
                    "because": str(c.get("because", "")).strip(),
                }
                for c in _criteria()
            ]
        return RunResult(
            arm=self.arm,
            board=self.spec.name,
            seed=self.spec.seed,
            lives=self.game.lives,
            kills=self.game.score // 10,
            ticks=self.game.ticks,
            decisions=self.decisions,
            preemptions=self.preemptions,
            cancellations=self.cancellations,
            reflexes=self.reflexes,
            lag_ticks=lag_stats(self.lags),
            overrun_ms=self.overrun_ms,
            catchup_ticks=self.catchup_ticks,
            verdicts=verdicts,
            journal=[],
            parse_rate=(self.parsed / self.replies if self.replies else None)
            if self.arm == "prompt"
            else None,
        )


# --- the real runs (the builders RUN_BUILDERS holds) -------------------------------------

#: ``PilotJsMachine`` keywords for the zeos arm; the tests and the tuning change them.
#: Empty means the machine's defaults (``DEFAULT_MAX_CHUNK``, ``DEFAULT_STALL_MS``,
#: the left/right/shoot payload grammar).
ZEOS_MACHINE_OPTIONS: dict[str, Any] = {}
#: ``BrowserPromptPlayer`` keywords for the prompt arm (``history`` defaults to the
#: board's own).
PROMPT_OPTIONS: dict[str, Any] = {}
#: Runner keywords for either arm (``max_ticks``, ``max_seconds``).
RUNNER_OPTIONS: dict[str, Any] = {}
#: ``KernelConfig`` fields to override on the zeos arm's kernel, for diagnosis only:
#: empty (the default) is the native case's kernel exactly. ``starvation_limit`` is the
#: one the integration measured: the kernel faults the pilot on its ninth preemption by
#: default, and a preemption-heavy real-time game reaches that within a minute.
ZEOS_KERNEL_OPTIONS: dict[str, Any] = {}


def configure_json(text: str) -> None:
    """Replace the arms' options from JSON text ``{"zeos": {...}, "prompt": {...},
    "kernel": {...}}``; a key left out goes back to its defaults. ``si_worker.js`` passes the page's ``?tune=``
    query parameter through it before each run, which is how the tuning runs set
    ``max_chunk`` or ``stall_ms`` without a rebuild."""
    raw = cast(dict[str, Any], json.loads(text or "{}"))
    unknown = sorted(set(raw) - {"zeos", "prompt", "kernel"})
    if unknown:
        raise ValueError(f"cannot tune {unknown}: only 'zeos', 'prompt' and 'kernel'")
    ZEOS_MACHINE_OPTIONS.clear()
    ZEOS_MACHINE_OPTIONS.update(cast(dict[str, Any], raw.get("zeos", {})))
    PROMPT_OPTIONS.clear()
    PROMPT_OPTIONS.update(cast(dict[str, Any], raw.get("prompt", {})))
    ZEOS_KERNEL_OPTIONS.clear()
    ZEOS_KERNEL_OPTIONS.update(cast(dict[str, Any], raw.get("kernel", {})))


def channel(worker: object) -> tuple[AsyncModelWorker, Bridge | None]:
    """The worker as the machine and the prompt arm take it, with the bridge for it.

    Under Pyodide ``si_worker.js`` hands over a JavaScript ``SyncModelWorker``, which
    ``pyodide_glue.attach`` wraps; a Python worker (``FakePilotWorker``, ``NodeWorker``)
    is used as it is, over the default ``PythonBridge``.
    """
    try:
        from pyodide.ffi import (  # pyright: ignore[reportMissingImports]
            JsProxy,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError:
        return cast(AsyncModelWorker, worker), None
    if isinstance(worker, JsProxy):
        from zeos_space_invaders_web.pyodide_glue import attach

        return attach(worker)
    return cast(AsyncModelWorker, worker), None


class LoopClock:
    """The zeos runner's clock: ``now()`` is ``time.monotonic()``, which is what
    ``ZeosDriver`` reads its deadlines on, and ``sleep`` is the given clock's -- under
    Pyodide an ``Atomics.wait`` on the control buffer, so Stop wakes a sleeping loop."""

    def __init__(self, sleeper: Clock) -> None:
        self._sleeper = sleeper

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._sleeper.sleep(seconds)


class _TimedPilot(PilotJsMachine):
    """``PilotJsMachine`` that notes the game tick and wall time of every splice the
    pager makes, so a long gap between pilot moves can be put next to the replay that
    caused it."""

    tick_of: Callable[[], int] = staticmethod(lambda: 0)
    splices: list[dict[str, float]]

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        if job not in self._native:  # pyright: ignore[reportPrivateUsage]
            self.splices.append(
                {
                    "tick": self.tick_of(),
                    "at": time.monotonic(),
                    "removed": end - start,
                    "inserted": len(tokens),
                }
            )
        return super().splice(job, start, end, tokens)


def _stopped(stop: StopFlag | None) -> bool:
    return stop is not None and bool(stop.is_set())


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def _overruns(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


class ZeosRun:
    """The zeos arm: ``PilotJsMachine`` over the channel, ``ZeosDriver`` over it,
    ``WallClockZeosRunner`` on the wall clock.

    ``warm`` prewarms the grammar's mask cache, brings the kernel up (which creates the
    pilot's context) and lets the runner prefill until the pilot blocks on its first
    board, all before the clock starts. ``close`` closes the machine, which cancels and
    drains a step left in flight and destroys every context.
    """

    def __init__(self, context: RunContext) -> None:
        if context.worker is None:
            raise ValueError("the zeos arm needs a worker: the model thread or the stub's")
        worker, bridge = channel(context.worker)
        self.context = context
        self.machine = _TimedPilot(worker, bridge=bridge, **ZEOS_MACHINE_OPTIONS)
        self.machine.splices = []
        self.driver = build_driver(self.machine, context.spec)
        if ZEOS_KERNEL_OPTIONS:
            kernel = self.driver.kernel
            kernel.config = dataclasses.replace(kernel.config, **ZEOS_KERNEL_OPTIONS)
        self.runner = WallClockZeosRunner(
            self.driver,
            context.spec,
            stop=context.stop,
            on_frame=context.on_frame,
            clock=LoopClock(context.clock),
            **RUNNER_OPTIONS,
        )
        game = self.runner.game
        self.machine.tick_of = lambda: int(game.ticks)
        self.prewarm_ms = 0.0
        self.closed = False

    def warm(self) -> None:
        if _stopped(self.context.stop):
            return
        began = time.monotonic()
        self.prewarm_ms = round(self.machine.prewarm(), 1)
        if _stopped(self.context.stop):
            return
        if not self.driver.started:
            self.driver.start()
        self.runner.warm()
        self.warm_s = round(time.monotonic() - began, 3)

    warm_s: float = 0.0

    def run(self) -> RunResult:
        result = self.runner.run()
        result.extras = self._extras()
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.driver.close()

    def _extras(self) -> dict[str, object]:
        machine = self.machine
        runner = self.runner
        tick_s = self.context.spec.tick_seconds
        faults: list[dict[str, object]] = []
        pilot_exits = 0
        journal_splices = 0
        for event in self.driver.kernel.events:
            kind = getattr(type(event), "KIND", "")
            if kind == "machine.splice":
                journal_splices += 1
            elif kind == "fault.raised":
                faults.append(
                    {
                        "job": str(getattr(event, "job", "")),
                        "fault": str(getattr(getattr(event, "fault", ""), "value", "")),
                        "detail": str(getattr(event, "detail", "")),
                    }
                )
            elif kind == "job.completed" and "pilot" in str(getattr(event, "job", "")):
                pilot_exits += 1
        pilot = [r for r in runner.records if r["by"] == "pilot"]
        valid = sum(1 for r in pilot if r["action"] in ACTIONS)
        # The longest stretch of game without a pilot move, from the clock's start (or
        # the previous move) to the next move (or the end), and the splices inside it.
        marks = [0, *(r["tick_applied"] for r in pilot), int(runner.game.ticks)]
        gap, gap_from, gap_to = 0, 0, 0
        for a, b in zip(marks, marks[1:], strict=False):
            if b - a > gap:
                gap, gap_from, gap_to = b - a, a, b
        splices = machine.splices
        in_gap = [s for s in splices if gap_from <= s["tick"] <= gap_to]
        steps = [e for e in machine.step_log if e["outcome"] in ("token", "cancelled")]
        biggest = max(steps, key=lambda e: e["positions"], default=None)
        return {
            "machine": "stub" if self.context.stub else "model",
            "kernel_options": dict(ZEOS_KERNEL_OPTIONS),
            "warm_s": self.warm_s,
            "runner_warm_s": runner.warm_s,
            "prewarm_ms": self.prewarm_ms,
            "overrun_ms": _overruns(runner.overruns_ms),
            "cancel_ms": _overruns(machine.cancel_ms),
            "roundtrip_s": _overruns(machine.roundtrips),
            "step_totals": dict(machine.step_totals),
            "positions_total": machine.positions_total,
            "fill_ms_total": round(machine.fill_ms_total, 1),
            "biggest_step": None if biggest is None else dict(biggest),
            "pilot_moves": len(pilot),
            "pilot_valid_moves": valid,
            "pilot_exits": pilot_exits,
            "faults": faults,
            "splices": len(splices),
            "journal_splices": journal_splices,
            # Model ids per served context at the end: what the pager's window is against.
            "context_ids": {
                str(job): len(ctx.ids)
                for job, ctx in machine._contexts.items()  # pyright: ignore[reportPrivateUsage]
            },
            "longest_pilot_gap": {
                "ticks": gap,
                "seconds": round(gap * tick_s, 3),
                "from_tick": gap_from,
                "to_tick": gap_to,
                "splices": len(in_gap),
            },
        }


class PromptRun:
    """The prompt arm: ``BrowserPromptPlayer`` on its own contexts, ``WallClockPromptRunner``
    on the wall clock. ``warm`` prefills the system prompt before the clock starts;
    ``close`` cancels a request under way and destroys both contexts."""

    def __init__(self, context: RunContext) -> None:
        if context.worker is None:
            raise ValueError("the prompt arm needs a worker: the model thread or the stub's")
        worker, bridge = channel(context.worker)
        spec = context.spec
        options: dict[str, Any] = {"history": spec.history, **PROMPT_OPTIONS}
        self.context = context
        self.player = BrowserPromptPlayer(
            worker, view=VIEWS[spec.view](), rules=spec.rules, bridge=bridge, **options
        )
        self.runner = WallClockPromptRunner(
            self.player,
            spec,
            stop=context.stop,
            on_frame=context.on_frame,
            clock=LoopClock(context.clock),
            **RUNNER_OPTIONS,
        )
        self.closed = False

    def warm(self) -> None:
        if not _stopped(self.context.stop):
            self.runner.warm()

    def run(self) -> RunResult:
        result = self.runner.run()
        replies = self.player.replies
        result.extras = {
            "machine": "stub" if self.context.stub else "model",
            "warm_s": self.runner.warm_s,
            "overrun_ms": _overruns(self.runner.overruns_ms),
            "replies": len(replies),
            "reply_latency_s": _overruns([r.latency for r in replies]),
            "unparsed": [r.text for r in replies if not r.parsed][:20],
        }
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.player.close()


RUN_BUILDERS["zeos"] = ZeosRun
RUN_BUILDERS["prompt"] = PromptRun


# --- JSON for the page ----------------------------------------------------------------


def json_sink(post: Callable[[str], object]) -> FrameSink:
    """A ``FrameSink`` that hands each frame to ``post`` as JSON text.

    The worker passes a JavaScript function; text crosses Pyodide's FFI with no proxy
    to destroy.
    """

    def sink(frame: Frame) -> None:
        post(json.dumps(frame, separators=(",", ":")))

    return sink


def payload_json(
    journal: Sequence[Mapping[str, Any]] | None = None, root: str | Path = CASE_ROOT
) -> str:
    """The debugger's input, as JSON text: the case's wiring and, given a journal, its frames.

    ``journal`` is ``RunResult.journal`` (encoded kernel records); ``None`` or empty
    draws the wiring only, which is all a ``FakeRun`` or the prompt arm has.
    """
    bundle = load_case(Path(root))
    records: list[JournalRecord] | None = None
    if journal:
        records = [JournalRecord(*decode_record(dict(raw))) for raw in journal]
    payload = build_payload(bundle, records=records, findings=findings(bundle))
    return json.dumps(payload, separators=(",", ":"))


def finished_json(result: RunResult, root: str | Path = CASE_ROOT) -> str:
    """The body of the ``finished`` message: ``{result, verdicts, journal, payload}``.

    ``result`` is ``RunResult.to_json()`` less its ``journal`` and ``verdicts``, which
    are sent once each, beside it. ``payload`` is itself JSON text, as coop-count-web's
    page passes it to the debugger.
    """
    data = result.to_json()
    journal = data.pop("journal")
    verdicts = data.pop("verdicts")
    return json.dumps(
        {
            "result": data,
            "verdicts": verdicts,
            "journal": journal,
            "payload": payload_json(result.journal, root),
        },
        separators=(",", ":"),
    )


def describe_json(root: str | Path = CASE_ROOT) -> str:
    """What the page shows before a run: each board's size and tick, the arms, the
    criteria, and the debugger's wiring-only payload."""
    boards: dict[str, dict[str, object]] = {}
    for name in BOARDS:
        spec = load_board(name)
        boards[name] = {
            "w": spec.rules.w,
            "h": spec.rules.h,
            "tick": spec.tick_seconds,
            "max_steps": spec.max_steps,
            "seed": spec.seed,
            "lives": spec.rules.lives,
            "monsters": spec.rules.monster_count,
        }
    return json.dumps(
        {
            "arms": list(ARMS),
            "boards": boards,
            "criteria": [str(c.get("id", "")) for c in _criteria()],
            "builders": sorted(RUN_BUILDERS),
            "payload": payload_json(None, root),
        },
        separators=(",", ":"),
    )
