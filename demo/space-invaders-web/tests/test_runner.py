# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The wall-clock runners' discipline, on a clock the test turns by hand.

A ``FakeClock`` moves only when the loop sleeps or a collaborator says time
passed, so how many ticks were applied, caught up or slept through is exact. The
driver and the prompt arm are fakes that spend fake time the way the real ones
spend wall time: a busy kernel runs to its deadline, a model reply arrives after
its latency.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pytest
from zeos_space_invaders.players.zeos.player import (
    threat_reading,  # pyright: ignore[reportUnknownVariableType]
)
from zeos_space_invaders.runlog import Decision

from zeos_space_invaders_web import boards
from zeos_space_invaders_web.contracts import (
    BOARDS,
    BoardName,
    BoardSpec,
    Frame,
    FrameSink,
    LocalStop,
    MonotonicClock,
    PromptReply,
    PromptRunnerFactory,
    ZeosRunnerFactory,
    load_board,
)
from zeos_space_invaders_web.metrics import (
    due_ticks,
    lag_stats,
    model_lags,
    overrun_stats,
    parse_rate,
    percentile,
)
from zeos_space_invaders_web.runner import WallClockPromptRunner, WallClockZeosRunner

# The constructors are the factories the contracts pin; pyright checks the claim.
_ZEOS: ZeosRunnerFactory = WallClockZeosRunner
_PROMPT: PromptRunnerFactory = WallClockPromptRunner

TICK = 0.5


class FakeClock:
    """Time that passes only when told to; it starts at ``time.monotonic()``,
    because the zeos runner refuses a clock that disagrees with the driver's."""

    def __init__(self, start: float | None = None) -> None:
        self.t = time.monotonic() if start is None else start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeKernel:
    def __init__(self) -> None:
        self.events: list[object] = []


class Preempted:
    KIND = "job.preempted"


class Read:
    KIND = "pipe.read"
    pipe = "game.state"


class Fired:
    KIND = "vector.fired"


class FakeMachine:
    cancellations = 3


class FakeDriver:
    """``ZeosDriverLike`` over fake time.

    Warm-up is ``warm_batches`` busy batches then idle. In the run, a busy batch
    runs to its deadline plus ``overrun_s`` (a last boundary that ran long); an idle
    one returns at once. After each board, the pilot answers ``answer_after``
    batches later with a move stamped with that board's tick.
    """

    def __init__(
        self,
        clock: FakeClock,
        *,
        busy: bool = False,
        overrun_s: float = 0.0,
        warm_batches: int = 2,
        answer_after: int | None = None,
        warm_move: bool = False,
        evade_after: int = 3,
    ) -> None:
        self.clock = clock
        self.busy = busy
        self.overrun_s = overrun_s
        self.warm_batches = warm_batches
        self.answer_after = answer_after
        self.warm_move = warm_move
        self.controls: Any = None
        self.last_applied = False
        self.kernel = FakeKernel()
        self.machine = FakeMachine()
        self.evade_after = evade_after
        self._evade_countdown: int | None = None
        self._sense_tick = 0
        self.started = False
        self.sensed: list[int] = []
        self.batches = 0
        self._board_tick = 0
        self._countdown: int | None = None

    @property
    def preemptions(self) -> int:
        return sum(1 for e in self.kernel.events if isinstance(e, Preempted))

    def start(self) -> None:
        self.started = True

    def run_kernel(self, deadline: float | None = None) -> Decision | None:
        assert deadline is not None
        self.batches += 1
        if self.warm_batches:
            self.warm_batches -= 1
            self.clock.t = max(self.clock.t, deadline)
            if self.warm_move and self.warm_batches == 0:
                return self._move("pilot", "left", 0)
            return None
        # Busy only once a board has arrived: until then the pilot is blocked on it.
        if self.busy and self.sensed:
            self.clock.t = max(self.clock.t, deadline) + self.overrun_s
        if self._evade_countdown is not None:
            self._evade_countdown -= 1
            if self._evade_countdown <= 0:
                self._evade_countdown = None
                # Stamped as the driver does: with the latest reading's tick.
                return self._move("evade", "shoot", self._sense_tick)
        if self._countdown is not None:
            self._countdown -= 1
            if self._countdown <= 0:
                self._countdown = None
                self.kernel.events.append(Preempted())
                return self._move("pilot", "right", self._board_tick)
        return None

    def _move(self, by: str, action: str, tick: int) -> Decision:
        self.last_applied = bool(self.controls.write(action))
        return Decision(by=by, action=action, tick=tick, latency=0.25, preempted=True)

    def sense(self, board: str, info: dict[str, Any], note: object = None) -> None:
        assert isinstance(board, str) and note is not None
        self._sense_tick = int(info["ticks"])
        self.sensed.append(self._sense_tick)
        if threat_reading(info) is not None:
            # The vector fires on the threat unless a reflex is already under way;
            # either way the board is not delivered this tick.
            if self._evade_countdown is None:
                self.kernel.events.append(Fired())
                self._evade_countdown = self.evade_after
            return
        self._board_tick = self._sense_tick
        if self.answer_after is not None and self._countdown is None:
            # A pilot between moves reads the board at once and starts on it.
            self.kernel.events.append(Read())
            self._countdown = self.answer_after

    def verdicts(self) -> list[dict[str, Any]]:
        return [{"id": "fake", "passed": True}]

    def journal(self) -> list[dict[str, Any]]:
        return [{"seq": 0}]


def collect(frames: list[Frame]) -> FrameSink:
    """A sink appending to ``frames``."""

    def sink(frame: Frame) -> None:
        frames.append(frame)

    return sink


def spec(name: BoardName = "default", **changes: Any) -> BoardSpec:
    """A board with a quiet sky unless asked otherwise, so a run lasts its ticks."""
    base = load_board(name, seed=7)
    rules = replace(base.rules, fire_chance=changes.pop("fire_chance", 0.0))
    return replace(base, rules=rules, tick_seconds=changes.pop("tick", TICK), **changes)


# --- metrics ----------------------------------------------------------------------


def test_metrics() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([3, 1, 2], 50) == 2.0
    assert percentile(list(range(1, 21)), 95) == 19.0
    assert lag_stats([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert lag_stats([1, 2, 3, 10]) == {"mean": 4.0, "p50": 2.0, "p95": 10.0, "max": 10.0}
    assert overrun_stats([]) == {"total": 0.0, "mean": 0.0, "max": 0.0, "count": 0}
    assert overrun_stats([1.0, 3.0]) == {"total": 4.0, "mean": 2.0, "max": 3.0, "count": 2}
    assert [due_ticks(t, 10.0, 0.5) for t in (9.9, 10.0, 10.49, 10.5, 11.6)] == [0, 1, 1, 2, 4]
    assert parse_rate(0, 0) is None
    assert parse_rate(2, 3) == 0.667
    record = {
        "action": "left",
        "tick": 0,
        "tick_applied": 2,
        "lag_ticks": 2,
        "latency": None,
        "preempted": False,
        "applied": True,
    }
    assert model_lags(
        [{**record, "by": "evade"}, {**record, "by": "pilot"}, {**record, "by": "prompt"}]  # pyright: ignore[reportArgumentType]
    ) == [2, 2]


@pytest.mark.parametrize("name", BOARDS)
def test_both_boards_load_and_run(name: BoardName) -> None:
    board = boards.load_board(name, seed=3)
    assert boards.board_text(name)
    clock = FakeClock()
    driver = FakeDriver(clock)
    frames: list[Frame] = []
    runner = WallClockZeosRunner(
        driver, replace(board, max_steps=4), clock=clock, on_frame=collect(frames)
    )
    result = runner.run()
    assert result.board == name and result.seed == 3
    assert result.ticks == 4
    assert frames[-1]["text"].count("\n") == board.rules.h - 1
    assert clock.sleeps and all(s == pytest.approx(board.tick_seconds) for s in clock.sleeps)


# --- the zeos runner --------------------------------------------------------------


def test_an_idle_kernel_sleeps_to_each_tick_and_never_overruns() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock)
    frames: list[Frame] = []
    runner = WallClockZeosRunner(driver, spec(), clock=clock, on_frame=collect(frames), max_ticks=6)
    start = clock.now()
    result = runner.run()
    assert runner.warmed and driver.started
    assert result.ticks == 6 and result.catchup_ticks == 0
    assert [f["tick"] for f in frames] == [0, 1, 2, 3, 4, 5, 6]
    # Warm-up spent two busy batches before the clock started; then one sleep a tick.
    assert len(clock.sleeps) == 6
    assert clock.now() - start == pytest.approx(2 * 0.05 + 6 * TICK)
    assert result.overrun_ms == 0.0
    # The board of every tick but the last is handed over; the last ends the run.
    assert driver.sensed == [0, 1, 2, 3, 4, 5]
    assert result.verdicts == [{"id": "fake", "passed": True}]
    assert result.journal == [{"seq": 0}]
    assert result.cancellations == 3
    assert result.arm == "zeos" and result.parse_rate is None


def test_a_busy_kernel_never_sleeps() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock, busy=True)
    result = WallClockZeosRunner(driver, spec(), clock=clock, max_ticks=5).run()
    assert result.ticks == 5
    assert clock.sleeps == []
    assert result.catchup_ticks == 0


def test_an_overrunning_loop_catches_up_in_one_frame() -> None:
    clock = FakeClock()
    # Every batch's last boundary runs 2.5 ticks past the deadline, so each time
    # the loop looks, three ticks have fallen due: one, plus two to catch up.
    driver = FakeDriver(clock, busy=True, overrun_s=2.5 * TICK)
    frames: list[Frame] = []
    runner = WallClockZeosRunner(driver, spec(), clock=clock, on_frame=collect(frames), max_ticks=9)
    result = runner.run()
    assert result.ticks == 9
    assert [f["catchup"] for f in frames] == [0, 2, 2, 2]
    assert [f["tick"] for f in frames] == [0, 3, 6, 9]
    assert result.catchup_ticks == 6
    assert result.overrun_ms == pytest.approx(3 * 2.5 * TICK * 1000)
    assert runner.overrun_stats()["max"] == pytest.approx(1250.0)
    # Only the newest board is handed over after a catch-up.
    assert driver.sensed == [0, 3, 6]


def test_a_catch_up_stops_at_the_last_tick() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock, busy=True, overrun_s=2.5 * TICK)
    result = WallClockZeosRunner(driver, spec(), clock=clock, max_ticks=4).run()
    assert result.ticks == 4
    assert result.catchup_ticks == 2


def test_a_pilot_move_is_lagged_against_the_board_it_answered() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock, busy=True, answer_after=5, warm_move=True)
    frames: list[Frame] = []
    runner = WallClockZeosRunner(
        driver, spec(), clock=clock, on_frame=collect(frames), max_ticks=12
    )
    result = runner.run()
    # The warm-up move lands before the clock starts: tick 0, no lag, in frame 0.
    assert frames[0]["decisions"][0]["tick_applied"] == 0
    moves = [r for r in runner.records if r["action"] == "right"]
    assert moves, "the pilot never answered"
    # Answered at the end of the fifth batch after its board was read, which is
    # before the fifth tick applies: four ticks late.
    assert [r["tick"] for r in moves] == [0, 5]
    assert all(r["lag_ticks"] == 4 and r["by"] == "pilot" for r in moves)
    assert all(r["tick_applied"] - r["tick"] == r["lag_ticks"] for r in runner.records)
    assert result.lag_ticks["max"] == 4.0 and result.lag_ticks["p50"] == 4.0
    assert result.preemptions == len(moves)
    assert frames[-1]["preemptions"] == len(moves)
    assert sum(len(f["decisions"]) for f in frames) == result.decisions
    assert result.reflexes == 0


def test_the_stop_flag_ends_the_run_within_one_tick() -> None:
    clock = FakeClock()
    stop = LocalStop()

    def on_frame(frame: Frame) -> None:
        if frame["tick"] == 3:
            stop.set()

    driver = FakeDriver(clock)
    result = WallClockZeosRunner(driver, spec(), clock=clock, stop=stop, on_frame=on_frame).run()
    assert result.ticks == 3


def test_a_stop_during_warm_up_ends_it() -> None:
    clock = FakeClock()
    stop = LocalStop()
    stop.set()
    driver = FakeDriver(clock, warm_batches=10**6)
    runner = WallClockZeosRunner(driver, spec(), clock=clock, stop=stop)
    runner.warm()
    assert not runner.warmed and driver.batches == 0
    assert runner.run().ticks == 0
    assert driver.sensed == [], "a stopped run handed the driver a board"


def test_a_kernel_that_never_quiesces_fails_warm_up() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock, warm_batches=10**6)
    runner = WallClockZeosRunner(driver, spec(), clock=clock, warm_timeout_s=1.0)
    with pytest.raises(TimeoutError, match="warm-up"):
        runner.warm()


def test_max_seconds_bounds_the_run() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock)
    result = WallClockZeosRunner(driver, spec(), clock=clock, max_seconds=2.2).run()
    # Ticks at 0.5 s to 2.0 s; the sleep to the fifth passes the limit.
    assert result.ticks == 4


def test_game_over_ends_the_run() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock)
    runner = WallClockZeosRunner(driver, spec(), clock=clock, max_ticks=1000)
    runner.game.lives = 1
    runner.game.dangers = [[runner.game.rules.h - 2, runner.game.player]]
    result = runner.run()
    assert result.lives == 0 and result.ticks == 1


# --- the prompt runner ------------------------------------------------------------


class FakeArm:
    """``PromptArm`` whose replies take ``latency`` fake seconds each."""

    def __init__(self, clock: FakeClock, latency: float, replies: list[str]) -> None:
        self.clock = clock
        self.latency = latency
        self.replies = replies
        self.asked: list[int] = []
        self.warmed = False
        self.closed = False
        self._ready: float | None = None

    @property
    def busy(self) -> bool:
        return self._ready is not None

    def warm(self) -> None:
        self.warmed = True
        self.clock.advance(3.0)

    def begin(self, obs: str, info: Any) -> None:
        assert not self.busy and isinstance(obs, str)
        self.asked.append(int(info["ticks"]))
        self._ready = self.clock.now() + self.latency

    def poll(self, timeout_s: float) -> PromptReply | None:
        if self._ready is None:
            return None
        if self._ready - self.clock.now() > timeout_s:
            self.clock.advance(timeout_s)
            return None
        self.clock.t = max(self.clock.t, self._ready)
        self._ready = None
        text = self.replies[(len(self.asked) - 1) % len(self.replies)]
        action = text if text in ("left", "right", "shoot") else None
        return PromptReply(text=text, action=action, latency=self.latency, tokens=2)

    def close(self) -> None:
        self.closed = True


def test_the_game_keeps_ticking_while_the_model_works() -> None:
    clock = FakeClock()
    arm = FakeArm(clock, latency=2.5 * TICK, replies=["left", "right", "huh"])
    frames: list[Frame] = []
    runner = WallClockPromptRunner(arm, spec(), clock=clock, on_frame=collect(frames), max_ticks=20)
    result = runner.run()
    assert arm.warmed and runner.warm_s == 3.0
    assert result.ticks == 20
    assert [f["tick"] for f in frames] == list(range(21))
    assert result.catchup_ticks == 0 and result.overrun_ms == 0.0
    assert all(r["by"] == "prompt" for r in runner.records)
    # Each reply lands 2.5 ticks after its board: on the third tick edge after it.
    assert {r["lag_ticks"] for r in runner.records} <= {2, 3}
    assert result.lag_ticks["mean"] == pytest.approx(2.5, abs=0.5)
    # A reply that names no action plays the fallback and counts against parsing.
    assert result.parse_rate == pytest.approx(2 / 3, abs=0.1)
    assert "huh" not in {r["action"] for r in runner.records}
    assert result.arm == "prompt" and result.verdicts == [] and result.preemptions == 0
    assert result.decisions == len(runner.replies)


def test_a_reply_over_budget_waits_for_the_next_tick() -> None:
    clock = FakeClock()
    # Four replies a tick against a budget of one: each tick plays one.
    arm = FakeArm(clock, latency=0.25 * TICK, replies=["left", "right"])
    runner = WallClockPromptRunner(arm, spec(), clock=clock, max_ticks=8)
    runner.run()
    applied = [r["tick_applied"] for r in runner.records]
    assert len(applied) == len(set(applied)), "two moves landed on one tick"
    assert all(r["applied"] for r in runner.records)


def test_the_prompt_runner_stops_within_one_tick() -> None:
    clock = FakeClock()
    stop = LocalStop()
    arm = FakeArm(clock, latency=10 * TICK, replies=["left"])

    def on_frame(frame: Frame) -> None:
        if frame["tick"] == 2:
            stop.set()

    result = WallClockPromptRunner(arm, spec(), clock=clock, stop=stop, on_frame=on_frame).run()
    assert result.ticks == 2 and result.decisions == 0
    assert result.parse_rate is None


def test_the_prompt_runner_on_the_real_clock() -> None:
    clock = MonotonicClock()
    tick = 0.02
    arm = RealArm(latency=0.05)
    runner = WallClockPromptRunner(arm, spec(tick=tick), clock=clock, max_ticks=25)
    began = clock.now()
    result = runner.run()
    elapsed = clock.now() - began
    assert result.ticks == 25
    assert elapsed < 25 * tick + 0.5
    assert result.decisions >= 3
    assert 1 <= result.lag_ticks["mean"] <= 5


class RealArm:
    """``PromptArm`` on the wall clock, replying ``latency`` seconds after a begin."""

    def __init__(self, latency: float) -> None:
        self.latency = latency
        self._ready: float | None = None
        self.clock = MonotonicClock()

    @property
    def busy(self) -> bool:
        return self._ready is not None

    def warm(self) -> None:
        pass

    def begin(self, obs: str, info: Any) -> None:
        self._ready = self.clock.now() + self.latency

    def poll(self, timeout_s: float) -> PromptReply | None:
        if self._ready is None:
            return None
        remaining = self._ready - self.clock.now()
        if remaining > timeout_s:
            self.clock.sleep(timeout_s)
            return None
        self.clock.sleep(remaining)
        self._ready = None
        return PromptReply(text="shoot", action="shoot", latency=self.latency, tokens=1)

    def close(self) -> None:
        pass


def test_a_clock_off_monotonic_time_is_refused() -> None:
    clock = FakeClock(start=100.0)
    with pytest.raises(ValueError, match="time.monotonic"):
        WallClockZeosRunner(FakeDriver(clock), spec(), clock=clock)


def test_a_reflex_is_lagged_against_the_threat_it_was_dispatched_for() -> None:
    clock = FakeClock()
    driver = FakeDriver(clock, busy=True, evade_after=3)
    runner = WallClockZeosRunner(driver, spec(), clock=clock, max_ticks=6)
    # A bomb two rows up in the ship's column: a threat on the tick-0 reading.
    runner.game.dangers = [[runner.game.rules.h - 3, runner.game.player]]
    result = runner.run()
    evades = [r for r in runner.records if r["by"] == "evade"]
    assert len(evades) == 1 and result.reflexes == 1
    # The tick-1 reading is a threat too, delivered while the reflex for tick 0 is
    # under way; the driver stamps the move with it, the runner keeps tick 0.
    assert evades[0]["tick"] == 0
    assert evades[0]["lag_ticks"] == evades[0]["tick_applied"] > 0


def _decision_frames(frames: list[Frame]) -> int:
    return sum(len(f["decisions"]) for f in frames)


@pytest.mark.parametrize("ending", ["stop", "max_seconds", "max_ticks"])
def test_every_decision_reaches_a_frame(ending: str) -> None:
    clock = FakeClock()
    stop = LocalStop()
    frames: list[Frame] = []

    def sink(frame: Frame) -> None:
        frames.append(frame)

    driver = FakeDriver(clock, busy=True, answer_after=1, warm_move=True)
    runner = WallClockZeosRunner(
        driver,
        spec(),
        clock=clock,
        on_frame=sink,
        stop=stop,
        max_ticks=7 if ending == "max_ticks" else None,
        max_seconds=2.6 if ending == "max_seconds" else None,
    )
    if ending == "stop":
        # Set by the move itself, after the tick's frame has gone out.
        original = driver.run_kernel

        def stopping(deadline: float | None = None) -> Decision | None:
            decision = original(deadline)
            if decision is not None and runner.game.ticks >= 4:
                stop.set()
            return decision

        driver.run_kernel = stopping
    result = runner.run()
    assert result.decisions > 0
    assert _decision_frames(frames) == result.decisions
    last = frames[-1]
    assert last["tick"] == result.ticks
    assert (last["preemptions"], last["cancellations"], last["reflexes"]) == (
        result.preemptions,
        result.cancellations,
        result.reflexes,
    )


@pytest.mark.parametrize("ending", ["stop", "max_seconds", "max_ticks"])
def test_every_prompt_decision_reaches_a_frame(ending: str) -> None:
    clock = FakeClock()
    stop = LocalStop()
    frames: list[Frame] = []
    arm = FakeArm(clock, latency=0.3 * TICK, replies=["left"])

    def sink(frame: Frame) -> None:
        frames.append(frame)
        if ending == "stop" and frame["tick"] >= 3:
            stop.set()

    runner = WallClockPromptRunner(
        arm,
        spec(),
        clock=clock,
        on_frame=sink,
        stop=stop,
        max_ticks=7 if ending == "max_ticks" else None,
        max_seconds=2.6 if ending == "max_seconds" else None,
    )
    result = runner.run()
    assert result.decisions > 0
    assert _decision_frames(frames) == result.decisions
    assert frames[-1]["tick"] == result.ticks
    # No request is left under way once the run is over.
    assert not arm.busy


def test_a_held_reply_is_not_played_on_the_last_tick() -> None:
    clock = FakeClock()
    arm = FakeArm(clock, latency=0.25 * TICK, replies=["left", "right"])
    runner = WallClockPromptRunner(arm, spec(), clock=clock, max_ticks=3)
    result = runner.run()
    assert all(r["tick_applied"] < 3 for r in runner.records)
    assert not arm.busy
    assert result.ticks == 3
