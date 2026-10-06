# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The zeos runner on the real clock, over the native driver and a scripted machine.

The machine is the native suite's ``stub_machine``: an ``APIMachineBase`` whose
completions stream from a thread, one word every ``delay`` seconds. Threads are
fine under CPython; what is under test is that one loop, with no thread of its
own, keeps the world on time and still lets the reflex preempt the pilot -- the
case judging itself as it does through the native threaded runner.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from zeos_space_invaders.players.zeos.player import ZeosDriver

from zeos_space_invaders_web.contracts import (
    BoardName,
    BoardSpec,
    Frame,
    LocalStop,
    MonotonicClock,
    load_board,
)
from zeos_space_invaders_web.runner import WallClockZeosRunner, build_driver

CRITERIA = {
    "fire-reaches-a-handler",
    "the-reflex-displaces-deliberation",
    "the-dodge-lands-inside-the-budget",
    "the-pilot-is-told-the-ship-moved",
}


def _native_stubs(native_demo: Path) -> ModuleType:
    """The native suite's ``tests/stubs.py``, loaded by path under its own name."""
    spec = importlib.util.spec_from_file_location(
        "native_space_invaders_stubs", native_demo / "tests" / "stubs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stub_machine(native_demo: Path) -> Any:
    return _native_stubs(native_demo).stub_machine


def board(name: BoardName = "default", **changes: Any) -> BoardSpec:
    base = load_board(name, seed=7)
    rules = replace(base.rules, fire_chance=changes.pop("fire_chance", 0.9))
    return replace(base, rules=rules, tick_seconds=changes.pop("tick", 0.05), **changes)


def test_the_case_judges_itself_on_the_wall_clock(stub_machine: Any) -> None:
    """Seed 7, fire 0.9, 0.05 s ticks: every criterion holds, as natively."""
    spec = board(max_steps=60)
    # 0.03 s a word: a turn outlasts a tick, so the pilot is mid-completion when
    # fire lands (the native suite's `stubs.SLOW`, scaled to this tick).
    driver = build_driver(stub_machine("shoot", delay=0.03), spec)
    frames: list[Frame] = []

    def collect(frame: Frame) -> None:
        frames.append(frame)

    runner = WallClockZeosRunner(driver, spec, on_frame=collect)
    try:
        result = runner.run()
    finally:
        driver.close()

    verdicts = {str(v["id"]): v for v in result.verdicts}
    assert set(verdicts) == CRITERIA
    failed = {k: v["detail"] for k, v in verdicts.items() if not v["passed"]}
    assert not failed, f"criteria the run did not meet: {failed}"
    assert result.preemptions >= 1
    assert result.reflexes >= 1 and result.cancellations >= 1
    assert result.journal and result.journal[0]["seq"] == 0

    pilot = [r for r in runner.records if r["by"] == "pilot"]
    assert pilot, "the pilot never moved"
    assert all(r["lag_ticks"] >= 0 for r in runner.records)
    lag = result.lag_ticks
    assert 0 <= lag["p50"] <= lag["p95"] <= lag["max"] <= 20
    assert lag["mean"] <= lag["max"]
    assert all(r["latency"] is not None for r in pilot)

    # The world kept time: one frame a tick at most, catch-up rare and small.
    assert result.ticks == spec.max_steps or result.lives == 0
    assert frames[0]["tick"] == 0 and frames[-1]["tick"] == result.ticks
    assert sum(1 + f["catchup"] for f in frames[1:]) == result.ticks
    assert result.catchup_ticks <= result.ticks // 4
    print(
        f"\nstub run: warm {runner.warm_s:.3f}s ticks {result.ticks} "
        f"overrun {runner.overrun_stats()} catchup {result.catchup_ticks} "
        f"lag {lag} preemptions {result.preemptions} "
        f"cancellations {result.cancellations} decisions {result.decisions} "
        f"pilot {len(pilot)} reflexes {result.reflexes}"
    )


def test_the_stop_flag_ends_a_live_run_within_one_tick(stub_machine: Any) -> None:
    spec = board(tick=0.05)
    driver = ZeosDriver(machine=stub_machine("left", delay=0.03))
    stop = LocalStop()
    clock = MonotonicClock()
    stopped_at: list[float] = []

    def on_frame(frame: Frame) -> None:
        if frame["tick"] == 5:
            stop.set()
            stopped_at.append(clock.now())

    runner = WallClockZeosRunner(driver, spec, stop=stop, on_frame=on_frame, clock=clock)
    try:
        result = runner.run()
    finally:
        driver.close()
    # No tick after the one that saw the flag set, and the loop out well inside it.
    assert result.ticks == 5
    assert runner.ended_at - stopped_at[0] < spec.tick_seconds


def test_the_ablation_board_runs_on_the_wall_clock(stub_machine: Any) -> None:
    spec = board("ablation", tick=0.03, max_steps=20, fire_chance=0.25)
    driver = build_driver(stub_machine("right", delay=0.005), spec)
    runner = WallClockZeosRunner(driver, spec)
    try:
        result = runner.run()
    finally:
        driver.close()
    assert result.board == "ablation"
    assert result.ticks == 20 or result.lives == 0
    assert any(r["by"] == "pilot" for r in runner.records)
