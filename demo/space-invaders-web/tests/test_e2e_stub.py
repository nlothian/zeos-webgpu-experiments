# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""End to end under CPython: ``page.open_run`` -> ``RUN_BUILDERS`` -> the real machine,
driver and runner, against ``FakePilotWorker`` on the wall clock.

The pilot's worker takes 20 ms a step and 1 ms a position, the board ticks every 0.05 s
and fire is at 0.9, so the reflex has to preempt a pilot step in flight. These are the
plan's Verification numbers; they are also what the page does with the stub machine, less
the JavaScript channel (``pyodide_run.mjs`` covers that).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import pytest

from zeos_space_invaders_web import page
from zeos_space_invaders_web.contracts import BoardName, BoardSpec, Frame, FrameSink, LocalStop
from zeos_space_invaders_web.fake_worker import FakePilotWorker

TICK_S = 0.05
FIRE = 0.9
SEED = 7
MAX_TICKS = 80
CRITERIA = {"vector_fired", "preemption_after", "latency", "resumed_dirty_naming"}


@pytest.fixture
def fast_board(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every board at ``TICK_S`` and ``FIRE``, and runs capped at ``MAX_TICKS``."""
    real = page.load_board

    def load(name: BoardName, seed: int | None = None) -> BoardSpec:
        spec = real(name, seed)
        rules = dataclasses.replace(spec.rules, fire_chance=FIRE)
        return dataclasses.replace(spec, tick_seconds=TICK_S, rules=rules)

    monkeypatch.setattr(page, "load_board", load)
    monkeypatch.setitem(page.RUNNER_OPTIONS, "max_ticks", MAX_TICKS)
    yield


def _worker(**kwargs: Any) -> FakePilotWorker:
    return FakePilotWorker(step_ms=20.0, position_ms=1.0, **kwargs)


def _sink(frames: list[Frame]) -> FrameSink:
    def sink(frame: Frame) -> None:
        frames.append(frame)

    return sink


@pytest.mark.usefixtures("fast_board")
def test_the_zeos_arm_passes_every_criterion_through_open_run() -> None:
    worker = _worker()
    frames: list[Frame] = []
    run = page.open_run("zeos", "default", SEED, worker, on_frame=_sink(frames))
    assert isinstance(run, page.ZeosRun)
    try:
        run.warm()
        assert run.runner.warmed and run.driver.started
        result = run.run()
    finally:
        run.close()
    assert not worker.inFlight, "close left a step in flight"

    verdicts = {str(v["kind"]): v for v in result.verdicts}
    assert CRITERIA <= set(verdicts)
    for kind in CRITERIA:
        assert verdicts[kind]["passed"], (kind, verdicts[kind]["detail"])
    assert result.preemptions >= 1
    assert result.cancellations >= 1
    assert any(d["by"] == "pilot" for f in frames for d in f["decisions"])
    assert any(d["by"] == "evade" for f in frames for d in f["decisions"])
    assert result.journal, "the zeos arm hands the debugger its journal"

    extras = result.extras
    assert extras["pilot_moves"] == extras["pilot_valid_moves"]
    assert extras["pilot_exits"] == 0
    assert isinstance(extras["prewarm_ms"], float)
    # The page's JSON carries it all.
    finished = page.finished_json(result)
    assert '"extras"' in finished and '"payload"' in finished


@pytest.mark.usefixtures("fast_board")
def test_the_prompt_arm_finishes_and_reports_its_parse_rate() -> None:
    worker = _worker(replies=("left", "shoot", "right", "jump"))
    frames: list[Frame] = []
    run = page.open_run("prompt", "default", SEED, worker, on_frame=_sink(frames))
    assert isinstance(run, page.PromptRun)
    try:
        run.warm()
        result = run.run()
    finally:
        run.close()
    assert not worker.inFlight
    assert result.arm == "prompt"
    assert result.decisions >= 1
    assert result.parse_rate is not None and 0.0 < result.parse_rate < 1.0
    assert result.extras["replies"] == result.decisions
    assert all(d["by"] == "prompt" for f in frames for d in f["decisions"])


@pytest.mark.usefixtures("fast_board")
def test_a_stop_mid_run_closes_cleanly() -> None:
    worker = _worker()
    stop = LocalStop()
    seen: list[Frame] = []

    def on_frame(frame: Frame) -> None:
        seen.append(frame)
        if frame["tick"] >= 10:
            stop.set()

    run = page.open_run("zeos", "ablation", SEED, worker, on_frame=on_frame, stop=stop)
    try:
        run.warm()
        result = run.run()
    finally:
        run.close()
    assert result.ticks == 10
    assert not worker.inFlight


def test_the_zeos_clock_tells_monotonic_time() -> None:
    """``LoopClock`` passes the runner's check whatever clock the page sleeps on."""

    class Late:
        def now(self) -> float:
            return 1e9

        def sleep(self, seconds: float) -> None:
            del seconds

    worker = _worker()
    run = page.open_run("zeos", "default", SEED, worker, clock=Late())
    assert isinstance(run, page.ZeosRun)
    run.close()


def test_a_long_run_with_many_preemptions_has_no_starvation_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web demo's kernel is built with ``page.STARVATION_LIMIT``: the kernel's own
    limit (8) would retire the pilot on its ninth preemption by the reflex."""
    real = page.load_board

    def load(name: BoardName, seed: int | None = None) -> BoardSpec:
        spec = real(name, seed)
        rules = dataclasses.replace(spec.rules, fire_chance=FIRE, lives=99)
        return dataclasses.replace(spec, tick_seconds=TICK_S, rules=rules)

    monkeypatch.setattr(page, "load_board", load)
    monkeypatch.setitem(page.RUNNER_OPTIONS, "max_ticks", 240)
    page.configure_json("")
    worker = _worker()
    run = page.open_run("zeos", "default", SEED, worker)
    assert isinstance(run, page.ZeosRun)
    assert run.driver.kernel.config.starvation_limit == page.STARVATION_LIMIT
    try:
        run.warm()
        result = run.run()
    finally:
        run.close()
    assert result.preemptions > 8, "the run did not preempt the pilot often enough to tell"
    assert result.extras["faults"] == []
    pilot = cast(int, result.extras["pilot_moves"])
    assert pilot >= 1


def test_tune_can_put_the_native_starvation_limit_back() -> None:
    try:
        page.configure_json('{"kernel": {"starvation_limit": 8}}')
        assert page.ZEOS_KERNEL_OPTIONS == {"starvation_limit": 8}
        run = page.open_run("zeos", "default", SEED, _worker())
        assert isinstance(run, page.ZeosRun)
        assert run.driver.kernel.config.starvation_limit == 8
        run.close()
    finally:
        page.configure_json("")
    assert page.ZEOS_KERNEL_OPTIONS == dict(page.DEFAULT_ZEOS_KERNEL_OPTIONS)
