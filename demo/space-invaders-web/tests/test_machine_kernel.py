# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The claim, at kernel level: a real zeos ``Kernel`` over the Space Invaders case, the
pilot served by ``PilotJsMachine`` over a slow worker, the reflex served natively.

The worker sleeps on a clock that moves only when it is slept on, so a 20 ms pilot step
is twenty 1 ms stalls of the machine and the run is the same every time.
"""

# pyright: reportPrivateUsage=false
# The game package is untyped, so what the tests read off it is partly unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import pytest
from machine_helpers import fake_worker
from zeos.machine.base import DecodeResult, MachineRequest, OpKind
from zeos_space_invaders.game import Controls, Game, H, snapshot
from zeos_space_invaders.players.zeos import REFLEX, ZeosDriver, evade_behaviour
from zeos_space_invaders.players.zeos.api_machine import Native

from zeos_space_invaders_web.fake_worker import FakePilotWorker
from zeos_space_invaders_web.machine import PilotJsMachine


def _driver(worker: FakePilotWorker) -> tuple[ZeosDriver, PilotJsMachine, Game]:
    machine = PilotJsMachine(worker, stall_ms=1.0)
    machine.register_behaviour(REFLEX, evade_behaviour)
    driver = ZeosDriver(machine=machine)  # pyright: ignore[reportArgumentType]
    game = Game(seed=1)
    game.rng.random = lambda: 1.0  # no unplanned monster fire
    game.player = 4
    driver.controls = Controls(game, per_tick=1)
    return driver, machine, game


def _run(driver: ZeosDriver, game: Game, ticks: int, threat_from: int) -> list[Any]:
    decisions: list[Any] = []
    for tick in range(ticks):
        game.dangers = [[H - 2, game.player]] if tick >= threat_from else []
        decisions.append(driver.step(game.render(), snapshot(game)))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        assert driver.controls is not None
        driver.controls.tick()
    return decisions


def test_the_pilot_flies_through_the_kernel() -> None:
    worker, _ = fake_worker(step_ms=5.0)
    driver, machine, game = _driver(worker)
    decisions = _run(driver, game, 4, threat_from=99)
    pilot = [d for d in decisions if d is not None and d.by == "pilot"]
    assert [d.action for d in pilot][:3] == ["left", "right", "shoot"]
    assert machine.roundtrips and machine.cancellations == 0
    driver.close()


def test_a_threat_preempts_the_pilot_mid_step_and_the_dodge_meets_its_deadline() -> None:
    worker, _ = fake_worker(step_ms=20.0, position_ms=0.05)
    driver, machine, game = _driver(worker)
    decisions = _run(driver, game, 8, threat_from=3)
    verdicts = {v["id"]: v for v in driver.verdicts()}  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert driver.preemptions > 0
    assert machine.cancellations > 0, "the preempted pilot's step was never cancelled"
    assert worker.cancelled > 0
    assert any(d is not None and d.by == "evade" and d.preempted for d in decisions)
    for name in (
        "fire-reaches-a-handler",
        "the-reflex-displaces-deliberation",
        "the-dodge-lands-inside-the-budget",
    ):
        assert verdicts[name]["passed"], verdicts[name]["detail"]
    assert all(not key.endswith(":evade") for key in worker.context_ids)
    assert machine.native_words > 0
    driver.close()
    assert worker.context_ids == ()


def _quiet_reflex(_native: Native) -> DecodeResult:
    """A reflex that dispatches, preempts and exits, and changes nothing in the world, so
    the pilot resumes with nothing injected."""
    return DecodeResult(tokens=(), request=MachineRequest(op=OpKind.EXIT))


@pytest.mark.parametrize("invalidating", [False, True])
def test_reflex_decodes_do_not_cancel_a_long_pilot_step(invalidating: bool) -> None:
    """Only the driver's ``invalidate`` drops the pilot's step when the reflex preempts it;
    the reflex's own decodes never touch the channel. Without the invalidate the step
    survives every dispatch and the move is made."""
    worker, _ = fake_worker(step_ms=150.0)
    machine = PilotJsMachine(worker, stall_ms=1.0)
    machine.register_behaviour(REFLEX, _quiet_reflex)
    driver = ZeosDriver(machine=machine)  # pyright: ignore[reportArgumentType]
    invalidated: list[object] = []
    if invalidating:
        real = machine.invalidate

        def spy(job: Any) -> None:
            invalidated.append(job)
            real(job)

        machine.invalidate = spy  # type: ignore[method-assign]
    else:
        driver._invalidate_preempted = lambda: None  # type: ignore[method-assign]
    game = Game(seed=1)
    game.rng.random = lambda: 1.0
    game.player = 4
    driver.controls = Controls(game, per_tick=1)
    decisions: list[Any] = []
    for tick in range(10):
        game.dangers = [[H - 2, game.player]] if tick % 2 else []
        decisions.append(driver.step(game.render(), snapshot(game)))
        driver.controls.tick()
    assert driver.preemptions >= 3
    reflex_steps = machine.step_totals["native"]
    assert reflex_steps >= driver.preemptions
    if invalidating:
        assert 0 < machine.cancellations <= len(invalidated)
    else:
        assert machine.cancellations == 0
        assert any(d is not None and d.by == "pilot" for d in decisions)
    driver.close()


def test_a_pilot_preempted_and_invalidated_resumes_and_plays_on() -> None:
    """``invalidate`` drops a pending automatic read as the native machine does; the
    resumed pilot starts a fresh command, and keeps flying once the fire stops."""
    worker, _ = fake_worker(step_ms=20.0, position_ms=0.05)
    driver, machine, game = _driver(worker)
    decisions: list[Any] = []
    for tick in range(16):
        game.dangers = [[H - 2, game.player]] if 3 <= tick < 6 else []
        decisions.append(driver.step(game.render(), snapshot(game)))
        assert driver.controls is not None
        driver.controls.tick()
    assert driver.preemptions > 0
    late = [d for d in decisions[7:] if d is not None and d.by == "pilot"]
    assert late, "the pilot never moved again after it was preempted"
    states = {str(job.descriptor.name): job.state.value for job in driver.kernel.sched.jobs()}
    assert states["pilot"] not in ("faulted", "done")
    malformed = [e for e in driver.kernel.events if "malformed" in getattr(type(e), "KIND", "")]
    assert not malformed
    assert machine.cancellations > 0
    driver.close()


def _pilot_job(machine: PilotJsMachine) -> Any:
    (job,) = machine._rounds  # the one served job
    return job


def test_an_invalidate_after_the_pilot_resumed_spares_its_fresh_command() -> None:
    """The driver invalidates at the end of a batch for each preemption in it. A pilot
    preempted and resumed before that may already be saying a new command, against the
    world as it is now: the invalidate leaves that step alone and counts no cancel."""
    worker, _ = fake_worker(step_ms=20.0, position_ms=0.05)
    driver, machine, game = _driver(worker)
    driver._invalidate_preempted = lambda: None  # held back, to be issued late
    fresh = False
    for tick in range(80):
        game.dangers = [[H - 2, game.player]] if 3 <= tick < 6 else []
        driver.step(game.render(), snapshot(game))
        assert driver.controls is not None
        driver.controls.tick()
        if driver.preemptions and machine._flight is not None:
            job = _pilot_job(machine)
            left = machine._left.get(job)
            if left is not None and machine._rounds[job].epoch > left:
                fresh = True
                break
    assert fresh, "the pilot never began a command after resuming"
    job = _pilot_job(machine)
    flight, cancellations = machine._flight, machine.cancellations
    machine.invalidate(job)
    assert machine._flight is flight and machine.cancellations == cancellations
    assert machine.invalidations_skipped == 1
    driver.close()


def test_an_invalidate_for_a_live_preemption_still_cancels_the_step() -> None:
    """Invalidated while the reflex holds the machine, the pilot's interrupted step goes."""
    worker, _ = fake_worker(step_ms=20.0, position_ms=0.05)
    driver, machine, game = _driver(worker)
    driver._invalidate_preempted = lambda: None
    seen: list[tuple[int, int]] = []
    real = machine.decode

    def decode(job: Any, *, allow_control: bool) -> DecodeResult:
        flight = machine._flight
        result = real(job, allow_control=allow_control)
        if job in machine._native and flight is not None and not seen:
            # The reflex has the machine; the pilot's step is the one it interrupted.
            before = machine.cancellations
            machine.invalidate(flight.job)
            seen.append((before, machine.cancellations))
        return result

    machine.decode = decode  # type: ignore[method-assign]
    _run(driver, game, 12, threat_from=3)
    assert seen, "the reflex never ran while a pilot step was in flight"
    before, after = seen[0]
    assert after == before + 1 and machine.invalidations_skipped == 0
    driver.close()
