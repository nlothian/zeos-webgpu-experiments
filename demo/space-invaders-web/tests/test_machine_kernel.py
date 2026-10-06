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

from machine_helpers import fake_worker
from zeos_space_invaders.game import Controls, Game, H, snapshot
from zeos_space_invaders.players.zeos import REFLEX, ZeosDriver, evade_behaviour

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
