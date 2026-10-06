# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""`ZeosDriver` asks its machine for `DriverMachine` and nothing more.

The browser port drives the same driver over a machine with no `APIMachineBase`
in its ancestry, so the claim is tested with an object that only delegates the
protocol's methods to a scripted machine it holds.
"""

from stubs import stub_machine

from zeos_space_invaders.game import Controls, Game, H, snapshot
from zeos_space_invaders.players.zeos import APIMachineBase, ZeosDriver
from zeos_space_invaders.players.zeos.player import DriverMachine


class Delegate:
    """`DriverMachine` by composition: every call forwarded, invalidations counted."""

    def __init__(self, inner):
        self.inner = inner
        self.invalidated = []

    @property
    def block_size(self):
        return self.inner.block_size

    @property
    def last_roundtrip(self):
        return self.inner.last_roundtrip

    def create_context(self, job, descriptor):
        self.inner.create_context(job, descriptor)

    def destroy_context(self, job):
        self.inner.destroy_context(job)

    def stats(self, job):
        return self.inner.stats(job)

    def decode(self, job, *, allow_control):
        return self.inner.decode(job, allow_control=allow_control)

    def inject(self, job, tokens):
        return self.inner.inject(job, tokens)

    def trunc(self, job, at):
        return self.inner.trunc(job, at)

    def fork(self, parent, child):
        return self.inner.fork(parent, child)

    def splice(self, job, start, end, tokens):
        return self.inner.splice(job, start, end, tokens)

    def set_mask(self, job, allowed_blocks):
        self.inner.set_mask(job, allowed_blocks)

    def visible_blocks(self, job):
        return self.inner.visible_blocks(job)

    def pad_to_block(self, job):
        return self.inner.pad_to_block(job)

    def blocks_for_range(self, job, start, end):
        return self.inner.blocks_for_range(job, start, end)

    def transcript(self, job):
        return self.inner.transcript(job)

    def invalidate(self, job):
        self.invalidated.append(job)
        self.inner.invalidate(job)

    def close(self):
        self.inner.close()


def test_a_machine_without_the_api_base_drives_the_driver():
    machine = Delegate(stub_machine("shoot", delay=0.01))
    assert isinstance(machine, DriverMachine)
    assert not isinstance(machine, APIMachineBase)

    game = Game(seed=1)
    game.rng.random = lambda: 1.0  # no unplanned monster fire
    game.player = 4
    driver = ZeosDriver(machine=machine)
    driver.controls = Controls(game, per_tick=1)
    assert driver.machine is machine
    for tick in range(12):
        game.dangers = [[H - 2, game.player]] if tick >= 2 else []
        driver.step(game.render(), snapshot(game))
        driver.controls.tick()
    verdicts = {v["id"]: v["passed"] for v in driver.verdicts()}
    driver.close()

    assert driver.reflexes > 0
    assert driver.preemptions > 0
    # The driver told this machine, not some base class, about each preemption.
    assert machine.invalidated
    assert verdicts["the-reflex-displaces-deliberation"]
    assert verdicts["the-dodge-lands-inside-the-budget"]
