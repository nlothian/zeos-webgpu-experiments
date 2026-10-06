# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What the machine and prompt-arm tests share: a clock the worker sleeps on, and
builders for a machine over ``FakePilotWorker``."""

from __future__ import annotations

from collections.abc import Sequence

from zeos.core.ids import JobId
from zeos.machine.base import DecodeResult, tokens_from_text

from zeos_space_invaders_web.fake_worker import FakePilotWorker
from zeos_space_invaders_web.machine import PilotJsMachine

BODY = "You are flying the ship."
PILOT = JobId(1)


class FakeClock:
    """Wall time that moves only when someone sleeps on it, so latency is exact."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


def fake_worker(
    moves: Sequence[str] = ("left", "right", "shoot"),
    *,
    step_ms: float = 0.0,
    position_ms: float = 0.0,
    clock: FakeClock | None = None,
    **kwargs: object,
) -> tuple[FakePilotWorker, FakeClock]:
    clock = clock or FakeClock()
    worker = FakePilotWorker(
        moves,
        step_ms=step_ms,
        position_ms=position_ms,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )
    return worker, clock


def pilot(
    worker: FakePilotWorker, job: JobId = PILOT, body: str = BODY, **kwargs: object
) -> PilotJsMachine:
    """A machine with a pilot context and its body injected, as the kernel leaves it."""
    machine = PilotJsMachine(worker, **kwargs)  # pyright: ignore[reportArgumentType]
    machine.create_context(job, "pilot")
    machine.inject(job, tokens_from_text(body))
    return machine


def until(
    machine: PilotJsMachine, job: JobId = PILOT, op: str = "write", limit: int = 2000
) -> list[DecodeResult]:
    """Decode until a request of ``op`` arrives; every result, stalls included."""
    out: list[DecodeResult] = []
    for _ in range(limit):
        result = machine.decode(job, allow_control=False)
        out.append(result)
        if result.request.op.value == op:
            return out
    raise AssertionError(f"no {op} in {limit} decodes")


def words(results: Sequence[DecodeResult]) -> list[str]:
    return [t.text for r in results for t in r.tokens]
