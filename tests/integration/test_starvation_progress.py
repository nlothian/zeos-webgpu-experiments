# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Starvation counts preemptions since the job last made progress, not in its lifetime.

Core §5.5 asks for a loud scheduler fault when a low-priority job is preempted more
than K times: an interrupt storm in which the job never gets anything done. A job that
is interrupted now and then but completes a request between interruptions is the
opposite of that, and a lifetime count faulted it anyway -- the Space Invaders pilot,
preempted by its ``evade`` reflex on every incoming bomb, died at the ninth bomb
however many moves it had made in between.

Progress is a request the kernel serviced without refusing it (``_handle_request``).
These tests pin both halves: a true storm still faults exactly as before, and a job
that keeps getting requests through never does, however often it is interrupted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FaultRaised, JobPreempted
from zeos.core.ids import DescriptorName, FaultKind, JobState, PipeName, Priority, VectorName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pcb import Job
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

IRQ = PipeName("irq")
OUT = PipeName("out")
K = KernelConfig().starvation_limit

WORK: dict[str, Any] = {"emit": "work"}
MOVE: dict[str, Any] = {"write": {"pipe": "stdout", "text": "move"}}
#: A write to a pipe the victim does not bind: refused with a capability fault.
REFUSED: dict[str, Any] = {"write": {"pipe": "elsewhere", "text": "move"}}


def _kernel(
    victim_steps: Sequence[Mapping[str, Any]], *, on_fault: str | None = None
) -> tuple[Kernel, Job, list[Event]]:
    """A victim that outlives any test, and a handler that outranks it on every ping."""
    front: dict[str, Any] = {"name": "victim", "priority": 80, "pipes": {"stdout": str(OUT)}}
    if on_fault is not None:
        front["on_fault"] = on_fault
    victim = Descriptor.from_frontmatter(front, body="v")
    handler = Descriptor.from_frontmatter(
        {"name": "handler", "priority": 5, "budget": {"tokens": 64}, "pipes": {"stdin": "irq"}},
        body="h",
    )
    machine = ScriptedMachine(
        {
            "victim": Script.from_spec([*victim_steps, {"exit": True}]),
            "handler": Script.from_spec([{"emit": "ack"}, {"exit": True}]),
        },
        block_size=16,
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("victim"): victim, DescriptorName("handler"): handler},
        machine=machine,
        pipes=PipeTable(
            [
                PipeSpec(name=IRQ, device=True, capacity_tokens=64),
                PipeSpec(name=OUT, sink=True, capacity_tokens=100_000),
            ]
        ),
        vectors=VectorTable(
            [
                VectorSpec(
                    name=VectorName("irq"),
                    source=IRQ,
                    handler=DescriptorName("handler"),
                    priority=Priority(5),
                )
            ]
        ),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="starvation-progress"),
    )
    kernel.start()
    job = kernel.spawn(DescriptorName("victim"))
    kernel.tick()  # the victim's start
    return kernel, job, events


def _interrupt(kernel: Kernel, victim: Job) -> None:
    """Preempt the victim once and run the handler to completion.

    Stops while the victim is still stacked, *before* it decodes again, so the caller
    decides exactly what the victim gets done between interruptions.
    """
    kernel.deliver(IRQ, "ping")
    for _ in range(16):
        kernel.tick()
        others = [j for j in kernel.sched.jobs() if j is not victim]
        if victim.state is JobState.FAULTED or all(j.state.is_terminal for j in others):
            return
    raise AssertionError("the handler never finished")


def _run_victim(kernel: Kernel, victim: Job, steps: int) -> None:
    """Let the victim decode ``steps`` script steps, one per tick."""
    for _ in range(steps):
        kernel.tick()
        assert victim.state is JobState.RUNNING


def _starvation(events: Sequence[Event]) -> list[FaultRaised]:
    return [e for e in events if isinstance(e, FaultRaised) and e.fault is FaultKind.STARVATION]


def _preemptions(events: Sequence[Event], victim: Job) -> int:
    return sum(1 for e in events if isinstance(e, JobPreempted) and e.job == victim.job_id)


def test_a_storm_still_starves_the_job() -> None:
    """K+1 preemptions with only decoded tokens between them: a decoded token is not
    progress, because a storm can let one through, so this faults exactly as before."""
    kernel, victim, events = _kernel([WORK] * 200)
    for _ in range(K + 1):
        _interrupt(kernel, victim)
        if victim.state is JobState.FAULTED:
            break
        _run_victim(kernel, victim, 1)

    [fault] = _starvation(events)
    assert fault.job == victim.job_id
    assert fault.detail == f"preempted {K + 1} times, over the limit of {K}"
    assert victim.state is JobState.FAULTED
    assert _preemptions(events, victim) == K + 1


def test_a_job_that_makes_progress_between_preemptions_never_starves() -> None:
    """Many more than K preemptions in all, a completed write between each pair."""
    rounds = 5 * K
    kernel, victim, events = _kernel([WORK, MOVE] * (rounds + 1))
    for _ in range(rounds):
        _interrupt(kernel, victim)
        _run_victim(kernel, victim, 2)
        assert victim.preempt_count == 0, "a serviced write is progress"

    assert _starvation(events) == []
    assert victim.state is JobState.RUNNING
    assert _preemptions(events, victim) == rounds


def test_exactly_k_preemptions_without_progress_is_within_the_limit() -> None:
    kernel, victim, events = _kernel([WORK] * 200)
    for _ in range(K):
        _interrupt(kernel, victim)
        _run_victim(kernel, victim, 1)

    assert _starvation(events) == []
    assert victim.preempt_count == K


def test_progress_resets_the_count_and_k_plus_one_after_it_still_faults() -> None:
    """The boundary on both sides of a reset: K, progress, K more is fine; one more
    after that is a storm counted from the reset, and faults."""
    kernel, victim, events = _kernel([WORK] * K + [MOVE] + [WORK] * 200)
    for _ in range(K):
        _interrupt(kernel, victim)
        _run_victim(kernel, victim, 1)
    assert victim.preempt_count == K
    _run_victim(kernel, victim, 1)  # the write
    assert victim.preempt_count == 0

    for _ in range(K):
        _interrupt(kernel, victim)
        _run_victim(kernel, victim, 1)
    assert _starvation(events) == []

    _interrupt(kernel, victim)
    [fault] = _starvation(events)
    assert fault.detail == f"preempted {K + 1} times, over the limit of {K}"
    assert victim.state is JobState.FAULTED
    assert _preemptions(events, victim) == 2 * K + 1, "faulted with a lifetime count past K"


def test_a_refused_request_is_not_progress() -> None:
    """A job retrying a call the kernel refuses is going nowhere, so the refusal must
    not reset the count -- under ``on_fault: retry`` it would otherwise never starve."""
    kernel, victim, events = _kernel([REFUSED] * 200, on_fault="retry")
    for _ in range(K + 1):
        _interrupt(kernel, victim)
        _run_victim(kernel, victim, 1)

    refusals = [e for e in events if isinstance(e, FaultRaised) and e.fault is FaultKind.CAPABILITY]
    assert len(refusals) >= K + 1
    [fault] = _starvation(events)
    assert fault.job == victim.job_id
