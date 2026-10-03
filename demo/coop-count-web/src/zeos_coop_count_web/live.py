# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A run the page can step one tick at a time, with keypresses arriving between ticks.

``zeos.driver.Driver.run`` takes the whole event schedule up front and runs to
quiescence in one call, which leaves no moment at which a person can press a key. The
``zeos-count`` CLI unrolls that loop so its console can deliver between ticks; this is
the same loop again, cut at each turn so the page decides when the next one happens.

One turn of ``step`` is one turn of the CLI's loop, in the same order: deliver every
scheduled event that is due, deliver what the page queued with ``press``, advance the
clock, tick, sample the machine's account, reap, and add one millisecond of virtual
time whether or not anything ran. With no presses, a run is the CLI's run and its
journal is byte for byte the one ``zeos-count run --machine scripted --events`` writes.

A press is delivered at the start of the next turn, before that turn advances the
clock -- exactly where the CLI delivers a scheduled event that has come due -- so the
journal stamps it with the previous turn's time. The page cannot slip an event between
two halves of a tick, and the journal records the delivery like any other.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zeos.core.events import Event
from zeos.core.ids import PipeName
from zeos.core.integrity import DEFAULT_THETA_READ
from zeos.core.kernel import KernelConfig
from zeos.core.pipes import PipeFull
from zeos.debugger.payload import build_payload
from zeos.descriptor.lint import Finding
from zeos.descriptor.loader import CaseBundle
from zeos.driver import Driver, ScheduledEvent, build_kernel
from zeos.journal.codec import to_line
from zeos.journal.writer import Journal
from zeos.machine.base import MachineBackend, TracesRaw
from zeos.trace import RawTrace

__all__ = ["LiveRun"]


class LiveRun:
    """A kernel, a machine and a schedule, advanced one turn per ``step``."""

    def __init__(
        self,
        bundle: CaseBundle,
        machine: MachineBackend,
        *,
        schedule: Sequence[ScheduledEvent] = (),
        seed: int = 0,
        max_ticks: int = 100_000,
        trace: bool = False,
        theta_read: float = DEFAULT_THETA_READ,
    ) -> None:
        self.bundle = bundle
        self.machine = machine
        self.events: list[Event] = []
        self.kernel, transport = build_kernel(
            bundle,
            machine=machine,
            journal_sink=self.events,
            config=KernelConfig(
                seed=seed, case=bundle.name, max_ticks=max_ticks, theta_read=theta_read
            ),
        )
        self.journal = Journal()
        self.trace = RawTrace() if trace and isinstance(machine, TracesRaw) else None
        self.driver = Driver(self.kernel, transport=transport, journal=self.journal)
        self.driver.boot(bundle.boot)
        self._sampled = len(self.events)
        self._pending = sorted(schedule, key=lambda e: e.at_ns)
        self._presses: list[tuple[PipeName, str]] = []
        self._device_pipes = frozenset(p.name for p in bundle.pipes if p.device)
        self._shown = 0
        self.max_ticks = max_ticks
        self.now_ns = 0
        self.ticks = 0
        self.finished = False
        self.reason = ""
        #: Deliveries the kernel refused because the pipe was full, in order.
        self.refused: list[tuple[PipeName, str]] = []

    def press(self, pipe: str, text: str) -> None:
        """Queue a write to a device pipe for the next turn, as a console would."""
        if self.finished:
            raise RuntimeError(f"the run has finished ({self.reason}); nothing reads a press")
        self._presses.append((PipeName(pipe), text))

    def awaiting_input(self) -> bool:
        """Whether a live job is parked on a device pipe, which only a press can fill."""
        return any(job.blocked_on in self._device_pipes for job in self.kernel.sched.live())

    def blocked_on(self, pipe: str) -> bool:
        return any(job.blocked_on == pipe for job in self.kernel.sched.live())

    def _deliver(self, pipe: PipeName, text: str) -> None:
        try:
            self.kernel.deliver(pipe, text)
        except PipeFull:
            self.refused.append((pipe, text))

    def step(self) -> list[str]:
        """One turn of the loop. Returns the journal lines it added."""
        if self.finished:
            return []
        while self._pending and self._pending[0].at_ns <= self.now_ns:
            event = self._pending.pop(0)
            self._deliver(event.pipe, event.text)
        presses, self._presses = self._presses, []
        for pipe, text in presses:
            self._deliver(pipe, text)

        self.kernel.advance_time(self.now_ns)
        ran = self.kernel.tick()
        if self.trace is not None and isinstance(self.machine, TracesRaw):
            self.trace.sample(self.machine, self.events[self._sampled :], self._sampled)
            self._sampled = len(self.events)
        self.driver.reap_finished()
        self.now_ns += Driver.DEFAULT_NS_PER_TICK
        if ran:
            self.ticks += 1
            if self.ticks >= self.max_ticks:
                self.finished, self.reason = True, f"reached {self.max_ticks} ticks"
        elif not self._pending and not self._presses and not self.awaiting_input():
            self.finished, self.reason = True, "quiescent"
        return self.lines()

    def lines(self) -> list[str]:
        """Journal lines not yet returned, each one JSON object as the journal file holds it."""
        self.journal.extend(self.events[len(self.journal) :])
        records = self.journal.records[self._shown :]
        self._shown = len(self.journal)
        return [to_line(r.seq, r.event) for r in records]

    def stop(self, reason: str = "stopped") -> None:
        if not self.finished:
            self.finished, self.reason = True, reason

    def journal_bytes(self) -> bytes:
        self.lines()
        return self.journal.to_bytes()

    def payload(self, findings: Sequence[Finding] = ()) -> dict[str, Any]:
        """What the debugger page draws: the case's wiring, and this run's frames."""
        self.lines()
        return build_payload(
            self.bundle,
            records=self.journal.records,
            findings=findings,
            trace=None if self.trace is None else self.trace.rows,
        )
