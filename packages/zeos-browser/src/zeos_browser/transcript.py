# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The lines ``zeos-count run`` prints to a terminal, collected for the page to show.

The CLI's transcript has two sources, and so does this one. What a job said comes from
the machine's ``on_command`` and ``on_arrival`` callbacks, the moment the seat sees it,
mid-tick. What the kernel decided -- a write that landed, a job that blocked -- comes
from the journal after each tick, in the order the journal recorded it. Rebuilding the
transcript from the journal alone would interleave the two by journal order, which is
not the order the CLI prints them in, so this takes the CLI's callbacks rather than
parsing the page's JSON lines. None of it touches the journal.

The formatting is ``_cmd_run`` in ``zeos_coop_count.cli``, line for line; the CLI's banner,
its closing summary and its interactive prompt are left out, since they are about the
terminal rather than the run.
"""

from __future__ import annotations

from collections.abc import Sequence

from zeos.core.events import Event, JobBlocked, PipeWritten
from zeos.core.ids import JobId
from zeos.core.kernel import Kernel
from zeos.machine.base import MachineRequest, OpKind

__all__ = ["Transcript"]


class Transcript:
    """Collects the CLI's lines for one run; ``take`` hands back the ones not yet taken."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._kernel: Kernel | None = None
        self._names: dict[JobId, str] = {}
        self._drained = 0
        self._taken = 0

    def attach(self, kernel: Kernel) -> None:
        """The kernel whose scheduler names the jobs; set once it has been built."""
        self._kernel = kernel

    def name_of(self, job: JobId) -> str:
        if job not in self._names:
            jobs = self._kernel.sched.jobs() if self._kernel is not None else ()
            found = next((j for j in jobs if j.job_id == job), None)
            self._names[job] = str(found.name) if found is not None else f"job {job}"
        return self._names[job]

    # -- the machine's callbacks, as the CLI passes them -----------------------

    def on_command(self, job: JobId, line: str, request: MachineRequest) -> None:
        who = f"{self.name_of(job):<10}"
        if request.op is OpKind.WRITE or request.op is OpKind.READ:
            # A write is shown from the journal once it has landed; a read is silent.
            return
        if request.op is OpKind.EXIT:
            self.lines.append(f"{who} exit")
        else:
            self.lines.append(f"{who} {line}")

    def on_arrival(self, job: JobId, text: str) -> None:
        self.lines.append(f"{self.name_of(job):<10} ◀── {text}")

    # -- what the run loop adds ----------------------------------------------

    def dropped(self, reason: str) -> None:
        """A delivery the kernel refused because the pipe was full."""
        self.lines.append(f"(dropped: {reason})")

    def drain(self, events: Sequence[Event]) -> None:
        """The facts only the kernel knows, from the events not yet drained."""
        for event in events[self._drained :]:
            if isinstance(event, PipeWritten) and event.job is not None:
                who = f"{self.name_of(event.job):<10}"
                self.lines.append(f"{who} ──▶ {event.pipe:<12} {' '.join(event.text)}")
            elif isinstance(event, JobBlocked):
                self.lines.append(f"{self.name_of(event.job):<10} ... waiting on {event.pipe}")
        self._drained = len(events)

    def take(self) -> list[str]:
        """Lines added since the last ``take``."""
        fresh = self.lines[self._taken :]
        self._taken = len(self.lines)
        return fresh
