# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Run a coop-count case under CPython, with the model worker running in Node.

The same ``JsMachine``, ``LiveRun`` loop and Transformers.js worker as the page, with
``NodeWorker`` standing in for the browser's model thread, so a case can be run, timed
and compared from a shell::

    uv run python -m zeos_coop_count_web.node_run ../coop-count/cases/coop-count-pipe \\
        --events ../coop-count/cases/coop-count-pipe/events.jsonl --journal out.jsonl

Besides the journal it writes an attention file: one JSON line per decode step, keyed
by the sequence number of the step's ``machine.decode`` event, with the measured mass
per kernel block and per segment as the kernel accounted it. The journal has no field
for attention -- its events are the kernel's -- so, like ``zeos.trace``'s raw trace, the
machine's measurement is kept beside the byte-compared record rather than in it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zeos.core.ids import JobId, Ring
from zeos.core.integrity import DEFAULT_THETA_READ
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.driver import load_schedule
from zeos.machine.base import DecodeResult
from zeos.machine.seat import seat_maps

from zeos_coop_count_web.js_machine import JsMachine
from zeos_coop_count_web.live import LiveRun
from zeos_coop_count_web.node_worker import DEFAULT_MODEL, NodeWorker

__all__ = ["MeasuredJsMachine", "main"]


class MeasuredJsMachine(JsMachine):
    """``JsMachine`` that keeps every step's measured attention, and can hide blocks.

    ``hidden`` maps a job to kernel blocks removed from every mask the kernel installs
    for it, through ``set_mask`` like any other narrowing, so the worker never lets the
    job attend them; it exists to show that a masked block's measured mass is zero. The
    kernel installs a mask after the descriptor body is injected and before the first
    decode, so a block of the body is hidden from the job's first step on.
    """

    def __init__(
        self, *args: Any, hidden: Mapping[int, frozenset[int]] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden = dict(hidden or {})
        self.rows: list[dict[str, Any]] = []
        self.run: LiveRun | None = None

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        super().set_mask(job, allowed_blocks - self.hidden.get(int(job), frozenset()))

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        result = super().decode(job, allow_control=allow_control)
        if result.attention is not None and self.run is not None:
            self.rows.append(self._row(job, result.attention))
        return result

    def _row(self, job: JobId, attention: Mapping[int, float]) -> dict[str, Any]:
        """The step's mass per block, and per segment as the kernel sums it: over each
        resident segment's blocks, dropping what lands outside the visible blocks."""
        assert self.run is not None
        kernel_job = next(j for j in self.run.kernel.sched.jobs() if j.job_id == job)
        visible = self.visible_blocks(job)
        segments: dict[str, float] = {}
        for record in kernel_job.segments.resident():
            blocks = kernel_job.segments.blocks_for(record)
            total = sum(attention.get(b, 0.0) for b in blocks)
            if total > 0 and record.readable and blocks & visible:
                segments[str(int(record.id))] = total
        mask = self._ctx_of(job).mask  # pyright: ignore[reportPrivateUsage]
        return {
            # The kernel journals this step's ``machine.decode`` next, so it takes the
            # sequence number one past every event so far.
            "seq": len(self.run.events),
            "job": int(job),
            "blocks": {str(b): attention[b] for b in sorted(attention)},
            "segments": segments,
            # Mass on blocks the installed mask was not computed over: the block the job
            # is writing into, visible until the next mask (``js_machine`` docstring).
            "unmasked": 0.0
            if mask is None
            else sum(v for b, v in attention.items() if b not in mask),
        }


def _hidden(specs: Sequence[str]) -> dict[int, frozenset[int]]:
    out: dict[int, set[int]] = {}
    for spec in specs:
        job, _, block = spec.partition(":")
        out.setdefault(int(job), set()).add(int(block))
    return {job: frozenset(blocks) for job, blocks in out.items()}


def with_rings(bundle: CaseBundle, specs: Sequence[str]) -> CaseBundle:
    """The case with some pipes declared at another ring, e.g. ``keys.number=EXTERNAL``:
    content from an untrusted principal, which is what gives integrity something to do."""
    rings = {name: Ring[ring] for name, _, ring in (spec.partition("=") for spec in specs)}
    unknown = set(rings) - {str(p.name) for p in bundle.pipes}
    if unknown:
        raise SystemExit(f"no such pipe in {bundle.name}: {sorted(unknown)}")
    pipes = tuple(
        dataclasses.replace(p, ring=rings[str(p.name)]) if str(p.name) in rings else p
        for p in bundle.pipes
    )
    return dataclasses.replace(bundle, pipes=pipes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--events", type=Path, help="a schedule of device writes (jsonl)")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--attention", type=Path, help="where the attention file goes (default: beside it)"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime", choices=("web", "node"), default="web")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--theta-read", type=float, default=DEFAULT_THETA_READ)
    parser.add_argument(
        "--hide",
        action="append",
        default=[],
        metavar="JOB:BLOCK",
        help="remove a kernel block from every mask a job is given (repeatable)",
    )
    parser.add_argument(
        "--ring",
        action="append",
        default=[],
        metavar="PIPE=RING",
        help="declare a pipe at another ring for this run, e.g. keys.number=EXTERNAL",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    bundle = with_rings(load_case(args.case), args.ring)
    descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
    schedule = load_schedule(args.events) if args.events else ()
    started = time.monotonic()
    with NodeWorker(args.model, runtime=args.runtime, threads=args.threads) as worker:

        def on_command(job: JobId, line: str, _request: object) -> None:
            if not args.quiet:
                print(f"{time.monotonic() - started:7.1f}s  job {job}  {line}", flush=True)

        machine = MeasuredJsMachine(
            worker,
            descriptors=descriptors,
            valued=valued,
            hidden=_hidden(args.hide),
            on_command=on_command,
        )
        run = LiveRun(
            bundle,
            machine,
            schedule=schedule,
            max_ticks=args.max_ticks,
            theta_read=args.theta_read,
        )
        machine.run = run
        while not run.finished:
            run.step()
            if run.waiting_for_a_press():
                # Nothing here presses a key, so a run parked on the console is over.
                run.stop("parked on the console with nothing left to deliver")
        args.journal.write_bytes(run.journal_bytes())
        attention = args.attention or args.journal.with_suffix(".attention.jsonl")
        attention.write_text("".join(json.dumps(row) + "\n" for row in machine.rows))
        print(
            f"{bundle.name}: {run.reason} after {run.ticks} ticks, "
            f"{len(machine.rows)} measured steps, {time.monotonic() - started:.0f}s "
            f"on {worker.backend}; journal {args.journal}, attention {attention}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
