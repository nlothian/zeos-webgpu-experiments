# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What the page's Web Worker calls into: lint a case, open a run, draw it.

Each function takes and returns plain values -- paths, strings, JSON text -- so the
JavaScript side needs no knowledge of the kernel's types. A run is handed back as a
``LiveRun`` proxy, which the page steps and presses.

Two machines are offered, named as the page names them:

``scripted``
    ``CommandSeat`` over ``TapeSource``, which is what ``zeos-count run --machine
    scripted`` runs: each descriptor's tape, one word per decode.
``js``
    ``JsMachine`` over a JavaScript ``ZeosModelWorker`` the caller passes in. The page
    passes the stub worker, which plays the same tapes through every method of the
    seam, so on a tape case the two machines write the same journal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zeos.core.integrity import DEFAULT_THETA_READ
from zeos.debugger.payload import build_payload
from zeos.descriptor.lint import Finding, Severity, lint
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.driver import load_schedule
from zeos.machine.base import MachineBackend
from zeos.machine.seat import CommandSeat, TapeSource, seat_maps

from zeos_coop_count_web.fake_worker import tapes_from_scripts
from zeos_coop_count_web.js_machine import Bridge, JsMachine, ZeosModelWorker
from zeos_coop_count_web.live import LiveRun

__all__ = ["MACHINES", "describe", "findings", "open_run", "payload_json", "tapes_json"]

MACHINES = ("scripted", "js")


def findings(bundle: CaseBundle) -> list[Finding]:
    """The case's lint findings, checked as ``zeos-count lint`` checks them."""
    return list(
        lint(
            bundle.descriptors,
            pipes=bundle.pipes,
            vectors=bundle.vectors,
            resources=bundle.resources,
            platforms=bundle.platforms,
            principals=bundle.principals,
            gates=bundle.gates,
        )
    )


def describe(case_dir: str) -> str:
    """A case as the page first shows it: its lint, whether it can run, and its wiring."""
    case = Path(case_dir)
    bundle = load_case(case)
    found = findings(bundle)
    return json.dumps(
        {
            "name": bundle.name,
            "lint": [f.render() for f in found],
            "errors": sum(1 for f in found if f.severity is Severity.ERROR),
            "descriptors": len(bundle.descriptors),
            "vectors": len(bundle.vectors),
            # Only a case whose every job has a tape can run without a model.
            "runnable": bool(bundle.descriptors)
            and all(str(name) in bundle.scripts for name in bundle.descriptors),
            "schedule": (case / "events.jsonl").is_file(),
            "devices": sorted(str(p.name) for p in bundle.pipes if p.device),
            "payload": build_payload(bundle, findings=found),
        },
        separators=(",", ":"),
    )


def tapes_json(case_dir: str) -> str:
    """Each descriptor's tape, for the stub worker to play."""
    return json.dumps(tapes_from_scripts(load_case(Path(case_dir)).scripts))


def _machine(
    bundle: CaseBundle, name: str, worker: ZeosModelWorker | None, bridge: Bridge | None
) -> MachineBackend:
    if name == "scripted":
        return CommandSeat(source=TapeSource(bundle.scripts))
    if name == "js":
        if worker is None:
            raise ValueError("the js machine needs a ZeosModelWorker")
        descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
        return JsMachine(worker, bridge=bridge, descriptors=descriptors, valued=valued)
    raise ValueError(f"unknown machine {name!r}; expected one of {MACHINES}")


def open_run(
    case_dir: str,
    machine: str,
    *,
    schedule: bool = True,
    worker: ZeosModelWorker | None = None,
    seed: int = 0,
    max_ticks: int = 100_000,
    theta_read: float = DEFAULT_THETA_READ,
) -> LiveRun:
    """A run of the case, booted and ready to step. Refuses a case that does not lint.

    ``schedule`` plays the case's ``events.jsonl``. ``worker`` is the JavaScript
    ``ZeosModelWorker`` for the ``js`` machine, reached across Pyodide's FFI. A model
    seat counts for ever, so ``max_ticks`` is what ends its run.
    """
    case = Path(case_dir)
    bundle = load_case(case)
    blocking = [f for f in findings(bundle) if f.severity is Severity.ERROR]
    if blocking:
        raise ValueError(
            "refusing to run a tree that does not lint:\n" + "\n".join(f.render() for f in blocking)
        )
    bridge: Bridge | None = None
    if worker is not None:
        from zeos_coop_count_web.pyodide_bridge import PyodideBridge

        bridge = PyodideBridge()
    events = case / "events.jsonl"
    return LiveRun(
        bundle,
        _machine(bundle, machine, worker, bridge),
        schedule=load_schedule(events) if schedule and events.is_file() else (),
        seed=seed,
        max_ticks=max_ticks,
        trace=True,
        theta_read=theta_read,
    )


def payload_json(run: LiveRun) -> str:
    """The debugger's input for a run, as JSON text."""
    payload: dict[str, Any] = run.payload(findings(run.bundle))
    return json.dumps(payload, separators=(",", ":"))
