# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The tape case, run through ``JsMachine`` over the fake worker, one page turn at a time."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from workers import RecordingWorker
from zeos.core.events import (
    Event,
    JobPreempted,
    JobResumed,
    VectorFired,
    WorldWritten,
)
from zeos.core.ids import ResumeKind
from zeos.descriptor.loader import load_case
from zeos.driver import load_schedule
from zeos.journal.writer import read_journal_lines
from zeos.machine.base import MachineBackend
from zeos.machine.seat import CommandSeat, TapeSource, seat_maps

from zeos_coop_count_web.fake_worker import FakeWorker, tapes_from_scripts
from zeos_coop_count_web.js_machine import JsMachine
from zeos_coop_count_web.live import LiveRun

CASES = Path(__file__).resolve().parents[2] / "coop-count" / "cases"
CASE = CASES / "coop-count-scripted"
#: The tape's interrupt, from the case's own schedule: the key at 39ms, the number at 40ms.
INTERRUPT_NS, NUMBER_NS = 39_000_000, 40_000_000
NEW_COUNT = "51"


def js_machine(worker_class: type[FakeWorker] = FakeWorker) -> JsMachine:
    bundle = load_case(CASE)
    descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
    worker = worker_class(tapes_from_scripts(bundle.scripts))
    return JsMachine(worker, descriptors=descriptors, valued=valued)


def seat() -> CommandSeat:
    return CommandSeat(source=TapeSource(load_case(CASE).scripts))


def run_to_end(machine: MachineBackend, *, schedule: bool = True) -> LiveRun:
    events = load_schedule(CASE / "events.jsonl") if schedule else ()
    run = LiveRun(load_case(CASE), machine, schedule=events)
    while not run.finished:
        run.step()
    return run


def _events(run: LiveRun) -> list[Event]:
    return [r.event for r in run.journal.records]


def _of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


@pytest.fixture(scope="module")
def run() -> LiveRun:
    return run_to_end(js_machine())


def test_the_run_ends_quiescent(run: LiveRun) -> None:
    assert run.reason == "quiescent"
    assert not run.awaiting_input()


def test_every_turn_reaches_its_handover(run: LiveRun) -> None:
    assert [(w.obj, w.after) for w in _of(_events(run), WorldWritten)] == [
        ("count.a", "10"),
        ("count.a", NEW_COUNT),
        ("count.b", NEW_COUNT),
        ("count.b", "60"),
    ]


def test_the_keypress_preempts_a_counter_and_it_resumes_dirty(run: LiveRun) -> None:
    events = _events(run)
    fired = _of(events, VectorFired)
    assert [f.vector for f in fired] == ["keyboard-interrupt"]
    preempted = _of(events, JobPreempted)
    assert preempted and preempted[0].by_priority == 5
    assert events.index(preempted[0]) > events.index(fired[0])
    dirty = [r for r in _of(events, JobResumed) if r.resume_kind is ResumeKind.DIRTY]
    assert {d.obj: (d.before, d.after) for d in dirty[0].dirty} == {"count.a": ("10", NEW_COUNT)}


def _facts(run: LiveRun, *, drop: str = "") -> list[dict[str, Any]]:
    """The journal as dicts without sequence numbers, optionally without one kind."""
    out: list[dict[str, Any]] = []
    for line in run.journal_bytes().decode().splitlines():
        record = json.loads(line)
        if record["kind"] != drop:
            del record["seq"]
            out.append(record)
    return out


def test_the_seam_differs_from_the_seat_only_in_what_the_mask_denied(run: LiveRun) -> None:
    """The fake decodes each tape word as one token, through every method of the seam and
    the grammar mask, so the kernel sees the seat's run -- except for attention.

    The seat's mask hides blocks written since the kernel's last refresh, so the hint it
    leaves lands partly on the job's own open segment, the kernel journals that as
    denied, and that segment's mass never reaches the working set. ``JsMachine`` counts
    blocks past the mask's horizon as visible, so nothing is denied, the working set
    includes what the job is writing, and every other event is the seat's.
    """
    ours, theirs = _facts(run), _facts(seat_run := run_to_end(seat()), drop="mask.denied")
    assert sum(r["kind"] == "mask.denied" for r in _facts(seat_run)) > 0
    assert all(r["kind"] != "mask.denied" for r in ours)
    assert [r["kind"] for r in ours] == [r["kind"] for r in theirs]
    differ = {a["kind"] for a, b in zip(ours, theirs, strict=True) if a != b}
    assert differ <= {"vm.working_set"}


def test_the_run_is_the_same_run_twice(run: LiveRun) -> None:
    assert run.journal_bytes() == run_to_end(js_machine()).journal_bytes()


def test_streamed_lines_are_the_journal(run: LiveRun) -> None:
    """What the page shows line by line is the file it offers for download."""
    again = LiveRun(load_case(CASE), js_machine(), schedule=load_schedule(CASE / "events.jsonl"))
    streamed = list(again.lines())
    while not again.finished:
        streamed += again.step()
    assert "".join(line + "\n" for line in streamed).encode() == again.journal_bytes()
    assert [r.seq for r in read_journal_lines(streamed)] == list(range(len(streamed)))


def test_a_press_lands_where_the_scheduled_event_would_have() -> None:
    """The page's keypress and the case's schedule are one delivery path: pressed at the
    turns the schedule names, the run is the scheduled run, byte for byte."""
    pressed = LiveRun(load_case(CASE), js_machine())
    while not pressed.finished:
        if pressed.now_ns == INTERRUPT_NS:
            pressed.press("keys.interrupt", "attention")
        if pressed.now_ns == NUMBER_NS:
            pressed.press("keys.number", NEW_COUNT)
        pressed.step()
    assert pressed.journal_bytes() == run_to_end(js_machine()).journal_bytes()


def test_a_parked_handler_holds_the_run_open_until_the_number_arrives() -> None:
    run = LiveRun(load_case(CASE), js_machine())
    while run.now_ns < INTERRUPT_NS:
        run.step()
    run.press("keys.interrupt", "attention")
    # Long past the end of both tapes: only the handler, parked on the console, is left.
    for _ in range(200):
        run.step()
    assert not run.finished and run.awaiting_input() and run.blocked_on("keys.number")
    run.press("keys.number", NEW_COUNT)
    while not run.finished:
        run.step()
    assert run.reason == "quiescent"
    written = [w for w in _of(_events(run), WorldWritten) if w.after == NEW_COUNT]
    assert {w.obj for w in written} == {"count.a", "count.b"}


def test_an_interrupt_while_the_handler_is_parked_is_withheld() -> None:
    """The console's rule, decided from the kernel when the press is delivered."""
    run = LiveRun(load_case(CASE), js_machine())
    while run.now_ns < INTERRUPT_NS:
        run.step()
    run.press("keys.interrupt", "attention", unless_waiting_on="keys.number")
    while not run.blocked_on("keys.number"):
        run.step()
    run.press("keys.interrupt", "attention", unless_waiting_on="keys.number")
    run.step()
    assert run.withheld == [("keys.interrupt", "attention")]
    run.press("keys.number", NEW_COUNT)
    while not run.finished:
        run.step()
    assert len(_of(_events(run), VectorFired)) == 1


def test_every_step_carries_both_masks() -> None:
    """Every step is told which ids it may emit and which blocks it may attend.

    The kernel installs its block mask at each refresh, naming the blocks of the
    segments that exist then; blocks the job decodes into afterwards lie past the
    horizon and are allowed. Nothing in this case hides a block, so no step is told to
    skip one -- least of all the one holding the position doing the attending.
    """
    machine = js_machine(RecordingWorker)
    run_to_end(machine)
    worker = machine._worker  # pyright: ignore[reportPrivateUsage]
    assert isinstance(worker, RecordingWorker)
    assert worker.seen
    assert all(len(opts["allowedTokens"]) == machine.vocabulary_size for opts in worker.seen)
    masked = [bytes(opts["allowedBlocks"]) for opts in worker.seen if opts["allowedBlocks"]]
    assert masked, "the kernel installs a mask in this case"
    assert all(b"\x00" not in flags for flags in masked)


@pytest.mark.skipif(
    importlib.util.find_spec("llama_cpp") is None,
    reason="zeos_coop_count imports llama_cpp at module level",
)
def test_the_page_loop_is_the_cli_loop(tmp_path: Path) -> None:
    """``LiveRun`` unrolls the ``zeos-count`` loop; without presses the two agree exactly."""
    from zeos_coop_count.cli import main

    journal = tmp_path / "cli.jsonl"
    argv = ["run", str(CASE), "--machine", "scripted", "--journal", str(journal), "--quiet"]
    assert main([*argv, "--events", str(CASE / "events.jsonl")]) == 0
    assert journal.read_bytes() == run_to_end(seat()).journal_bytes()
