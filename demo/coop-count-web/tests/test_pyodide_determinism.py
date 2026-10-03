# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The port's determinism claim: a run under Pyodide writes the bytes CPython writes.

The kernel reads no clock, starts no thread and opens no socket, the driver injects time
as one virtual millisecond per tick, and the journal is sorted-key JSON whose ids are
counters, so nothing about the interpreter should reach the journal. This runs the same
cases in both and compares the files byte for byte:

* ``tests/fixtures/smoke`` through ``zeos run``, the kernel repository's own fixture;
* ``coop-count-scripted`` through the page's run loop on the ``CommandSeat``;
* the same case through ``JsMachine`` over ``web/stub_worker.js`` in Pyodide, and over
  ``FakeWorker`` in CPython -- which also holds the two workers to one behaviour.

Pyodide runs in Node, from the ``pyodide`` npm package that ``npm install`` puts in
this demo's ``node_modules`` (or wherever ``ZEOS_PYODIDE_DIR`` points). Its postinstall
step caches PyYAML beside it, so the run needs no network. The test skips when Node, the
package or that cache is missing. Both wheels are built fresh with ``uv build`` and
installed by unpacking, as Pyodide installs any pure wheel; the case directories are
copied into Pyodide's in-memory filesystem, as the page does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from zeos.cli import main as zeos_main
from zeos.descriptor.loader import load_case
from zeos.driver import load_schedule
from zeos.machine.seat import CommandSeat, TapeSource, seat_maps

from zeos_coop_count_web.fake_worker import FakeWorker, tapes_from_scripts
from zeos_coop_count_web.js_machine import JsMachine
from zeos_coop_count_web.live import LiveRun

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parents[1]
SMOKE = REPO / "tests" / "fixtures" / "smoke"
CASE = REPO / "demo" / "coop-count" / "cases" / "coop-count-scripted"
PYODIDE = Path(os.environ.get("ZEOS_PYODIDE_DIR", WEB / "node_modules" / "pyodide"))
NODE = shutil.which("node")
UV = shutil.which("uv")

#: Run inside Pyodide. Paths are the in-memory filesystem's.
SCRIPT = """
import json
import sys

import js
from zeos.cli import main
from zeos_coop_count_web import page

print("Pyodide", sys.version.split()[0], sys.platform)
assert main([
    "run", "/cases/smoke", "--events", "/cases/smoke/events.jsonl",
    "--journal", "/out/smoke.jsonl", "--quiet",
]) == 0

case = "/cases/coop-count-scripted"
stub = js.createStubWorker(js.JSON.parse(page.tapes_json(case)))
for name, machine, worker in (("seat", "scripted", None), ("stub", "js", stub)):
    run = page.open_run(case, machine, worker=worker)
    while not run.finished:
        run.step()
    with open(f"/out/{name}.jsonl", "wb") as handle:
        handle.write(run.journal_bytes())
    print(name, run.reason, run.ticks, "ticks", len(run.journal), "events")
"""


def _skip_reason() -> str | None:
    if NODE is None:
        return "node is not installed"
    if UV is None:
        return "uv is not on PATH, and the test builds the wheels with it"
    if not (PYODIDE / "pyodide.mjs").is_file():
        return f"no pyodide npm package at {PYODIDE}; run 'npm install' in {WEB}"
    if not any(PYODIDE.glob("pyyaml-*.whl")):
        return f"PyYAML is not cached in {PYODIDE}; run 'npm install' in {WEB}"
    return None


pytestmark = [
    pytest.mark.determinism,
    pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason())),
]


@pytest.fixture(scope="module")
def pyodide_journals(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    assert NODE is not None and UV is not None
    wheels = tmp_path_factory.mktemp("wheels")
    for package in ("zeos", "zeos-coop-count-web"):
        subprocess.run(
            [UV, "build", "--wheel", "--package", package, "--out-dir", str(wheels)],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    out = tmp_path_factory.mktemp("pyodide")
    script = out / "run.py"
    script.write_text(SCRIPT, encoding="utf-8")
    names = ("smoke", "seat", "stub")
    argv = [NODE, str(WEB / "tests" / "pyodide_run.mjs")]
    for wheel in sorted(wheels.glob("*.whl")):
        argv += ["--wheel", str(wheel)]
    argv += ["--copy", f"{SMOKE}:/cases/smoke", "--copy", f"{CASE}:/cases/coop-count-scripted"]
    argv += ["--js", str(WEB / "web" / "stub_worker.js")]
    for name in names:
        argv += ["--fetch", f"/out/{name}.jsonl:{out / name}.jsonl"]
    argv.append(str(script))
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    print(done.stdout)
    return {name: (out / f"{name}.jsonl").read_bytes() for name in names}


def _run_to_end(run: LiveRun) -> bytes:
    while not run.finished:
        run.step()
    return run.journal_bytes()


def test_the_smoke_fixture_writes_the_same_bytes(
    pyodide_journals: dict[str, bytes], tmp_path: Path
) -> None:
    journal = tmp_path / "smoke.jsonl"
    argv = ["run", str(SMOKE), "--events", str(SMOKE / "events.jsonl")]
    assert zeos_main([*argv, "--journal", str(journal), "--quiet"]) == 0
    assert pyodide_journals["smoke"] == journal.read_bytes()


def test_the_page_loop_on_the_seat_writes_the_same_bytes(
    pyodide_journals: dict[str, bytes],
) -> None:
    bundle = load_case(CASE)
    run = LiveRun(
        bundle,
        CommandSeat(source=TapeSource(bundle.scripts)),
        schedule=load_schedule(CASE / "events.jsonl"),
        trace=True,
    )
    assert pyodide_journals["seat"] == _run_to_end(run)


def test_the_stub_worker_and_the_fake_write_the_same_bytes(
    pyodide_journals: dict[str, bytes],
) -> None:
    bundle = load_case(CASE)
    descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
    machine = JsMachine(
        FakeWorker(tapes_from_scripts(bundle.scripts)), descriptors=descriptors, valued=valued
    )
    run = LiveRun(bundle, machine, schedule=load_schedule(CASE / "events.jsonl"), trace=True)
    assert pyodide_journals["stub"] == _run_to_end(run)
    assert pyodide_journals["stub"] == pyodide_journals["seat"]
