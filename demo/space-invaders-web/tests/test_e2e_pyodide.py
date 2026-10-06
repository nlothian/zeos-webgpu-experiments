# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""End to end under Pyodide in Node: the built wheels, ``page.open_run`` and the real
machine and runner, against ``web/stub/pilot_stub_worker.js`` on a thread of its own,
reached through coop-count-web's channel as ``si_worker.js`` reaches it in the page.

The run sleeps on a control SharedArrayBuffer and Stop is written to it from another
thread, as the page writes it: the loop must keep the tick rate and end within one tick
of the Stop.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HERE = Path(__file__).resolve().parents[1]
COOP = HERE.parents[1] / "demo" / "coop-count-web"

TICK_S = 0.1
STOP_AFTER_MS = 6000
#: A synchronous call behind a step in flight (the kernel's inject of a resume notice, a
#: pager splice) waits for the step to stop at its next chunk boundary, so one chunk's
#: fill time is the longest the loop can stall: at ``DEFAULT_MAX_CHUNK`` (256) and 1 ms
#: a position that is 256 ms, more than two of these ticks. 0.25 ms a position keeps a
#: chunk inside a tick, which is what this test is about; the real model's chunk time
#: is what ``si_webgpu.mjs`` measures.
STUB_LATENCY = {"stepMs": 20, "positionMs": 0.25}

SCRIPT = f"""
import dataclasses, json, time
from js import Atomics, Date, zeosControl, zeosPilotStub, zeosStopAfter, zeosStoppedAt
from zeos_space_invaders_web import page

ARM = "__ARM__"
real = page.load_board
def load(name, seed=None):
    return dataclasses.replace(real(name, seed), tick_seconds={TICK_S})
page.load_board = load

class Stop:
    def is_set(self):
        return Atomics.load(zeosControl, 0) != 0

class Sleeper:
    def now(self):
        return time.monotonic()
    def sleep(self, seconds):
        if seconds > 0:
            Atomics.wait(zeosControl, 0, 0, seconds * 1000)

frames = []
run = page.open_run(ARM, "default", 7, zeosPilotStub, stub=True, on_frame=frames.append,
                    stop=Stop(), clock=Sleeper())
run.warm()
began = time.monotonic()
zeosStopAfter({STOP_AFTER_MS})
result = run.run()
ended = Date.now()
elapsed = time.monotonic() - began
run.close()
print(json.dumps({{
    "kind": type(run).__name__,
    "frames": len(frames),
    "ticks": result.ticks,
    "elapsed": elapsed,
    "stop_to_end_ms": ended - zeosStoppedAt(),
    "decisions": result.decisions,
    "pilot": sum(1 for f in frames for d in f["decisions"] if d["by"] in ("pilot", "prompt")),
    "cancellations": result.cancellations,
    "in_flight": bool(zeosPilotStub.inFlight),
    "extras": result.extras,
}}))
"""


def _pyodide_dir() -> Path | None:
    for base in (HERE, COOP):
        candidate = base / "node_modules" / "pyodide"
        if (candidate / "pyodide.mjs").is_file() and any(candidate.glob("pyyaml-*.whl")):
            return candidate
    return None


def _import_build() -> ModuleType:
    sys.path.insert(0, str(HERE))
    try:
        import build
    finally:
        sys.path.remove(str(HERE))
    return build


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, dict[str, Any]]]:
    if shutil.which("uv") is None:
        pytest.skip("build.py builds the wheels with uv")
    model = tmp_path_factory.mktemp("models") / "Tiny-ZEOS-OPT"
    model.mkdir()
    (model / "meta.json").write_text("{}", encoding="utf-8")
    dist = tmp_path_factory.mktemp("si") / "dist"
    yield dist, _import_build().build(dist, model=model)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
@pytest.mark.skipif(_pyodide_dir() is None, reason="needs npm install in coop-count-web")
@pytest.mark.parametrize("arm", ["zeos", "prompt"])
def test_the_stub_over_the_channel_keeps_time_and_stops_within_a_tick(
    built: tuple[Path, dict[str, Any]], tmp_path: Path, arm: str
) -> None:
    dist, manifest = built
    script = tmp_path / "run.py"
    script.write_text(SCRIPT.replace("__ARM__", arm), encoding="utf-8")
    command = ["node", str(HERE / "tests" / "pyodide_run.mjs")]
    for name in manifest["wheels"]:
        command += ["--wheel", str(dist / "wheels" / name)]
    command += ["--copy", f"{dist / 'cases' / 'space-invaders'}:/cases/space-invaders"]
    command += ["--pilot-stub", json.dumps(STUB_LATENCY), str(script)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["kind"] == ("ZeosRun" if arm == "zeos" else "PromptRun")
    expected = out["elapsed"] / TICK_S
    assert out["frames"] >= 0.9 * expected, out
    assert out["stop_to_end_ms"] <= TICK_S * 1000, out
    assert out["pilot"] >= 1, out
    assert not out["in_flight"], "close left a step in flight"
