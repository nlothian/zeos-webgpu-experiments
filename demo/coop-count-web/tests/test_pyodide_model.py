# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The page's arrangement under Node: the kernel in Pyodide, the model on a thread of its
own, called synchronously through ``SyncModelWorker``; and the journal it writes is the
one CPython writes over the same worker in a Node child process. Both run onnxruntime-web's
WebAssembly binary on one thread. Skipped without Node, the npm packages or the export."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from zeos_coop_count_web.node_run import main
from zeos_coop_count_web.node_worker import DEFAULT_MODEL, node_available

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parents[1]
CASE = REPO / "demo" / "coop-count" / "cases" / "coop-count-scripted"
PYODIDE = WEB / "node_modules" / "pyodide"
NODE, UV = shutil.which("node"), shutil.which("uv")
TICKS = 40

SCRIPT = f"""
import os
import js
from zeos_coop_count_web import page

run = page.open_run("/cases/coop-count-scripted", "js", worker=js.zeosModelWorker,
                    max_ticks={TICKS})
while not run.finished:
    run.step()
os.makedirs("/out", exist_ok=True)
with open("/out/model.jsonl", "wb") as handle:
    handle.write(run.journal_bytes())
"""

pytestmark = [
    pytest.mark.determinism,
    pytest.mark.skipif(
        NODE is None
        or UV is None
        or not node_available()
        or not (PYODIDE / "pyodide.mjs").is_file()
        or not (DEFAULT_MODEL / "meta.json").is_file(),
        reason="needs node, uv, `npm install` in demo/coop-count-web and the export",
    ),
]


def test_pyodide_over_the_model_thread_writes_cpythons_bytes(tmp_path: Path) -> None:
    assert NODE is not None and UV is not None
    wheels = tmp_path / "wheels"
    for package in ("zeos", "zeos-coop-count-web"):
        subprocess.run(
            [UV, "build", "--wheel", "--package", package, "--out-dir", str(wheels)],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    script = tmp_path / "run.py"
    script.write_text(SCRIPT, encoding="utf-8")
    pyodide = tmp_path / "pyodide.jsonl"
    argv = [NODE, str(WEB / "tests" / "pyodide_run.mjs")]
    for wheel in sorted(wheels.glob("*.whl")):
        argv += ["--wheel", str(wheel)]
    argv += ["--copy", f"{CASE}:/cases/coop-count-scripted", "--model", str(DEFAULT_MODEL)]
    argv += ["--fetch", f"/out/model.jsonl:{pyodide}", str(script)]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

    cpython = tmp_path / "cpython.jsonl"
    args = [str(CASE), "--events", str(CASE / "events.jsonl"), "--journal", str(cpython)]
    assert main([*args, "--max-ticks", str(TICKS), "--quiet"]) == 0
    assert pyodide.read_bytes() == cpython.read_bytes()
