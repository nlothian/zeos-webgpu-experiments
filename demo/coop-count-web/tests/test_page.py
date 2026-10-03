# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The page's Python half, and what build.py puts beside it."""

from __future__ import annotations

import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from zeos_coop_count_web import page

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parents[1]
CASES = REPO / "demo" / "coop-count" / "cases"


def test_both_pyodides_are_the_same_release() -> None:
    """The page loads Pyodide from the CDN and the Node test from npm; a determinism claim
    about one says nothing about the other unless they are the same release."""
    worker = (WEB / "web" / "pyodide_worker.js").read_text(encoding="utf-8")
    (cdn,) = re.findall(r"cdn\.jsdelivr\.net/pyodide/v([0-9.]+)/full/pyodide\.mjs", worker)
    npm = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
    assert npm["devDependencies"]["pyodide"] == cdn
    lock = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/pyodide"]["version"] == cdn


def test_only_the_tape_case_runs_without_a_model() -> None:
    described = {c.name: json.loads(page.describe(str(c))) for c in sorted(CASES.iterdir())}
    assert {name: d["runnable"] for name, d in described.items()} == {
        "coop-count-pipe": False,
        "coop-count-scripted": True,
        "coop-count-vector": False,
    }
    for info in described.values():
        assert info["errors"] == 0
        assert "frames" not in info["payload"], "a case not yet run is drawn as wiring only"
    assert described["coop-count-scripted"]["devices"] == ["keys.interrupt", "keys.number"]


def test_a_run_needs_a_worker_for_the_js_machine() -> None:
    with pytest.raises(ValueError, match="ZeosModelWorker"):
        page.open_run(str(CASES / "coop-count-scripted"), "js")


def test_a_finished_run_draws_its_frames() -> None:
    run = page.open_run(str(CASES / "coop-count-scripted"), "scripted")
    while not run.finished:
        run.step()
    payload = json.loads(page.payload_json(run))
    assert payload["structure"]["case"] == "coop-count-scripted"
    assert payload["frames"]["count"] == len(run.journal)
    assert payload["frames"]["trace"], "the machine's own account rides along for the model view"


@pytest.mark.skipif(shutil.which("uv") is None, reason="build.py builds the wheels with uv")
def test_build_assembles_everything_the_page_fetches(tmp_path: Path) -> None:
    sys.path.insert(0, str(WEB))
    try:
        import build
    finally:
        sys.path.remove(str(WEB))
    dist = tmp_path / "dist"
    manifest = build.build(dist)

    names = manifest["wheels"]
    assert isinstance(names, list) and len(names) == 2
    for name in names:
        assert (dist / "wheels" / name).is_file()
    (web_wheel,) = (dist / "wheels" / n for n in names if n.startswith("zeos_coop_count_web-"))
    archive = zipfile.ZipFile(web_wheel)
    assert all(n.startswith("zeos_coop_count_web") for n in archive.namelist())
    (metadata,) = (n for n in archive.namelist() if n.endswith("METADATA"))
    requires = [
        line
        for line in archive.read(metadata).decode().splitlines()
        if line.startswith("Requires-Dist")
    ]
    assert requires == ["Requires-Dist: zeos"], "nothing that cannot install under Pyodide"

    cases = manifest["cases"]
    assert isinstance(cases, dict) and sorted(cases) == sorted(c.name for c in CASES.iterdir())
    for name, files in cases.items():
        for file in files:
            assert (dist / "cases" / name / file).read_bytes() == (CASES / name / file).read_bytes()
    static = REPO / "src" / "zeos" / "debugger" / "static"
    for asset in static.iterdir():
        assert (dist / "debugger" / asset.name).read_bytes() == asset.read_bytes()

    shell = (dist / "index.html").read_text(encoding="utf-8")
    for script in re.findall(r'src="([^"]+)"', shell) + re.findall(r'href="([^":]+)"', shell):
        assert (dist / script).is_file(), f"index.html refers to {script}, which was not built"
    worker = (dist / "pyodide_worker.js").read_text(encoding="utf-8")
    for local in re.findall(r'import "\./([^"]+)"', worker):
        assert (dist / local).is_file()
