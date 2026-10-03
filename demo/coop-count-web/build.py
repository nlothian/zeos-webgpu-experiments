# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assemble ``web/dist/``: everything the page needs, ready for any static file host.

    uv run python demo/coop-count-web/build.py
    python -m http.server 8765 -d demo/coop-count-web/web/dist

It builds the ``zeos`` and ``zeos-coop-count-web`` wheels with ``uv build``, copies in
the coop-count cases, the debugger's static assets from ``src/zeos/debugger/static/``
and the page itself, and writes ``manifest.json`` naming the wheels and every case file,
which is how the page finds them: a static host lists no directories. The directory is
emptied first, so what is served is exactly what this run built.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = HERE / "web"
DIST = WEB / "dist"
CASES = REPO / "demo" / "coop-count" / "cases"
DEBUGGER = REPO / "src" / "zeos" / "debugger" / "static"
PACKAGES = ("zeos", "zeos-coop-count-web")
PAGE = ("index.html", "app.js", "pyodide_worker.js", "stub_worker.js", "style.css")


def build(dist: Path = DIST) -> dict[str, object]:
    if dist.exists():
        shutil.rmtree(dist)
    wheels = dist / "wheels"
    for package in PACKAGES:
        subprocess.run(
            ["uv", "build", "--wheel", "--package", package, "--out-dir", str(wheels)],
            cwd=REPO,
            check=True,
        )
    names = sorted(p.name for p in wheels.glob("*.whl"))

    cases: dict[str, list[str]] = {}
    for case in sorted(p for p in CASES.iterdir() if p.is_dir()):
        shutil.copytree(case, dist / "cases" / case.name)
        cases[case.name] = sorted(
            p.relative_to(case).as_posix() for p in case.rglob("*") if p.is_file()
        )
    shutil.copytree(DEBUGGER, dist / "debugger")
    for name in PAGE:
        shutil.copy2(WEB / name, dist / name)

    manifest: dict[str, object] = {"wheels": names, "cases": cases}
    (dist / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    manifest = build()
    print(f"built {DIST}: {len(manifest['wheels'])} wheels, {len(manifest['cases'])} cases")  # pyright: ignore[reportArgumentType]
    print(f"serve it with: python -m http.server 8765 -d {os.path.relpath(DIST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
