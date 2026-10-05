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

When ``npm install`` has been run it also copies ONNX Runtime Web and the tokenizer into
``vendor/``, and when the model has been exported it links the export into ``models/``
and names it in the manifest, which is what offers the model machine on the page. The
export is linked rather than copied because it is hundreds of megabytes; a host that
does not follow links needs ``--copy-model``.
"""

from __future__ import annotations

import argparse
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
PAGE = (
    "index.html",
    "app.js",
    "pyodide_worker.js",
    "stub_worker.js",
    "style.css",
    "coi_serviceworker.js",
    "transformers_worker.js",
    "model_channel.js",
    "model_host.js",
    "model_thread.js",
    "frames.js",
)
NODE_MODULES = HERE / "node_modules"
#: What the model thread imports, from the npm packages package.json pins.
VENDOR = {
    "onnxruntime-web": (
        NODE_MODULES / "onnxruntime-web" / "dist",
        (
            "ort.wasm.min.mjs",
            "ort.webgpu.min.mjs",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm",
            "ort-wasm-simd-threaded.asyncify.mjs",
            "ort-wasm-simd-threaded.asyncify.wasm",
        ),
    ),
    "tokenizers": (NODE_MODULES / "@huggingface" / "tokenizers" / "dist", ("tokenizers.min.mjs",)),
}
MODEL = HERE / "models" / "Qwen3.5-4B-zeos-q4"


def build(dist: Path = DIST, *, model: Path = MODEL, copy_model: bool = False) -> dict[str, object]:
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
    if all(source.is_dir() for source, _ in VENDOR.values()):
        for name, (source, files) in VENDOR.items():
            (dist / "vendor" / name).mkdir(parents=True)
            for file in files:
                shutil.copy2(source / file, dist / "vendor" / name / file)
        if (model / "meta.json").is_file():
            (dist / "models").mkdir()
            target = dist / "models" / model.name
            if copy_model:
                shutil.copytree(model, target)
            else:
                target.symlink_to(model.resolve(), target_is_directory=True)
            manifest["model"] = model.name
    (dist / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL, help="the export to offer")
    parser.add_argument("--copy-model", action="store_true", help="copy the export, not link")
    args = parser.parse_args(argv)
    manifest = build(model=args.model, copy_model=args.copy_model)
    print(f"built {DIST}: {len(manifest['wheels'])} wheels, {len(manifest['cases'])} cases")  # pyright: ignore[reportArgumentType]
    print(f"model: {manifest.get('model', 'none (run export/export_model.py and npm install)')}")
    print(f"serve it with: uv run python {os.path.relpath(HERE / 'serve.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
