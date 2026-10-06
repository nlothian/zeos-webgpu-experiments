# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assemble ``web/dist/``: everything the Space Invaders page needs, for any static host.

    uv run python demo/space-invaders-web/build.py
    uv run python demo/space-invaders-web/serve.py      # http://localhost:8766

It builds the ``zeos``, ``zeos-space-invaders``, ``zeos-coop-count-web`` and
``zeos-space-invaders-web`` wheels with ``uv build``; copies the page from ``web/``,
the generic model-thread JavaScript from ``../coop-count-web/web`` (copied at build
time, so there is one source for it), the Space Invaders case, and the debugger's
static assets; and writes ``manifest.json`` naming the wheels and every case file,
which is how the page finds them: a static host lists no directories. ``dist/`` is
emptied first, so what is served is exactly what this run built.

Under Pyodide the wheels' one third-party dependency, PyYAML (zeos's case loader),
comes from Pyodide's own package index: micropip resolves it when it installs the
``zeos`` wheel.

ONNX Runtime Web and the tokenizer come from coop-count-web's ``npm install`` and go
into ``vendor/``; with them present and the model exported, the export is linked into
``models/`` and named in the manifest, which is what offers the model machine on the page. A host
that does not follow symbolic links -- most static hosts -- needs ``--copy-model``.
``--link-node-modules`` links ``node_modules`` here to coop-count-web's, which is
where ``tests/pyodide_run.mjs`` looks for Pyodide.
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
COOP = REPO / "demo" / "coop-count-web"
COOP_WEB = COOP / "web"
CASE = REPO / "demo" / "space-invaders" / "src" / "zeos_space_invaders" / "cases" / "space-invaders"
DEBUGGER = REPO / "src" / "zeos" / "debugger" / "static"
PACKAGES = ("zeos", "zeos-space-invaders", "zeos-coop-count-web", "zeos-space-invaders-web")
#: The page's own files, from ``web/``.
PAGE = ("index.html", "app.js", "board.js", "si_worker.js", "stub_thread.js", "style.css")
#: The model thread and its channel, unchanged from coop-count-web.
GENERIC = (
    "model_host.js",
    "model_thread.js",
    "model_channel.js",
    "frames.js",
    "opt_zeos_worker.js",
    "transformers_worker.js",
    "stub_worker.js",
    "coi_serviceworker.js",
)
#: The Space Invaders stub, served over the channel from its own thread (stub_thread.js)
#: when ``web/stub/pilot_stub_worker.js`` exists; ``manifest.json`` says whether it does.
STUB = WEB / "stub"
STUB_WORKER = STUB / "pilot_stub_worker.js"
NODE_MODULES = COOP / "node_modules"
#: What the model thread imports, from the npm packages coop-count-web's package.json pins.
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
MODEL = COOP / "models" / "Qwen3.5-4B-ZEOS-OPT"


def link_node_modules(target: Path = HERE / "node_modules") -> Path:
    """Point ``node_modules`` here at coop-count-web's npm install, once."""
    if target.is_symlink() or target.exists():
        return target
    target.symlink_to(NODE_MODULES.resolve(), target_is_directory=True)
    return target


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

    shutil.copytree(CASE, dist / "cases" / CASE.name)
    cases = {
        CASE.name: sorted(p.relative_to(CASE).as_posix() for p in CASE.rglob("*") if p.is_file())
    }
    shutil.copytree(DEBUGGER, dist / "debugger")
    for name in PAGE:
        shutil.copy2(WEB / name, dist / name)
    for name in GENERIC:
        shutil.copy2(COOP_WEB / name, dist / name)
    stub = STUB_WORKER.is_file()
    if stub:
        shutil.copytree(STUB, dist / "stub")

    manifest: dict[str, object] = {"wheels": names, "cases": cases, "model": None, "stub": stub}
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
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL, help="the export to offer")
    parser.add_argument("--copy-model", action="store_true", help="copy the export, not link")
    parser.add_argument(
        "--link-node-modules",
        action="store_true",
        help="link node_modules here to coop-count-web's (for tests/pyodide_run.mjs)",
    )
    args = parser.parse_args(argv)
    if args.link_node_modules:
        print(f"node_modules: {link_node_modules()}")
    manifest = build(model=args.model, copy_model=args.copy_model)
    wheels = manifest["wheels"]
    assert isinstance(wheels, list)
    print(f"built {DIST}: {len(wheels)} wheels")  # pyright: ignore[reportUnknownArgumentType]
    model = manifest["model"] or "none (export the model and run npm install in coop-count-web)"
    print(f"model: {model}")
    print(f"serve it with: uv run python {os.path.relpath(HERE / 'serve.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
