# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assemble ``web/dist/``: everything the page needs, ready for any static file host.

    uv run python demo/coop-count-web/build.py
    python -m http.server 8765 -d demo/coop-count-web/web/dist

It builds the ``zeos`` and ``zeos-browser`` wheels with ``uv build``, copies in the
coop-count cases, the debugger's static assets from ``src/zeos/debugger/static/``, the
page itself and the generic model-thread JavaScript from ``packages/zeos-browser/web``
(copied at build time, so there is one source for it), and writes ``manifest.json`` naming the wheels and every case file,
which is how the page finds them: a static host lists no directories. The directory is
emptied first, so what is served is exactly what this run built.

When ``npm install`` has been run in ``packages/zeos-browser`` it also copies ONNX Runtime
Web and the tokenizer into ``vendor/`` and names the model in the manifest, which is what
offers the model machine on the page. The page downloads the model from the Hugging Face
Hub at a pinned commit, once, and keeps it in the browser's storage
(``zeos_browser.model_source``). ``--model [DIR]`` also links a local export (by default
``packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT``) into ``models/`` and loads that by
default; it is linked because it is gigabytes, and a host that does not follow links
needs ``--copy-model``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from zeos_browser import model_source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = HERE / "web"
BROWSER = REPO / "packages" / "zeos-browser"
BROWSER_WEB = BROWSER / "web"
DIST = WEB / "dist"
CASES = REPO / "demo" / "coop-count" / "cases"
DEBUGGER = REPO / "src" / "zeos" / "debugger" / "static"
PACKAGES = ("zeos", "zeos-browser")
#: The page's own files, from ``web/``.
PAGE = ("index.html", "app.js", "pyodide_worker.js", "style.css")
#: The stub, the model thread and its channel, from ``packages/zeos-browser/web``.
GENERIC = (
    "stub_worker.js",
    "coi_serviceworker.js",
    "transformers_worker.js",
    "opt_zeos_worker.js",
    "model_channel.js",
    "model_host.js",
    "model_thread.js",
    "model_cache.js",
    "opfs_store.js",
    "sha256.js",
    "frames.js",
)
NODE_MODULES = BROWSER / "node_modules"
#: What the model thread imports, from the npm packages zeos-browser's package.json pins.
VENDOR = {
    "onnxruntime-web": (
        NODE_MODULES / "onnxruntime-web" / "dist",
        (
            # The WebGPU build and the WebAssembly module it loads; nothing else runs.
            "ort.webgpu.min.mjs",
            "ort-wasm-simd-threaded.asyncify.mjs",
            "ort-wasm-simd-threaded.asyncify.wasm",
        ),
    ),
    "tokenizers": (NODE_MODULES / "@huggingface" / "tokenizers" / "dist", ("tokenizers.min.mjs",)),
}


def build(
    dist: Path = DIST,
    *,
    model: Path | None = None,
    copy_model: bool = False,
    hf_endpoint: str = model_source.HF_ENDPOINT,
    hf_repo: str = model_source.HF_REPO,
    hf_revision: str = model_source.HF_REVISION,
) -> dict[str, object]:
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
    for name in GENERIC:
        shutil.copy2(BROWSER_WEB / name, dist / name)

    manifest: dict[str, object] = {"wheels": names, "cases": cases}
    if all(source.is_dir() for source, _ in VENDOR.values()):
        for name, (source, files) in VENDOR.items():
            (dist / "vendor" / name).mkdir(parents=True)
            for file in files:
                shutil.copy2(source / file, dist / "vendor" / name / file)
        manifest |= model_source.model_fields(
            dist,
            local=model,
            copy=copy_model,
            endpoint=hf_endpoint,
            repo=hf_repo,
            revision=hf_revision,
        )
    (dist / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    model_source.add_arguments(parser)
    args = parser.parse_args(argv)
    manifest = build(
        model=args.model,
        copy_model=args.copy_model,
        hf_endpoint=args.hf_endpoint,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
    )
    print(f"built {DIST}: {len(manifest['wheels'])} wheels, {len(manifest['cases'])} cases")  # pyright: ignore[reportArgumentType]
    sources = manifest.get("model_sources")
    if isinstance(sources, dict):
        print(f"model: {manifest['model']} from {manifest['model_source']}; offered: {sources}")
    else:
        print("model: none (run npm install in packages/zeos-browser)")
    print(f"serve it with: uv run python {os.path.relpath(HERE / 'serve.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
