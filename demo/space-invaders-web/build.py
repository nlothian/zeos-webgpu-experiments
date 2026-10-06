# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assemble ``web/dist/``: everything the Space Invaders page needs, for any static host.

    uv run python demo/space-invaders-web/build.py
    uv run python demo/space-invaders-web/serve.py      # http://localhost:8766

It builds the ``zeos``, ``zeos-space-invaders``, ``zeos-browser`` and
``zeos-space-invaders-web`` wheels with ``uv build``; copies the page from ``web/``,
the generic model-thread JavaScript from ``packages/zeos-browser/web`` (copied at build
time, so there is one source for it), the Space Invaders case, and the debugger's
static assets; and writes ``manifest.json`` naming the wheels and every case file,
which is how the page finds them: a static host lists no directories. ``dist/`` is
emptied first, so what is served is exactly what this run built.

Under Pyodide the wheels' one third-party dependency, PyYAML (zeos's case loader),
comes from Pyodide's own package index: micropip resolves it when it installs the
``zeos`` wheel.

The tokenizer comes from ``npm install`` in ``packages/zeos-browser`` and goes into
``vendor/``; ONNX Runtime Web is loaded from jsDelivr at the version that install has,
named as ``ort_url``. With the install present the model is named in the manifest, which is
what offers the model machine on the page. The page downloads the model from the Hugging Face Hub
at a pinned commit, once, and keeps it in the browser's storage
(``zeos_browser.model_source``). ``--model [DIR]`` also links a local export (by default
``packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT``) into ``models/`` and loads that by
default; a host that does not follow symbolic links -- most static hosts -- needs
``--copy-model`` with it.
``--link-node-modules`` links ``node_modules`` here to zeos-browser's, which is
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

from zeos_browser import model_source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = HERE / "web"
DIST = WEB / "dist"
BROWSER = REPO / "packages" / "zeos-browser"
BROWSER_WEB = BROWSER / "web"
CASE = REPO / "demo" / "space-invaders" / "src" / "zeos_space_invaders" / "cases" / "space-invaders"
DEBUGGER = REPO / "src" / "zeos" / "debugger" / "static"
PACKAGES = ("zeos", "zeos-space-invaders", "zeos-browser", "zeos-space-invaders-web")
#: The page's own files, from ``web/``.
PAGE = ("index.html", "app.js", "board.js", "si_worker.js", "stub_thread.js", "style.css")
#: The model thread and its channel, unchanged from zeos-browser.
GENERIC = (
    "model_host.js",
    "model_thread.js",
    "model_cache.js",
    "opfs_store.js",
    "sha256.js",
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
NODE_MODULES = BROWSER / "node_modules"
#: What the model thread imports from the page, from the npm packages zeos-browser's
#: package.json pins. ONNX Runtime Web is loaded from jsDelivr (``model_source.ort_url``).
VENDOR = {
    "tokenizers": (NODE_MODULES / "@huggingface" / "tokenizers" / "dist", ("tokenizers.min.mjs",)),
}


def link_node_modules(target: Path = HERE / "node_modules") -> Path:
    """Point ``node_modules`` here at zeos-browser's npm install, once."""
    if target.is_symlink() or target.exists():
        return target
    target.symlink_to(NODE_MODULES.resolve(), target_is_directory=True)
    return target


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

    shutil.copytree(CASE, dist / "cases" / CASE.name)
    cases = {
        CASE.name: sorted(p.relative_to(CASE).as_posix() for p in CASE.rglob("*") if p.is_file())
    }
    shutil.copytree(DEBUGGER, dist / "debugger")
    for name in PAGE:
        shutil.copy2(WEB / name, dist / name)
    for name in GENERIC:
        shutil.copy2(BROWSER_WEB / name, dist / name)
    stub = STUB_WORKER.is_file()
    if stub:
        shutil.copytree(STUB, dist / "stub")

    manifest: dict[str, object] = {"wheels": names, "cases": cases, "model": None, "stub": stub}
    if all(source.is_dir() for source, _ in VENDOR.values()):
        for name, (source, files) in VENDOR.items():
            (dist / "vendor" / name).mkdir(parents=True)
            for file in files:
                shutil.copy2(source / file, dist / "vendor" / name / file)
        manifest["ort_url"] = model_source.ort_url(NODE_MODULES)
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
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    model_source.add_arguments(parser)
    parser.add_argument(
        "--link-node-modules",
        action="store_true",
        help="link node_modules here to zeos-browser's (for tests/pyodide_run.mjs)",
    )
    args = parser.parse_args(argv)
    if args.link_node_modules:
        print(f"node_modules: {link_node_modules()}")
    manifest = build(
        model=args.model,
        copy_model=args.copy_model,
        hf_endpoint=args.hf_endpoint,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
    )
    wheels = manifest["wheels"]
    assert isinstance(wheels, list)
    print(f"built {DIST}: {len(wheels)} wheels")  # pyright: ignore[reportUnknownArgumentType]
    if manifest["model"] is None:
        print("model: none (run npm install in packages/zeos-browser)")
    else:
        print(f"model: {manifest['model']} from {manifest['model_source']}")
    print(f"serve it with: uv run python {os.path.relpath(HERE / 'serve.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
