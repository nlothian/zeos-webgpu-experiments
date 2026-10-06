# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assemble ``dist/``: both browser demos on one static site, ready for Cloudflare Pages.

    uv run python demo/site/build.py [--out DIR]

Each demo's own ``build()`` writes its page into a subdirectory (``coop-count/``,
``space-invaders/``); the page's URLs are all relative, so it runs under any path.
Beside them go ``index.html``, which links to both, and ``_headers``, which has Pages send
every file cross-origin isolated, as each demo's ``serve.py`` does. Pages does not follow
symbolic links and refuses a file over 25 MiB, so either fails the build rather than the
deploy; that is why the demos' ``--model`` (a link to a local export) is not offered here.
The model comes from the Hugging Face Hub and ONNX Runtime Web from jsDelivr.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIST = HERE / "dist"
#: Subdirectory of the site -> the demo whose ``build.py`` fills it.
DEMOS = {
    "coop-count": REPO / "demo" / "coop-count-web",
    "space-invaders": REPO / "demo" / "space-invaders-web",
}
#: Cloudflare Pages' limits on what one deployment may hold.
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 20_000
HEADERS = """\
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
"""


def build_demo(demo: Path, dist: Path) -> dict[str, object]:
    """Run ``demo/<name>/build.py``'s ``build(dist)``.

    Every demo's script is called ``build``, so each is loaded under a name of its own.
    """
    name = f"{demo.name.replace('-', '_')}_build"
    spec = importlib.util.spec_from_file_location(name, demo / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(dist)


def check(site: Path) -> None:
    """Raise if Pages would refuse ``site``, or serve it other than as built."""
    problems: list[str] = []
    count = 0
    for root, dirs, files in os.walk(site):
        for entry in sorted(dirs + files):
            path = Path(root) / entry
            if path.is_symlink():
                problems.append(f"{path} is a symbolic link, which Pages does not follow")
            elif entry in files:
                count += 1
                size = path.stat().st_size
                if size > MAX_FILE_BYTES:
                    problems.append(f"{path} is {size} bytes; Pages' limit is {MAX_FILE_BYTES}")
    if count > MAX_FILES:
        problems.append(f"{site} holds {count} files; Pages' limit is {MAX_FILES}")
    if problems:
        raise ValueError("\n".join(problems))


def build(out: Path = DIST) -> dict[str, dict[str, object]]:
    """Assemble the site in ``out`` (emptied first); return each demo's manifest."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifests = {path: build_demo(demo, out / path) for path, demo in DEMOS.items()}
    shutil.copy2(HERE / "index.html", out / "index.html")
    (out / "_headers").write_text(HEADERS, encoding="utf-8")
    check(out)
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=DIST, help=f"default: {DIST}")
    args = parser.parse_args(argv)
    out: Path = args.out.resolve()
    build(out)
    files = [p for p in out.rglob("*") if p.is_file()]
    largest = max(files, key=lambda p: p.stat().st_size)
    total = sum(p.stat().st_size for p in files)
    print(f"built {out}: {len(files)} files, {total / 2**20:.1f} MiB")
    print(f"largest: {largest.relative_to(out)} ({largest.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
