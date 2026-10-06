# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What build.py assembles is what Cloudflare Pages will accept and serve isolated."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest

SITE = Path(__file__).resolve().parents[1]


def load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("site_build", SITE / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_build()


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("uv") is None:
        pytest.skip("the demos' build.py builds the wheels with uv")
    out = tmp_path_factory.mktemp("site") / "dist"
    build.build(out)
    return out


def test_both_demos_are_built_under_their_paths(site: Path) -> None:
    assert sorted(p.name for p in site.iterdir()) == [
        "_headers",
        "coop-count",
        "index.html",
        "space-invaders",
    ]
    for path in build.DEMOS:
        assert (site / path / "index.html").is_file()
        manifest = json.loads((site / path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["wheels"]


def test_the_landing_page_links_to_both_demos(site: Path) -> None:
    index = (site / "index.html").read_text(encoding="utf-8")
    for path in build.DEMOS:
        assert f'href="{path}/"' in index


def test_every_path_is_served_cross_origin_isolated(site: Path) -> None:
    assert (site / "_headers").read_text(encoding="utf-8") == (
        "/*\n"
        "  Cross-Origin-Opener-Policy: same-origin\n"
        "  Cross-Origin-Embedder-Policy: require-corp\n"
    )


def test_nothing_pages_would_refuse(site: Path) -> None:
    files = 0
    for root, dirs, names in os.walk(site):
        for name in dirs + names:
            assert not (Path(root) / name).is_symlink(), f"{name} is a symbolic link"
        for name in names:
            files += 1
            assert (Path(root) / name).stat().st_size <= 25 * 1024 * 1024, name
    assert files <= 20_000


def test_a_symbolic_link_fails_the_check(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="symbolic link"):
        build.check(tmp_path)


def test_a_file_over_the_limit_fails_the_check(tmp_path: Path) -> None:
    with (tmp_path / "large.wasm").open("wb") as handle:
        handle.truncate(build.MAX_FILE_BYTES)
    build.check(tmp_path)
    with (tmp_path / "large.wasm").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="large.wasm"):
        build.check(tmp_path)
