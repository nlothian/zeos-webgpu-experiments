# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Which model a built page loads, written into its ``manifest.json`` by each ``build.py``.

By default a page loads the OPT+ZEOS export from the Hugging Face Hub, at one pinned
commit, and keeps it in the browser's storage (``web/model_cache.js``). ``--model DIR``
links (or with ``--copy-model`` copies) a local export into the page as well and makes it
the default; ``?model=huggingface`` or ``?model=local`` in the page's URL picks either
(``web/model_host.js``'s ``modelSource``).

The manifest's fields::

    "model": "<the default source's name>",
    "model_source": "huggingface" | "local",
    "model_sources": {
        "huggingface": {"name": ..., "endpoint": "https://huggingface.co", "repo": ...,
                        "revision": "<40-hex commit>"},
        "local": {"name": ..., "path": "models/<name>/"},  # only with --model
    }
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

#: Where the Hub is. Another origin serving the same ``/<repo>/resolve/<revision>/<path>``
#: layout with CORS (a mirror) can stand in for it.
HF_ENDPOINT = "https://huggingface.co"
#: The export ``export/opt_zeos_surgery.py`` writes, published on the Hub.
HF_REPO = "nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16"
#: The commit of ``HF_REPO`` pages load. A commit, never a branch: the cache keys files by
#: it, and a branch's files can change under the same name.
HF_REVISION = "f71e07f80aa6d20f66c69575facae9813a053c5e"
#: Where ``export/opt_zeos_surgery.py`` writes the export, for ``--model`` with no value.
LOCAL_EXPORT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-4B-ZEOS-OPT"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENDPOINT = re.compile(r"https?://[^/?#]+")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The model options both demos' ``build.py`` take."""
    parser.add_argument(
        "--model",
        type=Path,
        nargs="?",
        const=LOCAL_EXPORT,
        default=None,
        metavar="DIR",
        help=f"also serve a local export and load it by default (DIR defaults to {LOCAL_EXPORT})",
    )
    parser.add_argument("--copy-model", action="store_true", help="copy the local export, not link")
    parser.add_argument(
        "--hf-endpoint", default=HF_ENDPOINT, help="the Hub, or a mirror with its URL layout"
    )
    parser.add_argument("--hf-repo", default=HF_REPO, help="the Hugging Face repo to load from")
    parser.add_argument(
        "--hf-revision", default=HF_REVISION, help="the commit of --hf-repo (40 hex digits)"
    )


def model_fields(
    dist: Path,
    *,
    local: Path | None = None,
    copy: bool = False,
    endpoint: str = HF_ENDPOINT,
    repo: str = HF_REPO,
    revision: str = HF_REVISION,
) -> dict[str, Any]:
    """The manifest's model fields; with ``local``, link or copy it to ``dist/models/``."""
    if not _COMMIT.fullmatch(revision):
        raise ValueError(f"--hf-revision must be a 40-hex commit, not {revision!r}")
    if not _ENDPOINT.fullmatch(endpoint):
        raise ValueError(f"--hf-endpoint must be an origin like {HF_ENDPOINT}, not {endpoint!r}")
    sources: dict[str, dict[str, str]] = {
        "huggingface": {
            "name": repo.rsplit("/", 1)[-1],
            "endpoint": endpoint,
            "repo": repo,
            "revision": revision,
        }
    }
    default = "huggingface"
    if local is not None:
        if not (local / "meta.json").is_file():
            raise ValueError(f"{local} is not an export: it has no meta.json")
        (dist / "models").mkdir(exist_ok=True)
        target = dist / "models" / local.name
        if copy:
            shutil.copytree(local, target)
        else:
            target.symlink_to(local.resolve(), target_is_directory=True)
        sources["local"] = {"name": local.name, "path": f"models/{local.name}/"}
        default = "local"
    return {"model": sources[default]["name"], "model_source": default, "model_sources": sources}
