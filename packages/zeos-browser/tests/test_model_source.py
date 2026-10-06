# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The model fields a page's build writes into its manifest (``model_source``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from zeos_browser import model_source


def test_by_default_the_page_loads_the_pinned_hub_commit_and_links_nothing(tmp_path: Path) -> None:
    fields = model_source.model_fields(tmp_path)
    assert fields == {
        "model": "Qwen3.5-4B-ZEOS-OPT_Q4F16",
        "model_source": "huggingface",
        "model_sources": {
            "huggingface": {
                "name": "Qwen3.5-4B-ZEOS-OPT_Q4F16",
                "endpoint": "https://huggingface.co",
                "repo": "nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16",
                "revision": model_source.HF_REVISION,
            }
        },
    }
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("revision", ["main", "f71e07f", "F" * 40, "g" * 40])
def test_the_revision_must_be_a_full_commit(tmp_path: Path, revision: str) -> None:
    with pytest.raises(ValueError, match="40-hex commit"):
        model_source.model_fields(tmp_path, revision=revision)


@pytest.mark.parametrize("endpoint", ["https://huggingface.co/", "huggingface.co", "ftp://x"])
def test_the_endpoint_must_be_an_origin(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError, match="origin"):
        model_source.model_fields(tmp_path, endpoint=endpoint)


def test_a_mirror_stands_in_for_the_hub(tmp_path: Path) -> None:
    fields = model_source.model_fields(tmp_path, endpoint="http://127.0.0.1:8777")
    assert fields["model_sources"]["huggingface"]["endpoint"] == "http://127.0.0.1:8777"


@pytest.mark.parametrize("copy", [False, True])
def test_a_local_export_is_served_too_and_becomes_the_default(tmp_path: Path, copy: bool) -> None:
    export = tmp_path / "exports" / "Tiny-ZEOS-OPT"
    export.mkdir(parents=True)
    (export / "meta.json").write_text("{}", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    fields = model_source.model_fields(dist, local=export, copy=copy)
    assert fields["model"] == "Tiny-ZEOS-OPT"
    assert fields["model_source"] == "local"
    assert fields["model_sources"]["local"] == {
        "name": "Tiny-ZEOS-OPT",
        "path": "models/Tiny-ZEOS-OPT/",
    }
    assert "huggingface" in fields["model_sources"]
    served = dist / "models" / "Tiny-ZEOS-OPT"
    assert served.is_symlink() != copy
    assert (served / "meta.json").read_text(encoding="utf-8") == "{}"


def test_a_directory_without_meta_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no meta.json"):
        model_source.model_fields(tmp_path, local=tmp_path)


def test_model_alone_means_the_default_export(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    model_source.add_arguments(parser)
    assert parser.parse_args([]).model is None
    assert parser.parse_args(["--model"]).model == model_source.LOCAL_EXPORT
    assert parser.parse_args(["--model", str(tmp_path)]).model == tmp_path
    assert model_source.LOCAL_EXPORT.parent.name == "models"
    assert model_source.LOCAL_EXPORT.parent.parent.name == "zeos-browser"
