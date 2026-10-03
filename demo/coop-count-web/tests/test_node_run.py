# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A kernel run over the real model worker: the case runner, its attention file, and a
block hidden through ``set_mask`` receiving exactly zero on every step after. Skipped
without Node, the npm packages or the export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeos_coop_count_web.evidence import report
from zeos_coop_count_web.node_run import main
from zeos_coop_count_web.node_worker import DEFAULT_MODEL, node_available

CASES = Path(__file__).resolve().parents[2] / "coop-count" / "cases"

pytestmark = pytest.mark.skipif(
    not node_available() or not (DEFAULT_MODEL / "meta.json").is_file(),
    reason="needs node, `npm install` in demo/coop-count-web and the export",
)


def test_a_hidden_block_receives_no_attention_and_the_rest_sums_to_one(tmp_path: Path) -> None:
    journal = tmp_path / "run.jsonl"
    case = CASES / "coop-count-scripted"
    argv = [str(case), "--events", str(case / "events.jsonl"), "--journal", str(journal)]
    assert main([*argv, "--max-ticks", "30", "--hide", "1:2", "--quiet"]) == 0

    rows = [json.loads(line) for line in journal.with_suffix(".attention.jsonl").open()]
    decodes = [json.loads(line) for line in journal.open() if '"machine.decode"' in line]
    assert {r["seq"] for r in rows} == {d["seq"] for d in decodes}, "one row per decode"
    mine = [r for r in rows if r["job"] == 1]
    assert mine, "counter-a decoded"
    for row in rows:
        assert sum(row["blocks"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(row["blocks"].get("2", 0.0) == 0.0 for row in mine)
    assert any(row["blocks"].get("1", 0.0) > 0.0 for row in mine), "its neighbour is read"
    assert "largest mass 0.0" in report(
        journal, journal.with_suffix(".attention.jsonl"), hidden=["1:2"]
    )
