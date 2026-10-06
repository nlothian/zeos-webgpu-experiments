# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The zeos arm on the real Qwen3.5-4B-ZEOS-OPT: ``page.open_run`` over ``NodeWorker``
(onnxruntime-node's CPU provider), 2 s ticks, 40 ticks of the default board.

The CPU is far slower than WebGPU, so this is about correctness, not lag: every command
the pilot finishes is a grammar-valid move, the pilot moves at least once, and no step is
left stuck (in flight past the end, or never completing). Needs the OPT+ZEOS export at
``demo/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT`` and ``npm install`` there; skips without
them. Several minutes on an M1 Max with 8 threads (``ZEOS_OPT_THREADS``), most of it the
system prompt's prefill.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, cast

import pytest
from zeos_coop_count_web.node_worker import DEMO, NodeWorker, node_available

from zeos_space_invaders_web import page
from zeos_space_invaders_web.contracts import BoardName, BoardSpec, Frame, FrameSink

MODEL = Path(os.environ.get("ZEOS_OPT_MODEL_DIR", DEMO / "models" / "Qwen3.5-4B-ZEOS-OPT"))
TICK_S = 2.0
TICKS = 40

pytestmark = pytest.mark.skipif(
    not (MODEL / "meta.json").is_file() or not node_available(),
    reason=f"needs the OPT+ZEOS export at {MODEL} and npm install",
)


def _sink(frames: list[Frame]) -> FrameSink:
    def sink(frame: Frame) -> None:
        frames.append(frame)

    return sink


def test_the_4b_pilot_flies_grammar_valid_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    real = page.load_board

    def load(name: BoardName, seed: int | None = None) -> BoardSpec:
        return dataclasses.replace(real(name, seed), tick_seconds=TICK_S)

    monkeypatch.setattr(page, "load_board", load)
    monkeypatch.setitem(page.RUNNER_OPTIONS, "max_ticks", TICKS)
    monkeypatch.setitem(page.RUNNER_OPTIONS, "warm_timeout_s", 1800.0)
    threads = int(os.environ.get("ZEOS_OPT_THREADS", "8"))
    frames: list[Frame] = []
    with NodeWorker(MODEL, runtime="node", threads=threads) as worker:
        run = page.open_run("zeos", "default", 7, cast(Any, worker), on_frame=_sink(frames))
        assert isinstance(run, page.ZeosRun)
        try:
            run.warm()
            assert run.runner.warmed
            result = run.run()
        finally:
            run.close()
        assert not worker.inFlight, "close left a step in flight"

    extras = cast(dict[str, Any], result.extras)
    print({k: extras[k] for k in ("warm_s", "pilot_moves", "step_totals", "roundtrip_s")})
    assert result.ticks == TICKS or result.lives == 0
    assert extras["pilot_moves"] >= 1, extras
    assert extras["pilot_valid_moves"] == extras["pilot_moves"], extras
    assert extras["pilot_exits"] == 0
    pilot = [d for f in frames for d in f["decisions"] if d["by"] == "pilot"]
    assert pilot and all(d["action"] in ("left", "right", "shoot") for d in pilot)
    totals = run.machine.step_totals
    assert totals["token"] > 0
    # A step that never completes would leave the pilot stalling to the end with no
    # token after its last move; the machine took a token within the run's last moves.
    assert run.machine.roundtrips, "no pilot command completed"
