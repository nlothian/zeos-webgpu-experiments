# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The contracts module imports, and the packaged boards are the native ones."""

from pathlib import Path

import pytest
from zeos_space_invaders.game import Rules

from zeos_space_invaders_web import contracts
from zeos_space_invaders_web.contracts import BOARDS, BoardName, load_board


@pytest.mark.parametrize("name", BOARDS)
def test_packaged_board_is_the_native_settings_file(name: BoardName, native_demo: Path) -> None:
    original = (native_demo / f"settings_{name}.json").read_bytes()
    assert contracts.board_text(name).encode() == original


def test_default_board() -> None:
    spec = load_board("default")
    assert spec.rules == Rules(
        w=12,
        h=16,
        monster_rows=3,
        monster_cols=5,
        monster_col_offset=1,
        lives=3,
        missile_rows=1,
        danger_rows=1,
        march_group=1,
        fire_chance=0.25,
    )
    assert (spec.tick_seconds, spec.max_steps, spec.history, spec.view) == (0.5, 600, 5, "lead")
    assert (spec.seed, spec.actions_per_tick) == (7, 1)


def test_ablation_board_takes_the_seed_asked_for() -> None:
    spec = load_board("ablation", seed=3)
    assert (spec.rules.w, spec.rules.h, spec.tick_seconds) == (9, 8, 0.2)
    assert (spec.seed, spec.actions_per_tick) == (3, None)
    assert load_board("ablation").seed is None


def test_unknown_board_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown board"):
        load_board("huge")  # pyright: ignore[reportArgumentType]


def test_channel_layout_matches_coop_count_web() -> None:
    """The frame offset is coop-count-web's ``DATA``; the port adds slots, not bytes."""
    channel = Path(__file__).resolve().parents[2] / "coop-count-web" / "web" / "model_channel.js"
    assert f"const DATA = {contracts.FRAME_OFFSET};" in channel.read_text()
    assert (
        len(
            {
                contracts.SLOT_STATE,
                contracts.SLOT_LENGTH,
                contracts.SLOT_ABORT,
                contracts.SLOT_IN_FLIGHT,
            }
        )
        == 4
    )
    assert max(contracts.SLOT_IN_FLIGHT, contracts.SLOT_ABORT) * 4 < contracts.FRAME_OFFSET


def test_local_stop_and_result_json() -> None:
    stop = contracts.LocalStop()
    assert not stop.is_set()
    stop.set()
    assert stop.is_set()
    lag: contracts.LagStats = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    result = contracts.RunResult(
        arm="zeos",
        board="default",
        seed=7,
        lives=3,
        kills=0,
        ticks=0,
        decisions=0,
        preemptions=0,
        cancellations=0,
        reflexes=0,
        lag_ticks=lag,
        overrun_ms=0.0,
        catchup_ticks=0,
    )
    assert result.to_json()["verdicts"] == []
    assert contracts.MonotonicClock().now() > 0
