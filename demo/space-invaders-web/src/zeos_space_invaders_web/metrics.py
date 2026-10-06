# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The numbers a wall-clock run is summarised by, as pure functions of its records.

Kept apart from the runners so the page, the benchmark and the tests compute a lag
or a parse rate the same way. Percentiles are nearest-rank, so every value
reported is one that was observed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypedDict

from .contracts import DecisionBy, DecisionRecord, LagStats

#: The authors whose moves a model made, and so whose lag is the measurement. The
#: reflex asks no model; averaging it in would flatter the run.
MODEL_AUTHORS: frozenset[DecisionBy] = frozenset({"pilot", "prompt"})


class OverrunStats(TypedDict):
    """How late the loop noticed each tick fall due, in milliseconds.

    One entry per time the loop found ticks due (a catch-up batch is one entry,
    measured from its earliest due time). ``total`` is ``RunResult.overrun_ms``;
    ``count`` is the number of entries.
    """

    total: float
    mean: float
    max: float
    count: int


def percentile(values: Sequence[float], q: float) -> float:
    """The nearest-rank ``q``-th percentile (0-100) of ``values``; 0.0 for none."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def lag_stats(lags: Iterable[float]) -> LagStats:
    """Mean, median, 95th percentile and maximum of ``lags``; all 0.0 for none."""
    values = [float(v) for v in lags]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def model_lags(records: Iterable[DecisionRecord]) -> list[int]:
    """``lag_ticks`` of every model-made move, in order."""
    return [r["lag_ticks"] for r in records if r["by"] in MODEL_AUTHORS]


def overrun_stats(overruns_ms: Sequence[float]) -> OverrunStats:
    """Total, mean and maximum lateness over the ticks the loop applied."""
    if not overruns_ms:
        return {"total": 0.0, "mean": 0.0, "max": 0.0, "count": 0}
    total = float(sum(overruns_ms))
    return {
        "total": round(total, 3),
        "mean": round(total / len(overruns_ms), 3),
        "max": round(float(max(overruns_ms)), 3),
        "count": len(overruns_ms),
    }


def due_ticks(now: float, due: float, interval: float) -> int:
    """How many tick intervals have fallen due by ``now``: 0 before ``due``, else
    one for ``due`` itself and one more for each whole interval since.

    Every one but the first is a catch-up tick.
    """
    if now < due:
        return 0
    return 1 + math.floor((now - due) / interval)


def parse_rate(parsed: int, replies: int) -> float | None:
    """The share of replies that named an action; ``None`` with no replies."""
    if replies == 0:
        return None
    return round(parsed / replies, 3)
