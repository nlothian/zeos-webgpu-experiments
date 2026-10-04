# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The exported graph, held to ``DecodeResult.attention``'s units under ONNX Runtime CPU
before any JavaScript runs it: a hidden position receives exactly zero, the positions
that were allowed sum to one, and hiding one changes the logits. For a hybrid model the
recurrent state is carried between runs as the worker carries it. Needs the export and
the ``export`` dependency group."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

DEMO = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("ZEOS_WEB_MODEL_DIR", DEMO / "models" / "Qwen3.5-2B-zeos-int8"))

ort = pytest.importorskip("onnxruntime", reason="needs the export dependency group")
np = pytest.importorskip("numpy", reason="needs the export dependency group")

pytestmark = pytest.mark.skipif(
    not (MODEL / "meta.json").is_file(), reason=f"no export at {MODEL}; run export_model.py"
)


@pytest.fixture(scope="module")
def graph() -> tuple[Any, dict[str, Any]]:
    meta = json.loads((MODEL / "meta.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(MODEL / "model.onnx"), options), meta


def _context() -> list[int]:
    """A deterministic id sequence longer than one prefill chunk."""
    return [(i * 7919) % 20000 + 100 for i in range(83)]


def _last_step(
    graph: tuple[Any, dict[str, Any]], ids: list[int], mask: Any
) -> tuple[Any, Any, Any]:
    """Prefill all but the last id in the worker's chunks, then one decode step: the
    logits, the new position's KV and the attention."""
    session, meta = graph
    kv = np.zeros((0, *meta["kvShape"]), np.float32)
    state = np.zeros(meta["stateShape"], np.float32)
    conv = np.zeros(meta["convShape"], np.float32)
    done = 0
    while done < len(ids) - 1:
        count = min(meta["maxChunk"], len(ids) - 1 - done)
        _, new, state, conv, _ = session.run(
            None,
            {
                "input_ids": np.array(ids[done : done + count], np.int64),
                "past_kv": kv,
                "state": state,
                "conv": conv,
                "key_mask": mask[: done + count],
            },
        )
        kv = np.concatenate((kv, new))
        done += count
    logits, new, _, _, attention = session.run(
        None,
        {
            "input_ids": np.array(ids[-1:], np.int64),
            "past_kv": kv,
            "state": state,
            "conv": conv,
            "key_mask": mask,
        },
    )
    return logits, new, attention


def test_a_hidden_position_receives_exactly_zero_and_the_rest_sum_to_one(
    graph: tuple[Any, dict[str, Any]],
) -> None:
    _, meta = graph
    ids = _context()
    for hidden in (0, 2, len(ids) // 2, len(ids) - 2):
        mask = np.ones(len(ids), np.bool_)
        mask[hidden] = False
        logits, new_kv, attention = _last_step(graph, ids, mask)
        assert attention.shape == (len(ids),)
        assert attention[hidden] == 0.0
        assert float(attention.sum()) == pytest.approx(1.0, abs=1e-5)
        assert (attention >= 0).all()
        assert logits.shape == (meta["logitsSize"],)
        assert new_kv.shape == (1, *meta["kvShape"])


def test_hiding_a_position_changes_what_the_step_computes(
    graph: tuple[Any, dict[str, Any]],
) -> None:
    """The mask acts inside the forward pass, not on the reported map: hiding a position
    moves the logits."""
    ids = _context()
    open_logits, _, open_attention = _last_step(graph, ids, np.ones(len(ids), np.bool_))
    hidden = np.ones(len(ids), np.bool_)
    hidden[0] = False
    hidden_logits, _, _ = _last_step(graph, ids, hidden)
    assert open_attention[0] > 0.0
    assert float(np.abs(open_logits - hidden_logits).max()) > 1e-3
