# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The exported decode graph, held to ``DecodeResult.attention``'s units under ONNX Runtime
CPU before any JavaScript runs it: a masked block receives exactly zero, and the blocks
that were allowed sum to one. Needs the export and the ``export`` dependency group."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

DEMO = Path(__file__).resolve().parent.parent
MODEL = Path(
    os.environ.get("ZEOS_WEB_MODEL_DIR", DEMO / "models" / "Qwen2.5-0.5B-Instruct-zeos-int8")
)

ort = pytest.importorskip("onnxruntime", reason="needs the export dependency group")
np = pytest.importorskip("numpy", reason="needs the export dependency group")

pytestmark = pytest.mark.skipif(
    not (MODEL / "meta.json").is_file(), reason=f"no export at {MODEL}; run export_model.py"
)


@pytest.fixture(scope="module")
def graphs() -> tuple[Any, Any, dict[str, Any]]:
    meta = json.loads((MODEL / "meta.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    prefill = ort.InferenceSession(str(MODEL / "prefill.onnx"), options)
    decode = ort.InferenceSession(str(MODEL / "decode.onnx"), options)
    return prefill, decode, meta


def _prefill(graphs: tuple[Any, Any, dict[str, Any]], ids: list[int]) -> Any:
    prefill, _, meta = graphs
    empty = np.zeros((0, meta["numLayers"], 2, meta["numKvHeads"], meta["headDim"]), np.float32)
    (kv,) = prefill.run(
        None,
        {
            "input_ids": np.array(ids, np.int64),
            "past_kv": empty,
            "key_mask": np.ones(len(ids), np.bool_),
        },
    )
    return kv


def _context(meta: dict[str, Any]) -> list[int]:
    """A deterministic id sequence long enough for several blocks."""
    return [(i * 7919) % 20000 + 100 for i in range(6 * meta["blockSize"] + 3)]


def test_a_masked_block_receives_exactly_zero_and_the_rest_sum_to_one(
    graphs: tuple[Any, Any, dict[str, Any]],
) -> None:
    _, decode, meta = graphs
    ids = _context(meta)
    kv = _prefill(graphs, ids[:-1])
    blocks = -(-len(ids) // meta["blockSize"])
    for masked in (0, 2, blocks - 2):
        allowed = np.ones(blocks, np.bool_)
        allowed[masked] = False
        logits, new_kv, attention = decode.run(
            None,
            {"input_ids": np.array(ids[-1:], np.int64), "past_kv": kv, "allowed_blocks": allowed},
        )
        assert attention.shape == (blocks,)
        assert attention[masked] == 0.0
        assert float(attention.sum()) == pytest.approx(1.0, abs=1e-5)
        assert (attention >= 0).all()
        assert logits.shape == (meta["logitsSize"],)
        assert new_kv.shape == (1, meta["numLayers"], 2, meta["numKvHeads"], meta["headDim"])


def test_masking_a_block_changes_what_the_step_computes(
    graphs: tuple[Any, Any, dict[str, Any]],
) -> None:
    """The mask acts inside the forward pass, not on the reported map: hiding a block
    moves the logits."""
    _, decode, meta = graphs
    ids = _context(meta)
    kv = _prefill(graphs, ids[:-1])
    blocks = -(-len(ids) // meta["blockSize"])
    feed: dict[str, Any] = {"input_ids": np.array(ids[-1:], np.int64), "past_kv": kv}
    open_logits, _, open_attention = decode.run(
        None, {**feed, "allowed_blocks": np.ones(blocks, np.bool_)}
    )
    hidden = np.ones(blocks, np.bool_)
    hidden[0] = False
    hidden_logits, _, _ = decode.run(None, {**feed, "allowed_blocks": hidden})
    assert open_attention[0] > 0.0
    assert float(np.abs(open_logits - hidden_logits).max()) > 1e-3
