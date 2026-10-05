# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Give ``onnx-community/Qwen3.5-4B-ONNX-OPT`` a per-position key mask and a measured
attention output, by graph surgery.

The ``-OPT`` export runs every Gated DeltaNet layer as the fused
``com.microsoft::LinearAttention`` and ``CausalConvWithState`` operators and every softmax
layer as ``GroupQueryAttention`` after ``RotaryEmbedding``, with 4-bit ``MatMulNBits``
weights, all of which ONNX Runtime Web runs on WebGPU. This script keeps every one of
those nodes and every weight byte, and changes only how the mask reaches them:

- **Input** ``key_mask`` (bool ``[1, past + new]``, true = visible) replaces
  ``attention_mask``. The graph casts it back to the int64 ``attention_mask`` the
  original nodes read their lengths from, so ``seqlens_k`` and ``total_sequence_length``
  are still the full length.
- **Softmax layers.** ``GroupQueryAttention`` takes an ``attention_bias``
  (``[1, 1, new, past + new]``), which it adds to the scores on top of its own causal
  mask. The ``-OPT`` graph fed it a padding bias; here it is 0 at a visible key and
  -65504 at a hidden one, so a hidden key gets exactly zero weight. A position whose
  every causal key is hidden sees itself, as in ``export_model.ZeosQwen``, so its
  softmax is defined.
- **DeltaNet layers**, at a hidden *new* position: beta (the write strength) is zero and
  the log decay is zero, so the state passes through unchanged and takes nothing of the
  position in; the position's projection enters the causal convolution's input (and so
  the carried window) as zeros, so no later position's taps read it; and the position
  still reads its own projection through the convolution's last tap, as it does in
  ``ZeosQwen``. The ``-OPT`` padding mask on ``z``, ``a`` and ``b`` is removed: it
  zeroed the output gate and set beta to 0.5, which is padding's business, not
  hiding's. A position already in the state when it is hidden cannot be taken out by
  the graph, since the state is not per position; the worker rewinds to a snapshot of
  the state from before it and runs the positions after it again under the new mask
  (``web/transformers_worker.js`` does this for the ``ZeosQwen`` export).
- **Output** ``attention`` (f32 ``[past + new]``): the last new position's attention
  probabilities over every position, averaged over the 16 heads of each of the 8
  softmax layers, recomputed from the rotated query and ``present.N.key``, the tensors
  ``GroupQueryAttention`` reads. A hidden position is exactly zero (the probabilities
  are multiplied by the visibility), and the vector sums to one.

The chunk of new positions is any length; ``LinearAttention`` solves its prefill
chunk-parallel, so the 16-position bound of the ``ZeosQwen`` export does not apply here.

The weights are not loaded: the decoder's two external-data files are copied byte for
byte under the new graph's name, and the few inline constants the new nodes need sit in
the graph. Run it with the export dependency group (see the demo README)::

    uv run python demo/coop-count-web/export/opt_zeos_surgery.py \\
        --src models/onnx-community/Qwen3.5-4B-ONNX-OPT \\
        --out demo/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DEMO = Path(__file__).resolve().parent.parent
SRC_DECODER = "decoder_model_merged_q4f16"
OUT_DECODER = "decoder_zeos_q4f16"
EMBED = "embed_tokens_q4f16"
#: The most new positions the worker sends in one run. The graph has no bound of its
#: own; this is the chunk the contract promises and the tests and benchmark exercise.
MAX_CHUNK = 2048
#: GroupQueryAttention's padding bias in the ``-OPT`` graph: the most negative float16.
HIDDEN_BIAS = -65504.0
#: The attention output's masked score, finite so no backend's exp meets an infinity.
HIDDEN_SCORE = -1.0e9
COPIED = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class Builder:
    """Appends nodes with fresh names under a ``/zeos/`` prefix."""

    def __init__(self, graph: onnx.GraphProto) -> None:
        self.graph = graph
        self.count = 0

    def name(self, stem: str) -> str:
        self.count += 1
        return f"/zeos/{stem}_{self.count}"

    def const(self, stem: str, value: np.ndarray) -> str:
        name = self.name(stem)
        self.graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    def node(self, op: str, inputs: list[str], stem: str, *, domain: str = "", **attrs: Any) -> str:
        out = self.name(stem)
        self.graph.node.append(
            helper.make_node(op, inputs, [out], name=out + "/node", domain=domain, **attrs)
        )
        return out


def _consumers(graph: onnx.GraphProto) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    found: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for slot, name in enumerate(node.input):
            found.setdefault(name, []).append((node, slot))
    return found


def _rewire(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for slot, name in enumerate(node.input):
            if name == old:
                node.input[slot] = new


def _producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto:
    for node in graph.node:
        if name in node.output:
            return node
    raise KeyError(name)


def _read_external(tensor: onnx.TensorProto, base: Path) -> np.ndarray:
    entries = {e.key: e.value for e in tensor.external_data}
    with (base / entries["location"]).open("rb") as handle:
        handle.seek(int(entries.get("offset", "0")))
        raw = handle.read(int(entries["length"]))
    dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(tensor.dims))


def _prune(graph: onnx.GraphProto) -> int:
    """Drop nodes nothing reads, and initialisers no node reads."""
    removed = 0
    keep_outputs = {o.name for o in graph.output}
    while True:
        used = {name for node in graph.node for name in node.input} | keep_outputs
        dead = [n for n in graph.node if not any(o in used for o in n.output if o)]
        if not dead:
            break
        for node in dead:
            graph.node.remove(node)
        removed += len(dead)
    used = {name for node in graph.node for name in node.input}
    for tensor in [t for t in graph.initializer if t.name not in used]:
        graph.initializer.remove(tensor)
    return removed


def _toposort(graph: onnx.GraphProto) -> None:
    ready = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {""}
    pending = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        rest: list[onnx.NodeProto] = []
        for node in pending:
            if all(name in ready for name in node.input):
                ordered.append(node)
                ready.update(node.output)
            else:
                rest.append(node)
        if len(rest) == len(pending):
            missing = sorted({n for node in rest for n in node.input if n not in ready})
            raise SystemExit(f"graph has a cycle or dangling inputs: {missing[:5]}")
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)


def _value_info(name: str, elem: int, shape: Iterable[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem, list(shape))


def operate(model: onnx.ModelProto, src_onnx_dir: Path) -> dict[str, Any]:
    """Rewrite ``model`` in place; returns what ``meta.json`` reports of the layers."""
    graph = model.graph
    b = Builder(graph)

    # -- the key mask replaces attention_mask ------------------------------------------
    old_mask = next(i for i in graph.input if i.name == "attention_mask")
    position = list(graph.input).index(old_mask)
    graph.input.remove(old_mask)
    graph.input.insert(
        position, _value_info("key_mask", TensorProto.BOOL, (1, "total_sequence_length"))
    )
    graph.node.insert(
        0,
        helper.make_node(
            "Cast", ["key_mask"], ["attention_mask"], name="/zeos/key_mask/Cast", to=7
        ),
    )

    total = "/model/shared_dims/attention_mask/Gather_1/output_0"  # int64 scalar T
    new = "/model/shared_dims/root_input/Gather_1/output_0"  # int64 scalar S
    for name in (total, new):
        _producer(graph, name)  # fails loudly if the -OPT graph has changed shape

    i64 = np.int64
    zero = b.const("zero", np.array(0, i64))
    one = b.const("one", np.array(1, i64))
    ax0 = b.const("axes0", np.array([0], i64))
    ax1 = b.const("axes1", np.array([1], i64))
    ax01 = b.const("axes01", np.array([0, 1], i64))

    # visible[i, j]: key j is visible to new position i, before causality, which
    # GroupQueryAttention applies itself. A position none of whose causal keys is
    # visible sees itself.
    mask_row = b.node("Squeeze", ["key_mask", ax0], "mask_row")  # [T] bool
    seen = b.node("CumSum", [b.node("Cast", [mask_row], "mask_i64", to=7), zero], "seen")  # [T]
    lonely = b.node("Equal", [seen, zero], "lonely")  # [T]
    keys = b.node("Range", [zero, total, one], "keys")  # [T]
    first_new = b.node("Sub", [total, new], "first_new")
    queries = b.node("Range", [first_new, total, one], "queries")  # [S]
    own = b.node(
        "Equal",
        [b.node("Unsqueeze", [queries, ax1], "q_col"), b.node("Unsqueeze", [keys, ax0], "k_row")],
        "own",
    )  # [S, T]
    lonely_q = b.node(
        "Unsqueeze", [b.node("Gather", [lonely, queries], "lonely_q", axis=0), ax1], "lonely_col"
    )
    visible = b.node(
        "Or",
        [b.node("Unsqueeze", [mask_row, ax0], "mask_2d"), b.node("And", [own, lonely_q], "self")],
        "visible",
    )  # [S, T]
    bias_2d = b.node(
        "Where",
        [
            visible,
            b.const("bias_open", np.array(0.0, np.float16)),
            b.const("bias_hidden", np.array(HIDDEN_BIAS, np.float16)),
        ],
        "bias_2d",
    )
    bias = b.node("Unsqueeze", [bias_2d, ax01], "gqa_bias")  # [1, 1, S, T] f16

    # The last new position's visibility, for the attention output.
    last_visible = b.node(
        "Squeeze",
        [
            b.node(
                "Slice",
                [
                    visible,
                    b.const("minus1", np.array([-1], i64)),
                    b.const("big", np.array([2**62], i64)),
                    ax0,
                ],
                "last_visible_2d",
            ),
            ax0,
        ],
        "last_visible",
    )  # [T] bool
    last_visible_f = b.node("Cast", [last_visible], "last_visible_f", to=1)

    # -- softmax layers ------------------------------------------------------------------
    gqa = [n for n in graph.node if n.op_type == "GroupQueryAttention"]
    attn_sum: str | None = None
    heads = kv_heads = head_dim = 0
    full_layers: list[int] = []
    for node in gqa:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        heads, kv_heads = int(attrs["num_heads"]), int(attrs["kv_num_heads"])
        scale = float(attrs["scale"])
        if attrs.get("softcap", 0.0) or attrs.get("do_rotary", 0):
            raise SystemExit(f"{node.name}: softcap or fused rotary; the bias would not match")
        while len(node.input) < 11:
            node.input.append("")
        node.input[10] = bias
        layer = int(node.name.split("/layers.")[1].split("/")[0])
        full_layers.append(layer)
        present_key = node.output[1]
        query = node.input[0]  # [1, S, heads * d] f16, rotated
        q_last = b.node(
            "Slice",
            [
                query,
                b.const("minus1", np.array([-1], i64)),
                b.const("big", np.array([2**62], i64)),
                ax1,
            ],
            f"l{layer}_q_last",
        )  # [1, 1, heads * d]
        # Head h reads KV head h // (heads / kv_heads), GroupQueryAttention's grouping.
        # In float32: computing K q^T in float16 instead measured no faster on WebGPU.
        qg = b.node(
            "Cast",
            [
                b.node(
                    "Reshape",
                    [q_last, b.const("qshape", np.array([kv_heads, heads // kv_heads, -1], i64))],
                    f"l{layer}_qg",
                )
            ],
            f"l{layer}_qg_f",
            to=1,
        )  # [kv, g, d]
        kt = b.node(
            "Transpose",
            [
                b.node(
                    "Cast",
                    [b.node("Squeeze", [present_key, ax0], f"l{layer}_k")],
                    f"l{layer}_k_f",
                    to=1,
                )
            ],
            f"l{layer}_kt",
            perm=[0, 2, 1],
        )  # [kv, d, T]
        scores = b.node(
            "Mul",
            [
                b.node("MatMul", [qg, kt], f"l{layer}_qk"),
                b.const("scale", np.array(scale, np.float32)),
            ],
            f"l{layer}_scores",
        )  # [kv, g, T]
        masked = b.node(
            "Where",
            [last_visible, scores, b.const("hidden_score", np.array(HIDDEN_SCORE, np.float32))],
            f"l{layer}_masked",
        )
        probs = b.node("Softmax", [masked], f"l{layer}_probs", axis=-1)
        mass = b.node("ReduceSum", [probs, ax01], f"l{layer}_mass", keepdims=0)  # [T]
        attn_sum = mass if attn_sum is None else b.node("Add", [attn_sum, mass], f"l{layer}_sum")
    if attn_sum is None:
        raise SystemExit("no GroupQueryAttention nodes: not the -OPT graph")
    averaged = b.node(
        "Mul",
        [attn_sum, b.const("per_head", np.array(1.0 / (len(gqa) * heads), np.float32))],
        "attention_mean",
    )
    graph.node.append(
        helper.make_node(
            "Mul", [averaged, last_visible_f], ["attention"], name="/zeos/attention/node"
        )
    )
    graph.output.append(_value_info("attention", TensorProto.FLOAT, ("total_sequence_length",)))

    # -- DeltaNet layers -----------------------------------------------------------------
    new_mask = "/model/gdn_mask/unsqueeze/output_0"  # [1, S, 1] f16, the new positions
    _producer(graph, new_mask)
    consumers = _consumers(graph)
    mask_muls = [node for node, _ in consumers[new_mask] if node.op_type == "Mul"]
    qkv_muls: dict[int, onnx.NodeProto] = {}
    for mul in mask_muls:
        other = next(x for x in mul.input if x != new_mask)
        layer = int(mul.name.split("/layers.")[1].split("/")[0])
        if mul.name.endswith("/mul_qkv"):
            qkv_muls[layer] = mul
        elif mul.name.endswith(("/mul_z", "/mul_a", "/mul_b")):
            _rewire(graph, mul.output[0], other)
        else:
            raise SystemExit(f"unexpected use of the GDN mask: {mul.name}")

    inits = {t.name: t for t in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "CausalConvWithState"]
    linear_layers: list[int] = []
    conv_dim = conv_kernel = 0
    for conv in convs:
        layer = int(conv.name.split("/layers.")[1].split("/")[0])
        linear_layers.append(layer)
        mul = qkv_muls[layer]
        raw = next(x for x in mul.input if x != new_mask)  # [1, S, C] f16, unmasked
        attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
        if attrs.get("activation", b"none") not in (b"silu", b"swish"):
            raise SystemExit(f"{conv.name}: expected a fused SiLU")
        del conv.attribute[:]
        conv.attribute.extend(
            [helper.make_attribute("ndim", 1), helper.make_attribute("activation", "none")]
        )
        weight = inits[conv.input[1]]
        w = (
            _read_external(weight, src_onnx_dir)
            if weight.data_location == 1
            else (numpy_helper.to_array(weight))
        )
        conv_dim, conv_kernel = int(w.shape[0]), int(w.shape[2])
        last_tap = b.const(f"l{layer}_last_tap", np.ascontiguousarray(w[:, 0, -1]))  # [C]
        back = next(n for n, _ in _consumers(graph)[conv.output[0]] if n.op_type == "Transpose")
        pre = back.output[0]  # [1, S, C] f16, before the activation
        readers = _consumers(graph)[pre]
        # A hidden position's input entered the convolution as zeros; give its own last
        # tap back, as ZeosQwen does: conv(masked) + (raw - masked) * w[:, -1].
        own = b.node(
            "Mul",
            [b.node("Sub", [raw, mul.output[0]], f"l{layer}_own_in"), last_tap],
            f"l{layer}_own",
        )
        pre_f = b.node(
            "Cast", [b.node("Add", [pre, own], f"l{layer}_pre")], f"l{layer}_pre_f", to=1
        )
        silu = b.node("Mul", [pre_f, b.node("Sigmoid", [pre_f], f"l{layer}_sig")], f"l{layer}_silu")
        act = b.node("Cast", [silu], f"l{layer}_act", to=10)
        for node, slot in readers:
            node.input[slot] = act

    las = [n for n in graph.node if n.op_type == "LinearAttention"]
    for la in las:
        layer = int(la.name.split("/layers.")[1].split("/")[0])
        # decay (log space) and beta are [1, S, heads]; a hidden position gets 0 and 0.
        la.input[4] = b.node("Mul", [la.input[4], new_mask], f"l{layer}_decay_masked")
        la.input[5] = b.node("Mul", [la.input[5], new_mask], f"l{layer}_beta_masked")

    removed = _prune(graph)
    _toposort(graph)

    present_key = next(o for o in graph.output if o.name == f"present.{full_layers[0]}.key")
    head_dim = present_key.type.tensor_type.shape.dim[3].dim_value
    recurrent = next(i for i in graph.input if i.name == "past_recurrent.0")
    rdims = [d.dim_value for d in recurrent.type.tensor_type.shape.dim[1:]]
    if len(las) != len(linear_layers) or sorted(linear_layers) != sorted(
        int(n.name.split("/layers.")[1].split("/")[0]) for n in las
    ):
        raise SystemExit("LinearAttention and CausalConvWithState layers disagree")
    print(f"removed {removed} padding-mask nodes; {len(graph.node)} nodes")
    return {
        "full": sorted(full_layers),
        "linear": sorted(linear_layers),
        "heads": heads,
        "kvHeads": kv_heads,
        "headDim": head_dim,
        "recurrentShape": rdims,
        "convDim": conv_dim,
        "convKernel": conv_kernel,
    }


def _shape(v: onnx.ValueInfoProto) -> list[int | str]:
    return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]


def _io(values: Iterable[onnx.ValueInfoProto]) -> list[dict[str, Any]]:
    return [
        {
            "name": v.name,
            "dtype": helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type).name,
            "shape": _shape(v),
        }
        for v in values
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="the -OPT export's directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEMO / "models" / "Qwen3.5-4B-ZEOS-OPT",
        help="output directory (default models/Qwen3.5-4B-ZEOS-OPT)",
    )
    args = parser.parse_args(argv)
    src: Path = args.src
    out: Path = args.out
    src_onnx = src / "onnx"

    model = onnx.load(str(src_onnx / f"{SRC_DECODER}.onnx"), load_external_data=False)
    layers = operate(model, src_onnx)

    shards = sorted(
        {e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"}
    )
    renamed = {name: name.replace(SRC_DECODER, OUT_DECODER) for name in shards}
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = renamed[entry.value]

    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    onnx.save(model, str(out / "onnx" / f"{OUT_DECODER}.onnx"))
    for old, new_name in renamed.items():
        shutil.copyfile(src_onnx / old, out / "onnx" / new_name)
    for file in sorted(p.name for p in src_onnx.iterdir() if p.name.startswith(EMBED)):
        shutil.copyfile(src_onnx / file, out / "onnx" / file)
    for file in COPIED:
        shutil.copyfile(src / file, out / file)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    special = {
        name: tokenizer.token_to_id(name)
        for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
    }
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config", config)
    logits = next(o for o in model.graph.output if o.name == "logits")
    meta: dict[str, Any] = {
        "model": "onnx-community/Qwen3.5-4B-ONNX-OPT",
        "derivedBy": "demo/coop-count-web/export/opt_zeos_surgery.py",
        "family": text["model_type"],
        "quant": "q4f16",
        "blockSize": 1,
        "maxChunk": MAX_CHUNK,
        "numLayers": len(text["layer_types"]),
        "layerTypes": text["layer_types"],
        "fullAttentionLayers": layers["full"],
        "linearAttentionLayers": layers["linear"],
        "numHeads": layers["heads"],
        "numKvHeads": layers["kvHeads"],
        "headDim": layers["headDim"],
        "hiddenSize": int(text["hidden_size"]),
        # The cache, as the decoder's inputs and outputs shape it (batch 1).
        "kvShape": [1, layers["kvHeads"], "past", layers["headDim"]],
        "recurrentShape": [1, *layers["recurrentShape"]],
        "convShape": [1, layers["convDim"], layers["convKernel"] - 1],
        "logitsSize": _shape(logits)[-1],
        "vocabSize": tokenizer.get_vocab_size(with_added_tokens=True),
        "eosId": special["<|im_end|>"],
        "padId": special["<|endoftext|>"],
        "controlIds": [special["<|im_start|>"], special["<|im_end|>"], special["<|endoftext|>"]],
        "thinkIds": [special["<think>"], special["</think>"]],
        "decoder": {
            "file": f"onnx/{OUT_DECODER}.onnx",
            "externalData": [f"onnx/{renamed[s]}" for s in shards],
            "inputs": _io(model.graph.input),
            "outputs": _io(model.graph.output),
        },
        "embedTokens": {
            "file": f"onnx/{EMBED}.onnx",
            "externalData": [f"onnx/{EMBED}.onnx_data"],
        },
        "files": {},
    }
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "meta.json"):
        rel = path.relative_to(out).as_posix()
        meta["files"][rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(f["bytes"] for f in meta["files"].values())
    print(f"wrote {out} ({total / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
