# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Export a Qwen2 instruct model as the two ONNX graphs the browser worker runs.

``prefill.onnx`` takes new token ids, the job's past KV and a per-position key mask, and
returns only the KV for the new positions: the worker never needs logits from a prefill,
because it keeps the last token of a context pending and runs it through the decode graph.

``decode.onnx`` takes exactly one token, the past KV and a per-block allowed mask, and
returns the logits, the new KV and the step's attention per KV block. The mask is applied
before the softmax in every layer, so a masked block receives exactly zero and contributes
nothing to the logits; the attention is summed over layers and heads and divided by their
product inside the graph, so it sums to one and only a vector of block masses leaves it.

Both graphs are written here rather than traced from ``transformers``: the KV layout,
the mask and the attention reduction are the point of the export, and the library's
cache and mask APIs move between releases. ``check_against_reference`` holds the
rewrite to the library's own logits before anything is exported.

Run it with the export dependency group (see the demo README)::

    uv sync --package zeos-coop-count-web --group export
    uv run --package zeos-coop-count-web --group export \\
        python demo/coop-count-web/export/export_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

DEMO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
#: One KV block per position. The kernel's blocks are counted in its own words, each of
#: which is a variable number of model tokens, so only a per-position map lets the
#: Python side sum mass and build masks for the kernel's blocks exactly.
DEFAULT_BLOCK_SIZE = 1
OPSET = 17
#: Initialisers at least this large go to the shared weights file; smaller ones stay
#: inline in each graph.
SHARE_THRESHOLD = 1024


class ZeosQwen2(nn.Module):
    """A Qwen2 decoder over a flat KV cache laid out ``[T, layers, 2, kv_heads, head_dim]``.

    Position-major so that appending, truncating and copying a job's cache are contiguous
    byte operations in JavaScript.
    """

    def __init__(self, hf: Any, *, block_size: int, int8_embedding: bool) -> None:
        super().__init__()
        cfg = hf.config
        self.layers_n = int(cfg.num_hidden_layers)
        self.heads = int(cfg.num_attention_heads)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // self.heads)
        self.group = self.heads // self.kv_heads
        self.eps = float(cfg.rms_norm_eps)
        self.block_size = block_size
        rope_theta = _rope_theta(cfg)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        base = hf.model
        weight = base.embed_tokens.weight.detach().float()
        self.int8_embedding = int8_embedding
        if int8_embedding:
            # Symmetric per-row int8: the table is a sixth of the model and a lookup, so
            # quantising it here halves the download without touching a matmul.
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("embed_q", torch.round(weight / scale).to(torch.int8))
            self.register_buffer("embed_scale", scale)
        else:
            self.register_buffer("embed_w", weight)
        self.blocks = base.layers
        self.norm = base.norm
        # Tied or not, the output projection is its own initialiser in the decode graph,
        # so the weight-only quantiser treats it like every other matmul.
        self.lm_head_t = nn.Parameter(hf.lm_head.weight.detach().float().t().contiguous())

    # -- pieces ----------------------------------------------------------------

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        if self.int8_embedding:
            return self.embed_q[ids].float() * self.embed_scale[ids]
        return self.embed_w[ids]

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    def project(
        self, layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = h.shape[1]
        attn = layer.self_attn
        q = attn.q_proj(h).reshape(n, self.heads, self.head_dim)
        k = attn.k_proj(h).reshape(n, self.kv_heads, self.head_dim)
        v = attn.v_proj(h).reshape(n, self.kv_heads, self.head_dim)
        q = self.rotate(q, cos[:, None, :], sin[:, None, :])
        k = self.rotate(k, cos[:, None, :], sin[:, None, :])
        return q, k, v

    def finish(self, layer: Any, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        x = x + layer.self_attn.o_proj(out)
        h = self.rms(x, layer.post_attention_layernorm.weight)
        mlp = layer.mlp
        return x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))

    # -- the two graphs ----------------------------------------------------------

    def prefill(
        self, input_ids: torch.Tensor, past_kv: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        """KV for ``input_ids`` appended after ``past_kv``. ``key_mask[j]`` false hides
        position ``j`` from every new query, on top of the causal mask."""
        past = past_kv.shape[0]
        n = input_ids.shape[0]
        positions = torch.arange(n, dtype=torch.int64) + past
        cos, sin = self.rope(positions)
        keys = torch.arange(past + n, dtype=torch.int64)
        visible = (keys[None, :] <= positions[:, None]) & key_mask[None, :]
        x = self.embed(input_ids)[None, :, :]
        new: list[torch.Tensor] = []
        for i, layer in enumerate(self.blocks):
            h = self.rms(x, layer.input_layernorm.weight)
            q, k, v = self.project(layer, h, cos, sin)
            new.append(torch.stack((k, v), dim=1))
            k_all = torch.cat((past_kv[:, i, 0], k), dim=0)
            v_all = torch.cat((past_kv[:, i, 1], v), dim=0)
            # The library's default attention path, which the exporter lowers to plain
            # operators; nothing about its weights leaves the graph.
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k_all.transpose(0, 1).repeat_interleave(self.group, dim=0)[None],
                v_all.transpose(0, 1).repeat_interleave(self.group, dim=0)[None],
                attn_mask=visible[None, None],
            )
            x = self.finish(layer, x, out[0].transpose(0, 1).reshape(1, n, -1))
        return torch.stack(new, dim=1)

    def decode(
        self, input_ids: torch.Tensor, past_kv: torch.Tensor, allowed_blocks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: logits for the next token, KV for this one, attention per block."""
        past = past_kv.shape[0]
        length = past + 1
        positions = torch.arange(1, dtype=torch.int64) + past
        cos, sin = self.rope(positions)
        block_of = torch.arange(length, dtype=torch.int64) // self.block_size
        allowed = allowed_blocks[block_of]
        zero = torch.zeros((), dtype=torch.float32)
        bias = torch.where(allowed, zero, torch.full((), float("-inf")))
        x = self.embed(input_ids)[None, :, :]
        mass = torch.zeros(length, dtype=torch.float32)
        new: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for i, layer in enumerate(self.blocks):
            h = self.rms(x, layer.input_layernorm.weight)
            q, k, v = self.project(layer, h, cos, sin)
            new.append(torch.stack((k, v), dim=1))
            k_all = torch.cat((past_kv[:, i, 0], k), dim=0)  # [S, kv, d]
            v_all = torch.cat((past_kv[:, i, 1], v), dim=0)
            # Head h = kv * group + g, the ordering the library's repeat_kv produces.
            qg = q.reshape(self.kv_heads, self.group, self.head_dim)
            scores = torch.matmul(qg, k_all.permute(1, 2, 0)) * scale  # [kv, g, S]
            # Before the softmax and in every layer: a masked key is never attended.
            probs = torch.softmax(scores + bias, dim=-1)
            mass = mass + probs.sum(dim=(0, 1))
            out = torch.matmul(probs, v_all.transpose(0, 1))  # [kv, g, d]
            x = self.finish(layer, x, out.reshape(1, 1, -1))
        mass = mass / float(self.layers_n * self.heads)
        blocks = allowed_blocks.shape[0]
        padded = F.pad(mass, (0, blocks * self.block_size - length))
        block_attention = padded.reshape(blocks, self.block_size).sum(dim=1)
        h = self.rms(x, self.norm.weight)
        logits = torch.matmul(h, self.lm_head_t).reshape(-1)
        return logits, torch.stack(new, dim=1), block_attention


class _Prefill(nn.Module):
    def __init__(self, core: ZeosQwen2) -> None:
        super().__init__()
        self.core = core

    def forward(self, input_ids, past_kv, key_mask):  # pyright: ignore[reportMissingParameterType]
        return self.core.prefill(input_ids, past_kv, key_mask)


class _Decode(nn.Module):
    def __init__(self, core: ZeosQwen2) -> None:
        super().__init__()
        self.core = core

    def forward(self, input_ids, past_kv, allowed_blocks):  # pyright: ignore[reportMissingParameterType]
        return self.core.decode(input_ids, past_kv, allowed_blocks)


def _rope_theta(cfg: Any) -> float:
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    if theta is None:
        raise ValueError("the model config names no rope_theta")
    return float(theta)


# -- checks ----------------------------------------------------------------------


def check_against_reference(hf: Any, core: ZeosQwen2, ids: list[int]) -> float:
    """Largest logit difference between the library's forward pass and ours, after a
    prefill of all but the last token and one decode step over everything."""
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0, -1].float()
        empty = _empty_kv(core)
        kv = core.prefill(torch.tensor(ids[:-1]), empty, torch.ones(len(ids) - 1, dtype=torch.bool))
        blocks = -(-len(ids) // core.block_size)
        got, _, attention = core.decode(
            torch.tensor(ids[-1:]), kv, torch.ones(blocks, dtype=torch.bool)
        )
    if abs(float(attention.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"attention sums to {float(attention.sum())}, not 1")
    return float((want - got).abs().max())


def _empty_kv(core: ZeosQwen2) -> torch.Tensor:
    return torch.zeros(0, core.layers_n, 2, core.kv_heads, core.head_dim)


def check_masked_block(decode_path: Path, meta: dict[str, Any], ids: list[int]) -> None:
    """Run the exported decode graph under ONNX Runtime CPU with one block masked, and
    hold it to the units ``DecodeResult.attention`` declares: zero on the masked block,
    a sum of one over the rest. ``tests/test_exported_graph.py`` repeats this."""
    import onnxruntime as ort

    prefill = ort.InferenceSession(
        str(decode_path.with_name("prefill.onnx")), providers=["CPUExecutionProvider"]
    )
    decode = ort.InferenceSession(str(decode_path), providers=["CPUExecutionProvider"])
    empty = np.zeros(
        (0, meta["numLayers"], 2, meta["numKvHeads"], meta["headDim"]), dtype=np.float32
    )
    past = ids[:-1]
    (kv,) = prefill.run(
        None,
        {
            "input_ids": np.array(past, dtype=np.int64),
            "past_kv": empty,
            "key_mask": np.ones(len(past), dtype=np.bool_),
        },
    )
    blocks = -(-len(ids) // meta["blockSize"])
    allowed = np.ones(blocks, dtype=np.bool_)
    masked = blocks // 2
    allowed[masked] = False
    _, _, attention = decode.run(
        None,
        {"input_ids": np.array(ids[-1:], dtype=np.int64), "past_kv": kv, "allowed_blocks": allowed},
    )
    if attention[masked] != 0.0:
        raise AssertionError(f"masked block {masked} received {attention[masked]}")
    if abs(float(attention.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"attention sums to {float(attention.sum())}")


# -- export ------------------------------------------------------------------------


def export_graphs(core: ZeosQwen2, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    kv = torch.randn(5, core.layers_n, 2, core.kv_heads, core.head_dim)
    prefill_path = out / "prefill.onnx"
    decode_path = out / "decode.onnx"
    with torch.no_grad():
        torch.onnx.export(
            _Prefill(core),
            (torch.tensor([1, 2, 3]), kv, torch.ones(8, dtype=torch.bool)),
            str(prefill_path),
            dynamo=False,
            opset_version=OPSET,
            input_names=["input_ids", "past_kv", "key_mask"],
            output_names=["new_kv"],
            dynamic_axes={
                "input_ids": {0: "new"},
                "past_kv": {0: "past"},
                "key_mask": {0: "total"},
                "new_kv": {0: "new"},
            },
        )
        blocks = -(-6 // core.block_size)
        torch.onnx.export(
            _Decode(core),
            (torch.tensor([1]), kv, torch.ones(blocks, dtype=torch.bool)),
            str(decode_path),
            dynamo=False,
            opset_version=OPSET,
            input_names=["input_ids", "past_kv", "allowed_blocks"],
            output_names=["logits", "new_kv", "block_attention"],
            dynamic_axes={
                "past_kv": {0: "past"},
                "allowed_blocks": {0: "blocks"},
                "block_attention": {0: "blocks"},
            },
        )
    return prefill_path, decode_path


def quantise(path: Path, quant: str) -> onnx.ModelProto:
    """Quantise every matmul with a constant weight, in an operator ORT Web runs."""
    model = onnx.load(str(path), load_external_data=True)
    if quant == "fp32":
        return model
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    if quant == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "int8.onnx"
            quantize_dynamic(
                path,
                target,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["MatMul"],
                use_external_data_format=True,
            )
            return onnx.load(str(target), load_external_data=True)
    bits, level = {"q4": (4, None), "q4a8": (4, 4), "q8": (8, None)}[quant]
    quantizer = MatMulNBitsQuantizer(
        model, bits=bits, block_size=32, is_symmetric=True, accuracy_level=level
    )
    quantizer.process()
    return quantizer.model.model


def write_shared(models: dict[str, onnx.ModelProto], out: Path, data_name: str) -> int:
    """Save the graphs with every large initialiser in one file both of them point at, so
    the browser downloads the weights once. Identical tensors are stored once."""
    data_path = out / data_name
    offsets: dict[str, tuple[int, int]] = {}
    with data_path.open("wb") as data:
        for name, model in models.items():
            for tensor in model.graph.initializer:
                raw = onnx.numpy_helper.to_array(tensor).tobytes()
                if len(raw) < SHARE_THRESHOLD:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in offsets:
                    pad = (-data.tell()) % 4096
                    data.write(b"\0" * pad)
                    offsets[digest] = (data.tell(), len(raw))
                    data.write(raw)
                offset, length = offsets[digest]
                tensor.ClearField("raw_data")
                for field in ("float_data", "int32_data", "int64_data", "double_data"):
                    tensor.ClearField(field)  # pyright: ignore[reportArgumentType]
                del tensor.external_data[:]
                tensor.data_location = onnx.TensorProto.EXTERNAL
                for key, value in (
                    ("location", data_name),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ):
                    entry = tensor.external_data.add()
                    entry.key = key
                    entry.value = value
            onnx.save(model, str(out / name))
    return data_path.stat().st_size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--quant",
        choices=("q4", "q4a8", "q8", "int8", "fp32"),
        default="int8",
        help=(
            "int8: per-channel int8 weights with dynamic int8 activations, the fast path on "
            "ONNX Runtime's WebAssembly backend (default); q4/q8: weight-only blocks "
            "(MatMulNBits), smaller but about fourteen times slower per decode step on ONNX "
            "Runtime Web 1.30's WebAssembly backend; q4a8: q4 "
            "with int8 compute; fp32: none"
        ),
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default models/<name>-zeos-<quant>)",
    )
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = args.model.split("/")[-1]
    source = DEMO / "models" / name
    snapshot_download(
        args.model,
        local_dir=source,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"],
    )
    out: Path = args.out or DEMO / "models" / f"{name}-zeos-{args.quant}"

    hf = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    core = ZeosQwen2(hf, block_size=args.block_size, int8_embedding=args.quant != "fp32")
    core.eval()

    sample = tokenizer("say 1; say 2; say 3; say 4;", add_special_tokens=False)["input_ids"]
    drift = check_against_reference(hf, core, sample)
    print(f"logit drift against transformers: {drift:.4g}")
    # The int8 embedding costs a little; anything larger is a wrong rewrite.
    if drift > (0.5 if core.int8_embedding else 1e-3):
        raise SystemExit(f"the rewrite disagrees with transformers by {drift}")

    with tempfile.TemporaryDirectory() as scratch:
        prefill_path, decode_path = export_graphs(core, Path(scratch))
        models = {
            "prefill.onnx": quantise(prefill_path, args.quant),
            "decode.onnx": quantise(decode_path, args.quant),
        }
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        weights = write_shared(models, out, "weights.bin")

    vocab = tokenizer.get_vocab()
    special = {t.content: i for i, t in tokenizer.added_tokens_decoder.items()}
    meta = {
        "model": args.model,
        "quant": args.quant,
        "blockSize": args.block_size,
        "numLayers": core.layers_n,
        "numHeads": core.heads,
        "numKvHeads": core.kv_heads,
        "headDim": core.head_dim,
        "logitsSize": int(hf.config.vocab_size),
        "tokenizerSize": len(vocab),
        "eosId": special["<|im_end|>"],
        "padId": special["<|endoftext|>"],
        "controlIds": [special["<|im_start|>"], special["<|im_end|>"], special["<|endoftext|>"]],
        "files": {},
    }
    for file in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / file, out / file)
    for file in sorted(p.name for p in out.iterdir() if p.name != "meta.json"):
        meta["files"][file] = {"bytes": (out / file).stat().st_size, "sha256": sha256(out / file)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    check_masked_block(out / "decode.onnx", meta, sample)
    print(f"wrote {out} ({weights / 1e6:.0f} MB of weights); masked-block check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
