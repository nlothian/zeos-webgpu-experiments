# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Export a Qwen language model as the ONNX graph the browser worker runs.

One graph, ``model.onnx``, runs a chunk of new tokens (one for a decode step, up to
``MAX_CHUNK`` for a prefill) after a job's cache and returns the logits at the last new
position, the cache for the new positions, and the last position's attention per KV
position. A per-position key mask is applied in every layer before anything mixes
positions, so a hidden position contributes nothing and, in the softmax layers, receives
exactly zero; the attention is averaged over those layers' heads inside the graph, so it
sums to one and only a vector of position masses leaves it.

Two families are written here rather than traced from ``transformers``: Qwen2 (every layer
softmax attention) and Qwen3.5 (``qwen3_5``: three Gated DeltaNet layers to each gated
softmax-attention layer). The cache is:

- ``past_kv`` -- keys and values of the softmax layers, position-major
  (``[positions, softmax layers, 2, kv heads, head dim]``), so appending, truncating and
  copying it are contiguous copies in JavaScript.
- ``state`` -- each Gated DeltaNet layer's recurrent state
  (``[linear layers, heads, key dim, value dim]``), a fixed size whatever the length.
- ``conv`` -- each Gated DeltaNet layer's last three pre-convolution projections
  (``[linear layers, 3, channels]``), the window its causal convolution reads.

The recurrent state is not per position, so it cannot be cut: the worker keeps snapshots
of ``state`` and ``conv`` and re-runs the tokens after the latest one at or before a cut
(see ``web/transformers_worker.js``). A hidden position is *skipped* by a DeltaNet layer:
its write gate and decay are zeroed, so the state carries nothing of it, and its
convolution taps are zeroed. The worker re-runs from a position whose visibility changed
so the state always matches the mask of the step that reads it.

``check_against_reference`` holds the rewrite to the library's own logits before
anything is exported.

Run it with the export dependency group (see the demo README)::

    uv sync --all-packages --group export
    uv run python demo/coop-count-web/export/export_model.py
"""

from __future__ import annotations

import argparse
import gc
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
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
#: The DeltaNet layers solve a unit lower-triangular system over each chunk: by repeated
#: squaring within blocks of ``SOLVE_BLOCK`` positions (exact for ``2 ** SOLVE_STEPS``
#: of them, and, in float32, accurate to 4e-5 at 16 where 32 already drifts by 1.5 in
#: the logits), and by forward substitution between blocks.
SOLVE_STEPS = 4
SOLVE_BLOCK = 2**SOLVE_STEPS
#: The most new positions one run of the graph takes. Also bounded by the int8 export:
#: ONNX Runtime's dynamic quantisation gives each matmul's activations one scale for the
#: whole chunk, and over 64 positions of a coop-count prompt that flips the model's
#: choice after a wake from `say` to `read` (full precision, and chunks of 16 or fewer,
#: keep `say`).
MAX_CHUNK = SOLVE_BLOCK
#: The window of a DeltaNet layer's causal convolution, minus the position itself.
CONV_WINDOW = 3
OPSET = 17
#: Initialisers at least this large go to the shared weights files; smaller ones stay
#: inline in the graph.
SHARE_THRESHOLD = 1024
#: The most bytes one weights file holds.
SHARD_BYTES = 1 << 30


def _text_config(cfg: Any) -> Any:
    return getattr(cfg, "text_config", None) or cfg


class ZeosQwen(nn.Module):
    """A Qwen decoder over the cache described in the module docstring."""

    def __init__(self, hf: Any, *, int8_embedding: bool) -> None:
        super().__init__()
        cfg = _text_config(hf.config)
        self.family = str(cfg.model_type)
        self.hybrid = self.family.startswith("qwen3_5")
        if not (self.hybrid or self.family == "qwen2"):
            raise SystemExit(f"model_type {self.family!r} is neither qwen2 nor qwen3_5")
        layers_n = int(cfg.num_hidden_layers)
        types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * layers_n)
        if any(t not in ("full_attention", "linear_attention") for t in types):
            raise SystemExit(f"unknown layer types {sorted(set(types))}")
        if not self.hybrid and any(t != "full_attention" for t in types):
            raise SystemExit("a qwen2 model with non-softmax layers")
        self.types = types
        self.layers_n = layers_n
        #: index of each layer among its kind
        self.slot = [types[: i + 1].count(t) - 1 for i, t in enumerate(types)]
        self.full_n = types.count("full_attention")
        self.linear_n = types.count("linear_attention")
        self.heads = int(cfg.num_attention_heads)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // self.heads)
        self.group = self.heads // self.kv_heads
        self.eps = float(cfg.rms_norm_eps)
        rope = dict(getattr(cfg, "rope_parameters", None) or {})
        if rope.get("rope_type", "default") != "default":
            raise SystemExit(f"rope_type {rope['rope_type']!r} is not supported")
        self.rotary_dim = int(self.head_dim * float(rope.get("partial_rotary_factor", 1.0)))
        rope_theta = _rope_theta(cfg)
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if self.linear_n:
            self.lin_k_heads = int(cfg.linear_num_key_heads)
            self.lin_v_heads = int(cfg.linear_num_value_heads)
            self.lin_k_dim = int(cfg.linear_key_head_dim)
            self.lin_v_dim = int(cfg.linear_value_head_dim)
            if int(cfg.linear_conv_kernel_dim) != CONV_WINDOW + 1:
                raise SystemExit(
                    f"conv kernel {cfg.linear_conv_kernel_dim} is not {CONV_WINDOW + 1}"
                )
        else:
            self.lin_k_heads = self.lin_v_heads = 1
            self.lin_k_dim = self.lin_v_dim = 1
        self.conv_dim = 2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim

        base = hf.model
        weight = base.embed_tokens.weight.detach().float()
        self.int8_embedding = int8_embedding
        if int8_embedding:
            # Symmetric per-row int8: the table is a quarter of the model and a lookup, so
            # quantising it here shrinks the download without touching a matmul.
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("embed_q", torch.round(weight / scale).to(torch.int8))
            self.register_buffer("embed_scale", scale)
        else:
            self.register_buffer("embed_w", weight)
        self.blocks = base.layers
        # Qwen3.5's RMSNorm scales by (1 + weight); fold that into one buffer per norm so
        # the graph does the same multiply for both families.
        self.final_norm = nn.Parameter(self._norm_weight(base.norm.weight), requires_grad=False)
        # Tied or not, the output projection is its own initialiser, so the weight-only
        # quantiser treats it like every other matmul.
        self.lm_head_t = nn.Parameter(hf.lm_head.weight.detach().float().t().contiguous())

    def _norm_weight(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight.detach().float()
        return 1.0 + weight if self.hybrid else weight

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

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotary embedding on the first ``rotary_dim`` channels of each head."""
        r = self.rotary_dim
        x_rot, x_pass = x[..., :r], x[..., r:]
        half = r // 2
        rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
        x_rot = x_rot * cos + rotated * sin
        if r == x.shape[-1]:
            return x_rot
        return torch.cat((x_rot, x_pass), dim=-1)

    def mlp(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        h = self.rms(x, self._norm_weight(layer.post_attention_layernorm.weight))
        mlp = layer.mlp
        return x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))

    # -- softmax attention -------------------------------------------------------

    def attention(
        self,
        layer: Any,
        h: torch.Tensor,
        past_kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Output, this chunk's KV ``[P, 2, kv, d]`` and the last query's probabilities
        summed over heads ``[T]``. ``bias`` is ``[P, T]``: 0 where visible, -inf where not."""
        p = h.shape[0]
        attn = layer.self_attn
        d = self.head_dim
        if self.hybrid:
            q, gate = torch.chunk(attn.q_proj(h).reshape(p, self.heads, 2 * d), 2, dim=-1)
            gate = gate.reshape(p, self.heads * d)
            q = self.rms(q, self._norm_weight(attn.q_norm.weight))
            k = self.rms(
                attn.k_proj(h).reshape(p, self.kv_heads, d), self._norm_weight(attn.k_norm.weight)
            )
        else:
            q = attn.q_proj(h).reshape(p, self.heads, d)
            k = attn.k_proj(h).reshape(p, self.kv_heads, d)
            gate = None
        v = attn.v_proj(h).reshape(p, self.kv_heads, d)
        q = self.rotate(q, cos[:, None, :], sin[:, None, :])
        k = self.rotate(k, cos[:, None, :], sin[:, None, :])
        k_all = torch.cat((past_kv[:, 0], k), dim=0)  # [T, kv, d]
        v_all = torch.cat((past_kv[:, 1], v), dim=0)
        # Head h = kv * group + g, the ordering the library's repeat_kv produces.
        qg = q.reshape(p, self.kv_heads, self.group, d).permute(1, 2, 0, 3)  # [kv, g, P, d]
        scores = torch.matmul(qg, k_all.permute(1, 2, 0)[:, None]) / math.sqrt(d)  # [kv, g, P, T]
        # Before the softmax and in every layer: a hidden key is never attended.
        probs = torch.softmax(scores + bias, dim=-1)
        out = torch.matmul(probs, v_all.transpose(0, 1)[:, None])  # [kv, g, P, d]
        out = out.permute(2, 0, 1, 3).reshape(p, self.heads * d)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        mass = probs[:, :, -1, :].sum(dim=(0, 1))
        return attn.o_proj(out), torch.stack((k, v), dim=1), mass

    # -- Gated DeltaNet ----------------------------------------------------------

    def delta(
        self,
        layer: Any,
        h: torch.Tensor,
        state: torch.Tensor,
        window: torch.Tensor,
        window_mask: torch.Tensor,
        new_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Output, final state ``[Hv, dk, dv]`` and the new convolution window ``[3, C]``.

        ``state`` is the state after every visible earlier position; ``window`` holds the
        raw projections of the three positions before the chunk and ``window_mask`` their
        visibility; ``new_mask`` the visibility of the chunk's own positions. A hidden
        position is skipped: write gate (beta) and decay (g) zero, convolution taps zero.

        The recurrence, per head, is S_t = a_t S_{t-1} + k_t u_t^T with the pseudo-value
        u_t = b_t (v_t - a_t S_{t-1}^T k_t) and output S_t^T q_t. Over a chunk the
        pseudo-values solve a unit lower-triangular system (I + L) U = B, here by
        ``solve``, so the whole chunk is a fixed sequence of matmuls.
        """
        la = layer.linear_attn
        p = h.shape[0]
        hk, hv, dk, dv = self.lin_k_heads, self.lin_v_heads, self.lin_k_dim, self.lin_v_dim
        pre = la.in_proj_qkv(h)  # [P, C]
        x = torch.cat((window, pre), dim=0)  # [3 + P, C]
        m = torch.cat((window_mask, new_mask), dim=0)[:, None]  # [3 + P, 1]
        xm = x * m
        w = la.conv1d.weight[:, 0, :]  # [C, 4]
        total = x.shape[0]
        conv = xm[CONV_WINDOW:] * w[:, CONV_WINDOW]
        for j in range(CONV_WINDOW):
            conv = conv + xm[j : total - CONV_WINDOW + j] * w[:, j]
        # A position always reads its own projection, hidden or not, as it always has its
        # own embedding in the residual stream.
        conv = conv + (x[CONV_WINDOW:] - xm[CONV_WINDOW:]) * w[:, CONV_WINDOW]
        conv = F.silu(conv)
        key_dim = hk * dk
        q = conv[:, :key_dim].reshape(p, hk, dk)
        k = conv[:, key_dim : 2 * key_dim].reshape(p, hk, dk)
        v = conv[:, 2 * key_dim :].reshape(p, hv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        if hv // hk > 1:
            q = q.repeat_interleave(hv // hk, dim=1)
            k = k.repeat_interleave(hv // hk, dim=1)
        q = q / math.sqrt(dk)
        visible = new_mask[:, None]
        beta = torch.sigmoid(la.in_proj_b(h)) * visible  # [P, Hv]
        g = -la.A_log.float().exp() * F.softplus(la.in_proj_a(h) + la.dt_bias) * visible

        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [Hv, P, d]
        big_g = torch.cumsum(g, dim=0).transpose(0, 1)  # [Hv, P], inclusive
        idx = torch.arange(p, dtype=torch.int64)
        lower = idx[:, None] >= idx[None, :]
        strict = idx[:, None] > idx[None, :]
        diff = big_g[:, :, None] - big_g[:, None, :]
        decay = torch.exp(torch.where(lower, diff, torch.full((), float("-inf"))))  # [Hv,P,P]
        e_g = torch.exp(big_g)[..., None]  # [Hv, P, 1]
        betah = beta.transpose(0, 1)[..., None]  # [Hv, P, 1]
        kk = torch.matmul(kh, kh.transpose(1, 2))
        lmat = betah * torch.where(strict, decay, torch.zeros(())) * kk
        rhs = betah * (vh - e_g * torch.matmul(kh, state))  # B
        u = self.solve(lmat, rhs)
        qk = torch.matmul(qh, kh.transpose(1, 2)) * decay
        out = e_g * torch.matmul(qh, state) + torch.matmul(qk, u)  # [Hv, P, dv]
        last = big_g[:, -1:]  # [Hv, 1]
        tail = torch.exp(last - big_g)[..., None]  # [Hv, P, 1]
        new_state = torch.exp(last)[..., None] * state + torch.matmul(
            (kh * tail).transpose(1, 2), u
        )
        out = out.transpose(0, 1)  # [P, Hv, dv]
        z = la.in_proj_z(h).reshape(p, hv, dv)
        normed = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps) * la.norm.weight
        normed = normed * F.silu(z)
        return la.out_proj(normed.reshape(p, hv * dv)), new_state, x[-CONV_WINDOW:]

    @staticmethod
    def solve(lmat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """``(I + lmat)^-1 rhs`` for strictly lower-triangular ``lmat`` ``[H, P, P]``,
        ``P <= MAX_CHUNK``. The chunk is padded to ``MAX_CHUNK`` (zero rows solve to
        zero), each diagonal block is inverted as the finite Neumann series
        (I - L)(I + L^2)(I + L^4)..., and the blocks are chained by forward
        substitution."""
        p = lmat.shape[1]
        pad = MAX_CHUNK - p
        lp = F.pad(lmat, (0, pad, 0, pad))
        bp = F.pad(rhs, (0, 0, 0, pad))
        solved: list[torch.Tensor] = []
        for i in range(MAX_CHUNK // SOLVE_BLOCK):
            rows = slice(i * SOLVE_BLOCK, (i + 1) * SOLVE_BLOCK)
            x = bp[:, rows]
            for j, done in enumerate(solved):
                x = x - torch.matmul(lp[:, rows, j * SOLVE_BLOCK : (j + 1) * SOLVE_BLOCK], done)
            neg = -lp[:, rows, rows]
            for step in range(SOLVE_STEPS):
                x = x + torch.matmul(neg, x)
                if step + 1 < SOLVE_STEPS:
                    neg = torch.matmul(neg, neg)
            solved.append(x)
        return torch.cat(solved, dim=1)[:, :p]

    # -- the graph ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: torch.Tensor,
        state: torch.Tensor,
        conv: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ``input_ids`` after the cache. ``key_mask[j]`` false hides position ``j``
        (past or new) from every position, on top of the causal mask; a hidden position
        sees itself only when nothing else is visible to it, so its softmax is defined.
        Returns the logits at the last position, the new positions'
        KV, the state and window after them, and the last position's attention over
        every position, averaged over the softmax layers' heads."""
        n = past_kv.shape[0]
        p = input_ids.shape[0]
        positions = torch.arange(p, dtype=torch.int64) + n
        cos, sin = self.rope(positions)
        keys = torch.arange(n + p, dtype=torch.int64)
        causal = keys[None, :] <= positions[:, None]
        own = keys[None, :] == positions[:, None]
        visible = causal & key_mask[None, :]
        visible = visible | (own & ~visible.any(dim=-1, keepdim=True))
        bias = torch.where(visible, torch.zeros(()), torch.full((), float("-inf")))
        maskf = key_mask.float()
        new_mask = maskf[n:]
        # The three positions before the chunk; before position 0 the window is zeros.
        window_mask = F.pad(maskf, (CONV_WINDOW, 0))[n : n + CONV_WINDOW]
        x = self.embed(input_ids)
        mass = torch.zeros(n + p, dtype=torch.float32)
        new_kv: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        windows: list[torch.Tensor] = []
        for i, layer in enumerate(self.blocks):
            h = self.rms(x, self._norm_weight(layer.input_layernorm.weight))
            s = self.slot[i]
            if self.types[i] == "full_attention":
                out, kv, layer_mass = self.attention(layer, h, past_kv[:, s], cos, sin, bias)
                new_kv.append(kv)
                mass = mass + layer_mass
            else:
                out, st, win = self.delta(layer, h, state[s], conv[s], window_mask, new_mask)
                states.append(st)
                windows.append(win)
            x = self.mlp(layer, x + out)
        mass = mass / float(self.full_n * self.heads)
        last = self.rms(x[-1:], self.final_norm)
        logits = torch.matmul(last, self.lm_head_t).reshape(-1)
        state_out = torch.stack(states) if states else state
        conv_out = torch.stack(windows) if windows else conv
        return logits, torch.stack(new_kv, dim=1), state_out, conv_out, mass

    # -- empties -----------------------------------------------------------------

    def empty_kv(self, n: int = 0) -> torch.Tensor:
        return torch.zeros(n, self.full_n, 2, self.kv_heads, self.head_dim)

    def empty_state(self) -> torch.Tensor:
        return torch.zeros(self.linear_n, self.lin_v_heads, self.lin_k_dim, self.lin_v_dim)

    def empty_conv(self) -> torch.Tensor:
        return torch.zeros(self.linear_n, CONV_WINDOW, self.conv_dim)

    def run(
        self, ids: list[int], *, mask: list[bool] | None = None, chunk: int = MAX_CHUNK
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits and attention after running ``ids`` from empty in chunks, as the worker
        does: every chunk but the last is a prefill, the last token a decode step."""
        mask_t = torch.ones(len(ids), dtype=torch.bool) if mask is None else torch.tensor(mask)
        kv, state, conv = self.empty_kv(), self.empty_state(), self.empty_conv()
        done = 0
        target = len(ids) - 1
        while done < target:
            count = min(chunk, target - done)
            _, new, state, conv, _ = self(
                torch.tensor(ids[done : done + count]), kv, state, conv, mask_t[: done + count]
            )
            kv = torch.cat((kv, new), dim=0)
            done += count
        logits, _, _, _, attention = self(torch.tensor(ids[-1:]), kv, state, conv, mask_t)
        return logits, attention


def _rope_theta(cfg: Any) -> float:
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    if theta is None:
        raise ValueError("the model config names no rope_theta")
    return float(theta)


# -- checks ----------------------------------------------------------------------


def check_against_reference(hf: Any, core: ZeosQwen, ids: list[int], chunk: int) -> float:
    """Largest logit difference between the library's forward pass and ours, running
    ``ids`` in chunks of ``chunk`` and the last token as a decode step."""
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0, -1].float()
        got, attention = core.run(ids, chunk=chunk)
    if abs(float(attention.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"attention sums to {float(attention.sum())}, not 1")
    return float((want - got).abs().max())


def check_masked_position(model_path: Path, meta: dict[str, Any], ids: list[int]) -> None:
    """Run the exported graph under ONNX Runtime CPU with one position hidden, and hold
    it to the units ``DecodeResult.attention`` declares: zero on the hidden position, a
    sum of one over the rest, and different logits. ``tests/test_exported_graph.py``
    repeats this."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    masked = len(ids) // 2

    def last_step(hidden: int | None) -> tuple[np.ndarray, np.ndarray]:
        mask = np.ones(len(ids), dtype=np.bool_)
        if hidden is not None:
            mask[hidden] = False
        feed = empty_feed(meta)
        done = 0
        while done < len(ids) - 1:
            count = min(MAX_CHUNK, len(ids) - 1 - done)
            _, new, feed["state"], feed["conv"], _ = session.run(
                None,
                {
                    **feed,
                    "input_ids": np.array(ids[done : done + count], np.int64),
                    "key_mask": mask[: done + count],
                },
            )
            feed["past_kv"] = np.concatenate((feed["past_kv"], new))
            done += count
        logits, _, _, _, attention = session.run(
            None, {**feed, "input_ids": np.array(ids[-1:], np.int64), "key_mask": mask}
        )
        return logits, attention

    open_logits, open_attention = last_step(None)
    logits, attention = last_step(masked)
    if attention[masked] != 0.0:
        raise AssertionError(f"hidden position {masked} received {attention[masked]}")
    for values in (attention, open_attention):
        if abs(float(values.sum()) - 1.0) > 1e-4:
            raise AssertionError(f"attention sums to {float(values.sum())}")
    if float(np.abs(open_logits - logits).max()) < 1e-3:
        raise AssertionError("hiding a position did not change the logits")


def empty_feed(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """The cache of an empty context, as ``meta.json`` shapes it."""
    return {
        "past_kv": np.zeros((0, *meta["kvShape"]), dtype=np.float32),
        "state": np.zeros(meta["stateShape"], dtype=np.float32),
        "conv": np.zeros(meta["convShape"], dtype=np.float32),
    }


# -- export ------------------------------------------------------------------------


def export_graph(core: ZeosQwen, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            core,
            (
                torch.tensor([1, 2, 3]),
                torch.randn(5, core.full_n, 2, core.kv_heads, core.head_dim),
                torch.randn(*core.empty_state().shape) * 0.01,
                torch.randn(*core.empty_conv().shape),
                torch.ones(8, dtype=torch.bool),
            ),
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=["input_ids", "past_kv", "state", "conv", "key_mask"],
            output_names=["logits", "new_kv", "state_out", "conv_out", "attention"],
            dynamic_axes={
                "input_ids": {0: "new"},
                "past_kv": {0: "past"},
                "key_mask": {0: "total"},
                "new_kv": {0: "new"},
                "attention": {0: "total"},
            },
        )
    return path


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


def write_shared(
    models: dict[str, onnx.ModelProto], out: Path, prefix: str = "weights"
) -> list[str]:
    """Save the graphs with every large initialiser in shared data files they point at,
    so the browser downloads the weights once. Identical tensors are stored once. A new
    file starts when one passes ``SHARD_BYTES``: Node reads no file over 2 GiB whole,
    and a browser holds smaller buffers more readily. Returns the data files' names."""
    offsets: dict[str, tuple[str, int, int]] = {}
    names: list[str] = []
    data = None
    try:
        for name, model in models.items():
            for tensor in model.graph.initializer:
                raw = onnx.numpy_helper.to_array(tensor).tobytes()
                if len(raw) < SHARE_THRESHOLD:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in offsets:
                    if data is None or data.tell() + len(raw) > SHARD_BYTES:
                        if data is not None:
                            data.close()
                        names.append(f"{prefix}-{len(names)}.bin")
                        data = (out / names[-1]).open("wb")
                    pad = (-data.tell()) % 4096
                    data.write(b"\0" * pad)
                    offsets[digest] = (names[-1], data.tell(), len(raw))
                    data.write(raw)
                location, offset, length = offsets[digest]
                tensor.ClearField("raw_data")
                for field in ("float_data", "int32_data", "int64_data", "double_data"):
                    tensor.ClearField(field)  # pyright: ignore[reportArgumentType]
                del tensor.external_data[:]
                tensor.data_location = onnx.TensorProto.EXTERNAL
                for key, value in (
                    ("location", location),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ):
                    entry = tensor.external_data.add()
                    entry.key = key
                    entry.value = value
            onnx.save(model, str(out / name))
    finally:
        if data is not None:
            data.close()
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Text the reference check runs: the command language, then the chat framing.
SAMPLES = (
    "say 1; say 2; say 3; say 4;",
    "<|im_start|>user\nYou are counter-a. Count from 1 to 10, one number per say command, "
    "then write your progress to count.progress_a and wake your peer with a message on "
    "count.a2b. Wait for count.b2a before you count again.<|im_end|>\n"
    "<|im_start|>assistant\nsay 1; say 2; say 3; say 4; say 5; say 6; say 7;",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--quant",
        choices=("q4", "q4a8", "q8", "int8", "fp32"),
        default="q4",
        help=(
            "q4/q8: weight-only blocks (MatMulNBits), smaller, and the form ONNX "
            "Runtime's WebGPU backend runs, but many times slower per decode step on its "
            "WebAssembly backend (q4 is the default); int8: per-channel int8 weights with "
            "dynamic int8 activations, the fast path on ONNX Runtime's WebAssembly "
            "backend; q4a8: q4 with int8 compute; fp32: none"
        ),
    )
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
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "merges.txt",
            "vocab.json",
            "LICENSE",
            "*.jinja",
        ],
    )
    out: Path = args.out or DEMO / "models" / f"{name}-zeos-{args.quant}"

    hf = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    core = ZeosQwen(hf, int8_embedding=args.quant != "fp32")
    core.eval()

    # The int8 embedding costs a little; anything larger is a wrong rewrite.
    limit = 0.5 if core.int8_embedding else 1e-3
    for text in SAMPLES:
        sample = tokenizer(text, add_special_tokens=False)["input_ids"]
        for chunk in (1, 7, MAX_CHUNK):
            drift = check_against_reference(hf, core, sample, chunk)
            print(
                f"logit drift against transformers ({len(sample)} tokens, chunk {chunk}): {drift:.4g}"
            )
            if drift > limit:
                raise SystemExit(f"the rewrite disagrees with transformers by {drift}")

    vocab = tokenizer.get_vocab()
    special = {t.content: i for i, t in tokenizer.added_tokens_decoder.items()}
    meta = {
        "model": args.model,
        "family": core.family,
        "quant": args.quant,
        "blockSize": 1,
        "maxChunk": MAX_CHUNK,
        "numLayers": core.layers_n,
        "layerTypes": core.types,
        "numHeads": core.heads,
        "numKvHeads": core.kv_heads,
        "headDim": core.head_dim,
        # past_kv is [positions, *kvShape]; state and conv have fixed shapes.
        "kvShape": [core.full_n, 2, core.kv_heads, core.head_dim],
        "stateShape": list(core.empty_state().shape),
        "convShape": list(core.empty_conv().shape),
        "logitsSize": int(core.lm_head_t.shape[1]),
    }

    with tempfile.TemporaryDirectory() as scratch:
        model_path = export_graph(core, Path(scratch))
        # The quantiser holds the whole float32 graph in memory; with the PyTorch model
        # still resident as well, a 4B model needs twice its float32 size at once.
        del core, hf
        gc.collect()
        models = {"model.onnx": quantise(model_path, args.quant)}
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        shards = write_shared(models, out)
        del models

    meta |= {
        "weights": shards,
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

    sample = tokenizer(SAMPLES[1], add_special_tokens=False)["input_ids"]
    check_masked_position(out / "model.onnx", meta, sample)
    weights = sum((out / shard).stat().st_size for shard in shards)
    print(f"wrote {out} ({weights / 1e6:.0f} MB of weights); masked-position check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
