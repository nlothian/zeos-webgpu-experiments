# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The OPT+ZEOS decoder written by ``export/opt_zeos_surgery.py``, under ONNX Runtime
CPU, which has kernels for every fused operator it keeps (``LinearAttention``,
``CausalConvWithState``, ``GroupQueryAttention``, ``RotaryEmbedding``,
``MatMulNBits``).

Prefilled in chunks of 1, 7, 16 and 512 with the cache carried between runs, as the
worker carries it, then one decode step:

- with every position visible, its logits are the unmodified ``-OPT`` graph's, up to
  float16 rounding, and agree with ``transformers``' Qwen3.5-4B as closely as the
  ``-OPT`` graph does (4-bit weights against bf16);
- a hidden position receives exactly zero attention, the rest sums to one, and hiding it
  moves the logits the way ``export_model.ZeosQwen``'s mask moves them.

Needs the surgery's output (``models/Qwen3.5-4B-ZEOS-OPT``) and the ``export``
dependency group. The comparisons against ``-OPT`` need its export at
``models/Qwen3.5-4B-ONNX-OPT`` and those against ``transformers`` the bf16 weights at
``models/Qwen3.5-4B``; each skips without them. The reference logits are computed once
(about 20 GB of memory, a few minutes) and cached under ``models/.reference/``.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

DEMO = Path(__file__).resolve().parent.parent
MODELS = DEMO / "models"
MODEL = Path(os.environ.get("ZEOS_OPT_MODEL_DIR", MODELS / "Qwen3.5-4B-ZEOS-OPT"))
OPT = Path(os.environ.get("ZEOS_OPT_SRC_DIR", MODELS / "Qwen3.5-4B-ONNX-OPT"))
HF = Path(os.environ.get("ZEOS_OPT_HF_DIR", MODELS / "Qwen3.5-4B"))
CACHE = MODELS / ".reference"

ort = pytest.importorskip("onnxruntime", reason="needs the export dependency group")
np = pytest.importorskip("numpy", reason="needs the export dependency group")
tokenizers = pytest.importorskip("tokenizers", reason="needs the export dependency group")

pytestmark = pytest.mark.skipif(
    not (MODEL / "meta.json").is_file(),
    reason=f"no export at {MODEL}; run export/opt_zeos_surgery.py",
)

#: A chat turn short enough to run one token at a time.
SHORT = (
    "<|im_start|>user\nYou are counter-a. Count from 1 to 10, one number per say command, "
    "then write your progress to count.progress_a and wake your peer with a message on "
    "count.a2b. Wait for count.b2a before you count again.<|im_end|>\n"
    "<|im_start|>assistant\nsay 1; say 2; say 3; say 4; say 5; say 6; say 7;"
)
#: Prose longer than one 512-position chunk, so that chunk carries a cache too.
LONG = (
    "<|im_start|>system\nYou are a careful assistant who answers questions about a small "
    "railway timetable. Read the notes below before you answer.<|im_end|>\n"
    "<|im_start|>user\nHere are my notes on the branch line.\n\n"
    "The branch line leaves the main station at the north end of the town, crosses the "
    "river on an iron bridge built in 1887, and climbs through three tunnels to the "
    "village of Ashby Cross. Trains run every forty minutes on weekdays, starting at "
    "six fifteen in the morning, and every hour on Saturdays. There is no service on "
    "Sundays except during the summer festival, when a steam train runs twice a day. "
    "The first train of the day waits at the junction for the connection from the coast, "
    "which is often late in winter, so passengers for the eight o'clock school bus at "
    "Ashby Cross are advised to take the second train instead. The station at Millford "
    "Halt is a request stop: tell the guard before the train leaves the bridge, or wave "
    "from the platform. Bicycles are carried free of charge, but only four on each train, "
    "and dogs travel free if they stay on the floor. The ticket office at the main "
    "station opens at a quarter to six and closes at nine in the evening; outside those "
    "hours tickets are bought from the guard, who takes cash and cards. Return tickets "
    "are valid for a month, and a weekly season ticket costs the same as four returns. "
    "Children under five travel free, and children under sixteen pay half the adult "
    "fare. The last train back from Ashby Cross leaves at ten past eleven on weekdays and "
    "at a quarter to ten on Saturdays. Engineering work closes the line for two weeks "
    "every February, when a replacement bus runs from the station forecourt and takes "
    "about twenty minutes longer than the train. The bus does not stop at Millford Halt; "
    "passengers for the halt should use the stop on the main road, a short walk away. "
    "During the summer festival the steam train is very popular and seats must be "
    "booked a week ahead; it leaves the main station at half past ten and at two in the "
    "afternoon, and returns from Ashby Cross an hour and a half later each time. Lost "
    "property is kept at the main station for three months. Complaints and compliments "
    "go to the branch line manager, whose office is above the ticket office.\n\n"
    "Question one: I want to reach Ashby Cross in time for the eight o'clock school bus "
    "on a Tuesday in January. Which train should I take, and why?\n"
    "Question two: I have two children, aged four and twelve, and we want return tickets "
    "on a Saturday in summer. How much do we pay compared with one adult return?\n"
    "Question three: I am travelling in February to Millford Halt with my dog and my "
    "bicycle. What should I expect, and is there anything I should do differently?\n"
    "Question four: what is the latest I can leave Ashby Cross on a Saturday to get "
    "back to the main station, and where would I buy a ticket at that hour?\n"
    "Please answer each question in one or two sentences, citing the notes."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nQuestion one: take the"
)

#: A fact the last position must recall: hiding the word moves the answer a long way.
SECRET = (
    "<|im_start|>user\nRemember this for later: the password for the shed is tangerine. "
    "The weather today is mild and the garden needs watering.<|im_end|>\n"
    "<|im_start|>assistant\nNoted.<|im_end|>\n<|im_start|>user\nWhat is the password for "
    "the shed? Answer with the word only.<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
#: The positions of " tangerine" in SECRET.
SECRET_HIDDEN = (14, 15, 16)


def _ids(text: str) -> list[int]:
    tok = tokenizers.Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    return list(tok.encode(text, add_special_tokens=False).ids)


class Decoder:
    """One of the two decoders, driven as the worker drives it: the embedding graph,
    then the decoder over a chunk with the cache of the chunks before it."""

    def __init__(self, directory: Path, decoder: str, *, zeos: bool) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.embed = ort.InferenceSession(
            str(directory / "onnx" / "embed_tokens_q4f16.onnx"), options
        )
        self.session = ort.InferenceSession(str(directory / "onnx" / decoder), options)
        self.zeos = zeos
        self.outputs = [o.name for o in self.session.get_outputs()]

    def empty(self) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        for value in self.session.get_inputs():
            if value.name.startswith("past_"):
                shape = [d if isinstance(d, int) else 0 for d in value.shape]
                shape[0] = 1
                cache[value.name] = np.zeros(shape, np.float16)
        return cache

    def step(self, cache: dict[str, Any], ids: list[int], start: int, mask: Any) -> dict[str, Any]:
        """Run ``ids`` at positions ``start...``; ``mask`` covers every position so far."""
        embeds = self.embed.run(None, {"input_ids": np.array([ids], np.int64)})[0]
        positions = np.arange(start, start + len(ids), dtype=np.int64)
        feed = {
            **cache,
            "inputs_embeds": embeds,
            "position_ids": np.broadcast_to(positions, (3, 1, len(ids))).copy(),
            "num_logits_to_keep": np.array(1, np.int64),
        }
        if self.zeos:
            feed["key_mask"] = np.asarray(mask, np.bool_)[None, : start + len(ids)]
        else:
            feed["attention_mask"] = np.ones((1, start + len(ids)), np.int64)
        result = dict(zip(self.outputs, self.session.run(None, feed), strict=True))
        for name, value in result.items():
            if name.startswith("present"):
                past = name.replace("present_", "past_").replace("present.", "past_key_values.")
                cache[past] = value
        return result

    def last_step(self, ids: list[int], chunk: int, mask: Any = None) -> dict[str, Any]:
        """Prefill all but the last id in chunks of ``chunk``, then one decode step."""
        mask = np.ones(len(ids), np.bool_) if mask is None else mask
        cache = self.empty()
        done = 0
        while done < len(ids) - 1:
            count = min(chunk, len(ids) - 1 - done)
            self.step(cache, ids[done : done + count], done, mask)
            done += count
        return self.step(cache, ids[-1:], done, mask)


def _logits(result: dict[str, Any]) -> Any:
    return result["logits"][0, -1].astype(np.float64)


def _kl(p_logits: Any, q_logits: Any) -> float:
    """KL(p || q) over the softmax of two logit vectors, in nats."""
    lp = p_logits - p_logits.max()
    lp = lp - np.log(np.exp(lp).sum())
    lq = q_logits - q_logits.max()
    lq = lq - np.log(np.exp(lq).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    return json.loads((MODEL / "meta.json").read_text())


@pytest.fixture(scope="module")
def cached_reference() -> dict[str, Any] | None:
    """``transformers``' last-position logits for both prompts, and ``ZeosQwen``'s with
    the password in SECRET hidden (and open, for the difference), or None without the
    weights. The ``zeos`` fixture asks for it first, so the float32 model has the
    memory to itself."""
    if not (HF / "config.json").is_file():
        return None
    short, long, secret = _ids(SHORT), _ids(LONG), _ids(SECRET)
    key = hashlib.sha256(json.dumps([short, long, secret, SECRET_HIDDEN]).encode()).hexdigest()
    path = CACHE / f"opt-zeos-{key[:16]}.npz"
    if not path.is_file():
        torch = pytest.importorskip("torch", reason="needs the export dependency group")
        transformers = pytest.importorskip("transformers")
        spec = importlib.util.spec_from_file_location(
            "export_model", DEMO / "export" / "export_model.py"
        )
        assert spec is not None and spec.loader is not None
        export_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_model)
        hf = transformers.AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32)
        hf.eval()
        with torch.no_grad():
            want = {
                name: hf(torch.tensor([ids])).logits[0, -1].float().numpy()
                for name, ids in (("short", short), ("long", long))
            }
            core = export_model.ZeosQwen(hf, int8_embedding=False)
            core.eval()
            mask = [i not in SECRET_HIDDEN for i in range(len(secret))]
            zeos_open, _ = core.run(secret)
            zeos_hidden, _ = core.run(secret, mask=mask)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            short=want["short"],
            long=want["long"],
            zeos_open=zeos_open.numpy(),
            zeos_hidden=zeos_hidden.numpy(),
        )
        del hf, core
        gc.collect()
    data = np.load(path)
    return {name: data[name].astype(np.float64) for name in data.files}


@pytest.fixture(scope="module")
def reference(cached_reference: dict[str, Any] | None) -> dict[str, Any]:
    if cached_reference is None:
        pytest.skip(f"no transformers weights at {HF}")
    return cached_reference


@pytest.fixture(scope="module")
def zeos(cached_reference: dict[str, Any] | None) -> Decoder:
    return Decoder(MODEL, "decoder_zeos_q4f16.onnx", zeos=True)


@pytest.fixture(scope="module")
def opt() -> Decoder:
    if not (OPT / "onnx" / "decoder_model_merged_q4f16.onnx").is_file():
        pytest.skip(f"no -OPT export at {OPT}")
    return Decoder(OPT, "decoder_model_merged_q4f16.onnx", zeos=False)


#: The position of SHORT hidden in the masking tests.
HIDDEN = 40
#: (prompt, chunk): one token at a time and 7 on the short prompt, 16 and 512 on the long
#: one, so every chunk size runs after a carried cache.
CASES = [("short", 1), ("short", 7), ("long", 16), ("long", 512)]


def test_the_graph_has_the_contracted_inputs_and_outputs(
    zeos: Decoder, meta: dict[str, Any]
) -> None:
    session = zeos.session
    inputs = {v.name: v for v in session.get_inputs()}
    outputs = {v.name: v for v in session.get_outputs()}
    assert "attention_mask" not in inputs
    assert inputs["key_mask"].type == "tensor(bool)"
    assert inputs["key_mask"].shape == [1, "total_sequence_length"]
    assert outputs["attention"].type == "tensor(float)"
    assert outputs["attention"].shape == ["total_sequence_length"]
    assert [v["name"] for v in meta["decoder"]["inputs"]] == list(inputs)
    assert [v["name"] for v in meta["decoder"]["outputs"]] == list(outputs)
    full = meta["fullAttentionLayers"]
    assert len(full) == 8 and len(meta["linearAttentionLayers"]) == 24
    for layer in full:
        assert inputs[f"past_key_values.{layer}.key"].shape == [
            "batch_size",
            4,
            "past_sequence_length",
            256,
        ]
    # The DeltaNet caches are named by layer index, like the softmax layers' KV.
    for slot in meta["linearAttentionLayers"]:
        assert inputs[f"past_conv.{slot}"].shape == ["batch_size", 8192, 3]
        assert inputs[f"past_recurrent.{slot}"].shape == ["batch_size", 32, 128, 128]


def test_meta_lists_every_file_with_its_size(meta: dict[str, Any]) -> None:
    listed = set(meta["files"])
    on_disk = {
        p.relative_to(MODEL).as_posix()
        for p in MODEL.rglob("*")
        if p.is_file() and p.name != "meta.json"
    }
    assert listed == on_disk
    for name, entry in meta["files"].items():
        assert (MODEL / name).stat().st_size == entry["bytes"]
    assert meta["eosId"] in meta["controlIds"] and meta["padId"] in meta["controlIds"]
    assert meta["vocabSize"] <= meta["logitsSize"]


@pytest.mark.parametrize(("prompt", "chunk"), CASES)
def test_with_nothing_hidden_it_is_the_opt_graph_and_agrees_with_transformers(
    zeos: Decoder, opt: Decoder, reference: dict[str, Any], prompt: str, chunk: int
) -> None:
    ids = _ids(SHORT if prompt == "short" else LONG)
    got = zeos.last_step(ids, chunk)
    base = _logits(opt.last_step(ids, chunk))
    mine = _logits(got)
    want = reference[prompt]
    # Against the graph it was cut from: float16 rounding of a reordered activation.
    assert float(np.abs(mine - base).max()) < 0.1
    assert int(mine.argmax()) == int(base.argmax())
    assert _kl(base, mine) < 1e-3
    # Against transformers: 4-bit weights cost what they cost the -OPT graph, no more.
    assert int(mine.argmax()) == int(want.argmax())
    assert _kl(want, mine) < max(0.1, 1.5 * _kl(want, base))
    attention = got["attention"]
    assert attention.shape == (len(ids),)
    assert float(attention.sum()) == pytest.approx(1.0, abs=1e-4)
    assert (attention >= 0).all()


@pytest.mark.parametrize("chunk", [1, 7, 16])
def test_a_hidden_position_receives_exactly_zero_and_the_rest_sum_to_one(
    zeos: Decoder, chunk: int
) -> None:
    ids = _ids(SHORT)
    for hidden in (0, 2, HIDDEN, len(ids) - 2):
        mask = np.ones(len(ids), np.bool_)
        mask[hidden] = False
        attention = zeos.last_step(ids, chunk, mask)["attention"]
        assert attention.shape == (len(ids),)
        assert attention[hidden] == 0.0
        assert float(attention.sum()) == pytest.approx(1.0, abs=1e-4)
        assert (attention >= 0).all()


def test_hiding_positions_moves_the_logits_as_zeos_qwen_does(
    zeos: Decoder, reference: dict[str, Any]
) -> None:
    """The mask acts inside the forward pass in both kinds of layer: hiding the password
    moves the answer, and to where the float32 ``ZeosQwen`` rewrite of ``transformers``
    moves it under the same mask, as closely as 4-bit weights allow."""
    ids = _ids(SECRET)
    pieces = tokenizers.Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    assert pieces.decode([ids[i] for i in SECRET_HIDDEN]) == " tangerine"
    mask = np.array([i not in SECRET_HIDDEN for i in range(len(ids))])
    for chunk in (1, 16):
        open_result = zeos.last_step(ids, chunk)
        hidden_result = zeos.last_step(ids, chunk, mask)
        mine_open, mine_hidden = _logits(open_result), _logits(hidden_result)
        assert all(open_result["attention"][i] > 0.0 for i in SECRET_HIDDEN)
        assert all(hidden_result["attention"][i] == 0.0 for i in SECRET_HIDDEN)
        # Hiding moves the distribution far further than quantisation does ...
        moved = _kl(reference["zeos_open"], reference["zeos_hidden"])
        assert moved > 1.0
        assert _kl(mine_open, mine_hidden) > 0.5 * moved
        # ... and to the same place.
        assert _kl(reference["zeos_hidden"], mine_hidden) < 0.1 * moved
        assert int(mine_hidden.argmax()) == int(reference["zeos_hidden"].argmax())
        assert int(mine_open.argmax()) == int(reference["zeos_open"].argmax())
        shift = mine_hidden - mine_open
        want = reference["zeos_hidden"] - reference["zeos_open"]
        assert float(np.corrcoef(shift, want)[0, 1]) > 0.8


def test_one_run_takes_a_chunk_of_2048_after_a_cache(zeos: Decoder) -> None:
    ids = (_ids(LONG) * 6)[:2100]
    cache = zeos.empty()
    mask = np.ones(len(ids), np.bool_)
    mask[100] = False
    zeos.step(cache, ids[:50], 0, mask)
    result = zeos.step(cache, ids[50:2098], 50, mask)
    assert cache["past_key_values.3.key"].shape[2] == 2098
    attention = result["attention"]
    assert attention.shape == (2098,)
    assert attention[100] == 0.0
    assert float(attention.sum()) == pytest.approx(1.0, abs=1e-4)
    assert np.isfinite(result["logits"]).all()
