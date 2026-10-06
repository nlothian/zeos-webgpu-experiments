# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# The benchmarks drive the native game, whose modules are untyped; strict mode would
# only restate that.

"""Pilot and prompt-arm prompt sizes in the model's real BPE tokens, for both boards.

* **Pilot prefix**: the kernel built by ``ZeosDriver`` (``build_kernel``: pilot.md plus
  ``rules_prompt`` for the board) over a ``JsMachine`` with the real tokenizer, booted
  until the pilot's first decode step. What the worker holds then is exactly what the
  page would prefill: the kernel's injections, word by word as ``JsMachine`` encodes
  them, inside its ChatML framing.
* **One board**: ``LeadView.state`` as the native zeos clock renders it, ``encode``-d
  into pipe words and encoded by ``JsMachine._encode`` (a leading space per word), plus
  the framing an arrival and the reopened turn add. Also the same text as one plain
  ``tokenize``, for comparison. Sampled over 40 ticks of a seeded game with a seeded
  random stick.
* **Prompt arm**: ``PromptPlayer.default_prompt()`` (rules_prompt + reply.md) as the
  system turn, and ``render_turn`` with 5 turns of history as the user turn.

Writes ``results/prompt_sizes.json``::

    UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/prompt_sizes.py
"""

from __future__ import annotations

import json
import random
import statistics
from typing import Any

from _common import BENCH, PrefixReady, TokenizerWorker, board, pilot_seat_maps, write_json
from zeos.machine.base import tokens_from_text
from zeos_coop_count_web.js_machine import JsMachine
from zeos_space_invaders.game import ACTIONS, Controls, Game, snapshot
from zeos_space_invaders.players.base import PromptPlayer
from zeos_space_invaders.players.zeos.player import ZeosDriver, encode
from zeos_space_invaders.utils.views import LeadView

from zeos_space_invaders_web.contracts import BOARDS, BoardName

TICKS = 40
#: The real pilot prefix and framed boards (arrival framing, the board, the reopened
#: turn, a move), per board, for ``prefill.js``; written to ``.cache/webgpu_inputs.json``.
WEBGPU_INPUTS: dict[str, dict[str, Any]] = {}


def _stats(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 1),
        "max": max(values),
    }


def pilot_prefix(worker: TokenizerWorker, name: BoardName) -> tuple[JsMachine, list[int]]:
    spec = board(name)
    descriptors, _ = pilot_seat_maps()
    machine = JsMachine(worker, descriptors=descriptors)
    driver = ZeosDriver(machine=machine, view=LeadView(), rules=spec.rules)  # pyright: ignore[reportArgumentType]
    driver.start()
    try:
        driver._pump()  # pyright: ignore[reportPrivateUsage]
    except PrefixReady as ready:
        return machine, ready.ids
    raise RuntimeError("the pilot was never decoded")


def measure(worker: TokenizerWorker, name: BoardName) -> dict[str, Any]:
    spec = board(name)
    machine, prefix = pilot_prefix(worker, name)
    pilot_job = next(iter(machine._contexts))  # pyright: ignore[reportPrivateUsage]
    ctx = machine._ctx_of(pilot_job)  # pyright: ignore[reportPrivateUsage]
    arrive = len(machine._framing_ids(machine._CHATML_ARRIVE))  # pyright: ignore[reportPrivateUsage]
    turn = len(machine._framing_ids(machine._CHATML_TURN))  # pyright: ignore[reportPrivateUsage]

    view = LeadView()
    player = PromptPlayer(view=LeadView(), rules=spec.rules, history=spec.history)
    system = player.default_prompt()
    game = Game(seed=spec.seed, rules=spec.rules)
    controls = Controls(game, per_tick=spec.actions_per_tick)
    stick = random.Random(0)
    pilot_board: list[int] = []
    plain_board: list[int] = []
    words: list[int] = []
    prompt_user: list[int] = []
    sample_board = ""
    arrive_ids = machine._framing_ids(machine._CHATML_ARRIVE)  # pyright: ignore[reportPrivateUsage]
    turn_ids = machine._framing_ids(machine._CHATML_TURN)  # pyright: ignore[reportPrivateUsage]
    move_ids = worker.tokenize("write stdout left;")
    framed: list[list[int]] = []
    for _ in range(TICKS):
        if game.over:
            break
        obs, info = game.render(), snapshot(game)
        now = view.state(obs, info)
        sample_board = sample_board or now
        tokens = tokens_from_text(encode(now))
        ids, _spans = machine._encode(tokens, ctx)  # pyright: ignore[reportPrivateUsage]
        pilot_board.append(len(ids))
        framed.append(arrive_ids + ids + turn_ids + move_ids)
        words.append(len(tokens))
        plain_board.append(worker.count(now))
        prompt_user.append(worker.count(player.render_turn(obs, info)))
        action = stick.choice(ACTIONS)
        info["steps"] = info["ticks"]
        player.turns.append((info["steps"], action, view.history(obs, info)))
        controls.write(action)
        controls.tick()
    # The user turns once the history is full, which is the steady state.
    full = prompt_user[spec.history :]
    system_ids = worker.count(system)
    # <|im_start|>system\n ... <|im_end|>\n<|im_start|>user\n ... <|im_end|>\n
    # <|im_start|>assistant\n<think>\n\n</think>\n\n
    framing = 2 + worker.count("system\n") + worker.count("\n") + 1 + worker.count("user\n")
    framing += 1 + worker.count("\n") + 1 + worker.count("assistant\n<think>\n\n</think>\n\n")
    WEBGPU_INPUTS[name] = {"prefix": prefix, "boards": framed[:12]}
    return {
        "board": name,
        "rules": {"w": spec.rules.w, "h": spec.rules.h, "tick_s": spec.tick_seconds},
        "pilot_prefix_tokens": len(prefix),
        "pilot_body_words": len(ctx.tokens),
        "pilot_board_tokens": _stats(pilot_board),
        "pilot_board_words": _stats(words),
        "pilot_arrival_framing_tokens": arrive + turn,
        "pilot_board_plain_bpe_tokens": _stats(plain_board),
        "prompt_system_tokens": system_ids,
        "prompt_user_tokens_history_full": _stats(full),
        "prompt_framing_tokens": framing,
        "prompt_total_per_decision": _stats([system_ids + framing + u for u in full]),
        "sample_board": sample_board,
    }


def main() -> None:
    worker = TokenizerWorker()
    out = {name: measure(worker, name) for name in BOARDS}
    worker.close()
    path = write_json("prompt_sizes.json", out)
    cache = BENCH / ".cache"
    cache.mkdir(exist_ok=True)
    (cache / "webgpu_inputs.json").write_text(json.dumps(WEBGPU_INPUTS))
    for name, row in out.items():
        print(name, {k: v for k, v in row.items() if k != "sample_board"})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
