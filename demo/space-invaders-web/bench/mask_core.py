# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# The benchmarks drive the native game, whose modules are untyped; strict mode would
# only restate that.

"""The grammar-mask timings, with nothing imported but ``token_mask`` and the ABI.

So the same function runs under CPython (``mask_cost.py``) and under Pyodide
(``mask_pyodide.mjs``), where the page's mask actually runs. ``measure`` takes the
vocabulary as pieces, the ids ``JsMachine`` reserves, the pilot's aliases and the
tokenizations of the commands to walk, and returns a JSON-able dict.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from zeos.machine.abi import DEFAULT
from zeos_browser.token_mask import CommandLanguage, RoundState, TokenMask

KEY = "pilot"


def _ms(fn: Callable[[], object], repeat: int = 1) -> tuple[float, object]:
    value: object = None
    began = time.perf_counter()
    for _ in range(repeat):
        value = fn()
    return (time.perf_counter() - began) * 1000 / repeat, value


def measure(
    pieces: Sequence[str],
    *,
    reserved: Sequence[int],
    control: Sequence[int],
    aliases: Sequence[str],
    valued: Sequence[str],
    walks: dict[str, list[list[int]]],
    copy: Callable[[bytes], object] | None = None,
) -> dict[str, Any]:
    """Cold and warm mask cost, the cost of walking each command, and the copy.

    ``walks`` maps a command to one or more tokenizations of it (lists of ids); each
    tokenization is walked from the round's start, asking for the mask before every
    token as ``JsMachine.decode`` does. ``copy`` turns a mask into what crosses into the
    worker (a ``Uint8Array`` under Pyodide); its cost is timed per call.
    """
    out: dict[str, Any] = {"vocab": len(pieces)}
    build_ms, language = _ms(lambda: CommandLanguage(DEFAULT, aliases, valued=valued))
    assert isinstance(language, CommandLanguage)
    out["language_build_ms"] = build_ms
    mask_ms, mask = _ms(lambda: TokenMask(pieces, reserved=reserved, control=control))
    assert isinstance(mask, TokenMask)
    out["tokenmask_build_ms"] = mask_ms

    start = language.start
    cold, first = _ms(lambda: mask.allowed(KEY, language, start, allow_control=False))
    warm, _ = _ms(lambda: mask.allowed(KEY, language, start, allow_control=False), repeat=2000)
    assert isinstance(first, bytes)
    out["start_state"] = {
        "cold_ms": cold,
        "warm_ms": warm,
        "allowed_ids": sum(first),
    }

    # Walk each command the way decode does: mask for the state, then advance by the
    # chosen piece. A fresh TokenMask, so the first walk pays every cold state.
    walk_mask = TokenMask(pieces, reserved=reserved, control=control)
    seen: set[RoundState] = set()
    per_command: dict[str, Any] = {}
    total_cold = 0.0
    for command, variants in walks.items():
        rows: list[dict[str, Any]] = []
        for ids in variants:
            state = start
            steps: list[dict[str, Any]] = []
            for token in ids:
                was_cached = state in seen
                ms, flags = _ms(
                    lambda state=state: walk_mask.allowed(KEY, language, state, allow_control=False)
                )
                assert isinstance(flags, bytes)
                if not was_cached:
                    total_cold += ms
                seen.add(state)
                piece = pieces[token]
                steps.append(
                    {
                        "piece": piece,
                        "cached": was_cached,
                        "ms": ms,
                        "allowed_ids": sum(flags),
                        "token_allowed": bool(flags[token]),
                    }
                )
                state = language.advance(state, piece)
            rows.append(
                {
                    "pieces": [pieces[t] for t in ids],
                    "steps": steps,
                    "states": len(ids),
                    "new_states": sum(not s["cached"] for s in steps),
                    "cold_ms": sum(s["ms"] for s in steps if not s["cached"]),
                }
            )
        per_command[command] = rows
    out["walks"] = per_command
    out["prewarm_distinct_states"] = len(seen)
    out["prewarm_cold_total_ms"] = total_cold

    # Every state again: all cache hits now.
    hit_ms: list[float] = []
    for state in sorted(seen):
        ms, _ = _ms(
            lambda state=state: walk_mask.allowed(KEY, language, state, allow_control=False),
            repeat=1000,
        )
        hit_ms.append(ms)
    out["warm_hit_ms_max"] = max(hit_ms)

    # Each payload length is its own round state (max_text bounds it), so a move the
    # model spells in other pieces than the prewarm walk lands on a cold one.
    payload: list[dict[str, Any]] = []
    head = language.advance(start, "write stdout ")
    for k in range(0, (DEFAULT.max_text or 16) + 1):
        state = language.advance(head, "x" * k)
        ms, flags = _ms(
            lambda state=state: walk_mask.allowed(KEY, language, state, allow_control=False)
        )
        assert isinstance(flags, bytes)
        payload.append({"chars": k, "cached": state in seen, "ms": ms, "allowed_ids": sum(flags)})
        seen.add(state)
    out["payload_states"] = payload
    out["payload_states_cold_total_ms"] = sum(p["ms"] for p in payload if not p["cached"])

    # What allowed() itself pays to build the bytes, and what crossing costs.
    flags = bytearray(len(pieces))
    out["bytes_from_bytearray_ms"], _ = _ms(lambda: bytes(flags), repeat=200)
    if copy is not None:
        out["copy_ms"], _ = _ms(lambda: copy(first), repeat=200)
    return out
