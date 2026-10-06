# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# The benchmarks drive the native game, whose modules are untyped; strict mode would
# only restate that.

"""Grammar-mask cost in CPython with the model's real 248k vocabulary.

The pilot's language is built by ``JsMachine._language`` from the case's own seat maps
(aliases ``stdin``/``stdout``), over ``TokenizerWorker``. Writes
``results/mask_cost.json`` and ``.cache/mask_inputs.json`` (the vocabulary and the walks,
for ``mask_pyodide.mjs``)::

    UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/mask_cost.py
"""

from __future__ import annotations

import json
import time
from array import array

from _common import BENCH, TokenizerWorker, pilot_seat_maps, write_json
from mask_core import measure
from zeos_coop_count_web.js_machine import JsMachine

from zeos_space_invaders_web.contracts import PREWARM_COMMANDS


def _representable(
    machine: JsMachine, seat_valued: dict[str, tuple[str, ...]], walks: dict[str, list[list[int]]]
) -> dict[str, bool]:
    """Whether each prewarm command is in the language built with the seat maps' valued."""
    from zeos_coop_count_web.token_mask import CommandLanguage

    language = CommandLanguage(machine.abi, machine.aliases("pilot"), valued=seat_valued["pilot"])
    pieces = machine._pieces  # pyright: ignore[reportPrivateUsage]
    return {
        cmd: bool(language.advance(language.start, "".join(pieces[t] for t in ids[0])))
        for cmd, ids in walks.items()
    }


def main() -> None:
    began = time.perf_counter()
    worker = TokenizerWorker()
    pieces_s = time.perf_counter() - began
    descriptors, seat_valued = pilot_seat_maps()
    # Not the seat maps' ``valued``: they mark ``stdout`` (game.controls is
    # world-backed) as an actuator, whose payload the language makes a number, so
    # ``write stdout left;`` would be unrepresentable. PilotMachineFactory takes no
    # ``valued``, i.e. JsMachine's default of none.
    began = time.perf_counter()
    machine = JsMachine(worker, descriptors=descriptors)
    machine_s = time.perf_counter() - began
    language = machine._language("pilot")  # pyright: ignore[reportPrivateUsage]
    info = worker.info()
    walks = {cmd: [worker.tokenize(cmd), worker.tokenize(" " + cmd)] for cmd in PREWARM_COMMANDS}
    aliases = list(machine.aliases("pilot"))
    valued_aliases = list(machine.valued("pilot"))
    result = measure(
        worker.pieces,
        reserved=[info.padId, info.eosId],
        control=info.controlIds,
        aliases=aliases,
        valued=valued_aliases,
        walks=walks,
        copy=lambda m: array("B", m),
    )
    result = {
        "runtime": "cpython",
        "aliases": aliases,
        "valued": valued_aliases,
        "seat_maps_valued_pilot": list(seat_valued["pilot"]),
        "prewarm_representable_with_seat_maps_valued": _representable(machine, seat_valued, walks),
        "language_alternatives": len(language._alternatives),  # pyright: ignore[reportPrivateUsage]
        "pieces_load_s": pieces_s,
        "jsmachine_init_s": machine_s,
        **result,
    }
    cache = BENCH / ".cache"
    cache.mkdir(exist_ok=True)
    (cache / "mask_inputs.json").write_text(
        json.dumps(
            {
                "pieces": worker.pieces,
                "reserved": [info.padId, info.eosId],
                "control": info.controlIds,
                "aliases": aliases,
                "valued": valued_aliases,
                "walks": walks,
            }
        )
    )
    path = write_json("mask_cost.json", result)
    worker.close()
    summary = {k: v for k, v in result.items() if k != "walks"}
    print(json.dumps(summary, indent=2))
    for cmd, rows in result["walks"].items():
        for row in rows:
            print(cmd, row["pieces"], f"new={row['new_states']}", f"cold={row['cold_ms']:.0f}ms")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
