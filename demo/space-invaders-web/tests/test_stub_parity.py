# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``web/stub/pilot_stub_worker.js`` and ``FakePilotWorker`` behave as one worker.

The same script of operations runs through both -- the JavaScript stub in Node, the
Python twin here -- and every answer must agree: the vocabulary, tokenization, every
token a step chooses, the positions and chunks it fills, and where a step stopped by a
cancel leaves the context resident. Timing is the one thing they model differently (the
stub sleeps between chunks and is told to stop by ``shouldStop``; the twin computes when
a step lands and is cancelled at a wall-clock time), so a stop is asked for as "before
run k", which both can express exactly. Skips without Node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from machine_helpers import FakeClock

from zeos_space_invaders_web.fake_worker import FakePilotWorker, fill_chunks

STUB = Path(__file__).resolve().parents[1] / "web" / "stub" / "pilot_stub_worker.js"
NODE = shutil.which("node")

OPTIONS: dict[str, Any] = {"moves": ["left", "shoot"], "replies": ["I go right", "dunno"]}

#: ``[op, ...args]``. ``step`` takes an allowed-piece list (``None`` for all) and a
#: ``maxChunk``; the harness appends a delivered token, as the machine would. ``stop``
#: begins a step and stops it before run ``k``.
OPS: list[list[Any]] = [
    ["tokenize", "write stdout left; hello  user\nassistant <|im_end|>"],
    ["create", "1:pilot"],
    ["append_text", "1:pilot", "you are the pilot of this ship"],
    ["step", "1:pilot", None, 4],
    ["step", "1:pilot", None, 4],
    ["step", "1:pilot", None, 4],
    ["step", "1:pilot", None, 4],
    ["step", "1:pilot", ["write", " write"], 4],
    ["append_text", "1:pilot", " a board of many many many many many words arrives now"],
    ["stop", "1:pilot", 2, 3],
    ["stop", "1:pilot", 0, 3],
    ["step", "1:pilot", None, 3],
    ["truncate", "1:pilot", 5],
    ["step", "1:pilot", None, 64],
    ["fork", "1:pilot", "2:pilot"],
    ["step", "2:pilot", None, 64],
    ["length", "2:pilot"],
    ["create", "p:prompt"],
    ["append_text", "p:prompt", "system rules"],
    ["step", "p:prompt", None, 64],
    ["append_ids", "p:prompt", [3]],
    ["append_text", "p:prompt", "assistant"],
    ["step", "p:prompt", None, 64],
    ["step", "p:prompt", None, 64],
    ["step", "p:prompt", None, 64],
    ["step", "p:prompt", None, 64],
    ["truncate", "p:prompt", 2],
    ["append_ids", "p:prompt", [3]],
    ["append_text", "p:prompt", "assistant"],
    ["step", "p:prompt", None, 64],
    ["step", "p:prompt", None, 64],
    ["destroy", "2:pilot"],
]

HARNESS = r"""
import { readFileSync } from "node:fs";
const [stubPath, options, ops] = JSON.parse(readFileSync(0, "utf8"));
await import(stubPath);
const w = globalThis.createPilotStubWorker(options);
const out = [{ info: w.info(), vocab: w.vocab }];
const mask = (pieces) => {
  if (pieces === null) return null;
  const flags = new Uint8Array(w.info().vocabSize);
  for (const p of pieces) flags[w.vocab.indexOf(p)] = 1;
  return flags;
};
for (const [op, ...args] of ops) {
  if (op === "tokenize") out.push(Array.from(w.tokenize(args[0])));
  else if (op === "create") w.createContext(args[0]);
  else if (op === "destroy") w.destroyContext(args[0]);
  else if (op === "append_text") w.append(args[0], w.tokenize(args[1]));
  else if (op === "append_ids") w.append(args[0], args[1]);
  else if (op === "truncate") w.truncate(args[0], args[1]);
  else if (op === "fork") w.fork(args[0], args[1]);
  else if (op === "length") out.push(w.length(args[0]));
  else if (op === "step") {
    const r = await w.decodeStep(args[0], { allowedBlocks: null, allowedTokens: mask(args[1]), maxChunk: args[2] });
    w.append(args[0], [r.tokenId]);
    out.push({ tokenId: r.tokenId, resident: r.resident, positions: r.stats.positions, chunks: r.stats.chunks });
  } else if (op === "stop") {
    let checks = 0;
    const r = await w.decodeStep(args[0], {
      allowedBlocks: null, allowedTokens: null, maxChunk: args[2], shouldStop: () => checks++ >= args[1],
    });
    out.push({ cancelled: r.cancelled === true, resident: r.resident, positions: r.stats.positions, chunks: r.stats.chunks });
  } else throw new Error(`unknown op ${op}`);
}
process.stdout.write(JSON.stringify(out));
"""


def run_python() -> list[Any]:
    clock = FakeClock()
    w = FakePilotWorker(
        OPTIONS["moves"],
        replies=OPTIONS["replies"],
        position_ms=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    vocab = [w.piece(i) for i in range(w.info().vocabSize)]
    info = w.info()
    out: list[Any] = [
        {
            "info": {
                "blockSize": info.blockSize,
                "padId": info.padId,
                "controlIds": list(info.controlIds),
                "eosId": info.eosId,
                "vocabSize": info.vocabSize,
            },
            "vocab": vocab,
        }
    ]

    def mask(pieces: list[str] | None) -> bytes | None:
        if pieces is None:
            return None
        flags = bytearray(len(vocab))
        for piece in pieces:
            flags[vocab.index(piece)] = 1
        return bytes(flags)

    for op, *args in OPS:
        if op == "tokenize":
            out.append(w.tokenize(args[0]))
        elif op == "create":
            w.createContext(args[0])
        elif op == "destroy":
            w.destroyContext(args[0])
        elif op == "append_text":
            w.append(args[0], w.tokenize(args[1]))
        elif op == "append_ids":
            w.append(args[0], args[1])
        elif op == "truncate":
            w.truncate(args[0], args[1])
        elif op == "fork":
            w.fork(args[0], args[1])
        elif op == "length":
            out.append(w.length(args[0]))
        elif op == "step":
            key, pieces, chunk = args
            r = w.decodeStep(
                key, {"allowedBlocks": None, "allowedTokens": mask(pieces), "maxChunk": chunk}
            )
            w.append(key, [r["tokenId"]])
            out.append(
                {
                    "tokenId": r["tokenId"],
                    "resident": w.resident(key),
                    "positions": r["stats"]["positions"],
                    "chunks": r["stats"]["chunks"],
                }
            )
        elif op == "stop":
            key, k, chunk = args
            pending = w.length(key) - w.resident(key)
            w.beginDecodeStep(
                key, {"allowedBlocks": None, "allowedTokens": None, "maxChunk": chunk}
            )
            # The check before run k happens k runs' worth of positions in.
            clock.now += sum(fill_chunks(pending, chunk)[:k]) * 0.001 - 0.0005 if k else 0.0
            w.cancelDecode()
            r = w.pollDecode(float("inf"))
            assert r is not None and r["cancelled"] is True
            out.append(
                {
                    "cancelled": True,
                    "resident": r["resident"],
                    "positions": r["stats"]["positions"],
                    "chunks": r["stats"]["chunks"],
                }
            )
        else:
            raise ValueError(op)
    return out


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_js_stub_and_the_python_twin_answer_alike() -> None:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", HARNESS],
        input=json.dumps([STUB.as_uri(), OPTIONS, OPS]),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    py = run_python()
    assert js[0] == py[0]
    for n, (left, right) in enumerate(zip(js[1:], py[1:], strict=True)):
        assert left == right, f"op {OPS[n]}"


def test_the_script_exercises_what_it_claims() -> None:
    """The cancel cases stop mid-fill, and the prompt context answers two turns."""
    py = run_python()
    steps: list[dict[str, Any]] = [r for r in py[1:] if isinstance(r, dict)]
    stops = [r for r in steps if r.get("cancelled")]
    assert [s["chunks"] for s in stops] == [2, 0]
    assert stops[0]["positions"] == 6
    vocab: list[str] = py[0]["vocab"]
    said: list[str] = [vocab[r["tokenId"]] for r in steps if "tokenId" in r]
    assert said[:4] == ["write", " stdout", " left;", " write"]
    assert " I" in said and " dunno" in said
