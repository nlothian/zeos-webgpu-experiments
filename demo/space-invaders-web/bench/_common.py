# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# The benchmarks drive the native game, whose modules are untyped; strict mode would
# only restate that.

"""What the benchmark scripts share: the real tokenizer, the case and the boards.

``TokenizerWorker`` is a ``ZeosModelWorker`` with the model's tokenizer and no model
(``tokenizer_server.mjs``): ``JsMachine`` can be built over it, the kernel can boot the
pilot through it and the prompt is counted in the ids the page would send, but asking
it for a decode step raises ``PrefixReady`` with the context's ids, which is how
``prompt_sizes.py`` measures a prefix.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from zeos.descriptor.loader import load_case
from zeos.machine.seat import seat_maps
from zeos_space_invaders.players.zeos.player import CASE_ROOT

from zeos_space_invaders_web.contracts import BoardName, BoardSpec, load_board

BENCH = Path(__file__).resolve().parent
DEMO = BENCH.parents[1]
COOP = DEMO / "coop-count-web"
MODEL = COOP / "models" / "Qwen3.5-4B-ZEOS-OPT"
RESULTS = BENCH / "results"


@dataclass(frozen=True, slots=True)
class Info:
    blockSize: int
    padId: int
    controlIds: list[int]
    eosId: int
    vocabSize: int


class PrefixReady(Exception):
    """A decode step was asked for: the context is the prompt the model would read."""

    def __init__(self, job: str, ids: Sequence[int]) -> None:
        super().__init__(job)
        self.job = job
        self.ids = list(ids)


class TokenizerWorker:
    """``ZeosModelWorker`` over the model's tokenizer only; see the module notes."""

    def __init__(self, model: Path = MODEL) -> None:
        self._process = subprocess.Popen(
            ["node", str(BENCH / "tokenizer_server.mjs"), "--model", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        raw = cast(dict[str, Any], self._call({"op": "info"}))
        self._info = Info(
            blockSize=int(raw["blockSize"]),
            padId=int(raw["padId"]),
            controlIds=[int(i) for i in raw["controlIds"]],
            eosId=int(raw["eosId"]),
            vocabSize=int(raw["vocabSize"]),
        )
        self.pieces: tuple[str, ...] = tuple(cast(list[str], self._call({"op": "pieces"})))
        self._contexts: dict[str, list[int]] = {}
        self._cache: dict[str, list[int]] = {}

    def _call(self, request: dict[str, object]) -> object:
        stdin = cast(IO[str], self._process.stdin)
        stdout = cast(IO[str], self._process.stdout)
        stdin.write(json.dumps(request) + "\n")
        stdin.flush()
        reply = cast(dict[str, Any], json.loads(stdout.readline()))
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply["value"]

    def info(self) -> Info:
        return self._info

    def tokenize(self, text: str) -> list[int]:
        ids = self._cache.get(text)
        if ids is None:
            ids = [int(i) for i in cast(list[int], self._call({"op": "tokenize", "text": text}))]
            self._cache[text] = ids
        return list(ids)

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def piece(self, tokenId: int) -> str:
        return self.pieces[tokenId]

    def createContext(self, jobId: str) -> None:
        self._contexts[jobId] = []

    def destroyContext(self, jobId: str) -> None:
        self._contexts.pop(jobId, None)

    def length(self, jobId: str) -> int:
        return len(self._contexts[jobId])

    def append(self, jobId: str, ids: object) -> None:
        self._contexts[jobId].extend(int(i) for i in cast(Iterable[int], ids))

    def truncate(self, jobId: str, n: int) -> None:
        del self._contexts[jobId][n:]

    def fork(self, parentId: str, childId: str) -> None:
        self._contexts[childId] = list(self._contexts[parentId])

    def decodeStep(self, jobId: str, opts: object) -> object:
        raise PrefixReady(jobId, self._contexts[jobId])

    def close(self) -> None:
        if self._process.poll() is None:
            cast(IO[str], self._process.stdin).close()
            self._process.wait(timeout=30)


def pilot_seat_maps() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """The case's descriptor -> aliases and descriptor -> valued-aliases maps, as
    ``node_run`` builds them for any case."""
    bundle = load_case(CASE_ROOT)
    return seat_maps(bundle.descriptors, bundle.pipes)


def board(name: BoardName, seed: int | None = 7) -> BoardSpec:
    return load_board(name, seed)


def write_json(name: str, value: object) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path
