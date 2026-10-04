# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The Transformers.js worker, run by Node in a child process, as a CPython object.

``NodeWorker`` has the ``ZeosModelWorker`` methods, so ``JsMachine`` drives it exactly as
it drives the worker in the page: the same JavaScript (``web/transformers_worker.js``)
over the same exported graphs, and by default the same ONNX Runtime WebAssembly kernels.
That is what lets the machine contract, the kernel cases and the attention evidence run
from pytest and a shell rather than only in a browser.

Each call writes one frame to the child and reads one back, so a call returns only when
the worker has finished, which is the synchronous interface ``JsMachine`` expects.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

__all__ = ["DEFAULT_MODEL", "DEMO", "ModelInfo", "NodeWorker", "Step", "node_available"]

DEMO = Path(__file__).resolve().parents[2]
BRIDGE = DEMO / "web" / "node_bridge.mjs"
DEFAULT_MODEL = DEMO / "models" / "Qwen3.5-2B-zeos-int8"

_TYPECODES = {"Uint8Array": "B", "Int32Array": "i", "Float32Array": "f"}


@dataclass(frozen=True, slots=True)
class ModelInfo:
    blockSize: int
    padId: int
    controlIds: list[int]
    eosId: int
    vocabSize: int


@dataclass(frozen=True, slots=True)
class Step:
    tokenId: int
    attention: list[float] | None


def node_available() -> bool:
    """Whether Node and this demo's npm dependencies are installed."""
    return shutil.which("node") is not None and (DEMO / "node_modules" / "onnxruntime-web").is_dir()


class NodeWorker:
    """A ``ZeosModelWorker`` whose methods are answered by ``web/node_bridge.mjs``."""

    def __init__(
        self, model: Path = DEFAULT_MODEL, *, runtime: str = "web", threads: int = 1
    ) -> None:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node is not on PATH")
        self._process = subprocess.Popen(
            [
                node,
                str(BRIDGE),
                "--model",
                str(model),
                "--runtime",
                runtime,
                "--threads",
                str(threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=DEMO,
        )
        self._next = 0
        ready, _ = self._read()
        self.backend = str(ready["backend"])
        self._info: ModelInfo | None = None
        self._pieces: tuple[str, ...] = tuple(cast(list[str], self._call("pieces")))

    # -- the interface -----------------------------------------------------------

    def info(self) -> ModelInfo:
        if self._info is None:
            raw = cast(dict[str, Any], self._call("info"))
            self._info = ModelInfo(
                blockSize=int(raw["blockSize"]),
                padId=int(raw["padId"]),
                controlIds=[int(i) for i in raw["controlIds"]],
                eosId=int(raw["eosId"]),
                vocabSize=int(raw["vocabSize"]),
            )
        return self._info

    def tokenize(self, text: str) -> list[int]:
        return list(cast(array[int], self._call("tokenize", text)))

    def piece(self, tokenId: int) -> str:
        # Asked once per id when a machine starts, so answered from the copy taken at
        # start-up; past the vocabulary it raises, as the worker's own ``piece`` throws.
        if not 0 <= tokenId < len(self._pieces):
            raise IndexError(f"token id {tokenId} is outside the vocabulary")
        return self._pieces[tokenId]

    def createContext(self, jobId: str) -> None:
        self._call("createContext", jobId)

    def destroyContext(self, jobId: str) -> None:
        self._call("destroyContext", jobId)

    def length(self, jobId: str) -> int:
        return int(cast(int, self._call("length", jobId)))

    def append(self, jobId: str, ids: Sequence[int]) -> None:
        self._call("append", jobId, array("i", ids))

    def truncate(self, jobId: str, n: int) -> None:
        self._call("truncate", jobId, n)

    def fork(self, parentId: str, childId: str) -> None:
        self._call("fork", parentId, childId)

    def decodeStep(self, jobId: str, opts: Any) -> Step:
        blocks = opts["allowedBlocks"]
        tokens = opts["allowedTokens"]
        raw = cast(
            dict[str, Any],
            self._call(
                "decodeStep",
                jobId,
                {
                    "allowedBlocks": None if blocks is None else array("B", blocks),
                    "allowedTokens": None if tokens is None else array("B", tokens),
                },
            ),
        )
        attention = raw["attention"]
        return Step(
            tokenId=int(raw["tokenId"]),
            attention=None if attention is None else list(cast(array[float], attention)),
        )

    def close(self) -> None:
        if self._process.poll() is None:
            stdin = cast(IO[bytes], self._process.stdin)
            stdin.close()
            self._process.wait(timeout=30)

    def __enter__(self) -> NodeWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- frames ------------------------------------------------------------------

    def _call(self, method: str, *args: object) -> object:
        blobs: list[bytes] = []
        header = {"id": self._next, "method": method, "args": [_encode(a, blobs) for a in args]}
        self._next += 1
        self._write(header, blobs)
        reply, reply_blobs = self._read()
        if not reply["ok"]:
            raise RuntimeError(f"worker {method}: {reply['error']}")
        return _decode(reply["value"], reply_blobs)

    def _write(self, header: dict[str, Any], blobs: list[bytes]) -> None:
        header["blobs"] = [len(b) for b in blobs]
        payload = json.dumps(header).encode()
        stdin = cast(IO[bytes], self._process.stdin)
        stdin.write(struct.pack("<I", len(payload)) + payload + b"".join(blobs))
        stdin.flush()

    def _read(self) -> tuple[dict[str, Any], list[bytes]]:
        stdout = cast(IO[bytes], self._process.stdout)
        head = stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(f"the node worker exited (status {self._process.poll()})")
        (length,) = struct.unpack("<I", head)
        header = cast(dict[str, Any], json.loads(stdout.read(length)))
        blobs = [stdout.read(n) for n in header.get("blobs", [])]
        return header, blobs


def _encode(value: object, blobs: list[bytes]) -> object:
    if isinstance(value, array):
        code = cast(array[Any], value).typecode
        name = {"B": "Uint8Array", "i": "Int32Array", "f": "Float32Array"}[code]
        blobs.append(cast(array[Any], value).tobytes())
        return {"$blob": len(blobs) - 1, "type": name}
    if isinstance(value, dict):
        return {k: _encode(v, blobs) for k, v in cast(dict[str, object], value).items()}
    return value


def _decode(value: object, blobs: list[bytes]) -> object:
    if isinstance(value, list):
        return [_decode(v, blobs) for v in cast(list[object], value)]
    if isinstance(value, dict):
        fields = cast(dict[str, object], value)
        if "$blob" in fields:
            out = array(_TYPECODES[str(fields["type"])])
            out.frombytes(blobs[int(cast(int, fields["$blob"]))])
            return out
        return {k: _decode(v, blobs) for k, v in fields.items()}
    return value
