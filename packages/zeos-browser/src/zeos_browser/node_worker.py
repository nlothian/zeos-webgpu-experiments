# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The stub model worker, run by Node in a child process, as a CPython object.

``NodeWorker`` has the ``ZeosModelWorker`` methods, so ``JsMachine`` drives it exactly as
it drives the worker in the page: ``web/stub_worker.js`` over a JSON file of
``{tapes, options}``, behind the same frames (``web/frames.js``) the page's channel
encodes. That is what lets the decode step that does not block, and its cancel, be
tested from pytest rather than only in a browser.

Each call writes one frame to the child and reads one back, so a call returns only when
the worker has finished, which is the synchronous interface ``JsMachine`` expects.

It also has the decode step that does not block, as ``SyncModelWorker`` has it in the
page (``web/model_channel.js``): ``beginDecodeStep`` writes the request and returns its
id, ``pollDecode(timeoutMs)`` waits for the reply with ``select`` for at most that long,
and ``cancelDecode`` writes a control frame ``{"cancel": id}`` that the bridge reads while
the step runs and that the step checks before every chunk. A cancelled step's result is
always dropped, even if the reply had already been written when the cancel was sent; and
while a step is in flight every other call that reaches the child raises
``CHANNEL_BUSY``.

Every reply names the request it answers. The pipe keeps replies in order, so a reply
to an earlier request (one whose caller was interrupted before reading it) is read and
discarded, and the wait goes on for the right one; a reply to a later request cannot
happen and raises.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import struct
import subprocess
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

__all__ = [
    "CHANNEL_BUSY",
    "PACKAGE",
    "ModelInfo",
    "NodeWorker",
    "Step",
    "node_available",
]

PACKAGE = Path(__file__).resolve().parents[2]
BRIDGE = PACKAGE / "web" / "node_bridge.mjs"

_TYPECODES = {"Uint8Array": "B", "Int32Array": "i", "Float32Array": "f"}

#: What a call raises while a begun decode step is in flight (``model_channel.js``'s).
CHANNEL_BUSY = "channel busy: decode in flight"


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
    """Whether Node is installed."""
    return shutil.which("node") is not None


class NodeWorker:
    """A ``ZeosModelWorker`` whose methods are answered by ``web/node_bridge.mjs``."""

    def __init__(self, *, stub: Path) -> None:
        """``stub``: a JSON file of ``{tapes, options}`` to serve ``web/stub_worker.js``
        over."""
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node is not on PATH")
        argv = [node, str(BRIDGE), "--stub", str(stub)]
        self._process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=PACKAGE
        )
        self._next = 0
        # Bytes read from the child and not yet parsed: frames are read off the raw pipe,
        # so ``select`` sees everything that has not been consumed.
        self._buffer = bytearray()
        self._in_flight: int | None = None
        self._cancelled = False
        frame = self._read(None)
        assert frame is not None  # no timeout
        ready, _ = frame
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
        raw = cast(dict[str, Any], self._call("decodeStep", jobId, _step_args(opts)))
        return Step(tokenId=int(raw["tokenId"]), attention=_floats(raw["attention"]))

    # -- the decode step that does not block -------------------------------------

    @property
    def inFlight(self) -> bool:
        return self._in_flight is not None

    def beginDecodeStep(self, jobId: str, opts: Any) -> int:
        """Write one decode step and return its request id; never waits. ``opts`` as
        ``decodeStep``'s, plus an optional ``maxChunk``."""
        if self._in_flight is not None:
            raise RuntimeError(CHANNEL_BUSY)
        blobs: list[bytes] = []
        request = self._next
        header = {
            "id": request,
            "method": "decodeStep",
            "args": [jobId, _encode(_step_args(opts), blobs)],
            "begun": True,
        }
        self._next += 1
        self._write(header, blobs)
        self._in_flight = request
        self._cancelled = False
        return request

    def pollDecode(self, timeoutMs: float = 0) -> dict[str, Any] | None:
        """The begun step's answer, or ``None`` if it has not arrived within ``timeoutMs``
        or nothing is in flight. ``{"tokenId", "attention", "cancelled": False,
        "resident", "stats"}`` for a step that finished, ``{"cancelled": True,
        "resident", "stats"}`` for one that was cancelled. An error reply raises."""
        if self._in_flight is None:
            return None
        frame = self._read(self._in_flight, timeout=max(0.0, timeoutMs) / 1000)
        if frame is None:
            return None
        reply, blobs = frame
        cancelled = self._cancelled
        self._in_flight = None
        self._cancelled = False
        if not reply["ok"]:
            raise RuntimeError(f"worker decodeStep: {reply['error']}")
        raw = cast(dict[str, Any], _decode(reply["value"], blobs))
        if cancelled or raw.get("cancelled") is True:
            return {"cancelled": True, "resident": raw["resident"], "stats": raw["stats"]}
        return {
            "tokenId": int(raw["tokenId"]),
            "attention": _floats(raw["attention"]),
            "cancelled": False,
            "resident": raw["resident"],
            "stats": raw["stats"],
        }

    def cancelDecode(self) -> None:
        """Ask the begun step to stop before its next chunk; never waits, and a no-op with
        nothing in flight."""
        if self._in_flight is None or self._cancelled:
            return
        self._cancelled = True
        self._write({"cancel": self._in_flight}, [])

    def close(self) -> None:
        """End the child: a step in flight is cancelled and drained first, and a child
        that has not exited 30 s after its stdin closed is killed."""
        if self._process.poll() is None:
            if self._in_flight is not None:
                self.cancelDecode()
                try:
                    self.pollDecode(30_000)
                except RuntimeError:
                    # The step's error, or the child gone: it is being shut down anyway.
                    pass
            stdin = cast(IO[bytes], self._process.stdin)
            stdin.close()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def __enter__(self) -> NodeWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- frames ------------------------------------------------------------------

    def _call(self, method: str, *args: object) -> object:
        if self._in_flight is not None:
            raise RuntimeError(CHANNEL_BUSY)
        blobs: list[bytes] = []
        header = {"id": self._next, "method": method, "args": [_encode(a, blobs) for a in args]}
        self._next += 1
        self._write(header, blobs)
        frame = self._read(cast(int, header["id"]))
        assert frame is not None  # no timeout
        reply, reply_blobs = frame
        if not reply["ok"]:
            raise RuntimeError(f"worker {method}: {reply['error']}")
        return _decode(reply["value"], reply_blobs)

    def _write(self, header: dict[str, Any], blobs: list[bytes]) -> None:
        header["blobs"] = [len(b) for b in blobs]
        payload = json.dumps(header).encode()
        stdin = cast(IO[bytes], self._process.stdin)
        stdin.write(struct.pack("<I", len(payload)) + payload + b"".join(blobs))
        stdin.flush()

    def _read(
        self, expect: int | None, timeout: float | None = None
    ) -> tuple[dict[str, Any], list[bytes]] | None:
        """The reply to request ``expect`` (``None``: the next frame, the ready frame);
        ``None`` if it is not complete within ``timeout`` seconds (``None``: wait as long
        as it takes). Replies to earlier requests are discarded. Whatever the timeout,
        what the pipe already holds is read."""
        fd = cast(IO[bytes], self._process.stdout).fileno()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            frame = self._frame()
            if frame is not None:
                answered = frame[0].get("id")
                if expect is None or answered == expect:
                    return frame
                if not (isinstance(answered, int) and answered < expect):
                    raise RuntimeError(f"a reply to request {answered} came for {expect}")
                continue
            if deadline is not None:
                left = max(0.0, deadline - time.monotonic())
                if not select.select([fd], [], [], left)[0]:
                    return None
            data = os.read(fd, 1 << 20)
            if not data:
                raise RuntimeError(f"the node worker exited (status {self._process.poll()})")
            self._buffer += data

    def _frame(self) -> tuple[dict[str, Any], list[bytes]] | None:
        """Take the first complete frame off the read buffer, if there is one."""
        buffer = self._buffer
        if len(buffer) < 4:
            return None
        (length,) = struct.unpack_from("<I", buffer)
        if len(buffer) < 4 + length:
            return None
        header = cast(dict[str, Any], json.loads(bytes(buffer[4 : 4 + length])))
        sizes = cast(list[int], header.pop("blobs", []))
        end = 4 + length + sum(sizes)
        if len(buffer) < end:
            return None
        blobs: list[bytes] = []
        at = 4 + length
        for n in sizes:
            blobs.append(bytes(buffer[at : at + n]))
            at += n
        del buffer[:end]
        return header, blobs


def _step_args(opts: Any) -> dict[str, object]:
    """A step's options as they cross the pipe: masks as byte arrays, the sample, and
    ``maxChunk`` only when it is set."""
    blocks = opts["allowedBlocks"]
    tokens = opts["allowedTokens"]
    args: dict[str, object] = {
        "allowedBlocks": None if blocks is None else array("B", blocks),
        "allowedTokens": None if tokens is None else array("B", tokens),
    }
    if opts.get("sample") is not None:
        args["sample"] = dict(opts["sample"])
    if opts.get("maxChunk") is not None:
        # Unchanged: the worker validates it, as it does from the page.
        args["maxChunk"] = opts["maxChunk"]
    return args


def _floats(value: object) -> list[float] | None:
    return None if value is None else list(cast(array[float], value))


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
            out: array[Any] = array(_TYPECODES[str(fields["type"])])
            out.frombytes(blobs[int(cast(int, fields["$blob"]))])
            return out
        return {k: _decode(v, blobs) for k, v in fields.items()}
    return value
