# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``PyodideAsyncWorker``'s conversions, against an object shaped like a JsProxy.

Under CPython there is no ``js`` module, so the converter for Python dicts is injected;
everything else the adapter does is attribute access and copying, which runs here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from machine_helpers import PILOT, fake_worker, pilot, until, words

from zeos_space_invaders_web.contracts import AsyncModelWorker
from zeos_space_invaders_web.pyodide_glue import PyodideAsyncWorker


class TypedArray(list[int]):
    """A JsProxy of an ``Int32Array``: iterable, and copied out through a memoryview."""

    def to_memoryview(self) -> memoryview:
        return memoryview(bytearray()).cast("i") if not self else _view(self)


def _view(ids: list[int]) -> memoryview:
    import array

    return memoryview(array.array("i", ids))


class JsChannel:
    """``SyncModelWorker`` as Pyodide shows it: attribute results, a JS getter."""

    def __init__(self) -> None:
        self.inner, _ = fake_worker()
        self.converted: list[dict[str, object]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def inFlight(self) -> int:  # a JS boolean arrives as a Python bool or int
        return int(self.inner.inFlight)

    def tokenize(self, text: str) -> TypedArray:
        return TypedArray(self.inner.tokenize(text))

    def pollDecode(self, timeoutMs: float) -> object:
        result = self.inner.pollDecode(timeoutMs)
        if result is None:
            return None
        stats = SimpleNamespace(**result["stats"])
        return SimpleNamespace(**{**result, "stats": stats})


def test_the_adapter_is_an_async_worker_the_machine_can_drive() -> None:
    channel = JsChannel()

    def to_js(value: dict[str, object]) -> object:
        channel.converted.append(value)
        return value

    worker: AsyncModelWorker = PyodideAsyncWorker(channel, to_js=to_js)
    assert worker.tokenize("write stdout") == channel.inner.tokenize("write stdout")
    machine = pilot(PyodideAsyncWorker(channel, to_js=to_js))  # pyright: ignore[reportArgumentType]
    assert words(until(machine, PILOT, op="read")) == ["write", " stdout", " left;"]
    assert channel.converted and all(c["maxChunk"] == 64 for c in channel.converted)
    assert worker.inFlight is False


def test_a_poll_answer_becomes_the_contract_dict() -> None:
    channel = JsChannel()
    worker = PyodideAsyncWorker(channel, to_js=lambda value: value)
    assert worker.pollDecode(0.0) is None
    channel.inner.createContext("1:pilot")
    channel.inner.append("1:pilot", channel.inner.tokenize("hello"))
    worker.beginDecodeStep("1:pilot", {"allowedBlocks": None, "allowedTokens": None})
    assert worker.inFlight is True
    worker.cancelDecode()
    assert worker.pollDecode(10.0) == {
        "cancelled": True,
        "resident": 0,
        "stats": {"positions": 0, "chunks": 0, "fillMs": 0.0},
    }
