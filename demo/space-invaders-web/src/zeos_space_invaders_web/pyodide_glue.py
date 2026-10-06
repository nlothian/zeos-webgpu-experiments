# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The JavaScript worker channel as an ``AsyncModelWorker``, for code running in Pyodide.

coop-count-web's ``SyncModelWorker`` (``web/model_channel.js``) already has the four
non-blocking methods; what crosses the Pyodide boundary does not have the shapes the
machine and the prompt arm are typed against. ``PyodideAsyncWorker`` wraps the JsProxy:

* step options built as a Python dict (``PythonBridge``) become a plain JavaScript object,
  flag arrays as ``Uint8Array``; options the ``PyodideBridge`` built pass through, with
  ``maxChunk`` already set on them by the caller;
* ``pollDecode``'s answer becomes a ``DecodeDone`` or ``DecodeCancelled`` dict (``None``
  while running), with ``attention`` left a JsProxy for ``PyodideBridge.floats`` to copy
  out in one go;
* ``tokenize``'s ``Int32Array`` becomes a list in one copy rather than one proxy call per
  id; ``inFlight`` is read as a Python bool.

Everything else is passed straight through. The module imports under CPython, so its
conversions can be tested there; ``attach`` is what the page calls under Pyodide.
"""

# Pyodide's modules exist only inside Pyodide, so a type checker outside it knows nothing
# of them.
# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from zeos_coop_count_web.js_machine import Bridge

from zeos_space_invaders_web.contracts import DecodeCancelled, DecodeDone
from zeos_space_invaders_web.machine import normalise_poll

__all__ = ["PyodideAsyncWorker", "attach"]


def _to_js_object(value: dict[str, object]) -> object:
    from js import Object
    from pyodide.ffi import to_js

    return to_js(value, dict_converter=Object.fromEntries)


class PyodideAsyncWorker:
    """``AsyncModelWorker`` over a JavaScript ``SyncModelWorker``."""

    def __init__(
        self, js_worker: Any, *, to_js: Callable[[dict[str, object]], object] = _to_js_object
    ) -> None:
        self._js = js_worker
        self._to_js = to_js

    # -- the synchronous interface, passed through ----------------------------

    def info(self) -> object:
        return self._js.info()

    def tokenize(self, text: str) -> list[int]:
        ids = self._js.tokenize(text)
        to_memoryview = getattr(ids, "to_memoryview", None)
        if to_memoryview is not None:
            return cast("list[int]", to_memoryview().tolist())
        return [int(i) for i in cast("Iterable[int]", ids)]

    def piece(self, tokenId: int) -> str:
        return str(self._js.piece(tokenId))

    def createContext(self, jobId: str) -> None:
        self._js.createContext(jobId)

    def destroyContext(self, jobId: str) -> None:
        self._js.destroyContext(jobId)

    def length(self, jobId: str) -> int:
        return int(self._js.length(jobId))

    def append(self, jobId: str, ids: object) -> None:
        self._js.append(jobId, ids)

    def truncate(self, jobId: str, n: int) -> None:
        self._js.truncate(jobId, n)

    def fork(self, parentId: str, childId: str) -> None:
        self._js.fork(parentId, childId)

    def decodeStep(self, jobId: str, opts: object) -> object:
        return self._js.decodeStep(jobId, self._options(opts))

    # -- the non-blocking step -------------------------------------------------

    def _options(self, opts: object) -> object:
        if isinstance(opts, dict):
            return self._to_js(cast("dict[str, object]", opts))
        return opts

    def beginDecodeStep(self, jobId: str, opts: object) -> int:
        return int(self._js.beginDecodeStep(jobId, self._options(opts)))

    def pollDecode(self, timeoutMs: float) -> DecodeDone | DecodeCancelled | None:
        return normalise_poll(self._js.pollDecode(timeoutMs))

    def cancelDecode(self) -> None:
        self._js.cancelDecode()

    @property
    def inFlight(self) -> bool:
        return bool(self._js.inFlight)

    def markBroken(self, reason: str) -> None:
        """Make the JavaScript channel refuse every later call (``model channel
        unusable``), as its own call timeout does; it is shared by later runs."""
        self._js.broken = reason


def attach(js_worker: Any) -> tuple[PyodideAsyncWorker, Bridge]:
    """The channel's worker and the bridge for it, as the machine and the prompt arm
    take them. Pyodide only."""
    from zeos_coop_count_web.pyodide_bridge import PyodideBridge

    return PyodideAsyncWorker(js_worker), PyodideBridge()
