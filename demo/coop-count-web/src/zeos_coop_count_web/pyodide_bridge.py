# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The ``Bridge`` for a worker that is a JavaScript object. Imports only under Pyodide.

Pyodide passes a Python list or dict to JavaScript as a proxy, which a worker reading
``opts.allowedTokens`` or indexing an ``Int32Array`` cannot use, and it hands JavaScript
``null`` back as ``jsnull`` rather than ``None``. Everything that crosses goes through
here so ``JsMachine`` itself stays plain Python.
"""

# Pyodide's modules exist only inside Pyodide, so a type checker outside it knows nothing
# of them.
# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

from js import Int32Array, Object, Uint8Array
from pyodide.ffi import JsNull, jsnull, to_js

__all__ = ["PyodideBridge"]


class PyodideBridge:
    def ids(self, ids: Sequence[int]) -> object:
        return Int32Array.new(to_js(list(ids)))

    def _flags(self, flags: bytes | None) -> object:
        return jsnull if flags is None else Uint8Array.new(to_js(flags))

    def options(self, allowed_blocks: bytes | None, allowed_tokens: bytes | None) -> object:
        return to_js(
            {
                "allowedBlocks": self._flags(allowed_blocks),
                "allowedTokens": self._flags(allowed_tokens),
            },
            dict_converter=Object.fromEntries,
        )

    def floats(self, value: object) -> list[float] | None:
        if value is None or isinstance(value, JsNull):
            return None
        return [float(v) for v in cast("Iterable[float]", value)]
