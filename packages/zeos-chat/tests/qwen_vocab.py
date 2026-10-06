# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The real Qwen3.5 vocabulary for the chat machine's tests, without the model.

``QwenTokenizer`` runs ``tests/tokenizer_bridge.mjs`` under Node: the export's
``tokenizer.json`` through ``@huggingface/tokenizers``, tokenized as the model workers
tokenize (``encodePlain``), with each byte-level token's bytes (``pieceBytes``).
``QwenChatWorker`` is a ``ScriptedChatWorker`` over it: its replies are spelled as the
tokenizer spells text with its added tokens parsed, so ``<tool_call>`` is the model's own
id, and everything delivered is tokenized as plain text, as the page's worker does.

Tests using it skip without the export's tokenizer files (``ZEOS_OPT_MODEL_DIR``, by
default zeos-browser's ``models/Qwen3.5-4B-ZEOS-OPT``) or without Node and zeos-browser's
``npm install``. They read the tokenizer only: no weights, and nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from chat_workers import ScriptedChatWorker, attend_first
from zeos_browser.node_worker import PACKAGE, ModelInfo, NodeWorker

MODEL = Path(os.environ.get("ZEOS_OPT_MODEL_DIR", PACKAGE / "models" / "Qwen3.5-4B-ZEOS-OPT"))
BRIDGE = Path(__file__).resolve().parent / "tokenizer_bridge.mjs"


def available() -> bool:
    return (
        shutil.which("node") is not None
        and (PACKAGE / "node_modules" / "@huggingface" / "tokenizers").is_dir()
        and all(
            (MODEL / f).is_file() for f in ("meta.json", "tokenizer.json", "tokenizer_config.json")
        )
    )


def added_ids() -> dict[str, int]:
    """Every added token of the export, special or not, by its text."""
    tokenizer = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    return {t["content"]: int(t["id"]) for t in tokenizer["added_tokens"]}


class QwenTokenizer(NodeWorker):
    """``NodeWorker``'s frames over ``tokenizer_bridge.mjs``: the vocabulary, no model."""

    def __init__(self, model: Path = MODEL) -> None:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node is not on PATH")
        self._process = subprocess.Popen(
            [node, str(BRIDGE), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=PACKAGE,
        )
        self._next = 0
        self._buffer = bytearray()
        self._in_flight = None
        self._cancelled = False
        frame = self._read(None)
        assert frame is not None  # no timeout
        ready, _ = frame
        self.backend = str(ready["backend"])
        self._info: ModelInfo | None = None
        self._pieces = tuple(cast(list[str], self._call("pieces")))
        self._partial = None
        self._tokenized: dict[str, list[int]] = {}

    def tokenize(self, text: str) -> list[int]:
        cached = self._tokenized.get(text)
        if cached is None:
            cached = self._tokenized[text] = super().tokenize(text)
        return list(cached)

    def spell(self, text: str) -> list[int]:
        return list(cast(array[int], self._call("spell", text)))

    @property
    def pieces(self) -> tuple[str, ...]:
        return self._pieces


class QwenChatWorker(ScriptedChatWorker):
    """A ``ScriptedChatWorker`` whose vocabulary, tokenizer and reserved ids are Qwen3.5's."""

    def __init__(self, tokenizer: QwenTokenizer, replies: Sequence[str], **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        self.vocab = tokenizer.pieces
        self.ids = {}
        info = tokenizer.info()
        self._info = info
        self._control = tuple(info.controlIds)
        self._attend = kwargs.pop("attend", attend_first)
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        im_end = added_ids()["<|im_end|>"]
        self._tape = []
        for reply in replies:
            self._tape += self.spell(reply)
            if not reply.endswith("</tool_call>"):
                self._tape.append(im_end)
        self.contexts = {}
        self.seen = []

    def spell(self, text: str) -> list[int]:
        return self.tokenizer.spell(text)

    def info(self) -> Any:
        return self._info

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.tokenize(text)

    def pieceBytes(self, tokenId: int) -> bytes:
        return self.tokenizer.pieceBytes(tokenId)
