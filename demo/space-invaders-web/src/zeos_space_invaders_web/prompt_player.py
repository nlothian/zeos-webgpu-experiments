# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The prompt-loop arm on the in-browser model, without blocking the game clock.

``BrowserPromptPlayer`` is the native ``PromptPlayer`` -- the same system prompt, the same
history of recent turns and the same reply parser -- with the request issued over the
worker channel a token at a time. It owns one worker context, framed in ChatML as
``JsMachine`` frames a job: the system prompt is a ``system`` turn prefilled once by
``warm`` into a prefix context that is kept; each request forks a fresh context from it
(a truncate would rewind to the worker's last KV snapshot below the prefix and replay the
rest every decision), appends a ``user`` turn holding the history
and the board, opens the ``assistant`` turn and decodes. Every step is begun and polled,
so ``poll`` returns within its timeout whether or not the reply is complete, and the run
loop keeps ticking the world while the model answers.

Replies are unconstrained, except that a control token other than ``<|im_end|>`` cannot
be emitted. A reply ends at ``<|im_end|>``, at the end-of-sequence id, at a newline, or
after ``max_new`` tokens, whichever comes first.
"""

# ``PromptPlayer`` is untyped, so what this class inherits from it is partly unknown.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

from zeos_coop_count_web.js_machine import IM_END, IM_START, Bridge, PythonBridge
from zeos_space_invaders.game import Rules
from zeos_space_invaders.players.base import PromptPlayer

from zeos_space_invaders_web.contracts import AsyncModelWorker, PromptReply
from zeos_space_invaders_web.machine import normalise_poll

__all__ = ["CONTEXT_ID", "PREFIX_ID", "BrowserPromptPlayer"]

#: The arm's worker contexts: the prefilled system prompt, and the decision forked from
#: it. The descriptor part says ``prompt`` to a stub worker.
PREFIX_ID = "prompt-prefix:prompt"
CONTEXT_ID = "prompt-arm:prompt"

#: How long one wait lasts when ``close`` drains a cancelled step.
_DRAIN_POLL_MS = 1000.0


class BrowserPromptPlayer(PromptPlayer):
    """``PromptArm`` over an ``AsyncModelWorker``; see the module docstring."""

    def __init__(
        self,
        worker: AsyncModelWorker,
        *,
        view: object,
        rules: Rules,
        history: int = 5,
        max_new: int = 6,
        bridge: Bridge | None = None,
        max_chunk: int | None = None,
    ) -> None:
        super().__init__(history=history, view=view, rules=rules)
        if max_new < 1:
            raise ValueError("max_new must be >= 1")
        self._worker = worker
        self._bridge: Bridge = bridge or PythonBridge()
        self.max_new = max_new
        self._max_chunk = max_chunk

        info: Any = worker.info()
        self._eos_id = int(info.eosId)
        self._pad_id = int(info.padId)
        control = [int(i) for i in info.controlIds]
        size = int(info.vocabSize)
        by_piece = {str(worker.piece(i)): i for i in control}
        missing = [m for m in (IM_START, IM_END) if m not in by_piece]
        if missing:
            raise ValueError(f"ChatML framing needs {missing} among the worker's controlIds")
        self._im_start = by_piece[IM_START]
        self._im_end = by_piece[IM_END]
        self._stops = frozenset({self._im_end, self._eos_id})
        flags = bytearray(b"\x01") * size
        for token_id in (self._pad_id, *control):
            flags[token_id] = 0
        flags[self._im_end] = 1
        flags[self._eos_id] = 1
        self._allowed = bytes(flags)
        self._options: object | None = None

        self._context = False
        self._decision = False
        self._prefix = 0
        self._ids: list[int] = []
        self._pieces: dict[int, str] = {}
        #: The request under way: its board, when it began, and what it has said.
        self._obs: str | None = None
        self._info: Mapping[str, object] = {}
        self._began = 0.0
        self._text: list[str] = []
        self._tokens = 0
        self._in_flight = False

        self.requests = 0
        self.parsed = 0
        self.replies: list[PromptReply] = []

    # -- framing -------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        return [int(i) for i in self._worker.tokenize(text)]

    def _turn(self, role: str, text: str, *, close: bool = True) -> list[int]:
        """``<|im_start|>role\\ntext<|im_end|>\\n``, the markers by their control ids."""
        ids = [self._im_start, *self._tokenize(f"{role}\n{text}")]
        if close:
            ids += [self._im_end, *self._tokenize("\n")]
        return ids

    def _piece(self, token_id: int) -> str:
        piece = self._pieces.get(token_id)
        if piece is None:
            piece = str(self._worker.piece(token_id))
            self._pieces[token_id] = piece
        return piece

    def _step_options(self) -> object:
        if self._options is None:
            options: object = self._bridge.options(None, self._allowed)
            if self._max_chunk is not None:
                if isinstance(options, dict):
                    options = cast("object", {**options, "maxChunk": self._max_chunk})
                else:
                    setattr(options, "maxChunk", self._max_chunk)  # noqa: B010 - a JsProxy attribute
            self._options = options
        return self._options

    # -- the arm -------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._obs is not None

    def warm(self) -> None:
        """Create the context and prefill the system prompt. A worker fills lazily, so
        one step is decoded and thrown away to make it compute the prefix now."""
        if self._context:
            return
        self._worker.createContext(PREFIX_ID)
        self._context = True
        prompt = cast("str", self.prompt)
        self._ids = self._turn("system", prompt)
        self._prefix = len(self._ids)
        self._worker.append(PREFIX_ID, self._bridge.ids(self._ids))
        self._worker.decodeStep(PREFIX_ID, self._bridge.options(None, None))

    def begin(self, obs: str, info: Mapping[str, object]) -> None:
        if self.busy:
            raise RuntimeError("a prompt request is already under way")
        self.warm()
        rendered = cast("str", self.render_turn(obs, info))
        turn = self._turn("user", rendered) + self._turn("assistant", "", close=False)
        if self._decision:
            self._worker.destroyContext(CONTEXT_ID)
        self._worker.fork(PREFIX_ID, CONTEXT_ID)
        self._decision = True
        self._ids = self._ids[: self._prefix] + turn
        self._worker.append(CONTEXT_ID, self._bridge.ids(turn))
        self._obs, self._info = obs, info
        self._began = time.monotonic()
        self._text, self._tokens = [], 0
        self.requests += 1
        self._begin_step()

    def _begin_step(self) -> None:
        self._worker.beginDecodeStep(CONTEXT_ID, self._step_options())
        self._in_flight = True

    def poll(self, timeout_s: float) -> PromptReply | None:
        if not self.busy:
            return None
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            remaining = max(deadline - time.monotonic(), 0.0)
            answer = normalise_poll(self._worker.pollDecode(remaining * 1000.0))
            if answer is None:
                return None
            self._in_flight = False
            if answer["cancelled"] is True:
                raise RuntimeError("the prompt arm's step was cancelled by someone else")
            token_id = answer["tokenId"]
            if token_id in self._stops:
                return self._finish()
            piece = self._piece(token_id)
            if "\n" in piece:
                self._text.append(piece.split("\n", 1)[0])
                self._tokens += 1
                return self._finish()
            self._text.append(piece)
            self._tokens += 1
            if self._tokens >= self.max_new:
                return self._finish()
            self._ids.append(token_id)
            self._worker.append(CONTEXT_ID, self._bridge.ids([token_id]))
            self._begin_step()
            if time.monotonic() >= deadline:
                return None

    def _finish(self) -> PromptReply:
        obs, info = self._obs, self._info
        assert obs is not None
        text = "".join(self._text).strip()
        action = cast("str | None", self.parse(text))
        self.record(obs, info, text, "", "stop", {})
        reply = PromptReply(
            text=text,
            action=action,
            latency=round(time.monotonic() - self._began, 4),
            tokens=self._tokens,
        )
        self._obs, self._info = None, {}
        self.parsed += reply.parsed
        self.replies.append(reply)
        return reply

    @property
    def parse_rate(self) -> float | None:
        """The share of finished requests whose reply named an action."""
        return None if not self.replies else self.parsed / len(self.replies)

    def close(self) -> None:
        """Cancel any request under way and destroy both contexts."""
        if self._in_flight:
            self._worker.cancelDecode()
            while normalise_poll(self._worker.pollDecode(_DRAIN_POLL_MS)) is None:
                pass
            self._in_flight = False
        self._obs, self._info = None, {}
        if self._decision:
            self._worker.destroyContext(CONTEXT_ID)
            self._decision = False
        if self._context:
            self._worker.destroyContext(PREFIX_ID)
            self._context = False
