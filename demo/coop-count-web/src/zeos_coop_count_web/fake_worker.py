# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A ``ZeosModelWorker`` with no model, in Python, for running ``JsMachine`` under CPython.

This is ``web/stub_worker.js`` written again in Python, and the two are kept to the same
behaviour: the Pyodide determinism test runs a case through ``JsMachine`` over each and
requires the same journal bytes. Neither has weights. What they have is:

* **a fixed vocabulary**: five reserved tokens (``<pad>``, ``<eos>``, ``<unk>``,
  ``<|im_start|>``, ``<|im_end|>``), then every word of the tapes and of the ChatML
  turn headers, each with and without a leading space, sorted;
* **a whitespace tokenizer**: each run of non-space characters is one token, written
  with a single leading space if any space preceded it, and ``<unk>`` if the vocabulary
  lacks it;
* **a tape per descriptor**: the ``emit`` steps of the case's ``script:`` blocks, the
  commands ``TapeSource`` plays. Each decode step gives the next word of the current
  command, split as ``zeos.machine.seat.words_of`` splits it, so a run decodes exactly
  the words the ``CommandSeat`` would;
* **no attention**: every step reports ``attention = None``.

The descriptor is read from the context id, which ``JsMachine`` writes as
``"<job>:<descriptor>"``. A step checks what it was given -- ``allowedBlocks`` sized to
the resident context, ``allowedTokens`` sized to the vocabulary -- and refuses to emit a
tape word the token mask forbids, so a run over this worker exercises every method of
the seam and the grammar mask on every step.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from zeos.machine.base import OpKind
from zeos.machine.scripted import Script, ScriptExhausted
from zeos.machine.seat import words_of

__all__ = ["FakeWorker", "ModelInfo", "Step", "SPECIAL", "tapes_from_scripts"]

#: The reserved tokens, in id order: pad, end of sequence, unknown, the ChatML markers.
SPECIAL = ("<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>")
PAD_ID, EOS_ID, UNK_ID = 0, 1, 2
CONTROL_IDS = (3, 4)
#: Words of the ChatML turn headers, so framing tokenizes to something other than unknown.
FRAMING_WORDS = ("user", "assistant")

#: ASCII whitespace only, written out because JavaScript's ``\\s`` and Python's differ
#: beyond it, and the two workers must tokenize alike.
_WORD = re.compile(r"[ \t\n\r\f\v]*[^ \t\n\r\f\v]+")
_SPACE = " \t\n\r\f\v"


def tapes_from_scripts(scripts: Mapping[str, Script]) -> dict[str, list[str]]:
    """Each descriptor's commands, from the ``emit`` steps ``TapeSource`` plays."""
    tapes: dict[str, list[str]] = {}
    for name, script in sorted(scripts.items()):
        commands: list[str] = []
        for index, step in enumerate(script.steps, start=1):
            if not step.emit or step.request.op is not OpKind.NONE:
                raise ValueError(
                    f"{name}: step {index} is not a command; a tape holds 'emit' steps only"
                )
            commands.append(step.emit)
        tapes[name] = commands
    return tapes


@dataclass(frozen=True, slots=True)
class ModelInfo:
    blockSize: int
    padId: int
    controlIds: tuple[int, ...]
    eosId: int


@dataclass(frozen=True, slots=True)
class Step:
    tokenId: int
    attention: None = None


@dataclass
class _Context:
    tape: Sequence[str]
    ids: list[int] = field(default_factory=list[int])
    #: Commands begun so far, so the next is ``tape[issued]``.
    issued: int = 0
    #: Words of the current command not yet emitted.
    pending: list[str] = field(default_factory=list[str])
    spoken: bool = False


class FakeWorker:
    """The stub worker in Python. Method names are the JavaScript interface's."""

    def __init__(
        self,
        tapes: Mapping[str, Sequence[str]],
        *,
        block_size: int = 16,
        terminator: str = ";",
    ) -> None:
        self._tapes = {k: tuple(v) for k, v in tapes.items()}
        self._block_size = block_size
        self._terminator = terminator
        words: set[str] = set(FRAMING_WORDS)
        for commands in self._tapes.values():
            for command in commands:
                words.update(
                    w.strip() for w in words_of(command, terminator=terminator, lead=False)
                )
        self._vocab = SPECIAL + tuple(sorted(words | {" " + w for w in words}))
        self._ids = {piece: i for i, piece in enumerate(self._vocab)}
        self._contexts: dict[str, _Context] = {}

    # -- identity ------------------------------------------------------------

    def info(self) -> ModelInfo:
        return ModelInfo(
            blockSize=self._block_size, padId=PAD_ID, controlIds=CONTROL_IDS, eosId=EOS_ID
        )

    def tokenize(self, text: str) -> list[int]:
        out: list[int] = []
        for match in _WORD.finditer(text):
            run = match.group(0)
            word = run.lstrip(_SPACE)
            piece = (" " + word) if len(word) < len(run) else word
            out.append(self._ids.get(piece, UNK_ID))
        return out

    def piece(self, tokenId: int) -> str:
        if not 0 <= tokenId < len(self._vocab):
            raise IndexError(f"token id {tokenId} is outside a vocabulary of {len(self._vocab)}")
        return self._vocab[tokenId]

    # -- contexts ------------------------------------------------------------

    def _ctx(self, job_id: str) -> _Context:
        ctx = self._contexts.get(job_id)
        if ctx is None:
            raise KeyError(f"no context {job_id!r}")
        return ctx

    def _fresh(self, job_id: str) -> _Context:
        if job_id in self._contexts:
            raise KeyError(f"context {job_id!r} already exists")
        _, _, descriptor = job_id.partition(":")
        return _Context(tape=self._tapes.get(descriptor, ()))

    def createContext(self, jobId: str) -> None:
        self._contexts[jobId] = self._fresh(jobId)

    def destroyContext(self, jobId: str) -> None:
        self._ctx(jobId)
        del self._contexts[jobId]

    def length(self, jobId: str) -> int:
        return len(self._ctx(jobId).ids)

    def append(self, jobId: str, ids: Sequence[int]) -> None:
        ctx = self._ctx(jobId)
        for token_id in ids:
            self.piece(int(token_id))
            ctx.ids.append(int(token_id))

    def truncate(self, jobId: str, n: int) -> None:
        ctx = self._ctx(jobId)
        if not 0 <= n <= len(ctx.ids):
            raise IndexError(f"truncate at {n} outside [0, {len(ctx.ids)}]")
        del ctx.ids[n:]

    def fork(self, parentId: str, childId: str) -> None:
        parent = self._ctx(parentId)
        child = self._fresh(childId)
        child.ids = list(parent.ids)
        child.spoken = parent.spoken
        self._contexts[childId] = child

    # -- decoding ------------------------------------------------------------

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> Step:
        ctx = self._ctx(jobId)
        if not ctx.ids:
            raise ValueError(f"decodeStep on {jobId!r}, which holds no tokens")
        blocks = opts["allowedBlocks"]
        expected = (len(ctx.ids) + self._block_size - 1) // self._block_size
        if blocks is not None and len(blocks) != expected:
            raise ValueError(f"allowedBlocks has {len(blocks)} entries for {expected} blocks")
        allowed = opts["allowedTokens"]
        if allowed is not None and len(allowed) != len(self._vocab):
            raise ValueError(f"allowedTokens has {len(allowed)} entries for {len(self._vocab)} ids")

        if not ctx.pending:
            if ctx.issued >= len(ctx.tape):
                raise ScriptExhausted(
                    f"{jobId} asked for command {ctx.issued + 1} of a tape with {len(ctx.tape)}"
                )
            ctx.pending = words_of(
                ctx.tape[ctx.issued], terminator=self._terminator, lead=ctx.spoken
            )
            ctx.issued += 1
        word = ctx.pending[0]
        token_id = self._ids[word]
        if allowed is not None and not allowed[token_id]:
            raise ValueError(f"{jobId}: the token mask refuses the tape's next word {word!r}")
        ctx.pending.pop(0)
        ctx.spoken = True
        return Step(tokenId=token_id)
