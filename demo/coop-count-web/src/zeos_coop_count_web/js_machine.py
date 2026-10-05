# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine backend whose model lives in JavaScript, reached through Pyodide.

``JsMachine`` is the browser's counterpart of ``LlamaMachine`` in
``demo/coop-count``: a ``SyscallSeat`` that keeps, per job, the kernel's words, the model
token ids behind them and how many ids each word occupies, and lets the inherited seat
turn decoded pieces into kernel requests one piece at a time. What it does not do is run
a model. Every token-level operation goes to a *worker*: a JavaScript object reached
through Pyodide's ``js`` module and passed in at construction, or under CPython a Python
object with the same methods (``zeos_coop_count_web.fake_worker.FakeWorker``).

The worker implements exactly this interface::

    interface ZeosModelWorker {
      // Model identity and reserved IDs. Call once.
      info(): { blockSize: number; padId: number; controlIds: number[]; eosId: number;
                vocabSize: number };
      tokenize(text: string): Int32Array;        // no BOS, no special-token parsing
      piece(tokenId: number): string;            // the text of one token
      createContext(jobId: string): void;
      destroyContext(jobId: string): void;
      length(jobId: string): number;             // model tokens currently resident
      // Prefill. Appends ids to the context and runs the forward pass for them.
      append(jobId: string, ids: Int32Array): void;
      // Drop every token at position >= n and the KV behind it.
      truncate(jobId: string, n: number): void;
      // Copy parent's tokens and KV into a fresh context for child.
      fork(parentId: string, childId: string): void;
      // One decode step, greedy unless opts.sample says otherwise. Returns the chosen
      // id and measured attention mass per KV block for this step, summed over layers
      // and heads and normalised so the values sum to 1.0 over blocks that received
      // attention. A backend that cannot measure returns attention = null.
      decodeStep(jobId: string, opts: {
        allowedBlocks: Uint8Array | null;       // 1 = may attend, indexed by block; null = all
        allowedTokens: Uint8Array | null;       // 1 = may emit, indexed by token id; null = all
        // Optional. Absent: the greedy choice. Present: sample among the topK allowed
        // ids at this temperature, taking the id whose cumulative probability first
        // exceeds u (see chat_machine.sample_index, which a worker must match).
        sample?: { temperature: number; topK: number; u: number };
      }): { tokenId: number; attention: Float32Array | null };
    }

How this class uses it, which is what a worker has to get right:

* **Context ids** are ``"<job>:<descriptor>"``. A worker treats them as opaque keys; the
  stub worker reads the descriptor from them to pick a tape. ``createContext`` and
  ``fork`` are only ever given an id that does not exist in the worker, and every other
  method only one that does.
* **Residency.** The worker's tokens for a context are always a prefix of the ids this
  class holds for it. Before each ``decodeStep`` it appends whatever lies past
  ``length()``, so the context is never empty when a step is asked for.
* **A decode step does not change the context.** The chosen id is appended by the
  ``append`` before the next step, as ``LlamaMachine`` feeds a sampled token on its
  next decode. The step computes the next-token distribution at the last resident
  position, attending only the allowed blocks, and reports that query's attention.
* **Blocks** in ``allowedBlocks`` and in ``attention`` are the worker's own:
  ``info().blockSize`` model tokens each, block ``b`` covering token positions
  ``[b * blockSize, (b + 1) * blockSize)``, one entry per block of the resident
  context. They are not the kernel's blocks, which count kernel words (``block_size``
  here) -- a word can be several model tokens. A worker block is allowed only if every
  word with a token in it lies in a kernel block the mask allows, so a block straddling
  a hidden segment is hidden whole; the mask fails closed. Measured mass on a worker
  block is shared among the kernel blocks with tokens in it, in proportion to how many.
  Both are exact when every word is one model token, as with the stub worker. A worker
  that reports mass on a block it was sent as 0 has attended what the mask hid, and
  this class raises ``MaskViolation``.
* **The mask's horizon.** The kernel builds its mask from the segments that exist when
  it installs it, at a block boundary or an inject, fork or splice, and decodes on until
  the next refresh. Kernel blocks at or past the block count at install time -- the
  horizon -- therefore cannot be named by the mask, and the only tokens that land there
  are the job's own decoded tokens and its padding, since every foreign arrival
  refreshes the mask first. They are allowed: hiding them would hide the very position
  doing the attending. ``visible_blocks`` reports the same set, so the attention the
  kernel sums is the attention the model could have paid. A ``trunc`` below the horizon
  lowers it, because the blocks it removed are gone and what is written there next is
  new.
* **Token reservation.** ``allowedTokens`` has one entry per vocabulary id. The pad id
  and the end-of-sequence id are always 0, and the ``controlIds`` are 0 unless the
  kernel enabled control tokens for the step; that is the sampler-side reservation the
  machine contract requires. A worker that returns an id the mask refused is a broken
  worker, and this class raises rather than letting the id through.
* **Vocabulary.** ``info().vocabSize`` is the number of ids, ``allowedTokens`` has that
  many entries, and ``piece`` is called once for every id below it at construction;
  every piece is kept, since the grammar mask needs them all (``token_mask``). The pad,
  end-of-sequence and control ids must lie below it.
* **Chat framing** (``chat_template="chatml"``, the default, as for ``LlamaMachine``)
  wraps the prompt and every arrival in ChatML turns. ``tokenize`` does not parse
  special tokens, so the turn markers are found among the ``controlIds``: the id whose
  ``piece`` is exactly ``<|im_start|>``, and the one whose piece is ``<|im_end|>``. The
  text between markers (``user\\n``, ``assistant\\n``, newlines) goes through
  ``tokenize``. A worker for a ChatML model must list both markers in ``controlIds``
  and return their literal text from ``piece``; this class refuses to start otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from zeos.core.ids import JobId, TokenKind
from zeos.machine.abi import DEFAULT, SyscallABI
from zeos.machine.base import (
    AttentionHint,
    ContextStats,
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    MaskViolation,
    OpKind,
    RawWindow,
    RawWord,
    SpliceResult,
    Token,
)
from zeos.machine.scripted import PAD_TOKEN
from zeos.machine.seat import SyscallParser, SyscallSeat

from zeos_coop_count_web.token_mask import CommandLanguage, RoundState, TokenMask

__all__ = [
    "Bridge",
    "DEFAULT_BLOCK_SIZE",
    "JsMachine",
    "PythonBridge",
    "WorkerViolation",
    "ZeosModelWorker",
]

DEFAULT_BLOCK_SIZE = 16

#: The ChatML turn markers, which must be control tokens of the worker's model.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: How far one step's attention may sum from 1.0. The worker reports float32, so a long
#: context accumulates rounding; a backend reporting raw weights is off by whole units.
_ATTENTION_TOLERANCE = 1e-3


class WorkerViolation(RuntimeError):
    """The worker answered outside the ZeosModelWorker contract."""


class ZeosModelWorker(Protocol):
    """The worker, as Python sees it. Method names are the JavaScript ones; arrays go in
    as whatever the ``Bridge`` makes of them."""

    def info(self) -> object: ...
    def tokenize(self, text: str) -> Iterable[int]: ...
    def piece(self, tokenId: int) -> str: ...
    def createContext(self, jobId: str) -> None: ...
    def destroyContext(self, jobId: str) -> None: ...
    def length(self, jobId: str) -> int: ...
    def append(self, jobId: str, ids: object) -> None: ...
    def truncate(self, jobId: str, n: int) -> None: ...
    def fork(self, parentId: str, childId: str) -> None: ...
    def decodeStep(self, jobId: str, opts: object) -> object: ...


class Bridge(Protocol):
    """How arguments cross into the worker and answers come back out.

    Under CPython the worker is a Python object and nothing needs converting
    (``PythonBridge``). Under Pyodide an id list must become an ``Int32Array``, a flag
    array a ``Uint8Array``, ``None`` a JavaScript ``null`` and the options a plain object
    (``zeos_coop_count_web.pyodide_bridge.PyodideBridge``).
    """

    def ids(self, ids: Sequence[int]) -> object: ...
    def options(
        self,
        allowed_blocks: bytes | None,
        allowed_tokens: bytes | None,
        sample: Mapping[str, float] | None = None,
    ) -> object: ...
    def floats(self, value: object) -> list[float] | None: ...


class PythonBridge:
    """The identity bridge, for a worker written in Python."""

    def ids(self, ids: Sequence[int]) -> object:
        return list(ids)

    def options(
        self,
        allowed_blocks: bytes | None,
        allowed_tokens: bytes | None,
        sample: Mapping[str, float] | None = None,
    ) -> object:
        opts: dict[str, object] = {"allowedBlocks": allowed_blocks, "allowedTokens": allowed_tokens}
        if sample is not None:
            opts["sample"] = dict(sample)
        return opts

    def floats(self, value: object) -> list[float] | None:
        if value is None:
            return None
        return [float(v) for v in cast("Iterable[float]", value)]


@dataclass
class _Context:
    """Per-job state: the worker's context id, the kernel's words and the ids behind them."""

    key: str
    descriptor: str
    parser: SyscallParser
    #: The kernel-visible token sequence T.
    tokens: list[Token] = field(default_factory=list[Token])
    #: The model token ids backing T, flat.
    ids: list[int] = field(default_factory=list[int])
    #: ``spans[i]`` is how many ids element ``i`` of T occupies, framing included.
    spans: list[int] = field(default_factory=list[int])
    #: ``framing[i]`` is how many of those ids are chat framing, before and after the word.
    framing: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    mask: frozenset[int] | None = None
    #: Kernel blocks at or past this index did not exist when the mask was installed.
    horizon: int = 0
    #: Where the current round stands in the command language.
    round: RoundState = ()
    tags: tuple[str, ...] = ("descriptor",)
    turn_open: bool = False
    turn_index: int | None = None
    spoken: bool = False

    def extend(self, tokens: Sequence[Token], ids: Sequence[int], spans: Sequence[int]) -> None:
        self.tokens.extend(tokens)
        self.ids.extend(ids)
        self.spans.extend(spans)
        self.framing.extend([(0, 0)] * len(spans))

    def kv_offset(self, at: int) -> int:
        """Turn a kernel offset into a model position, the one place the two units meet."""
        return sum(self.spans[:at])


class JsMachine(SyscallSeat):
    """A machine backend with one worker context per job, whose decode emits one token."""

    def __init__(
        self,
        worker: ZeosModelWorker,
        *,
        bridge: Bridge | None = None,
        abi: SyscallABI = DEFAULT,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        chat_template: str | None = "chatml",
        on_command: Callable[[JobId, str, MachineRequest], None] | None = None,
        on_arrival: Callable[[JobId, str], None] | None = None,
    ) -> None:
        super().__init__(
            abi=abi,
            descriptors=descriptors,
            valued=valued,
            on_command=on_command,
            on_arrival=on_arrival,
        )
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if chat_template not in (None, "chatml"):
            raise ValueError(f"unknown chat_template {chat_template!r}")
        self._worker = worker
        self._bridge: Bridge = bridge or PythonBridge()
        self._block_size = block_size
        self._chat_template = chat_template

        info = worker.info()
        self._worker_block = int(info.blockSize)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        if self._worker_block < 1:
            raise WorkerViolation(f"info().blockSize is {self._worker_block}; it must be >= 1")
        self._pad_id = int(info.padId)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        self._eos_id = int(info.eosId)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        self._control = frozenset(int(i) for i in info.controlIds)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
        size = int(info.vocabSize)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        if size < 1:
            raise WorkerViolation(f"info().vocabSize is {size}; it must be >= 1")
        outside = sorted(
            i for i in {self._pad_id, self._eos_id, *self._control} if not 0 <= i < size
        )
        if outside:
            raise WorkerViolation(
                f"info() names ids {outside} outside a vocabulary of {size} (vocabSize)"
            )
        self._pieces = tuple(str(worker.piece(i)) for i in range(size))
        self._mask = TokenMask(
            self._pieces, reserved=(self._pad_id, self._eos_id), control=self._control
        )
        self._languages: dict[str, CommandLanguage] = {}
        self._contexts: dict[JobId, _Context] = {}

        self._markers: dict[str, int] = {}
        if chat_template == "chatml":
            by_piece = {self._pieces[i]: i for i in sorted(self._control)}
            missing = [m for m in (IM_START, IM_END) if m not in by_piece]
            if missing:
                raise WorkerViolation(
                    f"chat_template='chatml' needs {missing} among the worker's controlIds, "
                    "each with its literal text as its piece; tokenize() parses no special "
                    "tokens, so a control id is the only way to reach one"
                )
            self._markers = {m: by_piece[m] for m in (IM_START, IM_END)}

    # -- lifecycle ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def vocabulary_size(self) -> int:
        return len(self._pieces)

    #: The chat turns, as LlamaMachine frames them.
    _CHATML_OPEN = f"{IM_START}user\n"
    _CHATML_TURN = f"{IM_END}\n{IM_START}assistant\n"
    _CHATML_ARRIVE = f"{IM_END}\n{IM_START}user\n"

    def _framing_ids(self, text: str) -> list[int]:
        """Framing text as ids: each turn marker by its control id, the rest tokenized."""
        out: list[int] = []
        rest = text
        while rest:
            found = [(rest.find(m), m) for m in self._markers if m in rest]
            if not found:
                out += self._tokenize(rest)
                break
            at, marker = min(found)
            out += self._tokenize(rest[:at])
            out.append(self._markers[marker])
            rest = rest[at + len(marker) :]
        return out

    def _tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        return [int(i) for i in self._worker.tokenize(text)]

    def _frame_into(self, ctx: _Context, text: str, index: int, *, before: bool) -> None:
        """Fold chat framing into the span of the token at ``index``, so no kernel offset moves."""
        ids = self._framing_ids(text)
        if not ids:
            return
        at = ctx.kv_offset(index) if before else ctx.kv_offset(index + 1)
        self._rewind(ctx, at)
        ctx.ids[at:at] = ids
        ctx.spans[index] += len(ids)
        head, tail = ctx.framing[index]
        ctx.framing[index] = (head + len(ids), tail) if before else (head, tail + len(ids))

    def _language(self, descriptor: str) -> CommandLanguage:
        language = self._languages.get(descriptor)
        if language is None:
            language = CommandLanguage(
                self.abi, self.aliases(descriptor), valued=self.valued(descriptor)
            )
            self._languages[descriptor] = language
        return language

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        if job in self._contexts:
            self.destroy_context(job)
        ctx = _Context(
            key=f"{job}:{descriptor}",
            descriptor=descriptor,
            parser=SyscallParser(self.abi),
            round=self._language(descriptor).start,
        )
        self._worker.createContext(ctx.key)
        self._contexts[job] = ctx

    def destroy_context(self, job: JobId) -> None:
        ctx = self._contexts.pop(job, None)
        if ctx is not None:
            self._worker.destroyContext(ctx.key)

    def _ctx_of(self, job: JobId) -> _Context:
        ctx = self._contexts.get(job)
        if ctx is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return ctx

    def stats(self, job: JobId) -> ContextStats:
        ctx = self._ctx_of(job)
        n = len(ctx.tokens)
        return ContextStats(
            resident_tokens=n,
            blocks=self._block_count(n),
            open_segment_tokens=n % self._block_size,
        )

    def _block_count(self, n: int) -> int:
        return (n + self._block_size - 1) // self._block_size

    # -- the worker's KV --------------------------------------------------------

    def _rewind(self, ctx: _Context, kv_at: int) -> None:
        """Drop the worker's tokens from ``kv_at`` on, if it holds any there."""
        if kv_at < self._worker.length(ctx.key):
            self._worker.truncate(ctx.key, kv_at)

    def _flush(self, ctx: _Context) -> None:
        """Append to the worker whatever ids it does not yet hold."""
        resident = int(self._worker.length(ctx.key))
        pending = ctx.ids[resident:]
        if pending:
            self._worker.append(ctx.key, self._bridge.ids(pending))

    def _encode(self, tokens: Sequence[Token], ctx: _Context) -> tuple[list[int], list[int]]:
        """Kernel tokens as ids and spans. A word's leading space marks a word boundary,
        which the very first word of a context lacks."""
        ids: list[int] = []
        spans: list[int] = []
        first = not ctx.ids
        for k, tok in enumerate(tokens):
            text = tok.text if (first and k == 0) else " " + tok.text
            word = self._tokenize(text) or [self._pad_id]
            ids.extend(word)
            spans.append(len(word))
        return ids, spans

    def _allowed_blocks(self, ctx: _Context) -> bytes | None:
        """The kernel's block mask in the worker's blocks, failing closed on a straddle."""
        if ctx.mask is None:
            return None
        size = self._worker_block
        count = (len(ctx.ids) + size - 1) // size
        hidden: set[int] = set()
        at = 0
        for index, span in enumerate(ctx.spans):
            block = index // self._block_size
            if span and block < ctx.horizon and block not in ctx.mask:
                hidden.update(range(at // size, (at + span - 1) // size + 1))
            at += span
        return bytes(0 if b in hidden else 1 for b in range(count))

    def _kernel_attention(
        self, ctx: _Context, mass: Sequence[float], sent: bytes | None
    ) -> dict[int, float]:
        """Measured mass per worker block, moved onto the kernel blocks under it. ``sent``
        is the ``allowedBlocks`` the step was given."""
        size = self._worker_block
        count = (len(ctx.ids) + size - 1) // size
        if len(mass) != count:
            raise WorkerViolation(
                f"attention has {len(mass)} entries for a context of {count} blocks"
            )
        total = sum(mass)
        if any(v < 0.0 for v in mass) or abs(total - 1.0) > _ATTENTION_TOLERANCE:
            raise WorkerViolation(
                f"attention must be non-negative and sum to 1.0 over blocks; it sums to {total}"
            )
        if sent is not None:
            attended = [b for b, value in enumerate(mass) if value > 0.0 and not sent[b]]
            if attended:
                raise MaskViolation(
                    f"the worker reported attention on blocks {attended}, which the step's "
                    "allowedBlocks hid"
                )
        # overlap[worker block][kernel block] = model tokens they share
        overlap: list[dict[int, int]] = [{} for _ in range(count)]
        at = 0
        for index, span in enumerate(ctx.spans):
            kernel_block = index // self._block_size
            for position in range(at, at + span):
                shares = overlap[position // size]
                shares[kernel_block] = shares.get(kernel_block, 0) + 1
            at += span
        out: dict[int, float] = {}
        for block, value in enumerate(mass):
            if value <= 0.0:
                continue
            shares = overlap[block]
            tokens = sum(shares.values())
            for kernel_block in sorted(shares):
                out[kernel_block] = (
                    out.get(kernel_block, 0.0) + value * shares[kernel_block] / tokens
                )
        return out

    # -- the five ops --------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        if self._chat_template == "chatml" and not ctx.turn_open and ctx.tokens:
            # Everything injected so far was the prompt, so close that turn and open the
            # model's own.
            ctx.turn_index = len(ctx.tokens) - 1
            self._frame_into(ctx, self._CHATML_TURN, ctx.turn_index, before=False)
            ctx.turn_open = True
        if not ctx.ids:
            raise RuntimeError(
                "cannot decode an empty context; the kernel injects the descriptor "
                "body before the first decode"
            )
        self._flush(ctx)

        language = self._language(ctx.descriptor)
        allowed = self._mask.allowed(
            ctx.descriptor, language, ctx.round, allow_control=allow_control
        )
        blocks = self._allowed_blocks(ctx)
        step = self._worker.decodeStep(ctx.key, self._bridge.options(blocks, allowed))
        tid = int(step.tokenId)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        if not (0 <= tid < len(allowed)) or not allowed[tid]:
            if tid in self._control and not allow_control:
                raise ControlTokenViolation(
                    f"job {job} was given control id {tid} while control tokens were disabled"
                )
            raise WorkerViolation(f"job {job}: the worker chose id {tid}, which the mask refused")
        # Read against the context the step attended, before the chosen id joins it.
        measured = self._bridge.floats(step.attention)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        attention = None if measured is None else self._kernel_attention(ctx, measured, blocks)

        piece = self._pieces[tid]
        token = Token(piece, TokenKind.CONTROL if tid in self._control else TokenKind.NORMAL)
        ctx.spoken = True

        before = len(ctx.tokens)
        ctx.extend([token], [tid], [1])
        after = len(ctx.tokens)
        ctx.round = language.advance(ctx.round, piece)

        request = self.consume(job, ctx.parser, piece)
        if request.op is not OpKind.NONE:
            # The round is over: the next command starts the language afresh.
            ctx.round = language.start
            ctx.parser.reset()

        return DecodeResult(
            tokens=(token,),
            request=request,
            attention=attention,
            # A worker that cannot measure leaves the kernel a guess, which it never
            # takes at its word: the hint is undeclared, so integrity demotes on
            # provenance alone.
            attention_hint=AttentionHint(tags=ctx.tags) if measured is None else None,
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        ctx = self._ctx_of(job)
        start = len(ctx.tokens)
        first = not ctx.tokens
        new_ids, new_spans = self._encode(tokens, ctx)
        ctx.extend(tokens, new_ids, new_spans)
        # An arrival lands mid-flight when the model's turn is open; anything before that
        # is still the prompt being put together.
        if tokens and self._chat_template == "chatml":
            if first:
                self._frame_into(ctx, self._CHATML_OPEN, start, before=True)
            elif ctx.turn_open:
                # Close the job's turn and let the sender speak; ``decode`` reopens the
                # job's turn later, so a run of arrivals shares one turn.
                self._frame_into(ctx, self._CHATML_ARRIVE, start, before=True)
                ctx.turn_open = False
                ctx.turn_index = None
        if ctx.spoken and tokens:
            self.note_arrival(job, " ".join(t.text for t in tokens))
        if tokens:
            # New content is the most likely thing the next step attends to.
            ctx.tags = ("self",)
        return start, len(ctx.tokens)

    def trunc(self, job: JobId, at: int) -> int:
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        dropped = len(ctx.tokens) - at
        if dropped == 0:
            return 0
        kv_at = ctx.kv_offset(at)
        self._rewind(ctx, kv_at)
        del ctx.tokens[at:]
        del ctx.ids[kv_at:]
        del ctx.spans[at:]
        del ctx.framing[at:]
        if ctx.turn_index is not None and at <= ctx.turn_index:
            ctx.turn_index = None
            ctx.turn_open = False
        ctx.horizon = min(ctx.horizon, self._block_count(at))
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        src = self._ctx_of(parent)
        self._flush(src)
        existing = self._contexts.get(child)
        if existing is None:
            existing = _Context(
                key=f"{child}:{src.descriptor}",
                descriptor=src.descriptor,
                parser=SyscallParser(self.abi),
                round=self._language(src.descriptor).start,
            )
            self._contexts[child] = existing
        else:
            # The worker forks into a fresh context only; the child keeps its own
            # descriptor, parser and round here, and only its tokens are replaced.
            self._worker.destroyContext(existing.key)
        self._worker.fork(src.key, existing.key)
        existing.tokens = list(src.tokens)
        existing.ids = list(src.ids)
        existing.spans = list(src.spans)
        existing.framing = list(src.framing)
        existing.turn_open = src.turn_open
        existing.turn_index = src.turn_index
        existing.spoken = src.spoken
        # A child starts from the parent's visibility and can only lose ground from there.
        existing.mask = src.mask
        existing.horizon = src.horizon
        return len(src.tokens)

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        """Replace a range of tokens, the one operation that moves the offsets after it."""
        ctx = self._ctx_of(job)
        if not 0 <= start <= end <= len(ctx.tokens):
            raise IndexError(f"splice [{start}, {end}) outside context of {len(ctx.tokens)}")
        downstream = len(ctx.tokens) - end

        kv_start = ctx.kv_offset(start)
        kv_end = ctx.kv_offset(end)
        tail_tokens = ctx.tokens[end:]
        tail_ids = ctx.ids[kv_end:]
        tail_spans = ctx.spans[end:]
        tail_framing = ctx.framing[end:]

        head = _Context(key=ctx.key, descriptor=ctx.descriptor, parser=ctx.parser)
        head.ids = ctx.ids[:kv_start]
        new_ids, new_spans = self._encode(tokens, head)

        self._rewind(ctx, kv_start)
        ctx.tokens[start:] = list(tokens) + tail_tokens
        ctx.ids[kv_start:] = new_ids + tail_ids
        ctx.spans[start:] = new_spans + tail_spans
        ctx.framing[start:] = [(0, 0)] * len(new_spans) + tail_framing
        return SpliceResult(tokens_in=len(tokens), invalidated_downstream=downstream)

    # -- the serving-stack contract ------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        """Install the allowed-block bitmap. The worker enforces it on every decode step."""
        ctx = self._ctx_of(job)
        ctx.mask = allowed_blocks
        ctx.horizon = self._block_count(len(ctx.tokens))

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        ctx = self._ctx_of(job)
        # No mask means every block, which is not the same as a mask allowing none.
        live = self._block_count(len(ctx.tokens))
        every = frozenset(range(live))
        if ctx.mask is None:
            return every
        # Blocks past the horizon hold only what the job decoded since; see the docstring.
        return (every & ctx.mask) | frozenset(range(ctx.horizon, live))

    def pad_to_block(self, job: JobId) -> int:
        ctx = self._ctx_of(job)
        remainder = len(ctx.tokens) % self._block_size
        if remainder == 0:
            return 0
        padding = self._block_size - remainder
        ctx.extend([PAD_TOKEN] * padding, [self._pad_id] * padding, [1] * padding)
        return padding

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        self._ctx_of(job)
        if end <= start:
            return frozenset()
        return frozenset(range(start // self._block_size, (end - 1) // self._block_size + 1))

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        return tuple(self._ctx_of(job).tokens)

    def raw(self, job: JobId) -> RawWindow:
        """Each word's pieces and the framing around them; framing after a word is
        reported ahead of the next, and after the last word as ``trailing``."""
        ctx = self._ctx_of(job)
        words: list[RawWord] = []
        carried: list[str] = []
        at = 0
        for span, (head, tail) in zip(ctx.spans, ctx.framing, strict=True):
            ids = ctx.ids[at : at + span]
            at += span
            own = span - head - tail
            framing = carried + [self._pieces[t] for t in ids[:head]]
            words.append(
                RawWord(
                    pieces=tuple(self._pieces[t] for t in ids[head : head + own]),
                    framing=tuple(framing),
                )
            )
            carried = [self._pieces[t] for t in ids[head + own :]]
        return RawWindow(
            words=tuple(words),
            kv_resident=int(self._worker.length(ctx.key)),
            trailing=tuple(carried),
        )

    # -- outside the Protocol -------------------------------------------------

    def lines(self, job: JobId) -> tuple[str, ...]:
        """The syscall lines this job has produced, for display only."""
        return tuple(self._ctx_of(job).parser.lines)

    def close(self) -> None:
        for job in list(self._contexts):
            self.destroy_context(job)
