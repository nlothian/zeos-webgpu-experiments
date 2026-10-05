# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine for a chat agent: free text and Qwen tool calls in, pipe requests out.

``JsMachine`` decodes the syscall ABI and nothing else. A chat agent speaks prose, and
asks for things the way Qwen3.5 was trained to: a ChatML assistant turn that ends either
with ``<|im_end|>`` or with a tool call::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

``ChatToolMachine`` keeps ``JsMachine``'s worker, bookkeeping and masks, and replaces the
two halves that know the ABI. It runs on any ``ZeosModelWorker``.

**What it asks the kernel for.** The model's words are never parsed as commands. The
machine turns three moments of a turn into requests itself:

* ``</tool_call>`` closing a call that parses: a ``WRITE_READ`` to ``tools.read`` or
  ``tools.effect`` -- the host's tool-class table decides which, and a tool it does not
  name is an effect -- with the JSON ``{"name": ..., "arguments": {...}}`` as payload,
  then a read of ``tools.results`` -- or of ``tools.results.trusted`` when the host's
  trusted-results table names the call (below). The arguments are decoded as the app's parser
  decodes them (``parseQwenToolCallBody`` in the gemma-data-agent site): a parameter
  the tool declares as a string is kept verbatim, anything else is JSON when it parses.
  A block that does not parse stays text, as it does there.
* ``<|im_end|>``: intercepted. It is never appended or reported as a token. The machine
  asks for a ``WRITE_READ`` of the turn's text to ``chat.out`` and a read of
  ``chat.user``, and the marker becomes framing in front of whatever arrives next.
* A write that was refused, so its read never happened: the kernel answers a refusal
  with a ``FAULT`` notice and lets the job run on (``on_fault: retry``). If nothing but
  kernel frames has arrived since the machine asked to read, its next step asks for the
  read again rather than decoding. So a refused tool call waits on ``tools.results`` for
  the host to settle it -- the result of running it with the user's approval, or a
  refusal -- and the model reads the notice and the settlement together.

**Tool classes** are ``"read"``, ``"effect"``, or a rule, ``{"read_if": {PARAM: PATTERN}}``:
the call is a read when its arguments are exactly the rule's parameters, each a string
that ``PATTERN`` matches in full (``re.fullmatch``, case-insensitive, ``.`` matching a
newline), and an effect otherwise. A tool whose class depends on what it is asked -- SQL
that only reads, say -- is classified here, from the call the model wrote, so the
kernel's capability check on ``tools.effect`` is the only thing that decides whether it
runs without the user. Patterns should stay inside the syntax Python's ``re`` and
JavaScript's ``RegExp`` share, so a host can show the same verdict before it asks.

**Trusted results.** A host whose tool returns text it wrote itself -- a reference card
bundled with the app, say -- can say so: ``trusted_results`` maps a tool name to
``{PARAM: [VALUE, ...]}``, and a call whose arguments are exactly the rule's parameters,
each a string equal to one of its values, reads its result from ``tools.results.trusted``
(TRUSTED, principal ``device``) rather than ``tools.results`` (EXTERNAL). The match is
exact and case-sensitive, with no patterns: unlike ``read_if``, which only routes a call
to a sink the kernel still checks, this rule raises the ring of what the job reads, so
``SQL`` or ``sql\n`` for a host that bundles ``sql`` is an unknown name and stays on
ring 3. The choice is made from the call the model wrote, before
anything is run, so the kernel knows which ring the answer will carry while the job
waits for it. The result is framed as any other: a ``<tool_response>`` in a ``user``
turn.

**History.** ``replay`` sets a job that has not spoken yet to read a list of pipes in
order before it decodes anything: a past conversation, delivered one turn at a time. A
delivery on ``chat.history`` or ``chat.history.trusted`` is a past assistant turn, framed
as the app's own renderer frames one (``<|im_start|>assistant\n`` and the text, with no
thinking prefix); its ring is the pipe's, so the host says how far it trusts its own
transcript by the pipe it picks. Once the list is read, a job whose context ends in a
replayed past reads ``chat.user`` next, whatever its last turn was.

A step that runs no forward pass reports ``attention={}``: it attended nothing. The
first step of a job that has had no message yet is such a step, a read of ``chat.user``.

**Framing.** The transcript is Qwen3.5's chat template, folded into the spans of the
kernel's words exactly as ``JsMachine`` folds its own, so no kernel offset moves:

* the descriptor body is the ``system`` turn;
* a message on ``chat.user`` is a ``user`` turn;
* a result on ``tools.results`` or ``tools.results.trusted`` is a ``user`` turn holding
  ``<tool_response>\\n...\\n</tool_response>``, and a second one before the model speaks
  joins it, as consecutive tool messages do in the template;
* a kernel frame (a ``FAULT`` notice, say) joins the ``user`` turn, opening one if the
  model's turn was open;
* the model's turn opens with ``<|im_start|>assistant\\n`` and a thinking prefix:
  ``<think>\\n`` with thinking on, the template's empty ``<think>\\n\\n</think>\\n\\n``
  with it off.

The turn markers, ``<think>``, ``</think>``, ``<tool_response>`` and ``</tool_response>``
are written as the worker's own ids for those pieces when its vocabulary has them, as a
prompt rendered by the template would be; ``tokenize`` parses no special tokens, so
nothing that arrives as content can produce one. A word the kernel injects is written
as it is when it carries its own leading whitespace (``KernelConfig.preserve_whitespace``)
or is the first of its injection, and after a space otherwise, so a newline in a tool
result reaches the model as a newline.

**What the model may emit.** Every id but: the pad id; the end-of-sequence id unless it
is ``<|im_end|>``; a control id, unless its piece is ``<|im_end|>``, ``<tool_call>``,
``</tool_call>``, ``<think>`` or ``</think>`` -- which the machine reads as text and never
reports to the kernel as ``CONTROL``, since none is a kernel frame; with thinking off, any
piece that is exactly ``<think>`` or ``</think>``, since the turn's think block is already
closed; an empty piece; and
any piece that would complete a banned tag, ``</?(KERNEL|RESUME|FAULT|STATUS|STUB)`` or
``</?tool_response``, across the pieces of the turn so far (``FrameGuard``), the first three
in any case and through invisible or look-alike characters, as the kernel matches them
(``zeos.core.framing``). The kernel's
frames ride on ``CONTROL`` tokens and already cannot be forged; what the guard adds is
that the model's own text never spells one, so nothing it wrote can later be mistaken
for one, and it cannot fake the shape of a tool result. Masks are cached per guard state.

**The tool's name, chosen masked** (``mask_tool_choice``). From the step after the model
emits ``<tool_call>`` until the piece that closes ``<function=NAME>``, the deliveries on
``hidden_pipes`` -- the EXTERNAL ones, tool results and replayed turns, as ``open_chat``
names them -- are hidden from the model: their own ids, not the framing around them, so
the model still sees that a tool answered and that its turn began. The arguments and the
rest of the turn see everything again. The tool is then chosen without reading what the
tools returned, though what the model wrote after reading it stays in view. On those
steps ``visible_blocks`` leaves out every kernel block that holds a hidden word, and the
step's measured mass on such a block -- which can only be on the framing or the model's
own text there, since the worker hid the rest -- is dropped rather than credited to the
hidden segment, so a hidden segment neither demotes the job nor raises ``mask.denied``
(on a worker that measures attention; one that cannot leaves the kernel its usual guess).
The call records it (``ToolCall.name_masked``, ``name_hidden``).

**Sampling.** Greedy unless ``sampling`` is given; then every step sends
``opts.sample = {temperature, topK, u}``, ``u`` drawn from a ``random.Random`` per job
seeded from the run's seed and the job id, so a run is as reproducible as its worker's
logits. ``sample_index`` is the rule a worker applies, written once here.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from zeos.core.framing import FOLDED_FRAMES, FRAMES, fold
from zeos.core.ids import JobId, PipeName, TokenKind
from zeos.machine.base import (
    AttentionHint,
    DecodeResult,
    MachineRequest,
    OpKind,
    SpliceResult,
    Token,
    tokens_from_text,
)
from zeos_browser.js_machine import (
    DEFAULT_BLOCK_SIZE,
    IM_END,
    IM_START,
    Bridge,
    JsMachine,
    WorkerViolation,
    ZeosModelWorker,
    _Context,  # pyright: ignore[reportPrivateUsage]
)

__all__ = [
    "BANNED_TAGS",
    "ArgumentRule",
    "EFFECT",
    "READ",
    "ToolClass",
    "ChatPipes",
    "ChatToolMachine",
    "FrameGuard",
    "GUARD_START",
    "GuardState",
    "Sampling",
    "ToolCall",
    "format_tool_call",
    "parse_tool_call_body",
    "sample_index",
    "thinking_prefix",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
EMPTY_THINK = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

#: Tag names whose opening or closing the model may never spell: the kernel's frames,
#: and the frame of a tool result.
BANNED_TAGS: tuple[str, ...] = (*FRAMES, "tool_response")

#: Control pieces the model may emit. None is a kernel frame; the machine reads them as
#: text, and intercepts ``<|im_end|>``.
EMITTABLE_CONTROL = frozenset({IM_END, TOOL_CALL_OPEN, TOOL_CALL_CLOSE, THINK_OPEN, THINK_CLOSE})

#: Pieces written as the worker's own id when it has one, rather than tokenized.
FRAMING_MARKERS = (
    IM_START,
    IM_END,
    THINK_OPEN,
    THINK_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
)

#: The two tool classes, as the host's table names them.
READ = "read"
EFFECT = "effect"

#: A rule over a call's arguments: exactly these parameters, each a string the pattern
#: matches in full.
ArgumentRule = Mapping[str, str]

#: A trusted-results rule: exactly these parameters, each a string equal to one of the
#: listed values -- compared exactly and case-sensitively, never as a pattern.
ExactRule = Mapping[str, Collection[str]]

#: A tool's class in the host's table: ``READ``, ``EFFECT``, or ``{"read_if": {param:
#: pattern}}``, a read only when the arguments are exactly those parameters and each is a
#: string the pattern matches in full.
ToolClass = str | Mapping[str, Mapping[str, str]]

_READ_IF = "read_if"


def thinking_prefix(thinking: bool) -> str:
    """What Qwen3.5's template pre-fills after ``<|im_start|>assistant\\n``."""
    return f"{THINK_OPEN}\n" if thinking else EMPTY_THINK


@dataclass(frozen=True, slots=True)
class ChatPipes:
    """The pipes a chat case binds, by role."""

    user: PipeName = PipeName("chat.user")
    out: PipeName = PipeName("chat.out")
    results: PipeName = PipeName("tools.results")
    #: Results the host wrote itself, for calls its trusted-results table names.
    results_trusted: PipeName = PipeName("tools.results.trusted")
    read: PipeName = PipeName("tools.read")
    effect: PipeName = PipeName("tools.effect")
    #: Past assistant turns, at EXTERNAL and at TRUSTED.
    history: PipeName = PipeName("chat.history")
    history_trusted: PipeName = PipeName("chat.history.trusted")

    @property
    def tool_sinks(self) -> tuple[PipeName, PipeName]:
        return (self.read, self.effect)

    @property
    def result_pipes(self) -> tuple[PipeName, PipeName]:
        return (self.results, self.results_trusted)

    @property
    def history_pipes(self) -> tuple[PipeName, PipeName]:
        return (self.history, self.history_trusted)


@dataclass(frozen=True, slots=True)
class Sampling:
    """Seeded sampling: among the ``top_k`` allowed ids, at this temperature."""

    temperature: float = 0.7
    top_k: int = 20

    def __post_init__(self) -> None:
        if not self.temperature > 0.0:
            raise ValueError("temperature must be above 0; greedy is sampling=None")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


def sample_index(
    logits: Sequence[float],
    allowed: bytes | None,
    *,
    temperature: float,
    top_k: int,
    u: float,
) -> int:
    """The id a worker chooses for ``opts.sample``, which every worker must match.

    Rank the allowed ids by logit, highest first and the lower id on a tie, and keep the
    first ``top_k``. Weight each by ``exp((logit - best) / temperature)``, and return the
    first whose running total exceeds ``u`` times the sum, or the last if rounding leaves
    none. ``web/transformers_worker.js`` implements the same in ``sampleToken``.
    """
    ranked = sorted(
        (i for i in range(len(logits)) if allowed is None or (i < len(allowed) and allowed[i])),
        key=lambda i: (-logits[i], i),
    )[:top_k]
    if not ranked:
        raise ValueError("allowedTokens permits no token")
    best = logits[ranked[0]]
    weights = [math.exp((logits[i] - best) / temperature) for i in ranked]
    threshold = u * sum(weights)
    running = 0.0
    for token_id, weight in zip(ranked, weights, strict=True):
        running += weight
        if running > threshold:
            return token_id
    return ranked[-1]


class _Track:
    """One half of a ``FrameGuard``: banned patterns over one view of the pieces.

    A state is the longest suffix of the view so far that is a proper prefix of a
    pattern, so ``""`` almost always. A piece is banned from a state when the state and
    the piece's view together contain a pattern: either inside the piece, which is the
    same from every state, or starting in the state and finished by the piece's head.
    """

    def __init__(self, patterns: Iterable[str], views: Sequence[str]) -> None:
        self.patterns = tuple(sorted(set(patterns)))
        self._prefixes = frozenset(p[:k] for p in self.patterns for k in range(1, len(p)))
        self._longest = max((len(p) for p in self.patterns), default=0)
        self._regex = (
            re.compile("|".join(re.escape(p) for p in self.patterns)) if self.patterns else None
        )
        self._views = tuple(views)
        self._inside = frozenset(i for i, v in enumerate(self._views) if self.spells(v))
        self._cache: dict[str, frozenset[int]] = {"": self._inside}

    def spells(self, view: str) -> bool:
        return self._regex is not None and self._regex.search(view) is not None

    def banned(self, state: str) -> frozenset[int]:
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        heads = tuple(
            sorted(
                {
                    pattern[len(tail) :]
                    for k in range(len(state))
                    if (tail := state[k:]) in self._prefixes
                    for pattern in self.patterns
                    if pattern.startswith(tail)
                }
            )
        )
        finished: frozenset[int] = frozenset(
            i for i, v in enumerate(self._views) if heads and v.startswith(heads)
        )
        result = self._inside | finished
        self._cache[state] = result
        return result

    def advance(self, state: str, view: str) -> str:
        text = state + view
        for length in range(min(len(text), self._longest - 1), 0, -1):
            if text[-length:] in self._prefixes:
                return text[-length:]
        return ""


#: A ``FrameGuard`` state: the exact track's, then the folded track's.
GuardState = tuple[str, str]

#: The state of a turn that has said nothing yet.
GUARD_START: GuardState = ("", "")


class FrameGuard:
    """Which pieces would complete a banned tag, given what the turn has said so far.

    A banned pattern is ``<NAME`` or ``</NAME``. The guard follows the kernel's imitation
    rule (``zeos.core.framing.spells_frame``) over its own names and in two tracks, so a
    state is a pair (``GuardState``). A name in ``FOLDED_FRAMES`` (``KERNEL``, ``RESUME``,
    ``FAULT``) is matched in each piece's fold (``zeos.core.framing.fold``): whatever its
    case, with invisible format characters dropped, NFKC look-alikes such as ``＜ＫＥＲＮＥＬ``
    and the Cyrillic and Greek homoglyphs read as Latin, and across pieces, since the fold
    of a concatenation is the concatenation of the folds. Every other name (``STATUS``,
    ``STUB``, ``tool_response``) is matched exactly and case-sensitively. Matching is
    anywhere in the text, so ``x<KERNEL`` is banned.

    It is at least as strict as the kernel: whatever the kernel alarms on contains a
    pattern of one track or the other, since ``<KERNEL`` in the text is ``<KERNEL`` in its
    fold. It is stricter in one way: the model's next piece is not yet known, so
    ``<KERNELS`` and ``<kernels`` are banned too, where the kernel would not alarm on them.
    """

    def __init__(self, pieces: Sequence[str], tags: Iterable[str] = BANNED_TAGS) -> None:
        names = sorted(set(tags))
        if not names:
            raise ValueError("a guard needs at least one tag")
        folded = [n for n in names if n in FOLDED_FRAMES]
        exact = [n for n in names if n not in FOLDED_FRAMES]
        self._pieces = tuple(pieces)
        self._exact = _Track(_tag_patterns(exact), self._pieces)
        self._folded = _Track(_tag_patterns(folded), tuple(map(fold, self._pieces)))
        self._cache: dict[GuardState, frozenset[int]] = {}

    @property
    def patterns(self) -> tuple[str, ...]:
        """Every banned pattern: the exact track's as written, the folded track's as
        they appear in a fold."""
        return tuple(sorted(self._exact.patterns + self._folded.patterns))

    def banned(self, state: GuardState) -> frozenset[int]:
        cached = self._cache.get(state)
        if cached is None:
            cached = self._exact.banned(state[0]) | self._folded.banned(state[1])
            self._cache[state] = cached
        return cached

    def advance(self, state: GuardState, piece: str) -> GuardState:
        """The state after ``piece``, which must not have been banned from ``state``."""
        return (self._exact.advance(state[0], piece), self._folded.advance(state[1], fold(piece)))

    def spells(self, text: str) -> bool:
        """Whether ``text`` contains a banned pattern anywhere."""
        return self._exact.spells(text) or self._folded.spells(fold(text))


def _tag_patterns(names: Iterable[str]) -> list[str]:
    return [f"{opening}{n}" for n in names for opening in ("<", "</")]


# -- tool calls, as the app's Qwen parser reads and writes them -------------------------

_FUNCTION = re.compile(r"\s*<function=([^>\n]+)>(.*?)</function>\s*", re.DOTALL)
_FUNCTION_OPEN = "<function="
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _strip_one_newline(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _convert(raw: str, kind: str | None) -> Any:
    if kind == "string":
        return raw
    try:
        return json.loads(raw.strip())
    except ValueError:
        return raw


def parse_tool_call_body(
    inner: str, param_types: Mapping[str, Mapping[str, str]] | None = None
) -> tuple[str, dict[str, Any]] | None:
    """The name and arguments inside ``<tool_call>...</tool_call>``, or None if it does
    not parse. ``param_types[tool][param]`` is the parameter's JSON-schema type."""
    match = _FUNCTION.fullmatch(inner)
    if match is None:
        return None
    name = match.group(1).strip()
    if not name:
        return None
    body = match.group(2)
    types = (param_types or {}).get(name, {})
    arguments: dict[str, Any] = {}
    last = 0
    for param in _PARAMETER.finditer(body):
        # Only whitespace may sit between parameters.
        if body[last : param.start()].strip():
            return None
        key = param.group(1).strip()
        kind = types.get(key)
        arguments[key] = _convert(
            _strip_one_newline(param.group(2)), kind.lower() if kind else None
        )
        last = param.end()
    if body[last:].strip():
        return None
    return name, arguments


def format_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """One call as Qwen3.5's template renders it. A string value is written as it is and
    anything else as the template's ``tojson``; neither is escaped here."""
    out = f"{TOOL_CALL_OPEN}\n<function={name}>\n"
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        out += f"<parameter={key}>\n{text}\n</parameter>\n"
    return out + f"</function>\n{TOOL_CALL_CLOSE}"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call the machine asked the kernel to write."""

    #: Counts this job's calls from 0.
    index: int
    name: str
    arguments: Mapping[str, Any]
    sink: PipeName
    #: The ``<tool_call>...</tool_call>`` text the model produced.
    raw: str
    #: Where its result is read from: ``tools.results``, or ``tools.results.trusted``
    #: for a call the trusted-results table names.
    results: PipeName = PipeName("tools.results")
    #: Whether the name was chosen with deliveries hidden (``mask_tool_choice``).
    name_masked: bool = False
    #: The kernel offsets ``[start, end)`` of the deliveries hidden while it was chosen.
    name_hidden: tuple[tuple[int, int], ...] = ()

    @property
    def payload(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False)


# -- the machine ----------------------------------------------------------------------

_SYSTEM = "system"
_USER = "user"
_ASSISTANT = "assistant"


@dataclass
class _Chat:
    """Per-job chat state, beside ``JsMachine``'s per-job context."""

    #: The role of the turn the context ends in, or None before anything was injected.
    role: str | None = None
    #: The pipe the machine last asked to read, until something other than a kernel
    #: frame arrives.
    awaiting: PipeName | None = None
    #: Whether content arrived on ``awaiting``.
    arrived: bool = False
    #: Whether the context ends in a tool response, so the next joins its turn.
    in_tool_response: bool = False
    #: What the model has decoded in the open assistant turn.
    text: str = ""
    #: Where in ``text`` the search for the next tool call starts.
    scan: int = 0
    #: The frame guard's state across the open turn.
    guard: GuardState = GUARD_START
    calls: list[ToolCall] = field(default_factory=list[ToolCall])
    rng: random.Random | None = None
    #: Pipes still to read, in order, before the job decodes (``replay``).
    replay: list[PipeName] = field(default_factory=list[PipeName])
    #: Whether a replay has yet to hand over to ``chat.user`` once its pipes are read.
    replay_tail: bool = False
    #: Kernel offsets ``[start, end)`` of deliveries on the hidden pipes, in order.
    hidden: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    #: Kernel blocks the latest step hid, which ``visible_blocks`` leaves out.
    narrowed: frozenset[int] = frozenset()
    #: Where in ``text`` the ``<tool_call>`` whose name was chosen masked opens, and what
    #: was hidden then.
    masked_call: int | None = None
    masked_hidden: tuple[tuple[int, int], ...] = ()

    def copy(self) -> _Chat:
        clone = _Chat(
            role=self.role,
            awaiting=self.awaiting,
            arrived=self.arrived,
            in_tool_response=self.in_tool_response,
            text=self.text,
            scan=self.scan,
            guard=self.guard,
            calls=list(self.calls),
            replay=list(self.replay),
            replay_tail=self.replay_tail,
            hidden=list(self.hidden),
            narrowed=self.narrowed,
            masked_call=self.masked_call,
            masked_hidden=self.masked_hidden,
        )
        if self.rng is not None:
            clone.rng = random.Random()
            clone.rng.setstate(self.rng.getstate())
        return clone


class ChatToolMachine(JsMachine):
    """A ``JsMachine`` whose job speaks Qwen3.5 chat and tool calls rather than the ABI."""

    def __init__(
        self,
        worker: ZeosModelWorker,
        *,
        tool_classes: Mapping[str, ToolClass],
        bridge: Bridge | None = None,
        pipes: ChatPipes | None = None,
        thinking: bool = False,
        assistant_prefix: str | None = None,
        sampling: Sampling | None = None,
        seed: int = 0,
        param_types: Mapping[str, Mapping[str, str]] | None = None,
        trusted_results: Mapping[str, ExactRule] | None = None,
        banned_tags: Iterable[str] = BANNED_TAGS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        mask_tool_choice: bool = False,
        hidden_pipes: Iterable[PipeName] | None = None,
    ) -> None:
        super().__init__(worker, bridge=bridge, block_size=block_size, chat_template="chatml")
        self.pipes = pipes or ChatPipes()
        self._tool_classes: dict[str, str] = {}
        self._read_if: dict[str, dict[str, re.Pattern[str]]] = {}
        for name, kind in tool_classes.items():
            if isinstance(kind, str):
                if kind not in (READ, EFFECT):
                    raise ValueError(
                        f"{name}: a tool class is {READ!r}, {EFFECT!r} or a rule, not {kind!r}"
                    )
                self._tool_classes[name] = kind
                continue
            if set(kind) != {_READ_IF} or not isinstance(kind[_READ_IF], Mapping):
                raise ValueError(f"{name}: a rule is {{{_READ_IF!r}: {{param: pattern}}}}")
            self._read_if[name] = _compile_rule(kind[_READ_IF])
        self._trusted: dict[str, dict[str, frozenset[str]]] = {
            name: _exact_rule(name, rule) for name, rule in (trusted_results or {}).items()
        }
        self.thinking = thinking
        self._prefix = thinking_prefix(thinking) if assistant_prefix is None else assistant_prefix
        self._sampling = sampling
        self._seed = seed
        self._param_types = {k: dict(v) for k, v in (param_types or {}).items()}
        self._chats: dict[JobId, _Chat] = {}
        self._tokenized: dict[str, tuple[int, ...]] = {}
        self._mask_tool_choice = mask_tool_choice
        self._hidden_pipes = frozenset(
            (self.pipes.results, self.pipes.history) if hidden_pipes is None else hidden_pipes
        )

        pieces = self._pieces
        # The markers JsMachine found among the control ids, and the rest of the
        # template's markers wherever the vocabulary has them, control ids first.
        for marker in FRAMING_MARKERS:
            if marker in self._markers:
                continue
            ids = [i for i in sorted(self._control) if pieces[i] == marker] or [
                i for i, p in enumerate(pieces) if p == marker
            ]
            if ids:
                self._markers[marker] = ids[0]
        self._im_end = self._markers[IM_END]

        base = bytearray(b"\x01" * len(pieces))
        for token_id, piece in enumerate(pieces):
            if (
                not piece
                or token_id == self._pad_id
                or (token_id == self._eos_id and piece != IM_END)
                or (token_id in self._control and piece not in EMITTABLE_CONTROL)
                # With thinking off the template closed the think block already; another
                # would start reasoning the host shows as the reply.
                or (not thinking and piece in (THINK_OPEN, THINK_CLOSE))
            ):
                base[token_id] = 0
        self._base = bytes(base)
        self._guard = FrameGuard(pieces, banned_tags)
        self._masks: dict[GuardState, bytes] = {}

    # -- lifecycle --------------------------------------------------------------

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        super().create_context(job, descriptor)
        chat = _Chat()
        if self._sampling is not None:
            chat.rng = random.Random(f"{self._seed}:{int(job)}")
        self._chats[job] = chat

    def destroy_context(self, job: JobId) -> None:
        super().destroy_context(job)
        self._chats.pop(job, None)

    def fork(self, parent: JobId, child: JobId) -> int:
        shared = super().fork(parent, child)
        clone = self._chat_of(parent).copy()
        if self._sampling is not None:
            clone.rng = random.Random(f"{self._seed}:{int(child)}")
        self._chats[child] = clone
        return shared

    def _chat_of(self, job: JobId) -> _Chat:
        chat = self._chats.get(job)
        if chat is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return chat

    # -- outside the Protocol ---------------------------------------------------

    def calls(self, job: JobId) -> tuple[ToolCall, ...]:
        """Every tool call this job has asked to write, in order."""
        return tuple(self._chat_of(job).calls)

    def awaiting(self, job: JobId) -> PipeName | None:
        """The pipe the job asked to read and has had nothing on yet, or None."""
        chat = self._chat_of(job)
        return None if chat.arrived else chat.awaiting

    def tool_class(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """``READ`` or ``EFFECT`` for a call; a rule needs the call's ``arguments``."""
        rule = self._read_if.get(name)
        if rule is None:
            return self._tool_classes.get(name, EFFECT)
        return READ if _rule_matches(rule, arguments) else EFFECT

    def results_pipe(self, name: str, arguments: Mapping[str, Any] | None = None) -> PipeName:
        """Where a call's result is read from: ``tools.results.trusted`` when the
        trusted-results table names the call exactly, ``tools.results`` otherwise."""
        rule = self._trusted.get(name)
        if rule is not None and _exact_matches(rule, arguments):
            return self.pipes.results_trusted
        return self.pipes.results

    def replay(self, job: JobId, pipes: Sequence[PipeName]) -> None:
        """Have a job that has not spoken read ``pipes`` in order before it decodes. The
        first is read where a fresh job reads anyway, so it must be ``chat.user``."""
        chat = self._chat_of(job)
        if chat.role != _SYSTEM or chat.calls or chat.replay_tail:
            raise RuntimeError(f"job {job} has already spoken; only a fresh job replays")
        if not pipes or pipes[0] != self.pipes.user:
            raise ValueError(f"a replay starts with a message on {self.pipes.user}")
        allowed = {self.pipes.user, *self.pipes.result_pipes, *self.pipes.history_pipes}
        stray = sorted({str(p) for p in pipes} - {str(p) for p in allowed})
        if stray:
            raise ValueError(f"a replay reads only {sorted(map(str, allowed))}, not {stray}")
        chat.replay = list(pipes[1:])
        chat.replay_tail = True

    def allowed_tokens(self, state: GuardState = GUARD_START) -> bytes:
        """The token mask from a guard state, one byte per vocabulary id."""
        mask = self._masks.get(state)
        if mask is None:
            flags = bytearray(self._base)
            for token_id in self._guard.banned(state):
                flags[token_id] = 0
            mask = bytes(flags)
            self._masks[state] = mask
        return mask

    def context_text(self, job: JobId) -> str:
        """Every id of the job's context as text, framing included: the prompt as the
        model reads it."""
        return "".join(self._pieces[i] for i in self._ctx_of(job).ids)

    # -- encoding ---------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        cached = self._tokenized.get(text)
        if cached is None:
            cached = tuple(int(i) for i in self._worker.tokenize(text))
            self._tokenized[text] = cached
        return list(cached)

    def _encode(self, tokens: Sequence[Token], ctx: _Context) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        spans: list[int] = []
        for k, tok in enumerate(tokens):
            text = tok.text if (k == 0 or tok.text[:1].isspace()) else " " + tok.text
            word = self._tokenize(text) or [self._pad_id]
            ids.extend(word)
            spans.append(len(word))
        return ids, spans

    # -- the ops ------------------------------------------------------------------

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        ctx = self._ctx_of(job)
        chat = self._chat_of(job)
        start = len(ctx.tokens)
        if not tokens:
            return start, start
        new_ids, new_spans = self._encode(tokens, ctx)
        ctx.extend(tokens, new_ids, new_spans)
        end = len(ctx.tokens)
        close = f"{IM_END}\n{IM_START}{_USER}"

        head, tail = "", ""
        if chat.role is None:
            head = f"{IM_START}{_SYSTEM}\n"
            chat.role = _SYSTEM
        elif tokens[0].kind is TokenKind.CONTROL:
            # A kernel frame: the system's while the prompt is being put together, and
            # otherwise the user turn's, opening one if the model was speaking.
            head = f"{close}\n" if chat.role == _ASSISTANT else "\n"
            if chat.role == _ASSISTANT:
                chat.role = _USER
            chat.in_tool_response = False
        elif chat.awaiting in self.pipes.history_pipes:
            head = f"{IM_END}\n{IM_START}{_ASSISTANT}\n"
            chat.role = _ASSISTANT
            chat.in_tool_response = False
            chat.arrived = True
        elif chat.awaiting in self.pipes.result_pipes:
            head = ("" if chat.role == _USER else close) + f"\n{TOOL_RESPONSE_OPEN}\n"
            tail = f"\n{TOOL_RESPONSE_CLOSE}"
            chat.role = _USER
            chat.in_tool_response = True
            chat.arrived = True
        else:
            head = "\n" if chat.role == _USER else f"{close}\n"
            chat.role = _USER
            chat.in_tool_response = False
            chat.arrived = chat.awaiting is not None

        if (
            self._mask_tool_choice
            and tokens[0].kind is not TokenKind.CONTROL
            and chat.awaiting in self._hidden_pipes
        ):
            chat.hidden.append((start, end))
        self._frame_into(ctx, head, start, before=True)
        if tail:
            self._frame_into(ctx, tail, end - 1, before=False)
        if start == 0:
            # The descriptor body: prefill it now, which for a pinned job is at boot,
            # so the first message costs only itself.
            self._flush(ctx)
        if tokens[0].kind is not TokenKind.CONTROL and start > 0:
            self.note_arrival(job, "".join(t.text for t in tokens))
        ctx.tags = ("self",)
        return start, end

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        """``JsMachine.visible_blocks``, less the blocks the latest step hid."""
        return super().visible_blocks(job) - self._chat_of(job).narrowed

    def trunc(self, job: JobId, at: int) -> int:
        chat = self._chat_of(job)
        chat.hidden = [(s, min(e, at)) for s, e in chat.hidden if s < at]
        return super().trunc(job, at)

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        """``JsMachine.splice``, keeping the framing on the edges of the range: a stub in
        place of a tool result still sits in its turn."""
        self._splice_hidden(self._chat_of(job), start, end, len(tokens))
        ctx = self._ctx_of(job)
        if not tokens or start == end:
            return super().splice(job, start, end, tokens)
        head_ids: list[int] = []
        tail_ids: list[int] = []
        if 0 <= start < len(ctx.spans):
            at = ctx.kv_offset(start)
            head_ids = ctx.ids[at : at + ctx.framing[start][0]]
        if 0 < end <= len(ctx.spans):
            stop = ctx.kv_offset(end)
            tail_ids = ctx.ids[stop - ctx.framing[end - 1][1] : stop]
        result = super().splice(job, start, end, tokens)
        last = start + len(tokens) - 1
        at = ctx.kv_offset(start)
        ctx.ids[at:at] = head_ids
        ctx.spans[start] += len(head_ids)
        stop = ctx.kv_offset(last + 1)
        ctx.ids[stop:stop] = tail_ids
        ctx.spans[last] += len(tail_ids)
        if start == last:
            ctx.framing[start] = (len(head_ids), len(tail_ids))
        else:
            ctx.framing[start] = (len(head_ids), 0)
            ctx.framing[last] = (0, len(tail_ids))
        return result

    def _splice_hidden(self, chat: _Chat, start: int, end: int, count: int) -> None:
        """Move the hidden ranges as a splice of ``count`` tokens over ``[start, end)``
        moves the words: what it replaces is the kernel's text, and is not hidden."""
        shift = count - (end - start)
        moved: list[tuple[int, int]] = []
        for s, e in chat.hidden:
            if s < start:
                moved.append((s, min(e, start)))
            if e > end:
                moved.append((max(s, end) + shift, e + shift))
        chat.hidden = [(s, e) for s, e in moved if e > s]

    def _in_name_span(self, chat: _Chat) -> bool:
        """Whether the turn so far ends inside an open call before its name is closed:
        after ``<tool_call>``, on the way to ``<function=NAME>``."""
        opened = chat.text.find(TOOL_CALL_OPEN, chat.scan)
        if opened < 0:
            return False
        head = chat.text[opened + len(TOOL_CALL_OPEN) :].lstrip()
        if len(head) <= len(_FUNCTION_OPEN):
            return _FUNCTION_OPEN.startswith(head)
        if not head.startswith(_FUNCTION_OPEN):
            return False
        name = head[len(_FUNCTION_OPEN) :]
        return ">" not in name and "\n" not in name

    def _narrowed_blocks(
        self, ctx: _Context, chat: _Chat, blocks: bytes | None
    ) -> tuple[bytes, frozenset[int]]:
        """``blocks`` with the ids of every hidden word hidden too, failing closed on a
        worker block that holds one, and the kernel blocks those words are in."""
        size = self._worker_block
        count = (len(ctx.ids) + size - 1) // size
        flags = bytearray(b"\x01") * count if blocks is None else bytearray(blocks)
        offsets = [0]
        for span in ctx.spans:
            offsets.append(offsets[-1] + span)
        kernel_blocks: set[int] = set()
        for start, end in chat.hidden:
            for index in range(start, end):
                head, tail = ctx.framing[index]
                first, last = offsets[index] + head, offsets[index + 1] - tail
                if last > first:
                    flags[first // size : (last - 1) // size + 1] = bytes(
                        (last - 1) // size + 1 - first // size
                    )
                kernel_blocks.add(index // self._block_size)
        return bytes(flags), frozenset(kernel_blocks)

    def _quiet(self, request: MachineRequest) -> DecodeResult:
        """A step that runs no forward pass, so attends nothing."""
        return DecodeResult(tokens=(), request=request, attention={})

    def _open_turn(self, ctx: _Context, chat: _Chat) -> None:
        close = "" if chat.role is None else f"{IM_END}\n"
        self._frame_into(
            ctx,
            f"{close}{IM_START}{_ASSISTANT}\n{self._prefix}",
            len(ctx.tokens) - 1,
            before=False,
        )
        chat.role = _ASSISTANT
        chat.in_tool_response = False
        chat.text = ""
        chat.scan = 0
        chat.guard = GUARD_START
        chat.masked_call, chat.masked_hidden = None, ()

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        chat = self._chat_of(job)
        chat.narrowed = frozenset()
        if not ctx.ids:
            raise RuntimeError(
                "cannot decode an empty context; the kernel injects the descriptor "
                "body before the first decode"
            )
        if chat.awaiting is not None and not chat.arrived:
            # Asked to read and nothing came: the kernel refused the write before it.
            return self._quiet(MachineRequest(op=OpKind.READ, pipe=chat.awaiting))
        if chat.replay_tail and chat.role != _SYSTEM:
            # Replaying: the next pipe of the past conversation, and once it is all read,
            # whatever the user says now. Nothing of the past is decoded.
            if chat.replay:
                pipe = chat.replay.pop(0)
            else:
                pipe, chat.replay_tail = self.pipes.user, False
            chat.awaiting, chat.arrived = pipe, False
            return self._quiet(MachineRequest(op=OpKind.READ, pipe=pipe))
        if chat.role == _SYSTEM:
            # Nobody has spoken yet.
            chat.awaiting, chat.arrived = self.pipes.user, False
            return self._quiet(MachineRequest(op=OpKind.READ, pipe=self.pipes.user))
        chat.awaiting, chat.arrived = None, False
        if chat.role != _ASSISTANT:
            self._open_turn(ctx, chat)
        self._flush(ctx)

        allowed = self.allowed_tokens(chat.guard)
        blocks = self._allowed_blocks(ctx)
        if self._mask_tool_choice and chat.hidden and self._in_name_span(chat):
            blocks, chat.narrowed = self._narrowed_blocks(ctx, chat, blocks)
            chat.masked_call = chat.text.find(TOOL_CALL_OPEN, chat.scan)
            chat.masked_hidden = tuple(chat.hidden)
        sample: dict[str, float] | None = None
        if self._sampling is not None:
            assert chat.rng is not None
            sample = {
                "temperature": self._sampling.temperature,
                "topK": self._sampling.top_k,
                "u": chat.rng.random(),
            }
        step = self._worker.decodeStep(ctx.key, self._bridge.options(blocks, allowed, sample))
        tid = int(step.tokenId)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        if not (0 <= tid < len(allowed)) or not allowed[tid]:
            raise WorkerViolation(f"job {job}: the worker chose id {tid}, which the mask refused")
        measured = self._bridge.floats(step.attention)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        attention = None if measured is None else self._kernel_attention(ctx, measured, blocks)
        if attention is not None and chat.narrowed:
            # What the step paid a hidden word's block went to its framing or to the
            # model's own text there; the kernel would credit it to the hidden segment.
            attention = {b: v for b, v in attention.items() if b not in chat.narrowed}
        hint = AttentionHint(tags=ctx.tags) if measured is None else None

        if tid == self._im_end:
            # The turn is over. The marker is not appended: it is the framing in front
            # of whatever arrives next.
            chat.awaiting, chat.arrived = self.pipes.user, False
            return DecodeResult(
                tokens=(),
                request=MachineRequest(
                    op=OpKind.WRITE_READ,
                    pipe=self.pipes.out,
                    payload=tokens_from_text(chat.text, preserve_whitespace=True),
                    read_pipe=self.pipes.user,
                ),
                attention=attention,
                attention_hint=hint,
            )

        piece = self._pieces[tid]
        token = Token(piece, TokenKind.NORMAL)
        ctx.spoken = True
        before = len(ctx.tokens)
        ctx.extend([token], [tid], [1])
        after = len(ctx.tokens)
        chat.text += piece
        chat.guard = self._guard.advance(chat.guard, piece)

        request = MachineRequest()
        call = self._closed_call(chat)
        if call is not None:
            chat.calls.append(call)
            chat.awaiting, chat.arrived = call.results, False
            request = MachineRequest(
                op=OpKind.WRITE_READ,
                pipe=call.sink,
                payload=tokens_from_text(call.payload, preserve_whitespace=True),
                read_pipe=call.results,
            )
        return DecodeResult(
            tokens=(token,),
            request=request,
            attention=attention,
            attention_hint=hint,
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def _closed_call(self, chat: _Chat) -> ToolCall | None:
        """The tool call the latest piece closed, if it closed one that parses."""
        text = chat.text
        opened = text.find(TOOL_CALL_OPEN, chat.scan)
        if opened < 0:
            return None
        inner_at = opened + len(TOOL_CALL_OPEN)
        closed = text.find(TOOL_CALL_CLOSE, inner_at)
        if closed < 0:
            return None
        chat.scan = closed + len(TOOL_CALL_CLOSE)
        parsed = parse_tool_call_body(text[inner_at:closed], self._param_types)
        if parsed is None:
            return None
        name, arguments = parsed
        sink = self.pipes.read if self.tool_class(name, arguments) == READ else self.pipes.effect
        masked = chat.masked_call == opened
        hidden = chat.masked_hidden if masked else ()
        chat.masked_call, chat.masked_hidden = None, ()
        return ToolCall(
            index=len(chat.calls),
            name=name,
            arguments=arguments,
            sink=sink,
            raw=text[opened : chat.scan],
            results=self.results_pipe(name, arguments),
            name_masked=masked,
            name_hidden=hidden,
        )


def _compile_rule(rule: Mapping[str, str]) -> dict[str, re.Pattern[str]]:
    return {
        param: re.compile(pattern, re.IGNORECASE | re.DOTALL) for param, pattern in rule.items()
    }


def _rule_matches(rule: Mapping[str, re.Pattern[str]], arguments: Mapping[str, Any] | None) -> bool:
    """Whether the arguments are exactly the rule's parameters, each a string its pattern
    matches in full."""
    if arguments is None or set(arguments) != set(rule):
        return False
    for param, pattern in rule.items():
        value = arguments[param]
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            return False
    return True


def _exact_rule(name: str, rule: object) -> dict[str, frozenset[str]]:
    shape = "a trusted-results rule is {param: [value, ...]}, exact values, not patterns"
    if not isinstance(rule, Mapping):
        raise ValueError(f"{name}: {shape}")
    exact: dict[str, frozenset[str]] = {}
    for param, values in rule.items():  # pyright: ignore[reportUnknownVariableType]
        if (
            not isinstance(param, str)
            or isinstance(values, str)
            or not isinstance(values, Collection)
            or not all(isinstance(v, str) for v in values)  # pyright: ignore[reportUnknownVariableType]
        ):
            raise ValueError(f"{name}: {shape}")
        exact[param] = frozenset(values)  # pyright: ignore[reportUnknownArgumentType]
    return exact


def _exact_matches(rule: Mapping[str, frozenset[str]], arguments: Mapping[str, Any] | None) -> bool:
    """Whether the arguments are exactly the rule's parameters, each a string equal to one
    of its values, compared exactly and case-sensitively."""
    if arguments is None or set(arguments) != set(rule):
        return False
    return all(isinstance(arguments[p], str) and arguments[p] in v for p, v in rule.items())
