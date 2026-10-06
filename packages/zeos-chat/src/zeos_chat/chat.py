# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A chat agent under the kernel, as a host page drives it: ``open_chat`` and ``ChatRun``.

The host owns the tools. The kernel owns who may ask for them. One ``ChatRun`` is one
conversation: the ``chat-agent`` case (``CHAT_CASE``, shipped in this package) with its
descriptor body replaced by the host's system prompt, a ``ChatToolMachine`` over the
host's model worker, and a ``LiveRun`` stepping the kernel one tick at a time with
``KernelConfig.preserve_whitespace`` on, so tool results keep their line breaks.

The loop a host runs::

    run = open_chat(worker, system_prompt=prompt, tool_classes={"ReadLines": "read", ...})
    run.send_user("How many rows are in train.csv?")
    while True:
        for event in run.step(16):
            ...                      # see below
        if run.waiting_on() == "chat.user":
            break                    # the turn is over

Events are plain dicts, one per thing the host can show or must act on, in journal
order. Every one has ``type``; the other fields are:

``token``         ``text``: one piece the model decoded, for streaming.
``tool_call``     ``call``, ``name``, ``arguments``, ``sink``, ``results``,
                  ``name_masked``, ``name_hidden``: a call that
                  landed on ``tools.read`` or ``tools.effect``. The host runs it, then
                  ``deliver_tool_result``. It also drains the sink (``drain``).
                  ``results`` is the pipe the job reads the answer from:
                  ``tools.results``, or ``tools.results.trusted`` for a call the
                  host's ``trusted_results`` table names. ``name_masked`` says the
                  tool's name was chosen with the EXTERNAL deliveries hidden
                  (``mask_tool_choice``), and ``name_hidden`` lists their segment ids.
``approval_required``
                  the ``tool_call`` fields, ``fault``, ``detail``,
                  ``integrity``, ``effective_integrity``, ``session_floor``,
                  ``demotions``: a call the kernel refused for privilege. Nothing landed.
                  On approval the host runs it under the user's authority and delivers
                  the result; on denial it calls ``deliver_refusal``. Either way the
                  job is waiting on ``tools.results``.
``tool_refused``  as ``approval_required``, for any other refusal of a tool write (a
                  payload larger than the sink, say). The host delivers a refusal.
``reply``         ``text``, ``reasoning``, ``raw``: the turn's reply on ``chat.out``.
                  ``raw`` is everything decoded in the turn; with thinking on,
                  ``reasoning`` is what came before ``</think>`` and ``text`` what came
                  after, and otherwise ``reasoning`` is None.
``arrived``       ``pipe``, ``segment``, ``ring``, ``integrity``: a delivery the job
                  read, as the kernel stamped it.
``demoted``       ``from_integrity``, ``to_integrity``, ``because``: the job's
                  watermark fell. ``because`` lists segments (``segment_info``).
``spoof``         ``pipe``, ``detail``: a delivery spelled a kernel frame. It is inert
                  data; the kernel alarms and the job carries on.
``fault``         ``fault``, ``detail``, ``pipe``: any other fault.
``waiting``       ``pipe``: the job blocked reading ``chat.user`` or a results pipe, or
                  a history pipe during ``import_history``.

``arrived`` and ``waiting`` cover the history pipes too, so a host that replays a past
conversation (``import_history``) learns each past turn's segment as it is read.

A segment is ``{"segment", "pipe", "principal", "tag", "ring", "integrity", "tokens",
"resident", "injected_at"}``, every value a string or an integer.

Integrities and rings are integers, 0 the most trusted. Two things can refuse a tool
write on ``tools.effect`` (``min_integrity: 2``): the job's watermark, which falls to 3
for good once it attends a tool result past ``theta_read``, and the session floor, the
ring of the pipe it last read (MP's confused-deputy rule), which is 3 from reading a
tool result until the next user message. ``effective_integrity`` is the worse of them.

``tools.results.trusted`` is for results the host wrote itself (``trusted_results``): it
is TRUSTED, so attending it never demotes, and it is declared ``session_floor: false``,
so reading it leaves the floor where it was -- it neither raises the floor to 3 nor
lowers a floor an earlier tool result raised.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from zeos.core.events import (
    CapabilityChecked,
    Decoded,
    Event,
    FaultRaised,
    Injected,
    IntegrityDemoted,
    JobBlocked,
    PipeWritten,
)
from zeos.core.ids import FaultKind, JobState, PipeName, Ring, SegmentId
from zeos.core.integrity import DEFAULT_THETA_READ
from zeos.core.pcb import Job
from zeos.descriptor.lint import Severity
from zeos.descriptor.loader import CaseBundle, load_case
from zeos_browser.js_machine import DEFAULT_BLOCK_SIZE, Bridge, PythonBridge, ZeosModelWorker
from zeos_browser.live import LiveRun
from zeos_browser.page import findings

from zeos_chat.chat_machine import (
    THINK_CLOSE,
    ChatPipes,
    ChatToolMachine,
    Sampling,
    ToolCall,
    ToolClass,
)

__all__ = [
    "ATTENTION",
    "CHAT_CASE",
    "DEFAULT_REFUSAL",
    "GATE_MODES",
    "STRICT",
    "ChatRun",
    "open_chat",
]

#: The case this package ships: one pinned chat job and its five pipes.
CHAT_CASE = Path(str(resources.files("zeos_chat") / "cases" / "chat-agent"))

#: What the model reads when the user declines a tool call.
DEFAULT_REFUSAL = "The user declined this tool call. It was not run."

#: How a tool result gates effects. ``STRICT``: reading one sets the session floor to 3
#: until the next user message, so any effect after it needs approval (MP's
#: confused-deputy rule). ``ATTENTION``: reading one leaves the floor alone, and only the
#: watermark -- whether the job measurably attended the result past ``theta_read`` --
#: refuses effects.
STRICT = "strict"
ATTENTION = "attention"
GATE_MODES = (STRICT, ATTENTION)

#: Lint rules about the syscall ABI's commands in a body. The chat machine speaks no ABI,
#: so a system prompt that quotes, say, a SQL statement ending in a semicolon is not a
#: command the model is being taught.
_ABI_BODY_RULES = frozenset({"unknown-body-verb", "unbound-body-pipe"})


def _default_bridge() -> Bridge:
    if sys.platform == "emscripten":
        from zeos_browser.pyodide_bridge import PyodideBridge

        return PyodideBridge()
    return PythonBridge()


def open_chat(
    worker: ZeosModelWorker,
    *,
    tool_classes: Mapping[str, ToolClass],
    case_dir: str | Path | None = None,
    system_prompt: str | None = None,
    thinking: bool = False,
    theta_read: float = DEFAULT_THETA_READ,
    seed: int = 0,
    sampling: Sampling | None = None,
    param_types: Mapping[str, Mapping[str, str]] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bridge: Bridge | None = None,
    pipes: ChatPipes | None = None,
    max_ticks: int = 10**9,
    gate_mode: str = STRICT,
    trusted_results: Mapping[str, Mapping[str, str]] | None = None,
    mask_tool_choice: bool = False,
) -> ChatRun:
    """A conversation, booted and waiting for its first message.

    ``tool_classes`` maps a tool name to ``"read"``, ``"effect"`` or a ``read_if`` rule
    (``chat_machine``); a tool it does not name is an effect. ``system_prompt`` replaces the case's one descriptor body, and
    should hold the tool declarations. ``param_types[tool][param]`` is a parameter's
    JSON-schema type, so a string parameter is never JSON-decoded. ``bridge`` defaults to
    Pyodide's under Pyodide and the identity bridge under CPython. ``gate_mode`` is
    ``STRICT`` or ``ATTENTION`` (see ``GATE_MODES``); ``ATTENTION`` declares the
    untrusted inbound pipes -- ``tools.results`` and ``chat.history`` -- with
    ``session_floor: false``. ``trusted_results[tool]`` is a ``{param: pattern}`` rule,
    matched as ``read_if`` is: a call it matches reads its result from
    ``tools.results.trusted``, for text the host wrote itself rather than fetched.
    ``mask_tool_choice`` hides every delivery on an EXTERNAL device pipe -- tool results
    and replayed turns -- while the model writes a tool's name (``chat_machine``).
    Refuses a case that does not lint.
    """
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode is one of {GATE_MODES}, not {gate_mode!r}")
    bundle = load_case(Path(case_dir) if case_dir is not None else CHAT_CASE)
    chat_pipes = pipes or ChatPipes()
    if gate_mode == ATTENTION:
        consulted = {chat_pipes.results, chat_pipes.history}
        bundle = dataclasses.replace(
            bundle,
            pipes=tuple(
                dataclasses.replace(p, session_floor=False) if p.name in consulted else p
                for p in bundle.pipes
            ),
        )
    if len(bundle.descriptors) != 1:
        raise ValueError(
            f"a chat case has exactly one descriptor; {bundle.name} has {len(bundle.descriptors)}"
        )
    if system_prompt is not None:
        ((name, descriptor),) = bundle.descriptors.items()
        bundle = dataclasses.replace(
            bundle, descriptors={name: dataclasses.replace(descriptor, body=system_prompt)}
        )
    blocking = [
        f
        for f in findings(bundle)
        if f.severity is Severity.ERROR and (system_prompt is None or f.rule not in _ABI_BODY_RULES)
    ]
    if blocking:
        raise ValueError(
            "refusing to run a tree that does not lint:\n" + "\n".join(f.render() for f in blocking)
        )
    machine = ChatToolMachine(
        worker,
        tool_classes=tool_classes,
        bridge=bridge or _default_bridge(),
        pipes=chat_pipes,
        thinking=thinking,
        sampling=sampling,
        seed=seed,
        param_types=param_types,
        trusted_results=trusted_results,
        block_size=block_size,
        mask_tool_choice=mask_tool_choice,
        hidden_pipes=[p.name for p in bundle.pipes if p.device and p.ring >= Ring.EXTERNAL],
    )
    return ChatRun(bundle, machine, seed=seed, theta_read=theta_read, max_ticks=max_ticks)


class ChatRun:
    """One conversation: a kernel, its chat machine, and the host's side of the pipes."""

    def __init__(
        self,
        bundle: CaseBundle,
        machine: ChatToolMachine,
        *,
        seed: int = 0,
        theta_read: float = DEFAULT_THETA_READ,
        max_ticks: int = 10**9,
    ) -> None:
        self.machine = machine
        self.pipes = machine.pipes
        self.run = LiveRun(
            bundle,
            machine,
            seed=seed,
            max_ticks=max_ticks,
            theta_read=theta_read,
            preserve_whitespace=True,
        )
        self.kernel = self.run.kernel
        self._seen = 0
        #: Whether a delivery is queued for the next tick.
        self._queued = False
        #: How many journal events ``drain`` has counted writes in.
        self._counted = 0
        #: Per sink, how many tokens each write not yet drained put there.
        self._writes: dict[PipeName, list[int]] = {}
        #: The refused call the host has yet to settle.
        self.pending_approval: dict[str, Any] | None = None
        self._last_check: CapabilityChecked | None = None

    # -- the job ----------------------------------------------------------------

    def job(self) -> Job:
        jobs = [j for j in self.kernel.sched.jobs() if not j.state.is_terminal]
        if len(jobs) != 1:
            raise RuntimeError(f"a chat run has one live job; this one has {len(jobs)}")
        return jobs[0]

    def waiting_on(self) -> str | None:
        """The pipe the job is blocked reading, or None while it can run. Before the
        first message the job is pinned and idle, which is waiting on ``chat.user``."""
        job = self.job()
        if job.state is JobState.PINNED_IDLE:
            return str(self.pipes.user)
        return None if job.blocked_on is None else str(job.blocked_on)

    # -- the host's side of the pipes ---------------------------------------------

    def send_user(self, text: str) -> None:
        """A message the user typed, delivered on ``chat.user`` before the next tick."""
        self._queued = True
        self.run.press(self.pipes.user, text)

    def deliver_tool_result(self, text: str, *, trusted: bool = False) -> None:
        """What a tool returned, as the answer to the latest call: on ``tools.results``
        (ring 3), or with ``trusted`` on ``tools.results.trusted`` (ring 2).

        The job already waits on the pipe the call's ``results`` names, chosen from the
        call when the model wrote it, so ``trusted`` states what the host believes it is
        delivering and must agree: a host that thinks a result trusted when the table
        did not name its call, or the reverse, is refused here rather than left to hang.
        """
        pipe = self._last_call().results
        if (pipe == self.pipes.results_trusted) != trusted:
            raise ValueError(
                f"the latest call's result is read from {pipe}; deliver it with "
                f"trusted={pipe == self.pipes.results_trusted}"
            )
        self._settle(pipe, text)

    def deliver_refusal(self, text: str = DEFAULT_REFUSAL) -> None:
        """The answer to a call the host will not run, on the pipe the call reads its
        result from."""
        self._settle(self._last_call().results, text)

    def _settle(self, pipe: PipeName, text: str) -> None:
        self.pending_approval = None
        self._queued = True
        self.run.press(pipe, text)

    def import_history(self, turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Replay a past conversation into a fresh run, one turn per delivery.

        Each turn is ``{"role": "user" | "assistant" | "tool", "text": ...}``. A user turn
        arrives on ``chat.user`` (TRUSTED), a tool result on ``tools.results``
        (EXTERNAL) or, with ``"trusted": True``, on ``tools.results.trusted`` (TRUSTED),
        and an assistant turn on ``chat.history`` (EXTERNAL) or, with
        ``"integrity": 2`` or less, on ``chat.history.trusted`` (TRUSTED): a host that
        recorded the integrity a turn was written at says so, and anything else is held
        as untrusted. The first turn must be the user's. Nothing is decoded, so nothing
        is attended and the watermark does not move; the run ends waiting on
        ``chat.user``. Returns the events of the replay.
        """
        if self.waiting_on() != str(self.pipes.user) or self.machine.calls(self.job().job_id):
            raise RuntimeError("history is imported into a fresh run, before its first message")
        plan: list[tuple[PipeName, str]] = []
        for turn in turns:
            role, text = turn["role"], str(turn["text"])
            if role == "user":
                pipe = self.pipes.user
            elif role == "tool":
                pipe = self.pipes.results_trusted if turn.get("trusted") else self.pipes.results
            elif role == "assistant":
                trusted = int(turn.get("integrity", 3)) <= 2
                pipe = self.pipes.history_trusted if trusted else self.pipes.history
            else:
                raise ValueError(f"a turn's role is user, assistant or tool, not {role!r}")
            plan.append((pipe, text))
        if not plan:
            return []
        self.machine.replay(self.job().job_id, [pipe for pipe, _ in plan])
        events: list[dict[str, Any]] = []
        for pipe, text in plan:
            for _ in range(64):
                if self.waiting_on() == str(pipe):
                    break
                events += self.step(16)
            else:
                raise RuntimeError(f"the replay never came to read {pipe}")
            self._queued = True
            self.run.press(pipe, text)
        while self.waiting_on() != str(self.pipes.user):
            before = self.run.ticks
            events += self.step(16)
            if self.run.ticks == before:
                raise RuntimeError(f"the replay stalled waiting on {self.waiting_on()}")
        return events

    def close(self) -> None:
        """End the run and free the worker's contexts, so another run can use it."""
        for job in self.kernel.sched.jobs():
            self.machine.destroy_context(job.job_id)
        self.run.stop("closed")

    def drain(self, pipe: str) -> list[str]:
        """Every write on a sink since it was last drained, one string per write."""
        name = PipeName(pipe)
        tokens = self.kernel.drain(name)
        events = self.run.events
        for event in events[self._counted :]:
            if isinstance(event, PipeWritten) and event.job is not None:
                self._writes.setdefault(event.pipe, []).append(event.tokens)
        self._counted = len(events)
        out: list[str] = []
        at = 0
        for count in self._writes.pop(name, []):
            out.append("".join(t.text for t in tokens[at : at + count]))
            at += count
        if at != len(tokens):
            raise RuntimeError(f"{pipe}: drained {len(tokens)} tokens, journalled {at}")
        return out

    # -- stepping -----------------------------------------------------------------

    def step(self, n: int = 1) -> list[dict[str, Any]]:
        """Up to ``n`` ticks, stopping early once only a delivery can move the job on."""
        for _ in range(n):
            # A pinned job parked on nothing reads as quiescent to ``LiveRun``, which
            # would end the run before the first message could reach it.
            if self.run.finished or (self.job().state is JobState.PINNED_IDLE and not self._queued):
                break
            self.run.step()
            self._queued = False
            if self.run.waiting_for_a_press():
                break
        return self._collect()

    def _collect(self) -> list[dict[str, Any]]:
        events = self.run.events
        out: list[dict[str, Any]] = []
        for event in events[self._seen :]:
            out.extend(self._translate(event))
        self._seen = len(events)
        return out

    def _translate(self, event: Event) -> list[dict[str, Any]]:
        pipes = self.pipes
        if isinstance(event, Decoded):
            return [{"type": "token", "text": "".join(event.text)}]
        if isinstance(event, CapabilityChecked):
            self._last_check = event
            return []
        if isinstance(event, PipeWritten) and event.job is not None:
            text = "".join(event.text)
            if event.pipe in pipes.tool_sinks:
                call = self._last_call()
                return [{"type": "tool_call", **self._call_fields(call)}]
            if event.pipe == pipes.out:
                reasoning, closed, answer = text.rpartition(THINK_CLOSE)
                if not closed:
                    reasoning, answer = None, text
                return [
                    {
                        "type": "reply",
                        "text": answer.lstrip("\n") if closed else answer,
                        "reasoning": reasoning,
                        "raw": text,
                    }
                ]
            return []
        replayed = (pipes.user, *pipes.result_pipes, *pipes.history_pipes)
        if isinstance(event, Injected) and event.pipe in replayed:
            return [
                {
                    "type": "arrived",
                    "pipe": str(event.pipe),
                    "segment": int(event.segment),
                    "ring": int(event.ring),
                    "integrity": int(event.integrity),
                }
            ]
        if isinstance(event, IntegrityDemoted):
            return [
                {
                    "type": "demoted",
                    "from_integrity": int(event.from_integrity),
                    "to_integrity": int(event.to_integrity),
                    "because": [self.segment_info(s) for s in event.because],
                }
            ]
        if isinstance(event, JobBlocked) and event.pipe in replayed:
            return [{"type": "waiting", "pipe": str(event.pipe)}]
        if isinstance(event, FaultRaised):
            return [self._fault(event)]
        return []

    def _fault(self, event: FaultRaised) -> dict[str, Any]:
        if event.fault is FaultKind.SPOOF:
            return {"type": "spoof", "pipe": _name(event.pipe), "detail": event.detail}
        if event.pipe in self.pipes.tool_sinks:
            job = self.job()
            check = self._last_check
            refusal: dict[str, Any] = {
                "type": "approval_required"
                if event.fault is FaultKind.PRIVILEGE
                else "tool_refused",
                **self._call_fields(self._last_call()),
                "fault": str(event.fault),
                "detail": event.detail,
                "integrity": int(job.current_integrity),
                "effective_integrity": None if check is None else int(check.effective_integrity),
                "session_floor": None if job.session_floor is None else int(job.session_floor),
                "demotions": self.demotions(),
            }
            self.pending_approval = refusal
            return refusal
        return {
            "type": "fault",
            "fault": str(event.fault),
            "detail": event.detail,
            "pipe": _name(event.pipe),
        }

    def _last_call(self) -> ToolCall:
        calls = self.machine.calls(self.job().job_id)
        if not calls:
            raise RuntimeError("a tool sink was written with no tool call behind it")
        return calls[-1]

    # -- what the host shows ------------------------------------------------------

    def segment_info(self, segment: SegmentId) -> dict[str, Any]:
        record = self.job().segments.get(segment)
        return {
            "segment": int(record.id),
            "pipe": str(record.provenance.pipe),
            "principal": str(record.provenance.principal),
            "tag": record.provenance.tag,
            "ring": int(record.ring),
            "integrity": int(record.integrity),
            "tokens": record.tokens,
            "resident": record.readable,
            "injected_at": record.provenance.injected_at,
        }

    def _call_fields(self, call: ToolCall) -> dict[str, Any]:
        starts = {record.start: int(record.id) for record in self.job().segments.all()}
        return {
            "call": call.index,
            "name": call.name,
            "arguments": dict(call.arguments),
            "sink": str(call.sink),
            "results": str(call.results),
            "name_masked": call.name_masked,
            "name_hidden": [starts[s] for s, _ in call.name_hidden if s in starts],
        }

    def demotions(self) -> list[dict[str, Any]]:
        """Every fall of the job's watermark so far, with the segments that caused it."""
        job = self.job().job_id
        return [
            {
                "from_integrity": int(e.from_integrity),
                "to_integrity": int(e.to_integrity),
                "because": [self.segment_info(s) for s in e.because],
            }
            for e in self.run.events
            if isinstance(e, IntegrityDemoted) and e.job == job
        ]

    def state(self) -> dict[str, Any]:
        """The job as the kernel holds it, for a trust indicator."""
        job = self.job()
        return {
            "integrity": int(job.current_integrity),
            "session_floor": None if job.session_floor is None else int(job.session_floor),
            "waiting_on": self.waiting_on(),
            "segments": [self.segment_info(s.id) for s in job.segments.all()],
            "demotions": self.demotions(),
            "pending_approval": self.pending_approval,
            "calls": [self._call_fields(c) for c in self.machine.calls(job.job_id)],
        }

    # -- the journal ----------------------------------------------------------------

    def journal_lines(self) -> list[str]:
        """Journal lines not yet returned, one JSON object each."""
        return self.run.lines()

    def journal_bytes(self) -> bytes:
        return self.run.journal_bytes()

    def state_json(self) -> str:
        return json.dumps(self.state(), separators=(",", ":"))


def _name(pipe: PipeName | None) -> str | None:
    return None if pipe is None else str(pipe)
