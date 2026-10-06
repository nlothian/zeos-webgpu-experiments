# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``PilotJsMachine`` as a ``MachineBackend``: the native API machine's semantics, kept
over a worker channel that is begun, polled and cancelled.

The worker is ``FakePilotWorker`` on a clock that moves only when the worker sleeps, so a
step that costs 10 ms is exactly ten 1 ms stalls and every test is exact.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

import pytest
from machine_helpers import BODY, PILOT, FakeClock, fake_worker, pilot, until, words
from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import (
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    OpKind,
    Token,
    tokens_from_text,
)
from zeos_space_invaders.players.zeos import REFLEX, evade_behaviour
from zeos_space_invaders.players.zeos.api_machine import Native

from zeos_space_invaders_web.contracts import (
    DEFAULT_MAX_CHUNK,
    PREWARM_COMMANDS,
    PilotMachine,
    PilotMachineFactory,
)
from zeos_space_invaders_web.fake_worker import FakePilotWorker
from zeos_space_invaders_web.machine import PilotJsMachine, normalise_poll, pilot_abi

OTHER = JobId(2)
EVADE = JobId(3)


def in_flight(step_ms: float = 10.0, **kwargs: Any) -> tuple[PilotJsMachine, FakePilotWorker]:
    """A pilot whose first word is decoded and whose second is in flight."""
    worker, _clock = fake_worker(step_ms=step_ms)
    machine = pilot(worker, **kwargs)
    first = until(machine, op="none", limit=1)
    assert not first[0].tokens, "a slow step must stall first"
    while not machine.decode(PILOT, allow_control=False).tokens:
        pass
    machine.decode(PILOT, allow_control=False)
    assert worker.inFlight
    return machine, worker


def test_the_classes_meet_their_contracts() -> None:
    factory: PilotMachineFactory = PilotJsMachine
    worker, _ = fake_worker()
    machine: PilotMachine = factory(worker)
    assert machine.cancellations == 0


# --- the step that does not block ---------------------------------------------------


def test_a_step_in_flight_is_a_stall_and_the_job_stays_preemptible() -> None:
    """The native machine's empty step: nothing decoded, nothing asked, still RUNNING."""
    worker, clock = fake_worker(step_ms=9.5)
    machine = pilot(worker, stall_ms=1.0)
    results = until(machine, op="none", limit=1)
    while not results[-1].tokens:
        assert results[-1] == DecodeResult(
            tokens=(), attention=None, attention_hint=results[-1].attention_hint
        )
        assert results[-1].request.op is OpKind.NONE
        assert worker.inFlight
        results.append(machine.decode(PILOT, allow_control=False))
    assert words(results) == ["write"]
    assert len(results) == 10, "a 9.5 ms step polled 1 ms at a time is ten decodes"
    assert machine.stalls == 9 and machine.steps == 1
    assert clock.slept == pytest.approx(0.0095)


def test_a_fast_worker_answers_in_the_decode_that_asked() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker)
    results = until(machine, op="read")
    assert [r.request.op.value for r in results] == ["none", "none", "write", "read"]
    assert words(results) == ["write", " stdout", " left;"]
    assert machine.stalls == 0


def test_a_completed_write_ends_the_turn_with_a_read_the_worker_never_sees() -> None:
    """One reply is one turn: the machine puts the pilot to sleep on stdin itself."""
    worker, _ = fake_worker()
    machine = pilot(worker)
    results = until(machine, op="read")
    read = results[-1]
    assert read.tokens == () and str(read.request.pipe) == "stdin"
    assert worker.delivered == 3, "the read cost no step"
    assert machine.lines(PILOT) == ("write stdout left",)


def test_without_turn_ends_at_call_the_model_reads_for_itself() -> None:
    worker, _ = fake_worker(reads=True)
    machine = pilot(worker, turn_ends_at_call=False)
    results = until(machine, op="read")
    assert words(results) == ["write", " stdout", " left;", " read", " stdin;"]
    assert worker.delivered == 5


def test_each_step_is_begun_with_the_machines_chunk() -> None:
    worker, _ = fake_worker()
    pilot(worker).decode(PILOT, allow_control=False)
    assert worker.last_options is not None
    assert worker.last_options["maxChunk"] == DEFAULT_MAX_CHUNK
    worker, _ = fake_worker()
    pilot(worker, max_chunk=7).decode(PILOT, allow_control=False)
    assert worker.last_options is not None and worker.last_options["maxChunk"] == 7


def test_the_round_trip_is_measured_from_the_first_step_to_the_call() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker)
    until(machine, op="read")
    until(machine, op="read")
    assert len(machine.roundtrips) == 2
    assert machine.last_roundtrip == machine.roundtrips[-1]


def test_the_step_log_records_every_decode() -> None:
    machine, _worker = in_flight()
    machine.inject(PILOT, tokens_from_text("a board"))
    until(machine, op="read")
    outcomes = {entry["outcome"] for entry in machine.step_log}
    assert {"stall", "token", "native"} <= outcomes
    tokens = [e for e in machine.step_log if e["outcome"] == "token"]
    assert tokens[0]["positions"] > 0 and tokens[0]["chunks"] >= 1
    assert all(e["job"] == str(PILOT) for e in machine.step_log)


# --- what cancels a step ------------------------------------------------------------


def test_an_injection_cancels_the_step_and_starts_the_command_afresh() -> None:
    """A resume notice, a fresh board: the prefix moved under the command being said."""
    machine, worker = in_flight()
    machine.inject(PILOT, tokens_from_text("a new board"))
    assert not worker.inFlight, "a synchronous call must find the channel drained"
    assert machine.cancellations == 1 and len(machine.cancel_ms) == 1
    ctx = machine._ctx_of(PILOT)
    assert ctx.parser.buffer == "" and ctx.round == machine._language("pilot").start
    assert words(until(machine)) == [" write", " stdout", " right;"]


def test_a_cancelled_step_never_delivers_its_token() -> None:
    machine, worker = in_flight()
    before = machine.transcript(PILOT)
    machine.trunc(PILOT, len(before) - 1)
    assert worker.cancelled == 1 and worker.delivered == 1
    assert machine.transcript(PILOT) == before[:-1]


def test_a_step_finished_before_its_cancel_was_read_is_dropped() -> None:
    """Race rule 1: the worker had the answer, but the caller cancelled first."""
    worker, clock = fake_worker(step_ms=10.0)
    machine = pilot(worker)
    machine.decode(PILOT, allow_control=False)
    clock.now += 1.0  # the step is long finished; nobody has polled it
    machine.inject(PILOT, tokens_from_text("interrupt"))
    assert worker.delivered == 0 and worker.cancelled == 1
    assert [t.text for t in machine.transcript(PILOT)] == [*BODY.split(), "interrupt"]


@pytest.mark.parametrize("trigger", ["trunc", "splice", "fork", "narrow", "pad"])
def test_every_change_to_the_context_cancels_the_step(trigger: str) -> None:
    machine, worker = in_flight()
    n = len(machine.transcript(PILOT))
    if trigger == "trunc":
        machine.trunc(PILOT, n - 1)
    elif trigger == "splice":
        machine.splice(PILOT, 0, 1, tokens_from_text("stub"))
    elif trigger == "fork":
        machine.fork(PILOT, OTHER)
    elif trigger == "narrow":
        machine.set_mask(PILOT, frozenset())
    else:
        machine.pad_to_block(PILOT)
    assert machine.cancellations == 1
    # The cancel is posted at once; the drain waits for the next use of the channel.
    machine._settle()
    assert not worker.inFlight
    assert worker.cancelled == 1 and worker.delivered == 1


def test_a_widening_mask_does_not_cancel_the_step() -> None:
    """A decoding job's mask widens at every block boundary as its segment grows."""
    machine, worker = in_flight()
    machine.set_mask(PILOT, frozenset(range(64)))
    assert machine.cancellations == 0 and worker.inFlight


def test_an_upstream_splice_keeps_the_command_being_said() -> None:
    worker, _ = fake_worker(step_ms=10.0)
    machine = pilot(worker, body="a b c d e f g h")
    while not machine.decode(PILOT, allow_control=False).tokens:
        pass
    machine.splice(PILOT, 0, 4, tokens_from_text("stub"))
    assert machine._ctx_of(PILOT).parser.buffer == "write"
    assert words(until(machine)) == [" stdout", " left;"]


def test_decoding_another_job_cancels_the_one_descheduled() -> None:
    machine, worker = in_flight()
    machine.create_context(OTHER, "pilot")
    assert machine.cancellations == 1, "creating a context needs the channel"
    machine.inject(OTHER, tokens_from_text(BODY))
    while worker.inFlight:
        machine.decode(PILOT, allow_control=False)
    machine.decode(PILOT, allow_control=False)
    assert worker.inFlight
    machine.decode(OTHER, allow_control=False)
    assert machine.cancellations == 2


def test_invalidate_drops_the_step_and_keeps_the_words() -> None:
    """Native ``partial="keep"``: what was said stays, the command starts again."""
    machine, worker = in_flight()
    machine.invalidate(PILOT)
    assert machine.cancellations == 1
    assert machine.transcript(PILOT)[-1].text == "write"
    results = until(machine)
    assert words(results) == [" write", " stdout", " right;"]
    assert results[-1].request.payload == tuple(tokens_from_text("right"))
    assert worker.delivered == 4


def test_invalidate_is_a_noop_without_a_context_or_for_a_reflex() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker)
    machine.register_behaviour(REFLEX, evade_behaviour)
    machine.create_context(EVADE, REFLEX)
    machine.invalidate(JobId(99))
    machine.invalidate(EVADE)
    assert machine.cancellations == 0


def test_a_cancelled_step_leaves_its_filled_positions_resident() -> None:
    """Race rule 3: the next step resumes the fill rather than starting it again."""
    worker, clock = fake_worker(position_ms=1.0)
    machine = pilot(worker, body=" ".join(["w"] * 40), max_chunk=8)
    machine.decode(PILOT, allow_control=False)  # 1 ms in: the first chunk is running
    pending = len(machine._ctx_of(PILOT).ids)
    clock.now += 0.0085  # 9.5 ms in: the second chunk is running
    machine.set_mask(PILOT, frozenset())
    machine._settle()
    assert worker.resident(machine._ctx_of(PILOT).key) == 16, "stopped at the next boundary"
    while not machine.decode(PILOT, allow_control=False).tokens:
        pass
    assert machine.step_log[-1]["positions"] == pending - 16


def test_close_drains_the_channel_and_destroys_every_context() -> None:
    machine, worker = in_flight()
    machine.close()
    assert not worker.inFlight and worker.context_ids == ()


# --- the reflex, served without the worker -------------------------------------------


def test_the_reflex_runs_while_a_pilot_step_is_in_flight() -> None:
    """No worker context and no worker call: the channel stays busy with the pilot."""
    machine, worker = in_flight()
    machine.register_behaviour(REFLEX, evade_behaviour)
    machine.create_context(EVADE, REFLEX)
    machine.inject(EVADE, tokens_from_text("fire lands in 1 turns in col 4; clear: left"))
    assert worker.inFlight and machine.cancellations == 0
    result = machine.decode(EVADE, allow_control=False)
    assert result.request.op is OpKind.WRITE
    assert [t.text for t in result.request.payload] == ["left"]
    assert machine.cancellations == 1, "the kernel took the machine off the pilot"
    assert all(not key.endswith(":evade") for key in worker.context_ids)
    assert machine.native_words == 1 and machine.step_log[-1]["outcome"] == "native"
    machine.destroy_context(EVADE)


def test_a_native_behaviour_sees_its_arrivals_once_without_framing() -> None:
    seen: list[Native] = []

    def watch(native: Native) -> DecodeResult:
        seen.append(native)
        return DecodeResult(tokens=(), request=MachineRequest(op=OpKind.EXIT))

    worker, _ = fake_worker()
    machine = PilotJsMachine(worker)
    machine.register_behaviour("watcher", watch)
    machine.create_context(EVADE, "watcher")
    machine.inject(EVADE, [Token("<pad>", TokenKind.CONTROL), *tokens_from_text("real")])
    machine.decode(EVADE, allow_control=False)
    machine.decode(EVADE, allow_control=False)
    assert [(n.step, n.arrived) for n in seen] == [(0, "real"), (1, "")]
    assert machine.stats(EVADE).resident_tokens == 2
    assert machine.pad_to_block(EVADE) == 14 and machine.stats(EVADE).blocks == 1


def test_a_native_control_token_is_refused() -> None:
    worker, _ = fake_worker()
    machine = PilotJsMachine(worker)
    machine.register_behaviour(
        "bad",
        lambda _n: DecodeResult(
            tokens=(Token("<x>", TokenKind.CONTROL),), request=MachineRequest()
        ),
    )
    machine.create_context(EVADE, "bad")
    with pytest.raises(ControlTokenViolation):
        machine.decode(EVADE, allow_control=False)


# --- the grammar --------------------------------------------------------------------


def test_prewarm_fills_the_mask_cache_for_the_pilots_commands() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker)
    ms = machine.prewarm()
    assert ms >= 0.0
    cached = len(machine._mask._cache)
    assert cached > 0
    for _ in PREWARM_COMMANDS:
        until(machine, op="read")
    assert len(machine._mask._cache) == cached, "a warmed command met a cold round state"


def test_forbidden_verbs_leave_the_grammar() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker, forbid_verbs=("exit",))
    assert machine.abi.verb("exit") is None
    language = machine._language("pilot")
    assert language.advance(language.start, "exit;") == ()
    assert language.advance(language.start, "write stdout left;") != ()
    with pytest.raises(ValueError, match="no such verb"):
        pilot_abi(("say",))


def test_the_pilot_may_name_only_its_own_pipes() -> None:
    worker, _ = fake_worker()
    machine = pilot(worker)
    language = machine._language("pilot")
    assert language.advance(language.start, "write tools x;") == ()


def test_poll_results_are_normalised_from_attributes_too() -> None:
    class Proxy:
        tokenId = 7
        attention = None
        cancelled = False

        class stats:  # noqa: N801 - the JavaScript field name
            positions = 3
            chunks = 1
            fillMs = 2.5

    assert normalise_poll(Proxy()) == {
        "tokenId": 7,
        "attention": None,
        "cancelled": False,
        "stats": {"positions": 3, "chunks": 1, "fillMs": 2.5},
    }
    assert normalise_poll(None) is None
    assert normalise_poll({"cancelled": True, "resident": 4, "stats": None}) == {
        "cancelled": True,
        "resident": 4,
        "stats": {"positions": 0, "chunks": 0, "fillMs": 0.0},
    }


def test_the_fake_clock_is_what_the_worker_sleeps_on() -> None:
    clock = FakeClock()
    worker, _ = fake_worker(step_ms=5.0, clock=clock)
    machine = pilot(worker, stall_ms=100.0)
    assert words([machine.decode(PILOT, allow_control=False)]) == ["write"]
    assert clock.slept == pytest.approx(0.005), "a poll returns the moment the step lands"
