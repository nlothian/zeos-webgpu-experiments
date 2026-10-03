# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The machine backend contract, run against the scripted backend and ``JsMachine``.

A copy of ``demo/coop-count/tests/test_machine_contract.py`` with the llama backend
swapped for ``JsMachine`` over the Python fake worker, twice: once as the stub is, with
no attention, and once over a worker that measures, so that the two attention clauses
bind rather than skip. The clauses themselves, and why each matters, are documented in
``tests/contract/test_machine_contract.py`` at the repository root.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from workers import MeasuringWorker
from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import MachineBackend, render, tokens_from_text
from zeos.machine.scripted import Script, ScriptedMachine

from zeos_coop_count_web.fake_worker import FakeWorker
from zeos_coop_count_web.js_machine import JsMachine

JOB = JobId(1)
CHILD = JobId(2)


@dataclass(frozen=True)
class Backend:
    name: str
    build: Callable[..., tuple[MachineBackend, JobId]]

    def make(
        self, *, block_size: int = 4, emits: Sequence[str] = ()
    ) -> tuple[MachineBackend, JobId]:
        return self.build(block_size=block_size, emits=emits)


def _scripted(
    *, block_size: int, emits: Sequence[str], **_: object
) -> tuple[MachineBackend, JobId]:
    script = Script.from_spec([{"emit": text} for text in emits])
    machine = ScriptedMachine({"d": script}, block_size=block_size)
    machine.create_context(JOB, "d")
    return machine, JOB


def _js(worker_class: type[FakeWorker]) -> Callable[..., tuple[MachineBackend, JobId]]:
    def build(*, block_size: int, emits: Sequence[str], **_: object):
        # The fake plays a tape through the grammar mask, so what it emits has to be a
        # command, so each emit is said. The machine is told no descriptors, so every
        # alias of the ABI is open to ``d``. The worker's blocks are the kernel's size, so
        # every clause about blocks reads the same in both units.
        tape = [f"say {text}" for text in emits] or ["say hello"]
        worker = worker_class({"d": tape, "child": ["say child-step"]}, block_size=block_size)
        machine = JsMachine(worker, block_size=block_size)
        machine.create_context(JOB, "d")
        return machine, JOB

    return build


BACKENDS = ["M0-scripted", "js-fake", "js-measuring"]


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> Backend:
    name = request.param  # pyright: ignore[reportAny]
    if name == "M0-scripted":
        return Backend(name, _scripted)
    if name == "js-fake":
        return Backend(name, _js(FakeWorker))
    return Backend(name, _js(MeasuringWorker))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", BACKENDS, indirect=True, ids=BACKENDS)


# -- the mask -----------------------------------------------------------------


def test_empty_mask_hides_everything_and_is_not_the_same_as_no_mask(backend: Backend) -> None:
    """An empty mask hides every block, unlike having no mask at all."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f"))
    assert m.visible_blocks(job) == frozenset({0, 1}), "unmasked means every live block"

    m.set_mask(job, frozenset())
    assert m.visible_blocks(job) == frozenset()


def test_a_mask_can_only_narrow_never_widen(backend: Backend) -> None:
    """Block indices outside the context do not add anything to the mask."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b"))
    m.set_mask(job, frozenset({0, 7, 99}))
    assert m.visible_blocks(job) <= frozenset({0})


def test_reported_attention_never_falls_outside_the_mask(backend: Backend) -> None:
    """Reported attention never covers a block the mask hides."""
    m, job = backend.make(block_size=4, emits=["x y"])
    m.inject(job, tokens_from_text("a b c d e f g h"))
    m.set_mask(job, frozenset({0}))
    result = m.decode(job, allow_control=False)
    if result.attention is None:
        pytest.skip("backend does not measure attention; C2 unenforceable here")
    assert set(result.attention) <= m.visible_blocks(job)


def test_measured_attention_is_normalised_per_block(backend: Backend) -> None:
    """Measured attention sums to one and is never negative."""
    m, job = backend.make(block_size=4, emits=["x y"])
    m.inject(job, tokens_from_text("a b c d e f g h"))
    result = m.decode(job, allow_control=False)
    if result.attention is None:
        pytest.skip("backend does not measure attention; C15 unenforceable here")
    assert sum(result.attention.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in result.attention.values())


# -- offset stability --------------------------------------------------------


def test_only_splice_moves_an_existing_offset(backend: Backend) -> None:
    """Only ``splice`` moves an existing offset; decode, inject, pad and fork do not."""
    m, job = backend.make(block_size=4, emits=["p q"])
    m.inject(job, tokens_from_text("a b c d e"))
    before = render(m.transcript(job))

    m.decode(job, allow_control=False)
    assert render(m.transcript(job)).startswith(before)

    m.inject(job, tokens_from_text("z"))
    assert render(m.transcript(job)).startswith(before)

    m.pad_to_block(job)
    assert render(m.transcript(job)).startswith(before)

    m.fork(job, CHILD)
    assert render(m.transcript(job)).startswith(before)

    m.trunc(job, 5)
    assert render(m.transcript(job)) == before


def test_splice_shifts_downstream_by_exactly_delta(backend: Backend) -> None:
    """A splice shifts the tokens after it by exactly the change in length."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f g h"))
    start, end, replacement = 2, 4, tokens_from_text("X")
    tail_before = render(m.transcript(job)[end:])

    result = m.splice(job, start, end, replacement)

    assert result.tokens_in == len(replacement)
    assert result.invalidated_downstream == 8 - end
    delta = result.tokens_in - (end - start)
    assert render(m.transcript(job)) == "a b X e f g h"
    assert render(m.transcript(job)[end + delta :]) == tail_before


def test_trunc_leaves_earlier_offsets_alone_and_shrinks_visibility(backend: Backend) -> None:
    """A truncation keeps earlier tokens and drops mask blocks that no longer exist."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f g h i j k l"))
    m.set_mask(job, frozenset({2}))
    assert m.visible_blocks(job) == frozenset({2})

    m.trunc(job, 4)

    assert render(m.transcript(job)) == "a b c d"
    assert m.visible_blocks(job) == frozenset()


# -- fork --------------------------------------------------------------------


def test_fork_carries_the_mask_to_the_child(backend: Backend) -> None:
    """A fork copies the parent's mask to the child."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("secret secret secret secret public public"))
    m.set_mask(job, frozenset({1}))
    m.fork(job, CHILD)
    assert m.visible_blocks(CHILD) == frozenset({1})


def test_fork_into_an_existing_context_keeps_the_childs_own_behaviour(backend: Backend) -> None:
    """A fork copies only tokens and mask, so the child keeps its own decode state."""
    m, job = backend.make(block_size=4, emits=["parent-step"])
    if isinstance(m, ScriptedMachine):
        m.register_script("child", Script.from_spec([{"emit": "child-step"}]))
        m.create_context(CHILD, "child")
        m.inject(job, tokens_from_text("shared"))
        assert m.fork(job, CHILD) == 1
        assert render(m.transcript(CHILD)) == "shared"
        m.decode(CHILD, allow_control=False)
        assert render(m.transcript(CHILD)) == "shared child-step"
        return

    assert isinstance(m, JsMachine)
    m.create_context(CHILD, "child")
    m.inject(job, tokens_from_text("shared"))

    assert m.fork(job, CHILD) == 1
    assert render(m.transcript(CHILD)) == "shared"
    m.decode(CHILD, allow_control=False)
    m.decode(CHILD, allow_control=False)
    assert [t.text.strip() for t in m.transcript(CHILD)] == ["shared", "say", "child-step;"], (
        "the child played its own tape; only tokens and mask were copied in"
    )


# -- control tokens ----------------------------------------------------------


def test_inject_may_carry_control_tokens_but_decode_may_not(backend: Backend) -> None:
    """``inject`` may add control tokens, and they are reported as control."""
    m, job = backend.make(block_size=4)
    start, end = m.inject(job, tokens_from_text("<stub 3fa2>", TokenKind.CONTROL))
    assert (start, end) == (0, 2)
    assert all(t.kind is TokenKind.CONTROL for t in m.transcript(job))


def test_padding_is_control_and_never_fills_a_whole_block(backend: Backend) -> None:
    """Padding is control tokens and never adds a whole block."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c"))
    added = m.pad_to_block(job)

    assert 0 < added < 4
    assert all(t.kind is TokenKind.CONTROL for t in m.transcript(job)[-added:])
    assert m.stats(job).resident_tokens % 4 == 0
    assert m.pad_to_block(job) == 0


# -- derived quantities ------------------------------------------------------


def test_block_arithmetic_is_derived_from_length_alone(backend: Backend) -> None:
    """Block counts and ranges follow from the token count alone, including when empty."""
    m, job = backend.make(block_size=4)
    assert m.stats(job).blocks == 0
    assert m.blocks_for_range(job, 0, 0) == frozenset()

    m.inject(job, tokens_from_text("a b c d e"))
    stats = m.stats(job)
    assert (stats.resident_tokens, stats.blocks) == (5, 2)
    assert m.blocks_for_range(job, 0, 1) == frozenset({0})
    assert m.blocks_for_range(job, 3, 5) == frozenset({0, 1})
    assert m.blocks_for_range(job, 4, 5) == frozenset({1})


# -- context lifecycle -------------------------------------------------------


def test_operations_on_an_absent_context_raise(backend: Backend) -> None:
    """Every operation on a destroyed context raises."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b"))
    m.destroy_context(job)

    with pytest.raises(Exception):
        m.stats(job)
    with pytest.raises(Exception):
        m.transcript(job)
    with pytest.raises(Exception):
        m.inject(job, tokens_from_text("c"))


def test_splice_and_trunc_reject_out_of_range_offsets(backend: Backend) -> None:
    """``splice`` and ``trunc`` raise ``IndexError`` on offsets outside the context."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c"))

    with pytest.raises(IndexError):
        m.trunc(job, 4)
    with pytest.raises(IndexError):
        m.splice(job, 2, 1, ())
    with pytest.raises(IndexError):
        m.splice(job, 0, 9, ())

    assert m.trunc(job, 3) == 0
