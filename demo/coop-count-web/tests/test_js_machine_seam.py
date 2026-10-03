# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What ``JsMachine`` asks of a worker, and what it refuses from one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from workers import ChoosingWorker, MeasuringWorker, RecordingWorker
from zeos.core.ids import JobId
from zeos.machine.base import ControlTokenViolation, MaskViolation, tokens_from_text

from zeos_coop_count_web.fake_worker import CONTROL_IDS, PAD_ID, UNK_ID, FakeWorker, ModelInfo
from zeos_coop_count_web.js_machine import JsMachine, WorkerViolation

JOB = JobId(1)
CHILD = JobId(2)
TAPES = {"d": ["say hello"]}


class DoublingWorker(MeasuringWorker):
    """Every word is two model tokens, so kernel offsets and model positions part ways."""

    def tokenize(self, text: str) -> list[int]:
        return [i for token_id in super().tokenize(text) for i in (token_id, token_id)]


@dataclass(frozen=True, slots=True)
class FixedStep:
    tokenId: int
    attention: list[float]


class FixedAttentionWorker(FakeWorker):
    def __init__(self, *args: Any, attention: Sequence[float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attention = list(attention)

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> FixedStep:  # type: ignore[override]
        return FixedStep(tokenId=super().decodeStep(jobId, opts).tokenId, attention=self.attention)


class NoMarkersWorker(FakeWorker):
    def info(self) -> ModelInfo:
        return ModelInfo(
            blockSize=16, padId=PAD_ID, controlIds=(), eosId=1, vocabSize=super().info().vocabSize
        )


def machine(worker: FakeWorker, **kwargs: Any) -> JsMachine:
    m = JsMachine(worker, **kwargs)
    m.create_context(JOB, "d")
    return m


def resident(worker: FakeWorker, m: JsMachine, job: JobId = JOB) -> list[int]:
    key = m._ctx_of(job).key  # pyright: ignore[reportPrivateUsage]
    return list(worker._contexts[key].ids)  # pyright: ignore[reportPrivateUsage]


# -- chat framing and reserved ids -----------------------------------------


def test_chatml_needs_both_turn_markers_among_the_control_ids() -> None:
    with pytest.raises(WorkerViolation, match="controlIds"):
        JsMachine(NoMarkersWorker(TAPES))
    JsMachine(NoMarkersWorker(TAPES), chat_template=None)


def test_framing_reaches_the_markers_by_control_id_and_moves_no_kernel_offset() -> None:
    worker = FakeWorker(TAPES)
    m = machine(worker)
    m.inject(JOB, tokens_from_text("say there"))
    m.decode(JOB, allow_control=False)
    raw = m.raw(JOB)
    # "say" is a tape word; "there" is not, so it is the unknown token.
    assert [w.pieces for w in raw.words][:2] == [("say",), ("<unk>",)]
    assert raw.words[0].framing == ("<|im_start|>", "user")
    assert raw.words[2].framing == ("<|im_end|>", "<|im_start|>", "assistant")
    assert m.stats(JOB).resident_tokens == 3
    start, end = CONTROL_IDS
    assert resident(worker, m)[:2] == [start, worker.tokenize("user")[0]]
    assert raw.kv_resident == len(resident(worker, m))


def test_a_control_id_from_the_worker_is_a_control_token_violation() -> None:
    m = machine(ChoosingWorker(TAPES, choose=CONTROL_IDS[0]))
    m.inject(JOB, tokens_from_text("go"))
    with pytest.raises(ControlTokenViolation):
        m.decode(JOB, allow_control=False)


@pytest.mark.parametrize("choose", [PAD_ID, UNK_ID, 10_000])
def test_an_id_the_mask_refused_is_a_worker_violation(choose: int) -> None:
    m = machine(ChoosingWorker(TAPES, choose=choose))
    m.inject(JOB, tokens_from_text("go"))
    with pytest.raises(WorkerViolation, match="refused"):
        m.decode(JOB, allow_control=False)


def test_the_vocabulary_is_the_size_info_declares() -> None:
    class Asked(FakeWorker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.asked: list[int] = []

        def piece(self, tokenId: int) -> str:
            self.asked.append(tokenId)
            return super().piece(tokenId)

    worker = Asked(TAPES)
    m = JsMachine(worker)
    size = worker.info().vocabSize
    assert m.vocabulary_size == size
    assert worker.asked[:size] == list(range(size)), "every id below vocabSize, nothing past it"
    assert max(worker.asked) == size - 1


@pytest.mark.parametrize("field", ["padId", "eosId", "vocabSize"])
def test_info_naming_ids_outside_the_vocabulary_is_a_worker_violation(field: str) -> None:
    class Short(FakeWorker):
        def info(self) -> ModelInfo:
            real = super().info()
            values = {
                "padId": real.padId,
                "eosId": real.eosId,
                "vocabSize": real.vocabSize,
                field: 0 if field == "vocabSize" else real.vocabSize,
            }
            return ModelInfo(blockSize=16, controlIds=real.controlIds, **values)

    with pytest.raises(WorkerViolation, match="vocab"):
        JsMachine(Short(TAPES))


# -- words against model tokens ----------------------------------------------


def test_offsets_stay_in_kernel_words_when_a_word_is_several_tokens() -> None:
    worker = DoublingWorker(TAPES, block_size=4)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f"))
    assert m.stats(JOB).resident_tokens == 6
    m.decode(JOB, allow_control=False)
    assert len(resident(worker, m)) == 12, "the six words, prefilled; the decoded id waits"

    m.splice(JOB, 1, 3, tokens_from_text("X"))
    assert [t.text for t in m.transcript(JOB)] == ["a", "X", "d", "e", "f", "say"]
    assert len(resident(worker, m)) == 2, "the worker keeps only what precedes the splice"
    m.decode(JOB, allow_control=False)
    machine_ids = m._ctx_of(JOB).ids  # pyright: ignore[reportPrivateUsage]
    assert resident(worker, m) == machine_ids[: len(resident(worker, m))]
    assert len(resident(worker, m)) == len(machine_ids) - 1

    m.trunc(JOB, 2)
    assert len(resident(worker, m)) == 4


def test_a_fork_copies_the_worker_context_and_keeps_the_childs_own() -> None:
    worker = FakeWorker({"d": ["say hello"], "other": ["exit"]})
    m = machine(worker)
    m.inject(JOB, tokens_from_text("shared words"))
    m.create_context(CHILD, "other")
    assert m.fork(JOB, CHILD) == 2
    assert resident(worker, m, CHILD) == resident(worker, m)
    assert m.decode(CHILD, allow_control=False).tokens[0].text == "exit;"


# -- the block mask and measured attention -----------------------------------


def test_a_worker_block_straddling_a_hidden_kernel_block_is_hidden() -> None:
    worker = MeasuringWorker(TAPES, block_size=3)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f g h"))

    m.set_mask(JOB, frozenset({1}))
    result = m.decode(JOB, allow_control=False)
    # Worker block 1 holds tokens 3, 4, 5: one from kernel block 0, so it is hidden too.
    assert result.attention == {1: pytest.approx(1.0)}

    m.trunc(JOB, 8)
    m.set_mask(JOB, frozenset({0}))
    assert m.decode(JOB, allow_control=False).attention == {0: pytest.approx(1.0)}


def test_mass_on_a_straddling_block_is_shared_by_token_count() -> None:
    worker = FixedAttentionWorker(TAPES, block_size=3, attention=[1 / 3, 1 / 3, 1 / 3])
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f g h"))
    result = m.decode(JOB, allow_control=False)
    assert result.attention_hint is None
    assert result.attention == {0: pytest.approx(4 / 9), 1: pytest.approx(5 / 9)}


@pytest.mark.parametrize(
    "attention", [[0.5, 0.5, 0.5], [1.5, -0.5, 0.0], [0.5, 0.5], [0.25, 0.25, 0.25, 0.25]]
)
def test_attention_outside_the_units_is_a_worker_violation(attention: list[float]) -> None:
    worker = FixedAttentionWorker(TAPES, block_size=3, attention=attention)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f g h"))
    with pytest.raises(WorkerViolation):
        m.decode(JOB, allow_control=False)


def test_a_worker_that_cannot_measure_leaves_an_undeclared_hint() -> None:
    m = machine(FakeWorker(TAPES))
    m.inject(JOB, tokens_from_text("go"))
    result = m.decode(JOB, allow_control=False)
    assert result.attention is None
    assert result.attention_hint is not None and not result.attention_hint.declared


def test_padding_is_the_workers_pad_id() -> None:
    worker = FakeWorker(TAPES, block_size=4)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a"))
    assert m.pad_to_block(JOB) == 3
    m.decode(JOB, allow_control=False)
    assert resident(worker, m)[1:] == [PAD_ID] * 3


def test_attention_on_a_block_the_step_hid_is_a_mask_violation() -> None:
    worker = FixedAttentionWorker(TAPES, block_size=4, attention=[0.5, 0.5])
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f g h"))
    m.set_mask(JOB, frozenset({1}))
    with pytest.raises(MaskViolation):
        m.decode(JOB, allow_control=False)


# -- the mask's horizon ------------------------------------------------------


def test_blocks_decoded_past_the_horizon_are_visible_and_sent_as_allowed() -> None:
    worker = RecordingWorker({"d": ["say a b c d e f"]}, block_size=4)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("p q r s t u v w"))
    m.set_mask(JOB, frozenset({1}))  # installed over two blocks; block 0 hidden
    for _ in range(4):
        m.decode(JOB, allow_control=False)
    assert m.stats(JOB).blocks == 3
    assert m.visible_blocks(JOB) == frozenset({1, 2}), "block 2 holds only the job's own words"
    assert bytes(worker.seen[-1]["allowedBlocks"]) == b"\x00\x01\x01"


def test_a_trunc_below_the_horizon_lowers_it() -> None:
    worker = FakeWorker({"d": ["say a b c d e f"]}, block_size=4)
    m = machine(worker, block_size=4, chat_template=None)
    m.inject(JOB, tokens_from_text("a b c d e f g h i j k l"))
    m.set_mask(JOB, frozenset({2}))
    m.trunc(JOB, 4)
    assert m.visible_blocks(JOB) == frozenset(), "the stale mask still hides block 0"
    for _ in range(2):
        m.decode(JOB, allow_control=False)
    assert m.visible_blocks(JOB) == frozenset({1}), "block 1 was written after the trunc"


# -- tapes the two workers could not agree on ----------------------------------


def test_a_non_ascii_tape_is_refused() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        FakeWorker({"d": ["say \u2192"]})
