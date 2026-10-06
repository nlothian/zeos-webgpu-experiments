# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``BrowserPromptPlayer``: the native prompt loop over the worker channel, polled."""

# pyright: reportPrivateUsage=false
# The game package is untyped, so what the tests read off it is partly unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import pytest
from machine_helpers import FakeClock, SilentWorker, fake_worker
from zeos_space_invaders.game import Game, Rules, snapshot
from zeos_space_invaders.utils.views import LeadView

from zeos_space_invaders_web.contracts import (
    ChannelBroken,
    PromptArm,
    PromptArmFactory,
    PromptReply,
)
from zeos_space_invaders_web.fake_worker import (
    EOS_ID,
    IM_END_ID,
    IM_START_ID,
    PAD_ID,
    FakePilotWorker,
)
from zeos_space_invaders_web.prompt_player import (
    ASSISTANT_OPEN,
    CONTEXT_ID,
    PREFIX_ID,
    BrowserPromptPlayer,
)


def arm(
    replies: Sequence[str] = ("left",), *, step_ms: float = 0.0, **kwargs: object
) -> tuple[BrowserPromptPlayer, FakePilotWorker, FakeClock]:
    worker, clock = fake_worker(replies=replies, step_ms=step_ms)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules(), **kwargs)  # pyright: ignore[reportArgumentType]
    return player, worker, clock


def board() -> tuple[str, dict[str, object]]:
    game = Game(seed=1)
    return game.render(), cast("dict[str, object]", snapshot(game))


def ask(player: BrowserPromptPlayer, timeout_s: float = 0.001) -> PromptReply:
    player.begin(*board())
    for _ in range(10_000):
        reply = player.poll(timeout_s)
        if reply is not None:
            return reply
    raise AssertionError("no reply")


def test_the_class_meets_its_contract() -> None:
    factory: PromptArmFactory = BrowserPromptPlayer
    worker, _ = fake_worker()
    player: PromptArm = factory(worker, view=LeadView(), rules=Rules())
    assert not player.busy


def test_warm_prefills_the_system_prompt_before_any_board() -> None:
    player, worker, _ = arm()
    player.warm()
    assert worker.context_ids == (PREFIX_ID,)
    assert worker.resident(PREFIX_ID) == player._prefix > 0
    assert worker.delivered == 1 and worker.filled == player._prefix
    player.warm()
    assert worker.delivered == 1, "warming twice prefills once"


def test_a_reply_arrives_over_several_polls_without_blocking() -> None:
    player, _worker, clock = arm(("I will go left",), step_ms=10.0)
    player.warm()
    player.begin(*board())
    assert player.busy
    started = clock.now
    assert player.poll(0.0) is None, "a zero timeout returns at once"
    assert clock.now == started
    polls = 0
    reply = None
    while reply is None:
        reply = player.poll(0.003)
        polls += 1
    assert reply.text == "I will go left" and reply.action == "left" and reply.tokens == 4
    assert polls > 5 and not player.busy
    assert player.poll(1.0) is None, "nothing under way"


def test_the_reply_is_parsed_with_the_native_fallback() -> None:
    player, _, _ = arm(("I'll go with shoot this turn", "dunno", "right"), max_new=10)
    assert [ask(player).action for _ in range(3)] == ["shoot", None, "right"]
    assert player.parse_rate == pytest.approx(2 / 3)
    assert len(player.turns) == 3, "every reply is remembered, the fallback for one"
    assert "Recent turns" in str(player.render_turn(*board()))


def test_a_reply_stops_at_max_new_tokens() -> None:
    player, _, _ = arm(("one two three four five six seven eight",))
    reply = ask(player)
    assert reply.tokens == 6 and reply.text == "one two three four five six"


def test_each_request_is_forked_from_the_prefilled_prefix() -> None:
    """Forked, not truncated: the prefix keeps its KV and is never refilled, and one
    request's board is not the next one's."""
    player, worker, _ = arm(("left",), history=0)
    ask(player)
    first = worker.length(CONTEXT_ID)
    filled = worker.filled
    ask(player)
    assert worker.length(CONTEXT_ID) == first
    assert worker.length(PREFIX_ID) == player._prefix
    assert worker._contexts[CONTEXT_ID].ids[: player._prefix] == player._ids[: player._prefix]
    assert worker.filled - filled == first - player._prefix, "only the turn was filled"
    assert sorted(worker.context_ids) == sorted((PREFIX_ID, CONTEXT_ID))


def test_only_im_end_and_eos_among_the_reserved_ids_may_be_said() -> None:
    player, worker, _ = arm()
    ask(player)
    allowed = worker.last_options and worker.last_options["allowedTokens"]
    assert isinstance(allowed, bytes)
    assert allowed[PAD_ID] == 0 and allowed[IM_START_ID] == 0
    assert allowed[IM_END_ID] == 1 and allowed[EOS_ID] == 1
    assert worker.last_options is not None and worker.last_options["allowedBlocks"] is None


def test_a_reply_ends_at_a_newline() -> None:
    class Multiline(FakePilotWorker):
        def piece(self, tokenId: int) -> str:
            text = super().piece(tokenId)
            return "left\nand more" if text == " left" else text

    clock = FakeClock()
    worker = Multiline(replies=("left right",), clock=clock, sleep=clock.sleep)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules())
    reply = ask(player)
    assert reply.text == "left" and reply.tokens == 1


def test_a_fence_before_the_move_does_not_end_the_reply() -> None:
    class Fenced(FakePilotWorker):
        def piece(self, tokenId: int) -> str:
            text = super().piece(tokenId)
            fence = {"left": "```\n", "right": "right\n```"}
            return fence.get(text.strip(), text)

    clock = FakeClock()
    worker = Fenced(replies=("left right",), clock=clock, sleep=clock.sleep)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules())
    reply = ask(player)
    assert reply.text == "```\nright" and reply.action == "right" and reply.tokens == 2


def test_the_assistant_turn_opens_with_an_empty_think_block() -> None:
    clock = FakeClock()
    worker = FakePilotWorker(replies=("left",), clock=clock, sleep=clock.sleep)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules())
    ask(player)
    tail = worker.tokenize("assistant\n" + ASSISTANT_OPEN)
    ids = player._ids  # pyright: ignore[reportPrivateUsage]
    at = max(i for i in range(len(ids) - len(tail) + 1) if ids[i : i + len(tail)] == tail)
    assert at > player._prefix  # pyright: ignore[reportPrivateUsage]


def test_begin_refuses_a_second_request() -> None:
    player, _, _ = arm(step_ms=5.0)
    player.begin(*board())
    with pytest.raises(RuntimeError, match="already under way"):
        player.begin(*board())


def test_close_cancels_the_request_and_destroys_the_context() -> None:
    player, worker, _ = arm(step_ms=50.0)
    player.begin(*board())
    player.close()
    assert not worker.inFlight and worker.cancelled == 1
    assert worker.context_ids == () and not player.busy


def test_the_step_chunk_is_passed_when_asked_for() -> None:
    player, worker, _ = arm(max_chunk=32)
    ask(player)
    assert worker.last_options is not None and worker.last_options["maxChunk"] == 32


def test_an_error_reply_ends_the_request_and_close_still_returns() -> None:
    class Failing(FakePilotWorker):
        fail = False

        def pollDecode(self, timeoutMs: float) -> Any:
            if self.fail:
                self._flight = None
                raise RuntimeError("model thread: out of memory")
            return super().pollDecode(timeoutMs)

    clock = FakeClock()
    worker = Failing(step_ms=10.0, clock=clock, sleep=clock.sleep)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules())
    player.begin(*board())
    worker.fail = True
    with pytest.raises(RuntimeError, match="out of memory"):
        player.poll(1.0)
    assert not player.busy
    player.close()
    assert worker.context_ids == ()


def test_close_gives_up_on_a_worker_that_never_answers() -> None:
    clock = FakeClock()
    worker = SilentWorker(replies=("left right",), step_ms=50.0, clock=clock, sleep=clock.sleep)
    player = BrowserPromptPlayer(worker, view=LeadView(), rules=Rules(), settle_timeout_s=0.05)
    player.begin(*board())
    worker.silent = True
    began = time.monotonic()
    with pytest.raises(ChannelBroken, match="not handed back within 0.05 s"):
        player.close()
    assert time.monotonic() - began < 5.0
    assert worker.broken is not None and not player.busy
    player.close()  # nothing left to wait for
