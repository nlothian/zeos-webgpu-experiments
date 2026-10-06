# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``NodeWorker``'s decode step that does not block: begin, poll with a timeout, cancel.

Over ``web/stub_worker.js`` with simulated latency (needs Node).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from zeos_coop_count_web.node_worker import CHANNEL_BUSY, NodeWorker, node_available

pytestmark = pytest.mark.skipif(not node_available(), reason="needs node")

TAPES = {"pilot": ["write stdout left;", "read stdin;"]}
WORDS = ["write", " stdout", " left;", " read", " stdin;"]
ALL: dict[str, Any] = {"allowedBlocks": None, "allowedTokens": None}


def stub_worker(tmp_path: Path, **options: float) -> NodeWorker:
    spec = tmp_path / "stub.json"
    spec.write_text(json.dumps({"tapes": TAPES, "options": options}))
    return NodeWorker(stub=spec)


@pytest.fixture
def slow(tmp_path: Path) -> Iterator[NodeWorker]:
    with stub_worker(tmp_path, positionMs=20, stepMs=5) as w:
        yield w


def context(w: NodeWorker, job: str, n: int) -> None:
    w.createContext(job)
    w.append(job, [w.tokenize("write")[0]] * n)


def drain(w: NodeWorker, timeout_ms: float = 10_000) -> dict[str, Any]:
    reply = w.pollDecode(timeout_ms)
    assert reply is not None, f"no reply within {timeout_ms} ms"
    return reply


def test_begin_poll_timeout_and_result(slow: NodeWorker) -> None:
    context(slow, "a:pilot", 11)
    began = time.monotonic()
    request = slow.beginDecodeStep("a:pilot", ALL)
    assert time.monotonic() - began < 0.05
    assert isinstance(request, int)
    assert slow.inFlight
    began = time.monotonic()
    assert slow.pollDecode(0) is None
    assert slow.pollDecode(30) is None
    assert time.monotonic() - began >= 0.025
    reply = drain(slow)
    assert reply["cancelled"] is False
    assert slow.piece(reply["tokenId"]) == "write"
    assert reply["attention"] is None
    assert reply["resident"] == 11
    assert reply["stats"]["positions"] == 11
    assert not slow.inFlight
    assert slow.pollDecode(1000) is None


def test_calls_refuse_while_a_step_is_in_flight(slow: NodeWorker) -> None:
    context(slow, "a:pilot", 4)
    info = slow.info()
    slow.beginDecodeStep("a:pilot", ALL)
    with pytest.raises(RuntimeError, match=CHANNEL_BUSY):
        slow.length("a:pilot")
    with pytest.raises(RuntimeError, match=CHANNEL_BUSY):
        slow.decodeStep("a:pilot", ALL)
    with pytest.raises(RuntimeError, match=CHANNEL_BUSY):
        slow.beginDecodeStep("a:pilot", ALL)
    assert slow.info() == info
    assert slow.piece(5) == slow.piece(5)
    drain(slow)
    assert slow.length("a:pilot") == 4


def test_cancel_lands_between_chunks_and_the_step_resumes(slow: NodeWorker) -> None:
    n = 41
    context(slow, "a:pilot", n)
    slow.beginDecodeStep("a:pilot", {**ALL, "maxChunk": 8})
    assert slow.pollDecode(60) is None
    cancelled_at = time.monotonic()
    slow.cancelDecode()
    slow.cancelDecode()
    assert slow.inFlight
    reply = drain(slow)
    assert time.monotonic() - cancelled_at < 8 * 0.020 + 0.15
    assert reply["cancelled"] is True
    assert "tokenId" not in reply
    assert 0 < reply["resident"] < n - 1 and reply["resident"] % 8 == 0
    assert not slow.inFlight
    slow.beginDecodeStep("a:pilot", {**ALL, "maxChunk": 8})
    resumed = drain(slow)
    assert slow.piece(resumed["tokenId"]) == "write"
    assert resumed["stats"]["positions"] == n - reply["resident"]


def test_a_result_that_raced_the_cancel_is_dropped(tmp_path: Path) -> None:
    with stub_worker(tmp_path) as w:
        context(w, "a:pilot", 3)
        w.beginDecodeStep("a:pilot", ALL)
        time.sleep(0.3)  # the stub answers at once, so the reply is in the pipe by now
        w.cancelDecode()
        reply = drain(w)
        assert reply["cancelled"] is True and reply["resident"] == 3
        # The dropped token was never appended, so it is chosen again.
        w.beginDecodeStep("a:pilot", ALL)
        assert w.piece(drain(w)["tokenId"]) == "write"


def test_an_error_reply_raises_and_clears_the_step(slow: NodeWorker) -> None:
    slow.beginDecodeStep("missing:pilot", ALL)
    with pytest.raises(RuntimeError, match="no context missing:pilot"):
        slow.pollDecode(5000)
    assert not slow.inFlight
    slow.cancelDecode()


def test_cancelled_and_resumed_steps_say_what_uncancelled_ones_do(tmp_path: Path) -> None:
    with stub_worker(tmp_path, positionMs=2, stepMs=4) as w:
        for job in ("plain:pilot", "cut:pilot"):
            context(w, job, 30)
        plain: list[int] = []
        cut: list[int] = []
        for i in range(len(WORDS)):
            w.beginDecodeStep("plain:pilot", {**ALL, "maxChunk": 4})
            plain.append(drain(w)["tokenId"])
            w.append("plain:pilot", [plain[-1]])
            w.beginDecodeStep("cut:pilot", {**ALL, "maxChunk": 4})
            time.sleep(i * 0.004)
            w.cancelDecode()
            assert drain(w)["cancelled"] is True
            w.beginDecodeStep("cut:pilot", {**ALL, "maxChunk": 4})
            cut.append(drain(w)["tokenId"])
            w.append("cut:pilot", [cut[-1]])
        assert cut == plain
        assert [w.piece(t) for t in plain] == WORDS


def test_polling_with_no_wait_takes_the_reply_once_it_lands(slow: NodeWorker) -> None:
    context(slow, "a:pilot", 6)
    slow.beginDecodeStep("a:pilot", ALL)
    polls = 0
    deadline = time.monotonic() + 10
    while (reply := slow.pollDecode(0)) is None:
        assert time.monotonic() < deadline, "pollDecode(0) never took the reply"
        polls += 1
        time.sleep(0.005)
    assert polls > 0
    assert slow.piece(reply["tokenId"]) == "write"
    assert not slow.inFlight


def test_a_reply_to_an_earlier_request_is_discarded(slow: NodeWorker) -> None:
    context(slow, "a:pilot", 3)
    # A request whose caller never read its reply, as if interrupted mid-call.
    slow._write({"id": slow._next, "method": "length", "args": ["missing"]}, [])  # pyright: ignore[reportPrivateUsage]
    slow._next += 1  # pyright: ignore[reportPrivateUsage]
    assert slow.length("a:pilot") == 3
    slow.beginDecodeStep("a:pilot", ALL)
    assert slow.piece(drain(slow)["tokenId"]) == "write"


def test_close_with_a_step_in_flight_cancels_and_drains_it(tmp_path: Path) -> None:
    w = stub_worker(tmp_path, positionMs=50, stepMs=5)
    context(w, "a:pilot", 200)  # 10 s uninterrupted
    w.beginDecodeStep("a:pilot", {**ALL, "maxChunk": 4})
    began = time.monotonic()
    with w:
        pass
    assert time.monotonic() - began < 5
    assert not w.inFlight
    assert w._process.poll() is not None  # pyright: ignore[reportPrivateUsage]
