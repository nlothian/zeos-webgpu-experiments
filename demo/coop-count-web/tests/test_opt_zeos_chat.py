# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A chat agent under the kernel on the real Qwen3.5-4B: ``open_chat`` over
``OptZeosWorker`` (``web/opt_zeos_worker.js``), run by Node with onnxruntime-node's CPU
provider through ``NodeWorker``.

One conversation: a plain question answered in prose, then a question only a tool can
answer, which must come back as a ``tool_call`` event the host runs, and a reply that
uses the result. Greedy, so the run is the same every time on one machine.

Needs the OPT+ZEOS export at ``models/Qwen3.5-4B-ZEOS-OPT`` and ``npm install``; skips
without them. About a minute on an M1 Max with 8 threads (``ZEOS_OPT_THREADS``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from zeos_coop_count_web.chat import ChatRun, open_chat
from zeos_coop_count_web.node_worker import DEMO, NodeWorker, node_available

MODEL = Path(os.environ.get("ZEOS_OPT_MODEL_DIR", DEMO / "models" / "Qwen3.5-4B-ZEOS-OPT"))

pytestmark = pytest.mark.skipif(
    not (MODEL / "meta.json").is_file() or not node_available(),
    reason=f"needs the OPT+ZEOS export at {MODEL} and npm install",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city's name."}},
                "required": ["city"],
            },
        },
    }
]


def system_prompt() -> str:
    """The tools section Qwen3.5's chat template renders, then the instructions."""
    tools = "\n".join(json.dumps(t) for t in TOOLS)
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{tools}\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:"
        "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n</function>\n</tool_call>\n\n"
        "You are a helpful assistant. Keep answers short."
    )


def turn(run: ChatRun) -> list[dict[str, Any]]:
    """Step until the job waits on a device pipe; every event but the tokens."""
    events: list[dict[str, Any]] = []
    for _ in range(64):
        events += [e for e in run.step(32) if e["type"] != "token"]
        if run.waiting_on() is not None:
            return events
    raise AssertionError("the job never waited")


@pytest.fixture(scope="module")
def worker() -> Any:
    with NodeWorker(
        MODEL, runtime="node", threads=int(os.environ.get("ZEOS_OPT_THREADS", "8"))
    ) as w:
        yield w


def test_a_chat_turn_and_a_tool_call_on_the_real_model(worker: NodeWorker) -> None:
    assert worker.backend == "cpu"
    run = open_chat(
        worker,
        tool_classes={"get_weather": "read"},
        param_types={"get_weather": {"city": "string"}},
        system_prompt=system_prompt(),
    )
    run.send_user("What is two plus two? Answer with one short sentence.")
    events = turn(run)
    (reply,) = [e for e in events if e["type"] == "reply"]
    assert "4" in reply["text"] or "four" in reply["text"].lower(), reply
    assert run.waiting_on() == "chat.user"

    run.send_user("What is the weather in Paris right now?")
    events = turn(run)
    calls = [e for e in events if e["type"] == "tool_call"]
    assert [c["name"] for c in calls] == ["get_weather"], events
    assert calls[0]["arguments"] == {"city": "Paris"}
    assert calls[0]["sink"] == "tools.read"
    assert run.waiting_on() == "tools.results"
    run.drain("tools.read")

    run.deliver_tool_result("Paris: sunny, 21 degrees Celsius, light wind from the west.")
    events = turn(run)
    (reply,) = [e for e in events if e["type"] == "reply"]
    assert "21" in reply["text"], reply
    assert run.waiting_on() == "chat.user"
    # The result arrived on ring 3, and the job read it.
    assert any(
        e["type"] == "arrived" and e["pipe"] == "tools.results" and e["ring"] == 3 for e in events
    )
