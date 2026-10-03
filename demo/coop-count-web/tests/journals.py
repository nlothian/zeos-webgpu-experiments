# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Comparing a ``JsMachine`` journal with the seat's, where the two may differ."""

from __future__ import annotations

import json

#: What a mask's horizon changes (``js_machine`` docstring, "A mask covers the blocks it
#: was computed for"): over the seat the kernel denies a job attention to the block it is
#: writing into and so leaves its open output out of the working set; over ``JsMachine``
#: that block is visible until the next mask decides it.
HORIZON_KINDS = frozenset({"mask.denied", "vm.working_set"})


def kinds(journal: bytes) -> list[str]:
    return [json.loads(line)["kind"] for line in journal.decode().splitlines()]


def beyond_the_horizon(journal: bytes) -> list[str]:
    """The journal with every event a mask's horizon can change left out and the
    sequence numbers dropped, so two journals compare on everything else."""
    out: list[str] = []
    for line in journal.decode().splitlines():
        event = json.loads(line)
        if event["kind"] in HORIZON_KINDS:
            continue
        del event["seq"]
        out.append(json.dumps(event, sort_keys=True))
    return out
