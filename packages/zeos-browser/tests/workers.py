# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Workers for tests that need what the stub does not do: measure attention, or misbehave."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from zeos_browser.fake_worker import FakeWorker, Step


@dataclass(frozen=True, slots=True)
class MeasuredStep:
    tokenId: int
    attention: list[float]


class MeasuringWorker(FakeWorker):
    """The fake, reporting attention spread evenly over the blocks it was allowed."""

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> MeasuredStep:  # type: ignore[override]
        step = super().decodeStep(jobId, opts)
        size = self.info().blockSize
        count = (self.length(jobId) + size - 1) // size
        blocks = opts["allowedBlocks"]
        allowed = [b for b in range(count) if blocks is None or blocks[b]]
        return MeasuredStep(
            tokenId=step.tokenId,
            attention=[1.0 / len(allowed) if b in allowed else 0.0 for b in range(count)],
        )


class RecordingWorker(FakeWorker):
    """The fake, keeping the options of every step so a test can read them back."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[Mapping[str, Any]] = []

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> Step:
        self.seen.append(opts)
        return super().decodeStep(jobId, opts)


class ChoosingWorker(FakeWorker):
    """The fake, but every step returns one fixed id regardless of the mask."""

    def __init__(self, *args: Any, choose: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.choose = choose

    def decodeStep(self, jobId: str, opts: Mapping[str, Any]) -> Step:
        return Step(tokenId=self.choose)
