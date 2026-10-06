# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures the browser port's tests share."""

from pathlib import Path

import pytest

#: The repository's ``demo/`` directory, for tests that compare against the native demo.
DEMO = Path(__file__).resolve().parents[2]


@pytest.fixture
def native_demo() -> Path:
    """``demo/space-invaders``, whose settings files the packaged boards copy."""
    return DEMO / "space-invaders"
