# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The two boards a run can be played on.

``contracts.load_board`` is the one implementation; this module is where the
runners and the page import it from.
"""

from .contracts import BOARDS, BoardName, BoardSpec, board_text, load_board

__all__ = ["BOARDS", "BoardName", "BoardSpec", "board_text", "load_board"]
