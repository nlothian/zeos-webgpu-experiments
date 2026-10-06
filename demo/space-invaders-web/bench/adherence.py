# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

# pyright: basic
# The benchmarks drive the native game, whose modules are untyped; strict mode would
# only restate that.

"""Syscall adherence of the 4B ZEOS-OPT on onnxruntime-node CPU, step-clocked.

**Zeos arm.** The real case (``ZeosDriver``/``build_kernel``: pilot.md plus the board's
rules) over the existing synchronous ``JsMachine`` with the grammar mask, on
``NodeWorker(runtime="node")``. No wall clock: each board is delivered to ``game.state``
the way ``ZeosDriver.sense`` delivers one (only boards; threats are never sensed, so
``evade`` never runs), the kernel is pumped until a write to ``game.controls`` lands, the
kernel idles, or ``--cap`` decode steps pass, and then the game ticks once. ``--mode``:

* ``plain``: ``JsMachine`` as it is; the model must say ``read stdin;`` itself.
* ``auto-read``: after each completed write the machine answers the next decode with
  ``read stdin`` without the model, as the native ``APIMachineBase`` does and as
  ``PilotJsMachine`` is specified to (CONTRACTS: "an automatic read stdin after a
  completed write").

Counted per board: every command and its decode steps, valid move payloads
(``write stdout left|right|shoot``) against other payloads, whether a move followed a
read, ``exit`` (the pilot is respawned, and the respawn's prefill counted), the grammar
mask's cache misses and their cost.

**Prompt arm.** ``PromptPlayer.default_prompt()`` as a system turn prefilled once, then
per board: truncate to it, append ``render_turn`` (5 turns of history) as the user turn
and an assistant turn with an empty think block, greedy-decode at most 6 unconstrained
tokens (ending at ``<|im_end|>``), ``PromptPlayer.parse``.

Writes ``results/adherence_<arm>_<board>[_<mode>].json``::

    UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/adherence.py \\
        --arm zeos --board default --mode auto-read --boards 15
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from _common import MODEL, board, pilot_seat_maps, write_json
from zeos.core.ids import JobId, ObjectName
from zeos.machine.base import (
    AttentionHint,
    DecodeResult,
    MachineRequest,
    OpKind,
    SpliceResult,
    Token,
)
from zeos_coop_count_web.js_machine import JsMachine
from zeos_coop_count_web.node_worker import NodeWorker
from zeos_coop_count_web.token_mask import CommandLanguage, RoundState
from zeos_space_invaders.game import ACTIONS, Controls, Game, snapshot
from zeos_space_invaders.players.base import FALLBACK_ACTION, PromptPlayer
from zeos_space_invaders.players.zeos.player import GAME_STATE, ZeosDriver, encode
from zeos_space_invaders.utils.views import LeadView

from zeos_space_invaders_web.contracts import BoardName

#: The moves a ``write stdout`` payload may be under ``--grammar moves``.
MOVES = ("left", "right", "shoot")
_MOVE = 7  # (7, alt, word, p): p characters of MOVES[word] matched


class MoveLanguage(CommandLanguage):
    """The pilot's language with ``write stdout``'s payload narrowed to one of ``MOVES``.

    A prototype of the payload sub-grammar the plan holds in reserve, built on
    ``CommandLanguage``'s automaton: the text tail of a ``write stdout`` alternative is
    replaced by a literal choice among the moves, and the terminator follows it.
    """

    def _tail(self, state: tuple[int, ...], char: str) -> tuple[tuple[int, ...], ...]:
        _, i, _k = state
        alt = self._alternatives[i]
        if alt.call and alt.head.startswith("write stdout"):
            return tuple((_MOVE, i, w, 1) for w, word in enumerate(MOVES) if word[0] == char)
        return super()._tail(state, char)

    def _step(self, state: tuple[int, ...], char: str) -> Iterable[tuple[int, ...]]:
        if state[0] == _MOVE:
            _, i, w, p = state
            word = MOVES[w]
            if p == len(word):
                return self._terminate(i) if char == self.abi.terminator[0] else ()
            return ((_MOVE, i, w, p + 1),) if word[p] == char else ()
        return super()._step(state, char)


class BenchMachine(JsMachine):
    """``JsMachine`` that logs every step and command, times the mask, and optionally
    reads ``stdin`` for the pilot after each write."""

    def __init__(self, *args: Any, auto_read: bool, moves: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, on_command=self._command, **kwargs)
        self.auto_read = auto_read
        self.moves = moves
        self.last_roundtrip: float | None = None
        self.steps = 0
        self.since_command = 0
        self.commands: list[dict[str, Any]] = []
        self.mask_calls: list[dict[str, Any]] = []
        self.step_ms: list[float] = []
        self.splices: list[dict[str, int]] = []
        self._read_due: set[JobId] = set()
        allowed = self._mask.allowed

        def timed(key: str, language: Any, state: RoundState, *, allow_control: bool) -> bytes:
            hit = (key, state, allow_control) in self._mask._cache  # pyright: ignore[reportPrivateUsage]
            began = time.perf_counter()
            flags = allowed(key, language, state, allow_control=allow_control)
            self.mask_calls.append(
                {"hit": hit, "ms": (time.perf_counter() - began) * 1000, "state": repr(state)}
            )
            return flags

        self._mask.allowed = timed  # type: ignore[method-assign]

    def _command(self, job: JobId, line: str, request: MachineRequest) -> None:
        self.commands.append(
            {
                "line": line,
                "op": request.op.name,
                "pipe": None if request.pipe is None else str(request.pipe),
                "payload": " ".join(t.text for t in request.payload),
                "tokens": self.since_command,
                "auto": False,
            }
        )
        self.since_command = 0

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        if self.auto_read and job in self._read_due:
            self._read_due.discard(job)
            ctx.parser.reset()
            ctx.round = self._language(ctx.descriptor).start
            self.commands.append(
                {
                    "line": "read stdin",
                    "op": "READ",
                    "pipe": "stdin",
                    "payload": "",
                    "tokens": 0,
                    "auto": True,
                }
            )
            return DecodeResult(
                tokens=(),
                request=self.abi.parse("read stdin"),
                attention=None,
                attention_hint=AttentionHint(tags=("self",)),
            )
        began = time.perf_counter()
        result = super().decode(job, allow_control=allow_control)
        self.step_ms.append((time.perf_counter() - began) * 1000)
        self.steps += 1
        self.since_command += 1
        if result.request.op is OpKind.WRITE:
            self._read_due.add(job)
        return result

    def _language(self, descriptor: str) -> CommandLanguage:
        if not self.moves or descriptor != "pilot":
            return super()._language(descriptor)
        language = self._languages.get(descriptor)
        if language is None:
            language = MoveLanguage(self.abi, self.aliases(descriptor))
            self._languages[descriptor] = language
        return language

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        ctx = self._ctx_of(job)
        kv_start = ctx.kv_offset(start)
        result = super().splice(job, start, end, tokens)
        # Everything after the splice point is re-run by the next step.
        self.splices.append(
            {
                "step": self.steps,
                "kv_start": kv_start,
                "replay": len(ctx.ids) - kv_start,
                "removed_words": end - start,
                "inserted_words": len(tokens),
            }
        )
        return result

    def invalidate(self, job: JobId) -> None:
        """Nothing is ever preempted here: no threat is sensed."""


def zeos_arm(
    name: BoardName, mode: str, boards: int, cap: int, threads: int, grammar: str
) -> dict[str, Any]:
    spec = board(name)
    descriptors, _ = pilot_seat_maps()
    started = time.monotonic()
    worker = NodeWorker(MODEL, runtime="node", threads=threads)
    load_s = time.monotonic() - started
    machine = BenchMachine(
        worker, descriptors=descriptors, auto_read=mode == "auto-read", moves=grammar == "moves"
    )
    game = Game(seed=spec.seed, rules=spec.rules)
    controls = Controls(game, per_tick=spec.actions_per_tick)
    view = LeadView()
    driver = ZeosDriver(machine=machine, view=LeadView(), rules=spec.rules, max_ticks=cap)  # pyright: ignore[reportArgumentType]
    driver.controls = controls
    kernel = driver.kernel
    began = time.monotonic()
    driver.start()
    rows: list[dict[str, Any]] = []
    respawns = 0
    for b in range(boards):
        if game.over:
            break
        obs, info = game.render(), snapshot(game)
        now = view.state(obs, info)
        first_command = len(machine.commands)
        first_step = machine.steps
        board_began = time.monotonic()
        driver._seen_events = len(kernel.events)  # pyright: ignore[reportPrivateUsage]
        driver._publish_world(info)  # pyright: ignore[reportPrivateUsage]
        stale = kernel.pipes.get(GAME_STATE)
        if stale.readable:
            stale.read()
        kernel.deliver(GAME_STATE, encode(now))
        moved = None
        while machine.steps - first_step < cap:
            ticks = driver._pump()  # pyright: ignore[reportPrivateUsage]
            for job_id in kernel.unreaped():
                kernel.reap(job_id)
            if driver._author(since=driver._seen_events) is not None:  # pyright: ignore[reportPrivateUsage]
                moved = kernel.world.get(ObjectName("game.action"), "shoot")
                break
            if not any(j.descriptor.name == "pilot" for j in kernel.sched.jobs()):
                # The pilot exited: boot it again, as the case boots it.
                kernel.spawn(next(iter(driver.boot)))  # pyright: ignore[reportArgumentType]
                respawns += 1
                continue
            if ticks < cap:
                # The batch ended short of its bound with no write: the kernel idled,
                # i.e. the pilot is asleep on a drained pipe.
                break
        commands = machine.commands[first_command:]
        rows.append(
            {
                "board": b,
                "tick": info["ticks"],
                "moved": moved,
                "steps": machine.steps - first_step,
                "seconds": round(time.monotonic() - board_began, 2),
                "commands": commands,
            }
        )
        print(
            f"board {b} tick {info['ticks']}: {[c['line'] for c in commands]} "
            f"({machine.steps - first_step} steps, {time.monotonic() - board_began:.1f}s)",
            flush=True,
        )
        controls.tick()
    total_s = time.monotonic() - began
    worker.close()
    every = [c for r in rows for c in r["commands"]]
    model_cmds = [c for c in every if not c["auto"]]
    writes = [c for c in model_cmds if c["op"] == "WRITE"]
    moves = [c for c in writes if c["pipe"] == "stdout" and c["payload"] in ACTIONS]
    move_after_read = 0
    for i, c in enumerate(every):
        if c in moves and i > 0 and every[i - 1]["op"] == "READ":
            move_after_read += 1
    misses = [m for m in machine.mask_calls if not m["hit"]]
    return {
        "arm": "zeos",
        "board": name,
        "mode": mode,
        "grammar": grammar,
        "seed": spec.seed,
        "boards": len(rows),
        "load_s": round(load_s, 1),
        "run_s": round(total_s, 1),
        "decode_steps": machine.steps,
        "step_ms_mean": round(sum(machine.step_ms) / max(1, len(machine.step_ms)), 1),
        "boards_with_a_move": sum(r["moved"] is not None for r in rows),
        "model_commands": Counter(f"{c['op']} {c['pipe'] or ''}".strip() for c in model_cmds),
        "valid_moves": len(moves),
        "move_payloads": Counter(c["payload"] for c in moves),
        "other_writes": [c["line"] for c in writes if c not in moves],
        "model_reads": sum(c["op"] == "READ" for c in model_cmds),
        "moves_after_a_read": move_after_read,
        "exits": sum(c["op"] == "EXIT" for c in model_cmds),
        "respawns": respawns,
        "says": [c["line"] for c in model_cmds if c["line"].startswith("say")],
        "tokens_per_command": Counter(c["tokens"] for c in model_cmds),
        "splices": machine.splices,
        "context_tokens_at_end": max((len(c.ids) for c in machine._contexts.values()), default=0),  # pyright: ignore[reportPrivateUsage]
        "mask_calls": len(machine.mask_calls),
        "mask_misses": len(misses),
        "mask_miss_ms": [round(m["ms"]) for m in misses],
        "mask_distinct_states": len({m["state"] for m in machine.mask_calls}),
        "per_board": rows,
    }


def prompt_arm(name: BoardName, boards: int, max_new: int, threads: int) -> dict[str, Any]:
    spec = board(name)
    started = time.monotonic()
    worker = NodeWorker(MODEL, runtime="node", threads=threads)
    load_s = time.monotonic() - started
    info_ = worker.info()
    by_piece = {worker.piece(i): i for i in info_.controlIds}
    im_start, im_end = by_piece["<|im_start|>"], by_piece["<|im_end|>"]
    player = PromptPlayer(view=LeadView(), rules=spec.rules, history=spec.history)
    prefix = [im_start, *worker.tokenize("system\n" + player.default_prompt()), im_end]
    prefix += worker.tokenize("\n")
    worker.createContext("prompt")
    worker.append("prompt", prefix)
    began = time.monotonic()
    worker.decodeStep("prompt", {"allowedBlocks": None, "allowedTokens": None})
    prefix_s = time.monotonic() - began
    game = Game(seed=spec.seed, rules=spec.rules)
    controls = Controls(game, per_tick=spec.actions_per_tick)
    rows: list[dict[str, Any]] = []
    for b in range(boards):
        if game.over:
            break
        obs, info = game.render(), snapshot(game)
        info["steps"] = info["ticks"]
        turn = player.render_turn(obs, info)
        ids = [im_start, *worker.tokenize("user\n" + turn), im_end, *worker.tokenize("\n")]
        ids += [im_start, *worker.tokenize("assistant\n<think>\n\n</think>\n\n")]
        board_began = time.monotonic()
        worker.truncate("prompt", len(prefix))
        worker.append("prompt", ids)
        out: list[int] = []
        for _ in range(max_new):
            step = worker.decodeStep("prompt", {"allowedBlocks": None, "allowedTokens": None})
            if step.tokenId in (im_end, info_.eosId):
                break
            out.append(step.tokenId)
            worker.append("prompt", [step.tokenId])
        text = "".join(worker.piece(t) for t in out)
        action = PromptPlayer.parse(text)
        played = action or FALLBACK_ACTION
        player.turns.append((info["steps"], played, player.view.history(obs, info)))
        rows.append(
            {
                "board": b,
                "tick": info["ticks"],
                "prompt_tokens": len(prefix) + len(ids),
                "user_tokens": len(ids),
                "reply": text,
                "reply_tokens": len(out),
                "action": action,
                "seconds": round(time.monotonic() - board_began, 2),
            }
        )
        print(f"board {b} tick {info['ticks']}: {text!r} -> {action}", flush=True)
        controls.write(played)
        controls.tick()
    worker.close()
    parsed = [r for r in rows if r["action"] is not None]
    return {
        "arm": "prompt",
        "board": name,
        "seed": spec.seed,
        "boards": len(rows),
        "load_s": round(load_s, 1),
        "prefix_tokens": len(prefix),
        "prefix_s": round(prefix_s, 1),
        "parse_rate": len(parsed) / max(1, len(rows)),
        "actions": Counter(r["action"] for r in rows),
        "exact_one_word": sum(r["reply"].strip() in ACTIONS for r in rows),
        "per_board": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--arm", choices=("zeos", "prompt"), required=True)
    parser.add_argument("--board", choices=("default", "ablation"), default="default")
    parser.add_argument("--mode", choices=("auto-read", "plain"), default="auto-read")
    parser.add_argument(
        "--grammar",
        choices=("syscall", "moves"),
        default="syscall",
        help="moves: write stdout's payload limited to left/right/shoot (MoveLanguage)",
    )
    parser.add_argument("--boards", type=int, default=15)
    parser.add_argument("--cap", type=int, default=40, help="decode steps per board")
    parser.add_argument("--max-new", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.arm == "zeos":
        result = zeos_arm(args.board, args.mode, args.boards, args.cap, args.threads, args.grammar)
        suffix = "" if args.grammar == "syscall" else f"_{args.grammar}"
        name = f"adherence_zeos_{args.board}_{args.mode}{suffix}.json"
    else:
        result = prompt_arm(args.board, args.boards, args.max_new, args.threads)
        name = f"adherence_prompt_{args.board}.json"
    path = write_json(name, result)
    print({k: v for k, v in result.items() if k != "per_board"})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
