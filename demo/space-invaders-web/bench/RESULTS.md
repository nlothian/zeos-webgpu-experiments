# Benchmarks for the Space Invaders browser port

Measured on 6 October 2026, on the user's M1 Max (32 GB, 10 cores), against
`demo/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT` and the existing synchronous APIs
(`JsMachine`, `OptZeosWorker`, `NodeWorker`). Nothing outside `bench/` was changed.

**Caveat on CPU timings.** Up to four other agents ran Node model workers at the same
time (load average 12–33). The CPython/Pyodide mask timings and every wall-clock
number from onnxruntime-node are therefore pessimistic and noisy. The adherence counts
are step-clocked and are not affected. The WebGPU runs had the GPU to themselves.

Raw outputs are in `results/`. Each section names the script that produced it. Run
everything from the repository root with `UV_NO_CONFIG=1`.

## Headline findings (read these first)

1. **The 4B does not play under the syscall grammar as specified.** With `JsMachine`'s
   grammar and an automatic `read stdin` after each write, it wrote
   `write stdout left left left 1` on all 15 boards. That is 0 valid moves out of 15,
   9 tokens per command, and the same reply every time whatever the board said. The
   payload is free text up to `max_text` = 16 characters. After `left` the model keeps
   going until the 16-character bound forces the `;`. The game ignores the
   unrecognised action, so the pilot never moves.
   - With a **payload sub-grammar** (`left|right|shoot`, `MoveLanguage` in
     `adherence.py`), it made 15/15 valid moves at 5 tokens each: 10 `left`, 5 `right`,
     0 `shoot`. It changes its answer with the board, but weakly.
   - **A payload sub-grammar is required, not optional.**
2. **There is a trap in the seat maps.** `seat_maps()` (what `node_run` uses for any
   case) marks the pilot's `stdout` as *valued*, because `game.controls` is
   world-backed. `CommandLanguage` then makes its payload a **number**, and
   `write stdout left;` is unrepresentable (`mask_cost.json`:
   `prewarm_representable_with_seat_maps_valued`). `PilotJsMachine` must not pass the
   seat maps' `valued` for the pilot. The contract's factory takes no `valued`, which is
   correct; keep it that way.
3. **Board prefill at small chunks is much slower than the plan assumed.** At about
   3k context, one prefill run costs about 350 ms plus 2.9 ms per position. Most of
   that fixed cost is there for any run of 8 or more positions; a 1-position decode
   costs only 55–67 ms. So:
   - Reading a ~290-token board takes 4.5 s at chunk 32, 2.7 s at 64, 2.0 s at 128 and
     1.6 s at 256.
   - The per-chunk overhead is about 65% of a 64-position run. The plan's rule
     ("raise the chunk if overhead > 30%") is never met below about 800.
   - The plan's 290–300 tok/s holds only for 256-position runs near the start of a
     context. A board read at 3k context runs at about 180 tok/s even at chunk 256.
4. **A pager splice replays far more than 255 positions.** `JsMachine.splice` truncates
   the worker at the splice point and re-appends the tail, so everything after the
   oldest stubbed board is re-run.
   - In both 15-board runs the pager stubbed 6 boards in one step once the context
     passed its 4096-word window, at about board 10. That replays about 1.5k positions:
     8.5 s at chunk 256 or 2048, and 13.5 s at chunk 64, on WebGPU.
   - The pilot's context reaches about 5.4k BPE tokens (4096 kernel words), not the
     3–4k the plan assumed.
5. **Cold grammar states cost seconds under Pyodide.**
   - The start state takes 0.77 s. A free-text payload state takes 1.6–2.1 s, and there
     are 17 of them, one per payload length: 22 s to warm them all.
   - The 4 prewarm commands visit 7 states, which take 6.7 s cold under Pyodide
     (3.3 s in CPython).
   - The per-step copy into a `Uint8Array` is 0.22 ms, which is negligible.

## Summary table

| Quantity | Default 12×16 @ 0.5 s | Ablation 9×8 @ 0.2 s | Source |
|---|---|---|---|
| Pilot prefix (kernel injection + ChatML), BPE | 2,251 | 2,095 | `prompt_sizes.json` |
| One board as `JsMachine` encodes it, BPE (min/mean/max) | 265 / 305 / 336 | 213 / 250 / 291 | ″ |
| + arrival/turn framing per board | 10 | 10 | ″ |
| Pilot context at the end of 15 boards | 5,384 | — | `adherence_zeos_default_auto-read.json` |
| Prompt arm system turn, BPE | 1,957 | 1,796 | `prompt_sizes.json` |
| Prompt arm user turn with 5 history turns (min/mean/max) | 454 / 582 / 648 | 407 / 475 / 536 | ″ |
| Prefix prefill from empty, WebGPU (2.8k / 2.6k) | 11.6 s, 243 tok/s | 10.8 s, 238 tok/s | `prefill_webgpu.json` |
| Board read after ~2.8k prefix, chunk 32 | 4.55 s (10 runs) | 3.94 s (9.3 runs) | ″ |
| ″ chunk 64 | 2.66 s (6 runs) | 2.37 s (5 runs) | ″ |
| ″ chunk 128 | 1.99 s (4 runs) | 1.67 s (3 runs) | ″ |
| ″ chunk 256 | 1.63 s (3 runs) | 1.33 s (2 runs) | ″ |
| ″ chunk 2048 (cut at 256 boundaries anyway) | 1.62 s | 1.36 s | ″ |
| Decode step at 3.7–4.0k context, 248k token mask | 54.8 ms | 53.9 ms | ″ |
| Splice of the oldest board: replay 1,672 / 1,411 positions, chunk 2048 | 8.5 s | 7.1 s | ″ |
| ″ chunk 256 | 9.1 s | 7.6 s | ″ |
| ″ chunk 64 | 13.5 s | 12.3 s | ″ |
| Mask, cold start state: CPython / Pyodide | 0.27 s / 0.77 s | same | `mask_cost*.json` |
| Mask, cold free-text payload state: CPython / Pyodide | 0.5–0.87 s / 1.6–2.1 s | same | ″ |
| Mask, prewarm 4 commands (7 states): CPython / Pyodide | 3.3 s / 6.7 s | same | ″ |
| Mask, warm hit | 0.1–0.3 µs | same | ″ |
| Mask copy per step: `array('B')` / Pyodide `Uint8Array.new(to_js())` | 0.005 ms / 0.22 ms | same | ″ |
| Zeos adherence, syscall grammar, auto-read: valid moves | **0/15** (`left left left 1` ×15) | not run | `adherence_zeos_default_auto-read.json` |
| Zeos adherence, move sub-grammar, auto-read: valid moves | 15/15 (10 left, 5 right) | not run | `adherence_zeos_default_auto-read_moves.json` |
| Prompt arm, ≤6 tokens: parse rate | 14/15 (all `left`, fenced in a code block, 5 tokens) | not run | `adherence_prompt_default.json` |

### One run's cost by context behind it (WebGPU, `prefill_scaling_webgpu.json`, ms, median)

| Positions in the run → / past ↓ | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 0 | 67 | 239 | 255 | 329 | 508 | 939 |
| 512 | 67 | 238 | 279 | 355 | 532 | 1054 |
| 1024 | 48 | 243 | 310 | 387 | 569 | 1136 |
| 2048 | 55 | 257 | 376 | 453 | 636 | 1102 |
| 3072 | 59 | 270 | 440 | 516 | 690 | 1111 |

Every run here reads the logits back, as the last run of a step does.

## 1. Grammar mask cost: `mask_cost.py`, `mask_pyodide.mjs`

```
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/mask_cost.py
node demo/space-invaders-web/bench/mask_pyodide.mjs        # after mask_cost.py
```

**Setup.**
- The vocabulary is the model's own: 248,077 pieces, from `tokenizer_server.mjs`, which
  decodes exactly as `OptZeosWorker.piece` does.
- The language is built by `JsMachine._language("pilot")` from the case's seat maps
  (aliases `stdin`, `stdout`), without `valued`; see headline 2.
- The pilot's language has 6 alternatives (`say`, `write stdin|stdout`, `read stdin`,
  `read stdout`, `exit`). `mask_core.measure` is shared by both runtimes.
- Pyodide is 314.0.7 (Python 3.14.2) under Node, the interpreter the page's mask runs in.

**Results.**

- **Cold vs warm round state.**
  - The start state takes 267 ms cold in CPython and 770 ms in Pyodide; 28 ids are
    allowed there.
  - A hit takes 0.1–0.3 µs, a dictionary lookup.
  - Constructing `JsMachine` over the 248k pieces takes 15–18 ms. Loading the pieces
    over the tokenizer process takes about 1.1 s.
- **States per move.**
  - `write stdout left;` tokenizes as `write`, ` stdout`, ` left`, `;`. It visits 4
    states: start, after `write`, after ` stdout`, and the payload at 4 characters.
  - ` right` and ` shoot` share one state, the payload at 5 characters, so `right` adds
    1 new state and `shoot` none.
  - `read stdin;` adds 2 states.
  - The leading-space variants (` write …`) reach the same states.
  - In total the four `PREWARM_COMMANDS` visit **7 distinct states**: 3.3 s cold in
    CPython and 6.7 s in Pyodide.
- **The payload is the expensive part.**
  - Inside `write stdout <payload>` almost every piece (up to 246k ids) is accepted.
    Each piece must be walked character by character, and every payload length 0–16 is
    a separate state.
  - Cold, each costs 0.2–0.87 s in CPython and 0.3–2.1 s in Pyodide. The 15 not visited
    by the prewarm total 10.9 s in CPython and 22 s in Pyodide.
  - Any move the model spells in other pieces than the prewarm walk lands on a cold
    state. The syscall-grammar run below visited 9 states (its 16-character payloads),
    not 7.
- **Per-step `allowed` and copy.**
  - `bytes(bytearray(248k))` takes 0.005 ms.
  - `array('B', mask)` (NodeWorker) takes 0.005–0.009 ms.
  - Pyodide `Uint8Array.new(to_js(mask))` (`PyodideBridge._flags`) takes **0.22 ms**.
  - No need to keep the mask in the SAB.
- **The move sub-grammar** (`MoveLanguage`) used 6 states in the adherence run. Cold, in
  CPython under load, they cost 0.19–0.94 s each.

## 2. Prompt sizes: `prompt_sizes.py`

```
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/prompt_sizes.py
```

**How each size was measured.**
- **Pilot prefix.** `ZeosDriver(machine=JsMachine(TokenizerWorker), view=LeadView(),
  rules=board)` boots the case. That is `build_kernel`: pilot.md plus
  `rules_prompt(LeadView, rules)`. It is pumped until the pilot's first decode step, and
  the ids the worker holds then are counted. These are the ids the page prefills:
  1,768 kernel words become 2,251 BPE tokens (default) and 1,686 words become 2,095
  (ablation). This is not the ~3k the plan assumed.
- **Board.** `LeadView.state` is rendered as the native zeos clock does. It is
  `encode`-d into pipe words with `<nl>` and encoded by `JsMachine._encode`, which puts
  a leading space before each word. Samples cover 40 ticks of the seeded game (seed 7)
  with a seeded random stick.
  - Encoding the board word by word gives about the same count as one plain
    `tokenize`: 305 vs 306 (default), 250 vs 242 (ablation).
  - The arrival and the reopened assistant turn add 10 framing tokens.
- **Prompt arm.**
  - System turn: `PromptPlayer.default_prompt()` (rules_prompt + reply.md), 1,957 /
    1,796 tokens.
  - User turn: `render_turn` with a full 5-turn history, 582 / 475 tokens on average.
  - ChatML framing: 21 tokens.
  - Total per decision: about 2,560 / 2,290 tokens, of which about 600 / 500 are new
    each decision.

## 3. WebGPU prefill: `prefill.html`, `prefill.js`, `prefill_webgpu.mjs`

```
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/prompt_sizes.py   # writes .cache/webgpu_inputs.json
PLAYWRIGHT_MODULE=<any node_modules/playwright> node demo/space-invaders-web/bench/prefill_webgpu.mjs \
    --query "scaling="                                                       # -> results/prefill_webgpu.json
PLAYWRIGHT_MODULE=… node demo/space-invaders-web/bench/prefill_webgpu.mjs \
    --query "boards=" --out demo/space-invaders-web/bench/results/prefill_scaling_webgpu.json
```

**Setup.**
- Headed Chrome via Playwright, as `tests/opt_zeos_webgpu.mjs` runs it. `demo/` is
  served by coop-count-web's `serve.py`.
- `OptZeosWorker` on WebGPU with the page's own onnxruntime-web 1.30.0 (local
  `node_modules`, no CDN). The adapter is apple metal-3, and the model loads in 9.3 s.
- The inputs are real ids: the pilot prefix from §2, then framed boards (arrival
  framing + board + turn framing + `write stdout left;`).
- The chunk size is `worker.chunk`, the field `fill()` cuts runs at. It is what
  `maxChunk` will set. `fill()` also cuts at every multiple of 256, the snapshot
  boundary.
- Each trial forks a base context holding prefix + 2 boards (2,809 / 2,582 positions).
  It appends the next board and times one `decodeStep` with a full 248k
  `allowedTokens`.
- There is one untimed warm-up pass per chunk size, and 3 trials per chunk size, of
  which the median is reported.

**Board read by chunk** (table above):
- 32 → 4.5 s, 64 → 2.7 s, 128 → 2.0 s, 256 → 1.6 s (default board).
- Runs per board: 10 / 6 / 4 / 3, including the 256-boundary cut.
- Each run costs 430–500 ms at 32 or 64 positions, about 680 ms at 128, and about
  680 ms for 256, with 780–930 ms for the last run, which reads back the logits and the
  attention.
- From the scaling table, a run at 3k context costs about **350 ms + 2.9 ms/position**.
  - At chunk 64 the overhead is about 65% of a run. At 256 it is about 33%.
  - So the cancel granularity at chunk 64 is about 0.5 s, and the read costs 1.65× what
    it costs at 256.
- **Decode step**, one position, at 3.7–4.0k context: 54–55 ms (max 61). That agrees
  with the plan's ~50 ms.
- **Splice and replay.** On a context of prefix + 6 boards (3,720 positions), the
  oldest board is replaced by an 8-token stub the way `JsMachine.splice` does it:
  `truncate` to the board's start, then re-append the stub and the tail.
  - The next step runs 1,672 positions: the snapshot rewind (2,251 → 2,048) plus the
    whole tail.
  - That takes 8.5 s at chunk 2048, 9.1 s at 256 and 13.5 s at 64.
  - The ablation board: 1,411 positions, 7.1 / 7.6 / 12.3 s.
  - The "≤255-position replay" holds for a *mask* change, served from a snapshot. It
    does not hold for a splice, which changes the token sequence.

## 4. Syscall adherence on onnxruntime-node CPU: `adherence.py`

`adherence.py` ran the model on onnxruntime-node's CPU provider, which the repository no
longer has: the model runs on WebGPU only. The script is in the history at `a397805`; its
results stay in `results/` and are reported here as measured.

```
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/adherence.py --arm zeos --board default --mode auto-read
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/adherence.py --arm zeos --board default --mode auto-read --grammar moves
UV_NO_CONFIG=1 uv run python demo/space-invaders-web/bench/adherence.py --arm prompt --board default
```

**Setup.**
- `NodeWorker(runtime="node", threads=8)` with the 4B ZEOS-OPT and the real case:
  `ZeosDriver` over a `JsMachine` subclass that only logs, plus the auto-read.
- It is step-clocked: each board is delivered to `game.state` as `ZeosDriver.sense`
  delivers one. Only boards are sensed, never threats, so `evade` never runs.
- The kernel is pumped until a write to `game.controls` lands (at most 40 decode steps
  per board), and then the game ticks once.
- Seed 7, 15 consecutive boards. Greedy decoding.
- `auto-read` answers the decode after each completed write with `read stdin`, without
  the model, as the native machine does and `PilotJsMachine` is specified to.

**Zeos arm, the syscall grammar as `JsMachine` builds it (no `valued`).**
- **0/15 valid move payloads.** All 15 commands were `write stdout left left left 1`:
  the payload filled to the 16-character bound, 9 tokens per command (8 on one board).
- There were no `say`, no `exit` and no other verbs, and the model never emitted
  `read stdin` itself. Every read was the machine's automatic one.
- Board 0's move was made before any board had been read, as natively. Every later
  move followed the automatic read (14/15).
- There were 9 mask cache misses (0.2–1.0 s each, in CPython under load) and 9 distinct
  states.

**Zeos arm with the move sub-grammar (`--grammar moves`).**
- **15/15 valid moves**, 5 tokens per command (4 on one board): 10 `left`, 5 `right`,
  never `shoot`.
- There were no `exit`, no `say` and no model-issued `read`.
- There were 6 distinct mask states and 6 misses.
- Replies vary with the board, but the model leans heavily to `left`. Whether it plays
  *well* is the integrator's run to judge.

**Splices.**
- Both runs splice once, at about board 10, removing 6 boards in one step. The
  replacements are 32–36 words each.
- That means re-running from about 2.6–2.9k to the 5.4k end. On WebGPU, §3 puts that
  at about 8–13 s.

**Prompt arm** (`--arm prompt`, ≤6 unconstrained tokens, history 5), from
`adherence_prompt_default.json`:

- **Parse rate 14/15 (0.93).** Every parsed reply was `left`.
- Only the first reply was the bare word `left`. From the second board on, the model
  answered with a code fence around `left`, which is 5 of the 6 allowed tokens.
- The unparsed reply was a fence opening onto `incoming fire in your`, cut at 6 tokens.
- So the 6-token cap is only just enough for this habit. A 4-token cap would fail
  14/15.
- The system prefix was 1,962 tokens. The user turn grew from 281 to 616 tokens as the
  history filled.
- Like the zeos arm, the prompt arm collapses to one action whatever the board says.
  This is the 4B's behaviour with greedy decoding on these prompts, not something the
  grammar did.

**Not run, for time:**
- `--mode plain` (does the model read stdin unprompted?).
- Both arms on the ablation board.

The ablation zeos run was started and stopped by hand. Under the CPU contention each
15-board zeos run took about 27 minutes of wall time. The commands above run them.

## 5. Against the plan's predictions, and recommendations

**Predicted per-move cost on WebGPU from these numbers.**
- A pilot move is one board read, plus 5 decode steps with the move sub-grammar (9 with
  the free-text payload), plus mask lookups that are free once warm.
- The automatic read costs nothing.

| Arm / board | Plan | From these numbers (chunk 256) | Chunk 64 |
|---|---|---|---|
| Zeos pilot, default @ 0.5 s | ~1.2 s ≈ 2.4 ticks late | 1.63 + 0.27 ≈ **1.9 s ≈ 3.8 ticks** | 2.66 + 0.27 ≈ 2.9 s ≈ 5.9 ticks |
| Zeos pilot, ablation @ 0.2 s | ≈ 6 ticks | 1.33 + 0.27 ≈ **1.6 s ≈ 8 ticks** | 2.37 + 0.27 ≈ 2.6 s ≈ 13 ticks |
| Zeos pilot, the move after the pager's splice | — | **+8–9 s ≈ 17 / 43 ticks**, once per ~10 boards | +13 s ≈ 27 / 65 ticks |
| Prompt arm, default | ~5 s ≈ 10 ticks | **~3.9 s ≈ 8 ticks** (see below) | — |
| Prompt arm, ablation | ≈ 26 ticks | **~2.5 s ≈ 12–13 ticks** | — |

**Prompt arm cost.**
- A decision truncates the context to the system prefix. Its end (about 1.96k / 1.80k)
  lies past the 1,792 snapshot, so it replays about 170 / 8 positions.
- It then appends about 600 / 500 new tokens and runs them at chunk 256: about 3 runs of
  about 1.1 s plus a short one.
- The reply is at most 6 decode steps at 55 ms.

**What this means for the plan.**
- The zeos pilot is still about 2× faster per move than the prompt arm in the default
  board's ticks.
- But both are later than predicted for the zeos arm and earlier than predicted for the
  prompt arm.
- The plan's 1.2 s/move assumed 290 tok/s board reads and did not include the per-run
  overhead.

**Recommendations.**

1. **`maxChunk` for board prefill: 256, not 64.**
   - 64 makes every board read 1.65× slower (2.7 s vs 1.6 s), because each run carries
     about 350 ms of fixed cost at 3k context. It buys a cancel latency of about 0.5 s
     instead of about 0.7–0.9 s.
   - A threat is handled by the native `evade` reflex anyway, which never waits on the
     worker except to cancel. So the extra 0.2–0.4 s of cancel latency costs less than
     the 1 s added to every move.
   - If finer cancellation is wanted, 128 is the compromise (2.0 s per read, about 0.7 s
     per run).
   - Warm-up and splice replay: keep 2048, which is cut at 256 boundaries anyway.
2. **Payload sub-grammar: needed.**
   - Without it the 4B never produces a valid move (0/15). With it, 15/15.
   - Implement it in the machine's language: `MoveLanguage` in `adherence.py` is a
     working prototype of about 25 lines over `CommandLanguage`.
   - It also cuts the command from 9 to 5 tokens, about 0.2 s per move.
   - It also makes the cold payload states cheap: they accept only a handful of ids. A
     full vocabulary pass is still needed per cold state, about 0.2–0.4 s in CPython and
     about 0.5–0.8 s in Pyodide.
3. **`forbid_verbs=("exit",)`: not needed on this evidence.** There were 0 `exit` in 30
   pilot commands (and 0 `say`). Keep the switch off by default. Turn it on only if the
   integrator's wall-clock runs show exits after preemption, which this step-clocked
   bench could not provoke.
4. **Prewarm the moves grammar's 6 states before the clock**, not just
   `PREWARM_COMMANDS`' walk. That is 7 states for the free-text grammar, or about 6.7 s
   in Pyodide.
   - With the free-text grammar a single cold payload state stalls the loop for up to
     2 s.
5. **`stall_ms`: 5.** Decode steps are about 55 ms and board reads 1.3–2.7 s, so polling
   at 1 ms gives about 50 stalls per decode step for nothing. 5 ms keeps the journal
   about 5× smaller and adds at most 5 ms of latency to a 55 ms step. Use 10 if the
   journal is still large.
6. **Prompt arm: history 5 is fine; fork rather than truncate.**
   - With 5 turns the user turn is about 580 / 475 tokens, 2–3 runs at chunk 256. That
     is about 2.3–3.6 s per decision.
   - History 3 would save about 1 run (about 1 s) on the default board, if latency
     matters more than context for that arm.
   - **Fork from a prefix context per decision instead of truncating.** Truncating to
     1,962 rewinds to the 1,792 snapshot and replays 170 positions (about 0.5 s) every
     decision. A fork of a context that ends exactly at the prefix carries its state
     with no replay.
7. **The splice.**
   - This is the biggest unbudgeted stall: about 8.5 s (17 ticks default, 43 ablation)
     once the pilot's context fills, at about board 10. It is cancellable per chunk, but
     the work comes back on the next step.
   - The integrator should either size `context.window` so the pager splices little and
     often, or budget the one long move.
   - A worker-level splice that keeps the KV before the splice point and replays only
     from the latest snapshot would be a `src/`-free change in `opt_zeos_worker.js`, but
     it still has to re-run the tail, because DeltaNet state is not positional.
8. **Watch the pilot's context length.**
   - It reaches about 5.4k tokens before the pager acts, not 3–4k.
   - At 5k the scaling table predicts about 60 ms more per prefill run than at 3k.
   - The decode step stays at about 55 ms.

## Files

| File | What it is |
|---|---|
| `_common.py` | `TokenizerWorker` (the model's tokenizer as a `ZeosModelWorker`), seat maps, paths |
| `tokenizer_server.mjs` | the tokenizer without the model, for `TokenizerWorker` |
| `mask_core.py` | the mask timings, runtime-independent |
| `mask_cost.py` / `mask_pyodide.mjs` | §1 under CPython / Pyodide |
| `prompt_sizes.py` | §2, and the WebGPU inputs (`.cache/webgpu_inputs.json`) |
| `prefill.html`, `prefill.js`, `prefill_webgpu.mjs` | §3 |
| `adherence.py` (removed; at `a397805`) | §4, with `BenchMachine` and `MoveLanguage` |
| `results/*.json` | the raw outputs quoted here |

`.cache/` holds the large regenerable inputs (the 248k pieces, the ids) and is ignored.
