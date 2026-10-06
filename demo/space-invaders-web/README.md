# Space Invaders in the browser

**Status: playable on WebGPU.** Both players run on the real model in the browser. The
ZEOS arm plays a full 600-tick default game, with the four criteria passing. One
setting differs from the native demo: the pilot's context window is enlarged so the
pager does not act (see *Settings that differ from the native demo*).

The port puts [`../space-invaders`](../space-invaders/) in a static web page: the ZEOS
kernel and the game run under Pyodide, and the pilot is served by
`Qwen3.5-4B-ZEOS-OPT` over WebGPU through the worker channel of
[`../coop-count-web`](../coop-count-web/). The game keeps real time, so a slow model falls
behind and the reflex preempts it, as in the native demo. A prompt-loop arm on the same
model is the comparison.

## Getting started

The page borrows coop-count-web's npm install (ONNX Runtime Web, the tokenizer, and the
Pyodide the Node tests use) and its model export. From the repository root:

```bash
(cd demo/coop-count-web && npm install)              # once; also caches PyYAML for Node
uv run python demo/space-invaders-web/build.py       # wheels, page, case, vendor -> web/dist
uv run python demo/space-invaders-web/serve.py       # http://localhost:8766
```

Open <http://localhost:8766> in a browser with WebGPU (recent Chrome or Edge), choose
*Qwen3.5-4B-ZEOS-OPT on WebGPU* as the machine, and press **run**. The first run loads
the model (about 5 s from a local disk), and the model stays loaded for later runs. Each
run then warms up before the clock starts, taking about 15 s for the ZEOS arm and 7 s
for the prompt loop. The
server sends the two headers that make the page cross-origin isolated; without them
there is no SharedArrayBuffer, and the page says so. It uses port 8766 so that it can run
beside coop-count-web's page on 8765. `serve.py --port` and `--dir` change both.

`build.py` links the model export (by default
`../coop-count-web/models/Qwen3.5-4B-ZEOS-OPT`) into `web/dist/models/` and names it in
`manifest.json`; `--model DIR` offers another export. The link works with `serve.py`;
**to put `web/dist` on a static host, build with `--copy-model`**, which copies the
export (gigabytes) instead, because a static host does not follow symbolic links.
Without the npm install or the export the build still works, and the page offers only
the stub. `--link-node-modules` links `node_modules`
here to coop-count-web's.

## What you'll see

- **Controls:** the player (the ZEOS kernel, with its `pilot` and `evade` reflex, or the
  prompt loop), the board (default 12×16 at 0.5 s a tick, or the ablation board, 9×8 at
  0.2 s), the seed, and the machine. *Language model on WebGPU* needs a WebGPU adapter;
  there is no WebAssembly fallback, because the 4B does not fit it. *Stub* needs no GPU:
  when the build has `web/stub/pilot_stub_worker.js` it runs on a thread of its own
  (`stub_thread.js`) and answers through the same channel as the model, with simulated
  latency; without it the stub machine runs on no worker, which only a `FakeRun` accepts.
  The header shows whether the page is cross-origin isolated, whether WebGPU offered an
  adapter, and whether the build has the model.
- **Phases:** *loading model* (with a download bar, model machine only), *warming* (the
  system prompt is prefilled before the game clock starts), *playing*, *finished*.
  **Stop** is read between 20 ms slices of the loop, so it ends a run within about 20 ms
  plus the step under way, which can be the lead view's aim search (see *Known issues*). If the
  model thread stops answering, Stop gives up after 30 s and the page shows the error;
  later runs on that model refuse until the page is reloaded.
- **The board:** drawn on a canvas, sharp at any zoom. Under it, the last move: who made
  it (`pilot`, `evade` or `prompt`), how many ticks late it landed, and `PREEMPTED` when
  the reflex took the machine from the pilot. A late move outlines the column the ship
  was in when the board was read. Live counts: lives, kills, ticks, preemptions,
  cancellations, reflexes, lag (mean and 95th percentile, in ticks), and catch-up ticks.
- **The result:**
  - the case's four criteria, with a verdict each;
  - the run's measurements: warm-up and mask prewarm, loop overrun, catch-up ticks,
    pilot moves and how many were valid, request-to-move time, cancel latency, the
    longest gap between pilot moves, pager splices and kernel faults (or, for the prompt
    loop, its replies and any it could not parse);
  - a table of every run this session (player × board × seed: lives, kills, lag,
    preemptions, cancellations, model moves, warm-up, parse rate);
  - downloads of the journal and the result;
  - the ZEOS debugger, which steps through the run's journal.

### The real runs and the FakeRun

`page.open_run` looks the player up in `page.RUN_BUILDERS`. The ZEOS arm (`ZeosRun`) is
`PilotJsMachine` over the worker channel, the native `ZeosDriver` and
`WallClockZeosRunner`; the prompt loop (`PromptRun`) is `BrowserPromptPlayer` and
`WallClockPromptRunner`. Before the clock starts, the ZEOS arm fills the grammar's mask
cache and prefills the pilot's system prompt until it waits for its first board; the
prompt loop prefills its system prompt.

A run with no worker -- the stub machine in a build without `web/stub/` -- is a
`FakeRun`: a real `Game` on the real board against the wall clock, with random moves and
simulated model latency. It runs no kernel, so its verdicts read *not judged* and the
debugger shows the case's wiring only.

`?tune=` on the page's URL overrides the arms' options for that page load
(`page.configure_json`). It takes JSON with four keys, and the header shows when one is
set:
- `zeos`: `PilotJsMachine` keywords;
- `prompt`: `BrowserPromptPlayer` keywords;
- `kernel`: the zeos arm's `KernelConfig` fields;
- `context`: the pilot's `ContextPolicy` fields.

For example, `?tune={"zeos":{"max_chunk":256},"context":{"window":4096}}` brings back the
earlier chunk and the native window.

## Settings that differ from the native demo

These are set on the kernel the web demo builds. The native case files and
`build_kernel` are unchanged, and there is no copy of the case. The kernel itself is the
native one, at its own starvation limit (8): it counts a job's preemptions since the job
last made progress ([`docs/notes/starvation-progress.md`](../../docs/notes/starvation-progress.md)),
and the pilot makes progress with every move and every read of a board, so the reflex's
preemptions do not retire it. Under the earlier rule, which never reset the count, the
pilot was faulted 6–12 s into an ablation game (the *before* ablation row below). At the
native limit with the current rule, a full ablation game on a quiet GPU
(`bench/results/webgpu/native_limit_ablation.json`) ran to game over at tick 83 with 10
preemptions, no fault, and 7 pilot moves, the longest gap between them 4 s.

- **Pilot context window 32,768, not 4096, `page.DEFAULT_PILOT_CONTEXT`.**
  - When the pager splices a span out, the model must compute every position after the
    splice point again: the DeltaNet layers' state cannot be cut. The earliest span the
    pager stubs lies just after the pinned body, so a splice replays nearly the whole
    context.
  - On WebGPU a 3k-position replay took 16–19 s. With the native window the pilot went
    43–53 s without a move in a 120 s default game.
  - Lowering the watermarks, shrinking the window or using smaller blocks did not
    shorten the replay (CPython sweep). The tail is mostly padding and stubs, which the
    pager keeps.
  - The native window exists because each API request resends the whole transcript. In
    the browser the KV cache stays resident instead.
  - The price is context growth, about 22k model ids by the end of a 600-tick default
    game, so prefill runs and cancels slow down as a game goes on.
- **Prefill chunk 128, not 256, `contracts.DEFAULT_MAX_CHUNK`.** Chosen end to end;
  see *Measured*.

## Measured

Qwen3.5-4B-ZEOS-OPT on WebGPU, Chrome, M1 Max, seed 7, `tests/si_webgpu.mjs`.

**Quiet** means `si_webgpu.mjs` found no other GPU user (another browser's GPU process,
Playwright run or dev server) before or after the run; WebGPU is shared, and another
session's real-model tests ran on the same GPU during part of the tuning. The *before*
rows predate the contention check. They ran while another session's tests may have been
running, so whether the GPU was quiet is unknown.

Lag is in ticks, between the board a move answered and the tick it landed on. It counts
only moves that were made, so a pilot stuck in a replay shows as a long gap rather than
as lag.

**Before → after**:
- *before*: Wave 2, with chunk 256, window 4096 and the native starvation limit;
- *after*: the current defaults.

| run | GPU | lives / kills / ticks | lag mean / p95 / max | model moves | preemptions | longest gap between pilot moves | notes |
|---|---|---|---|---|---|---|---|
| ZEOS, default, 60 s, before | unknown | 3 / 1 / 121 | 5.0 / 8 / 8 | 23 | 7 | 4 s | |
| ZEOS, default, 60 s, after | quiet | 3 / 3 / 121 | 6.2 / 9 / 9 | 19 | 12 | 4.5 s | |
| ZEOS, default, 120 s, before¹ | unknown | 3 / 4 / 240 | 8.7 / 12 / 87 | 25 | 16 | 43.5 s (2 splices) | one 2,961-position replay took 18.6 s |
| ZEOS, default, 120 s, after | quiet | 3 / 2 / 241 | 6.9 / 9 / 10 | 34 | 6 | 5 s | no splices; context 13.4k ids |
| ZEOS, default, full game, after | quiet | 2 / 12 / 600 | 9.3 / 17 / 23 | 64 | 43 | 11.5 s | no splices or faults; context 22k ids; cancel max 2.4 s |
| ZEOS, ablation, 60 s, before | unknown | 0 / 3 / 64 | 9.8 / 17 / 17 | 5 | 9 | 3.4 s, then none | pilot faulted for starvation |
| ZEOS, ablation, after (game over at 98 ticks) | quiet | 3 / 7 / 98 | 11.6 / 20 / 20 | 8 | 15 | 4 s | no faults |
| ZEOS, ablation, after, 2nd run (game over at 93 ticks) | quiet | 2 / 3 / 93 | 12.1 / 23 / 23 | 7 | 11 | 4.6 s | |
| prompt, default, 60 s, before² | unknown | 2 / 0 / 120 | 5.5 / 7 / 7 | 22 | – | – | parse 100% |
| prompt, default, 60 s, after | quiet | 2 / 0 / 120 | 5.1 / 6 / 6 | 22 | – | – | parse 100%; loop overrun max 205 ms (was 535 ms) |
| prompt, ablation, 60 s, before² | unknown | 0 / 0 / 66 | 9.3 / 11 / 11 | 7 | – | – | parse 86% |
| prompt, ablation, after | quiet | 0 / 1 / 73 | 8.9 / 12 / 12 | 7 | – | – | parse 86% |

Sources, in [`bench/results/webgpu/`](bench/results/webgpu/):
- *before*: `wave2.json`, `wave2_prompt.json` (prompt rows), `diag_nostarve.json` (¹);
- *after*: `final_60.json`, `final_120.json`, `final_300.json` (the full game).

¹ With the starvation limit lifted, as a diagnosis; the kernel's starvation rule of the
time (a count that never reset) would have faulted the pilot. The replay rate it implies (6.3 ms a position) is close to the benchmark's
quiet 5.4 ms at chunk 256 (about 16 s for 2,961 positions), so contention, if any, was
mild.

² After the prompt-arm fix: the reply opens with an empty think block. Before that fix
every reply began `<think>` and none parsed.

**Warm-up** (before the clock starts):
- ZEOS arm: 14–19 s, of which about 5 s (default) or 3.5 s (ablation) is mask prewarm.
- Prompt loop: about 7.5 s.

**Against the plan's predictions**:
- ZEOS arm: about 3.8 ticks late on default and 8 on ablation were predicted; measured
  6–7 and 11–12.
- Prompt loop: about 8 and 13 were predicted; measured about 5 and 9.

### How the defaults were chosen

These are quiet-GPU runs: seed 7, 120 s on the default board unless stated, ablation runs
lasting until game over.

| setting | default board | ablation board |
|---|---|---|
| chunk 256, window 4096 | 25 moves, lag 5.4 / 8 / 17, **53 s gap** (4 splices), overrun max 2.6 s | – |
| chunk 128, window 4096 | 25 moves, lag 6.2 / 8 / 8, **43.5 s gap** (4 splices), overrun max 1.1 s | – |
| chunk 256, window 32768 | 36 moves, lag 6.6 / 12 / 21, 10.5 s gap, cancel max 2.2 s | lag p95 28 and 34 (2 runs), 7 and 5 moves |
| chunk 128, window 32768 | 33 moves, lag 7.2 / 12 / 14, 7 s gap, cancel max 1.5 s | lag p95 16 and 19, 8 and 8 moves |

Sources, in [`bench/results/webgpu/`](bench/results/webgpu/): `q_c256w4.json`,
`q_c128w4.json`, `q_c256w32.json`, `q_c128w32.json` (default board) and
`qa_c256w32.json`, `qa_c256w32b.json`, `qa_c128w32.json`, `qa_c128w32b.json` (ablation).
Each file's `note` names its settings, and each row carries the contention check.

**Kept as they were:**
- `stall_ms` stays 5. The loop's overrun p95 was about 7 ms on the default board, so
  nothing pointed at it.
- The mask prewarm stays on. A cold grammar state costs 1.6–2.1 s under Pyodide, which
  would otherwise land mid-game.
- The prompt loop keeps a history of 5. With 2, lag fell slightly (4.2 vs 5.1 ticks on
  default) but the parse rate fell to 80–89% (`qp_h5.json`, `qp_h2.json`).

## Known issues

- **Lag grows over a long game.** With the larger window the context grows. By tick 600
  of the default board, cancel latency reaches 2.4 s and the loop overruns by up to 1.9 s.
  Fixing it needs a way to shorten the context that the model's recurrent state can
  follow; a pager splice cannot.
- **The `lead` view's aim search** (`zeos_space_invaders/game/aim.py`, native code) plays
  the game forward breadth-first. When no shot lands within 16 turns it visits about
  6,800 states, which takes 300–750 ms under Pyodide. That stalls the loop: the prompt
  loop's `begin` and the ZEOS arm's `sense` both render the board with it. The prompt
  loop now renders each turn once rather than twice.
- **A synchronous call behind a step in flight waits for one chunk.** The kernel's
  inject of a resume notice and the pager's splice must stop the step first: about
  0.3–0.6 s on average, up to 1–2 s at long context. This is most of the ablation
  board's overrun.

## Layout

- `src/zeos_space_invaders_web/contracts.py`: the Python seams between the pieces (the
  non-blocking worker, the machine, the prompt arm, the runners, frames and results), and
  `load_board`. [`CONTRACTS.md`](CONTRACTS.md) is the JavaScript half (the channel's slot
  layout, the begin/poll/cancel decode step and its race rules, the page and worker
  messages) and who owns which file.
- `src/zeos_space_invaders_web/page.py`: `open_run`, the two real runs and `FakeRun`,
  and the JSON the page reads (`finished_json`, `payload_json`, `describe_json`).
- `machine.py` (`PilotJsMachine`), `prompt_player.py` (`BrowserPromptPlayer`),
  `runner.py` (the wall-clock runners), `metrics.py`, `pyodide_glue.py` and
  `fake_worker.py` (`FakePilotWorker`); `web/stub/pilot_stub_worker.js` is its twin.
- `bench/`: the measurements the defaults came from ([`bench/RESULTS.md`](bench/RESULTS.md)).
- `src/zeos_space_invaders_web/boards/`: byte-for-byte copies of the native
  `settings_default.json` and `settings_ablation.json`.
- `web/`: the page (`index.html`, `app.js`, `board.js`, `style.css`) and `si_worker.js`,
  the Web Worker that hosts Pyodide and the run loop. The model thread and its channel
  are coop-count-web's files, copied in by `build.py`.

## Tests

```bash
uv run pytest demo/space-invaders-web
node --test demo/space-invaders-web/tests/js/*.test.mjs
uv run python demo/space-invaders-web/build.py && \
  PLAYWRIGHT_MODULE=/path/to/node_modules/playwright node demo/space-invaders-web/tests/si_webgpu.mjs
```

The pytest suite builds `web/dist` into a temporary directory and, when
coop-count-web's npm install is present, plays under Pyodide in Node
(`tests/pyodide_run.mjs`): a `FakeRun`, and both arms against the JavaScript pilot stub
over the real channel (`test_e2e_pyodide.py`). `test_e2e_stub.py` plays both arms under
CPython against `FakePilotWorker`; `test_e2e_node_model.py` plays the ZEOS arm on the
4B through onnxruntime-node when the export is present (several minutes).
`tests/si_webgpu.mjs` drives the built page in headed Chrome over both boards and both
players and prints each run's metrics; it needs Playwright and Chrome, which are not
dependencies of this demo.
