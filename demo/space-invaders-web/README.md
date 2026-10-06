# Space Invaders in the browser

**Status: playable, with two known problems** (see *Measured* below): the kernel faults
the pilot for starvation on its ninth preemption, which ends its play within about ten
seconds on the ablation board, and a pager splice makes the pilot replay most of its
context, about 19 s on WebGPU.

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

Open <http://localhost:8766> in a browser with WebGPU (recent Chrome or Edge). The
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
  **Stop** ends a run at the next tick.
- **The board:** drawn on a canvas, sharp at any zoom. Under it, the last move: who made
  it (`pilot`, `evade` or `prompt`), how many ticks late it landed, and `PREEMPTED` when
  the reflex took the machine from the pilot. A late move outlines the column the ship
  was in when the board was read. Live counts: lives, kills, ticks, preemptions,
  cancellations, reflexes, lag (mean and 95th percentile, in ticks), and catch-up ticks.
- **The result:** the case's four criteria with a verdict each, a table of every run this
  session (player × board × seed: lives, kills, lag, preemptions, cancellations, the
  prompt loop's parse rate), downloads of the journal and the result, and the ZEOS
  debugger on the Space Invaders case.

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

`?tune=` on the page's URL overrides the arms' options for that page load, as JSON
`{"zeos": {...}, "prompt": {...}, "kernel": {...}}` (`page.configure_json`): for example
`?tune={"zeos":{"max_chunk":128}}`. `kernel` overrides the zeos arm's `KernelConfig`, for
diagnosis only.

## Measured

Qwen3.5-4B-ZEOS-OPT on WebGPU in Chrome on an M1 Max, seed 7, `tests/si_webgpu.mjs`,
60 s of play each, machine defaults (`max_chunk` 256, `stall_ms` 5):

| player | board | lives | kills | ticks | lag mean / p95 (ticks) | preemptions | warm-up | notes |
|---|---|---|---|---|---|---|---|---|
| ZEOS | default | 3 | 1 | 121 | 5.0 / 8 | 7 | 16 s | 23 pilot moves, all valid; four criteria pass |
| ZEOS | ablation | 0 | 3 | 64 | 9.8 / 17 | 9 | 12 s | pilot faulted for starvation on its 9th preemption; no pilot move after |
| prompt | default | 2 | 0 | 120 | 5.5 / 7 | – | 8 s | parse rate 100% |
| prompt | ablation | 0 | 0 | 66 | 9.3 / 11 | – | 7 s | parse rate 86% |

Warm-up includes about 5 s (default) or 3.5 s (ablation) of mask prewarm. With the
kernel's starvation limit lifted (`?tune={"kernel":{"starvation_limit":100000}}`, a
diagnosis, not a fix), 120 s of each: on the default board the pilot made 25 moves, lag
8.7 / 12 ticks, but a pager splice made one step replay 2,961 positions (18.6 s), and
the longest gap between pilot moves was 43.5 s; on the ablation board the ZEOS arm won
(3 lives, 8 kills, 81 ticks).

**Starvation.** `KernelConfig.starvation_limit` (8) is compared with a job's preemption
count, which the scheduler only ever increments. The case does not set it. A pilot that
takes several ticks a move is running when most threats arrive, so it is preempted by
nearly every one; the ablation board reaches nine threats in about 40 ticks and the
default board in about 120 (simulated over 40 seeds).

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
