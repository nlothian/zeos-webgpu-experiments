# Space Invaders in the browser

**Status: under construction.** The page, its build and its server work end to end, but
every run is a `FakeRun` (below): the pieces that let the kernel and the model play
are on other branches and are not wired in yet.

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

### The FakeRun

`page.open_run` looks the player up in `page.RUN_BUILDERS`, which is empty on this
branch, and falls back to `FakeRun`: a real `Game` on the real board against the wall
clock, with random moves (biased to shoot under a monster) and simulated model latency
(about 1.2 s a pilot move, 5 s a prompt reply; quicker with the stub). The ZEOS arm's
reflex is the native `threat_reading` and `dodge`, and it preempts a pilot request in
flight. A `FakeRun` never calls its worker and runs no kernel, so it has no journal: its
verdicts read *not judged* and the debugger shows the case's wiring only.

## Layout

- `src/zeos_space_invaders_web/contracts.py`: the Python seams between the pieces (the
  non-blocking worker, the machine, the prompt arm, the runners, frames and results), and
  `load_board`. [`CONTRACTS.md`](CONTRACTS.md) is the JavaScript half (the channel's slot
  layout, the begin/poll/cancel decode step and its race rules, the page and worker
  messages) and who owns which file.
- `src/zeos_space_invaders_web/page.py`: `open_run`, `FakeRun`, and the JSON the page
  reads (`finished_json`, `payload_json`, `describe_json`).
- `src/zeos_space_invaders_web/boards/`: byte-for-byte copies of the native
  `settings_default.json` and `settings_ablation.json`.
- `web/`: the page (`index.html`, `app.js`, `board.js`, `style.css`) and `si_worker.js`,
  the Web Worker that hosts Pyodide and the run loop. The model thread and its channel
  are coop-count-web's files, copied in by `build.py`.

## Tests

```bash
uv run pytest demo/space-invaders-web
node --test demo/space-invaders-web/tests/js/*.test.mjs
```

The pytest suite builds `web/dist` into a temporary directory, and plays a `FakeRun`
under Pyodide in Node (`tests/pyodide_run.mjs`) when coop-count-web's npm install is
present.
