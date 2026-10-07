# Space Invaders in the browser

A language model plays Space Invaders in a web page, against a clock that does not wait
for it. It plays two ways, and you can watch both:

- **Prompt loop**: the usual approach. Show the model the board, wait for its reply,
  make the move.
- **ZEOS**: the ZEOS kernel runs the model as a `pilot` job beside an `evade` reflex.
  When a bomb is about to hit, the kernel preempts the pilot, the reflex dodges, and the
  pilot is told the ship moved.

Everything runs in the page. The kernel and the game run under Pyodide, and the model,
`Qwen3.5-4B-ZEOS-OPT`, runs on WebGPU through
[`zeos-browser`](../../packages/zeos-browser/README.md). It is the browser version of
[`../space-invaders`](../space-invaders/), which plays against a model server.

## How ZEOS changes the outcome

The same model (on WebGPU, Chrome, M1 Max) plays the same games both ways. A move still
being thought about when the bomb lands is too late. Seeds 7, 11 and 23; each cell
lists the three games in that order.

| | Default board, prompt loop | Default board, ZEOS | Hard board, prompt loop | Hard board, ZEOS |
|---|---|---|---|---|
| Ship destroyed | 3 of 3 games | **0 of 3** | 3 of 3 | **0 of 3** |
| Lives left (of 3) | 0 / 0 / 0 | **3 / 3 / 3** | 0 / 0 / 0 | **1 / 3 / 3**² |
| Game survival length in ticks | 74 / 409 / 230 | **600 / 600 / 600** (full game) | 44 / 91 / 20 | **93 / 76 / 90**¹ |
| Invaders shot | 0 / 6 / 3 (of 15) | **11 / 11 / 13** | 1 / 3 / 0 (of 10) | **3 / 6 / 4**² |
| Dodges the reflex took over | – | 46 / 47 / 60 | – | 12 / 9 / 10 |
| Model's reaction delay (mean ticks between seeing a board and its move landing) | 5.1 | 9.0 | 7.6 | 8.5 |
| Warm-up before the game starts | 7 s | 17 s | 7 s | 14 s |

- **Default board**: 12×16, a tick every 0.5 s, 600 ticks (5 minutes).
- **Hard board** (*ablation* in the page's board menu): 9×8, a tick every 0.2 s, more
  bombs. The game ends when the invaders reach the bottom.

¹ The game ended because the invaders landed; ZEOS's ship was still alive in every game.

² In seeds 11 and 23 the kernel stopped the pilot for starvation (preempted 9 times
without completing a move, over the limit of 8). From then on only the reflex played,
so it kept the ship alive but shot no more invaders. See *Known issues*.

**How to read it.** ZEOS does not make the model faster: its moves land later, because
its context grows over the game. The ship survives because the kernel does not wait for
the model. It takes the machine away at the moment of danger, and the reflex dodges in
time.

Sources: `compare_s7.json`, `compare_s11.json` and `compare_s23.json` in
[`bench/results/webgpu/`](bench/results/webgpu/), from
`tests/si_webgpu.mjs --seconds 320 --seed N` (see *Tests*).

## Getting started

You need a browser with WebGPU (a recent Chrome or Edge). From the repository root:

```bash
uv sync --all-packages
npm install --prefix packages/zeos-browser           # once
uv run python demo/space-invaders-web/build.py       # builds the page into web/dist
uv run python demo/space-invaders-web/serve.py       # http://localhost:8766
```

Open <http://localhost:8766>, set **machine** to the model on WebGPU, and press **run**.

- **The first run downloads the model** from
  [`nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16`](https://huggingface.co/nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16),
  about 2.4 GB. The browser keeps it, so later runs, reloads and restarts load it from
  storage. A line under the progress bar shows how much is stored, with a **clear cached
  model** button.
- **Each run warms up** before the clock starts: about 15 s for ZEOS and 7 s for the
  prompt loop.
- **Use `serve.py`, not any static server.** It sends the two headers that make the page
  cross-origin isolated, which the model channel needs (SharedArrayBuffer); the page
  says when they are missing. It listens on 8766, so it can run beside coop-count-web's
  page on 8765; `--port` and `--dir` change that.
- **No GPU?** Choose the **stub** machine. It answers through the same channel as the
  model, with simulated latency, so you can see the kernel and the page work.

### Build options

- `--model` loads a local export of the model instead of the Hugging Face copy:
  `packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT` by default (made as *Getting the
  model* in [zeos-browser](../../packages/zeos-browser/README.md) describes), or
  `--model DIR`. It links the export into `web/dist/models/`, and the page loads it by
  default; `?model=huggingface` or `?model=local` in the URL picks either.
- `--copy-model` copies the export instead of linking it, for a static host, which does
  not follow symbolic links. The copy is gigabytes.
- `--link-node-modules` links this demo's `node_modules` to zeos-browser's.
- Without zeos-browser's npm install the build still works, and the page offers only
  the stub.

## What you'll see

- **Controls**: the player (ZEOS or the prompt loop), the board, the seed and the
  machine. The header shows whether the page is cross-origin isolated, whether WebGPU
  offered an adapter, and whether the build has the model.
- **Phases**: *loading model*, with a progress bar; *warming*, while the system prompt
  is prefilled before the clock starts; *playing*; and *finished*. **Stop** ends a run
  within a fraction of a second.
- **The board**, and under it the last move: who made it (`pilot`, `evade` or
  `prompt`), how many ticks late it landed, and `PREEMPTED` when the reflex took the
  machine from the pilot. A late move outlines the column the ship was in when the board
  was read. Live counts beside it: lives, kills, ticks, preemptions, cancellations,
  reflexes, lag and catch-up ticks.
- **The result**:
  - the case's four criteria, each with a verdict;
  - the run's measurements, such as warm-up, how late moves landed, how long cancels
    took, the longest gap between pilot moves, and any kernel faults (for the prompt
    loop, its replies and any it could not parse);
  - a table of every run this session;
  - downloads of the journal and the result;
  - the ZEOS debugger, which steps through the run's journal.

## How it differs from the native demo

The case files and the kernel are the native demo's. The web demo sets two things
differently on the kernel it builds:

- **The pilot's context window is 32,768, not 4096** (`page.DEFAULT_PILOT_CONTEXT`). The
  model keeps its state on the GPU between steps, and its recurrent layers cannot cut a
  span out of the middle. So when the pager removes old context, the model must
  recompute everything after the cut, which takes 15–20 s on WebGPU. The larger window
  keeps the pager from acting in a full game. The cost is that the context reaches about
  22k ids by the end of a default game.
- **The model reads input in chunks of 128 positions, not 256**
  (`contracts.DEFAULT_MAX_CHUNK`). A cancel can only land between chunks, so smaller
  chunks let a preemption stop the pilot sooner, at a small cost in reading speed.

The kernel's starvation limit is its own (8): a job preempted more than 8 times without
completing a request in between is faulted
([`docs/notes/starvation-progress.md`](../../docs/notes/starvation-progress.md)). The
pilot completes one with every move and every board it reads.

## Known issues

- **The hard board can starve the pilot.** The threat sensor reports a falling bomb on
  every tick until it lands, so at 0.2 s a tick the reflex preempts the pilot about
  once a tick while bombs fall. Each preemption cancels the pilot's command, and a move
  takes the 4B longer than a tick in the browser, so the pilot can go 9 preemptions
  without completing one, and the kernel faults it. The native demo runs the same sensor
  and kernel, but its model server answers in about half a second and cancels by
  dropping a stream, so its pilot completes moves between bombs.
- **The pilot slows down over a long game.** Its context grows, so by the end of a
  default game a cancel can take 2.4 s and the game loop can fall up to 1.9 s behind.
- **Inserting a notice waits for the chunk in flight.** When the kernel tells a resumed
  pilot what changed, it must first let the model finish the chunk it is running:
  0.3–0.6 s on average, up to 1–2 s late in a game. The game loop stalls meanwhile. On
  the hard board this is most of the time the loop falls behind.
- **The board's aim hint can stall the loop.** The `lead` view
  (`zeos_space_invaders/game/aim.py`) searches ahead for a shot. When none lands within
  16 turns it visits about 6,800 states, which takes 300–750 ms under Pyodide. Both
  players render the board with it.

## How it fits together

- **Python**, under Pyodide in a Web Worker (`web/si_worker.js`):
  - `page.open_run` assembles a run for the chosen player.
  - ZEOS is `PilotJsMachine` (`machine.py`) driven by the native `ZeosDriver` and a
    wall-clock runner (`runner.py`).
  - The prompt loop is `BrowserPromptPlayer` (`prompt_player.py`) and its runner.
  - Before the clock starts, ZEOS warms the grammar's mask cache and prefills the
    pilot's system prompt; the prompt loop prefills its system prompt.
- **The model** runs on a thread of its own, behind zeos-browser's worker channel.
  `build.py` copies those files in from `packages/zeos-browser/web`.
- **The stub machine** runs `web/stub/pilot_stub_worker.js` on the same channel.
  - A build without `web/stub/` falls back to a `FakeRun`: the real game against the
    clock with random moves and no kernel, so its verdicts read *not judged*.

`?tune=` in the page's URL overrides options for that page load (`page.configure_json`),
and the header shows when it is set. It takes JSON with up to four keys:

- `zeos`: `PilotJsMachine` keywords;
- `prompt`: `BrowserPromptPlayer` keywords;
- `kernel`: the ZEOS run's `KernelConfig` fields;
- `context`: the pilot's `ContextPolicy` fields.

For example, `?tune={"zeos":{"max_chunk":256},"context":{"window":4096}}` runs with
256-position chunks and the native demo's 4096 window.

## Layout

- `src/zeos_space_invaders_web/`:
  - `contracts.py`: the Python seams between the pieces, and `load_board`;
  - `page.py`: `open_run`, the runs, and the JSON the page reads;
  - `machine.py`, `prompt_player.py`, `runner.py`, `metrics.py`, `pyodide_glue.py`;
  - `fake_worker.py`: `FakePilotWorker`, the CPython twin of the JavaScript stub;
  - `boards/`: byte-for-byte copies of the native `settings_default.json` and
    `settings_ablation.json`.
- [`CONTRACTS.md`](CONTRACTS.md): the JavaScript side, which covers the channel's
  layout, the begin/poll/cancel decode step, and the page and worker messages.
- `web/`: the page (`index.html`, `app.js`, `board.js`, `style.css`) and `si_worker.js`.
- `bench/`: component benchmarks and the measurements the settings above were chosen
  from ([`bench/RESULTS.md`](bench/RESULTS.md)).

## Tests

```bash
uv run pytest demo/space-invaders-web
node --test demo/space-invaders-web/tests/js/*.test.mjs
```

- **No GPU needed.** The pytest suite plays both players under CPython against
  `FakePilotWorker` (`test_e2e_stub.py`). When zeos-browser's npm install is present, it
  also builds the page and plays under Pyodide in Node against the JavaScript stub
  (`test_e2e_pyodide.py`).
- **The real model** only runs on WebGPU, so it is driven separately.
  `tests/si_webgpu.mjs` plays the built page in headed Chrome, over both boards and both
  players, and prints each run's measurements. It needs Playwright and Chrome, which
  are not dependencies of this demo:

  ```bash
  uv run python demo/space-invaders-web/build.py --model
  PLAYWRIGHT_MODULE=/path/to/node_modules/playwright node demo/space-invaders-web/tests/si_webgpu.mjs
  ```

  Build with `--model` so each run loads the local export rather than downloading the
  model into a fresh browser profile. Run one at a time with nothing else on the GPU,
  and leave its Chrome window alone until it closes.
