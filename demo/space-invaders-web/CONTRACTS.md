# Contracts: the JavaScript side

The Python half of these seams is `src/zeos_space_invaders_web/contracts.py`; its
constants (`SLOT_*`, `FRAME_OFFSET`, `CHANNEL_BUSY`, `PageMessage`, `WorkerMessage`,
`CONTROL_STOP`) are the names used here. If this file and that module disagree, that is
a bug in one of them.

## The shape of a run

One synchronous loop in the Pyodide worker owns both the game clock and the kernel. A
model decode step never blocks it: the machine *begins* a step over the channel, *polls*
for its result with a bounded `Atomics.wait` (the loop's only sleep while a step is in
flight), and *cancels* it when the job is preempted or its context changes. The model
thread checks for a cancel before every prefill chunk, so a cancel lands within one
chunk (`maxChunk`, 256 positions for a pilot step). Stop reaches the loop through a
control SharedArrayBuffer the page writes, because the loop never yields to the worker's
event loop.

## 1. The channel (`demo/coop-count-web/web/model_channel.js`)

The changes are additive: `SyncModelWorker`'s existing synchronous methods, including
`decodeStep`, behave as before, and coop-count-web's page and tests are unaffected.

### Buffer layout

| Int32 slot | Name | Values |
|---|---|---|
| 0 | `SLOT_STATE` | 0 idle, 1 waiting, 2 answered (unchanged) |
| 1 | `SLOT_LENGTH` | byte length of the reply frame (unchanged) |
| 2 | `SLOT_ABORT` | `requestId + 1` of the step asked to stop; 0 none |
| 3 | `SLOT_IN_FLIGHT` | `requestId + 1` of the begun step; 0 none |

The reply frame (`frames.js`) still starts at byte 16 (`FRAME_OFFSET`, `DATA`). Slots
hold `id + 1` so that 0 always means "none" (request ids start at 0).

### Python side: `SyncModelWorker`

```js
beginDecodeStep(jobId, {allowedBlocks, allowedTokens, sample, maxChunk}) -> requestId
pollDecode(timeoutMs) -> null
                       | {tokenId, attention, cancelled: false, stats}
                       | {cancelled: true, resident, stats}
cancelDecode() -> void
get inFlight -> boolean
```

`stats` is `{positions, chunks, fillMs}`: KV positions this step computed, the prefill
runs they took, and the milliseconds spent filling.

- **`beginDecodeStep`** never waits. It throws `channel busy: decode in flight` if a step
  is already in flight. Otherwise it clears slot 2, stores `id + 1` in slot 3, sets
  slot 0 to waiting, posts `{id, method: "decodeStep", args: [jobId, opts], begun: true}`
  and returns `id`. `allowedBlocks`/`allowedTokens` are copied to `Uint8Array`s as
  `decodeStep` does.
- **`pollDecode(timeoutMs)`** returns `null` at once when nothing is in flight. Otherwise
  it waits on slot 0 for at most `timeoutMs` and returns `null` if the reply has not
  landed; it returns as soon as it does. Taking a reply clears slots 0, 2 and 3, so
  `inFlight` is false afterwards. An error reply is thrown, after clearing.
- **`cancelDecode()`** stores the in-flight `id + 1` in slot 2 and returns; a no-op when
  nothing is in flight. The step stays in flight until `pollDecode` returns it.
- **`inFlight`** is `slot 3 !== 0`.
- **`call()`** (every synchronous method that reaches the model thread) throws
  `channel busy: decode in flight` while a step is in flight. `piece` and a cached
  `info` do not reach the model thread and keep working. A caller that needs the channel
  cancels and drains first: `cancelDecode()`, then `pollDecode` until non-null (the
  Python machine's `_settle()`).

### Race rules

1. **The result of a cancelled step is always dropped.** If slot 2 holds the step's
   `id + 1` when `pollDecode` takes the reply, the answer is
   `{cancelled: true, resident, stats}`, even when the model thread had finished the
   step and replied with a token. A token is never delivered for a step its caller
   cancelled.
2. **Cancel is checked before each chunk**, i.e. before every `session.run`, including
   the final one-position decode. A cancel posted while a chunk runs takes effect after
   that chunk.
3. **Resident positions stay valid.** A cancelled step leaves the context's KV cache at
   `resident` positions (`track.kvLength`). The next step for the same context resumes
   filling from there; nothing the cancelled step computed is recomputed unless the mask
   history requires it (the existing rewind rule).
4. **A step never changes the context's tokens.** `decodeStep` chooses a token and
   returns it; appending it is the caller's business. Dropping a result therefore leaves
   nothing to undo.
5. **One step in flight per channel.** The model thread serves requests in order, and
   `call()` refuses while a step is in flight, so no request queues behind a step.

### Model side: `serveChannel`

For a request with `begun: true`, `serveChannel` adds
`shouldStop = () => Atomics.load(state, SLOT_ABORT) === id + 1` to the options before
`serveRequest`, and the reply carries `resident` and `stats` whether the step finished
or stopped. The reply for a step that finished is `{tokenId, attention, resident, stats}`;
`pollDecode` adds `cancelled: false`, or converts it under rule 1.

## 2. The model workers

### `OptZeosWorker.decodeStep` (`opt_zeos_worker.js`)

```js
decodeStep(jobId, {allowedBlocks, allowedTokens, sample, shouldStop = () => false, maxChunk = this.chunk})
  -> {tokenId, attention, resident, stats}
   | {cancelled: true, resident, stats}
```

`fill()` cuts the positions to compute into runs of at most `maxChunk` and calls
`shouldStop()` before each `run()`. On a stop it returns at once with
`{cancelled: true, resident: track.kvLength, stats}`; the next `fill` for the context
resumes from `kvLength`. Called without the two new options it behaves as before
(`maxChunk` defaults to the graph's own, 2048), and the extra `resident` and `stats`
fields are ignored by the existing `JsMachine`.

### The stub and transformers workers

`stub_worker.js` and `transformers_worker.js` honour `shouldStop` the same way and
report `resident` and `stats`. The stub also takes simulated latency,
`{stepMs = 0, positionMs = 0}`: a step costs `stepMs` plus `positionMs` for each position
it fills, spread over `maxChunk`-sized chunks with `shouldStop()` checked between them.
With both 0 it is as fast and as deterministic as before. The Space Invaders stub
(`demo/space-invaders-web/web/stub/pilot_stub_worker.js`) is served over the same
channel, so begin/poll/cancel behave identically for the stub and the model.

## 3. Page and worker messages (`si_worker.js`)

Messages are `{type, ...body}`, the coop-count-web convention.

Page to worker:

| `type` | Body | Meaning |
|---|---|---|
| `boot` | `{controlSab}` | Load Pyodide and the wheels; keep the control buffer. |
| `attachModel` | `{backend, buffer, port, name}` | Wrap the model thread the page started (as coop-count-web). |
| `start` | `{arm, board, seed, machine}` | `arm`: `"zeos"` or `"prompt"`; `board`: `"default"` or `"ablation"`; `machine`: `"model"` or `"stub"`. |

Stop is not a message: the page does `Atomics.store(control, CONTROL_STOP, 1)` and
`Atomics.notify(control, CONTROL_STOP)` on the Int32 view of `controlSab`
(`CONTROL_BYTES` long). The loop reads it every iteration, so a run ends within one
tick; `start` clears it.

Worker to page:

| `type` | Body |
|---|---|
| `ready` | `{pyodide, isolated}` |
| `warming` | `{arm, board}`: system-prompt prefill and mask prewarm, before the clock starts |
| `started` | `{arm, board, seed, machine}`: the clock has started |
| `frame` | a `Frame` (`contracts.py`) |
| `decision` | a `DecisionRecord`, one per entry of each frame's `decisions` |
| `finished` | `{result, verdicts, journal, payload}`: `result` is `RunResult.to_json()` |
| `error` | `{message, stack}` |

`page.py` exposes `open_run(arm, board, seed, worker, *, stub, on_frame, stop)`, which
returns a `Run` (`warm()`, `run()`, `close()`); `si_worker.js` posts `warming`, calls
`warm()`, posts `started`, calls `run()` and posts `finished`.

## 4. File ownership

Each workstream edits only its own files; the table is the plan's.

| Branch / worktree (siblings of zeos) | Agent | Owns |
|---|---|---|
| `feat/si-web` / `zeos-si-web` | orchestrator, then integrator | scaffold: root `pyproject.toml` member, `uv.lock` member hunk only, `demo/space-invaders-web/{pyproject.toml, README.md, contracts.py, boards/*.json, tests/conftest.py}`; later the e2e tests and `si_webgpu.mjs` |
| `si/channel` / `zeos-si-channel` | Opus: channel | coop-count-web `web/{model_channel.js, opt_zeos_worker.js, frames.js, stub_worker.js, transformers_worker.js, node_model_thread.mjs}`, `node_worker.py` (async methods), JS + pytest channel tests |
| `si/machine` / `zeos-si-machine` | Opus: machine | `js_machine.py` (refactor only); SI `machine.py`, `prompt_player.py`, `fake_worker.py` (`FakePilotWorker`), `pyodide_glue.py`, `web/stub/pilot_stub_worker.js`, machine/prompt/stub-parity tests |
| `si/runner` / `zeos-si-runner` | Opus: runner | native `player.py` (Protocol annotations only) plus one native test; SI `runner.py`, `boards.py`, `metrics.py`, runner tests (against `tests/stubs.py` `stub_machine`) |
| `si/page` / `zeos-si-page` | Opus: page | SI `build.py`, `serve.py`, `web/{index.html, app.js, si_worker.js, board.js, style.css}`, `page.py`, page tests, `pyodide_run.mjs`; codes against a `FakeRun` |
| `si/bench` / `zeos-si-bench` | Opus: benchmark | `demo/space-invaders-web/bench/*` only |

`load_board` already lives in `contracts.py`; `boards.py` re-exports or extends it rather
than reimplementing it.

## Machine notes

What the machine workstream (`machine.py`, `prompt_player.py`, `fake_worker.py`,
`pyodide_glue.py`, `web/stub/pilot_stub_worker.js`) settled that the sections above leave
open, and the one change it made to `contracts.py`:

- **`PilotMachine.prewarm`** is `prewarm(descriptor="pilot", commands=PREWARM_COMMANDS)
  -> float`: it returns the milliseconds the walk took, so the page can report it. It was
  `prewarm() -> None`; a call with no arguments is unchanged.
- **Optional extras.** `PilotJsMachine` also takes `tokenize_cache` (words kept, default
  4096: a board is tokenized a word at a time, and each tokenize is a round trip to the
  model thread). `BrowserPromptPlayer` also takes `bridge` and `max_chunk` (`None`, the
  default, leaves `maxChunk` out, so the worker fills in its own chunk). Both still match
  their `*Factory` Protocols.
- **Defaults from the benchmark** (`bench/RESULTS.md`): `DEFAULT_MAX_CHUNK` is 256 (it
  was 64; below about 800 positions a run's fixed overhead made chunk 64 read a board
  1.65x slower), and `stall_ms` defaults to `DEFAULT_STALL_MS`, 5 (it was 1), both in
  `contracts.py`. A step with more than `uncapped_above` (1024) positions to fill -- the
  first step over a descriptor body, a replay after a splice -- is sent without
  `maxChunk`, so warm-up and replay run at the worker's own chunk; such a step is
  cancelled only between the worker's own runs. `uncapped_above=None` caps every step.
- **The pilot's grammar** is the native pilot ABI (`write`, `read`, `exit`) less
  `forbid_verbs` (off by default), over the aliases `descriptors` binds (default
  `{"pilot": ("stdin", "stdout")}`), with the payload of `payloads`' aliases narrowed to
  one of their words: by default `{"stdout": ("left", "right", "shoot")}`, so `write
  stdout left;` is the only shape a move takes (free text let the 4B write `write stdout
  left left left 1` on every board). `payloads={}` gives the free-text payload back. The
  machine passes no `valued` aliases: `seat_maps` marks the pilot's `stdout` valued
  (`game.controls` is world-backed), and a numeric payload could not name a move.
- **Prewarm** walks each command as the worker tokenizes it and also warms the state
  after each space, which covers the move states (about 6 per descriptor); each cold
  state costs 1.6-2.1 s under Pyodide, so expect seconds of warm-up.
- **What cancels, and what restarts the command.** Inject, trunc, splice, fork, a
  narrowing `set_mask`, block padding and `invalidate` cancel the job's step; the decode
  of another job (native or served) cancels whatever is in flight. Inject, trunc, fork,
  a narrowing `set_mask`, `invalidate`, and a splice that reaches the current command
  also reset the job's parser and grammar round (the native `_cancel`); an upstream
  splice renumbers the command instead. `invalidate` does this even with nothing in
  flight (resetting an empty round changes nothing), and leaves a pending automatic
  `read stdin` in place: a turn whose `write` completed has made its move.
- **Settling.** Only operations that call the worker synchronously wait for a cancelled
  step to drain (create/destroy a served context, inject, trunc, splice, fork, `raw`,
  `prewarm`, `close`). The rest post the cancel and let a later `decode` drain it, polling
  for at most `stall_ms` and stalling meanwhile; native jobs never touch the worker.
- **Step log.** A `decode` that drains a cancelled step and then begins the next logs a
  `cancelled` entry and then the entry for its own outcome. The automatic `read stdin`
  is logged as `native` (no worker call).
- **The prompt arm** prefills the system prompt into `prompt-prefix:prompt`, keeps it,
  and forks each decision into `prompt-arm:prompt` (a truncate would rewind to the KV
  snapshot below the prefix and replay ~170 positions every decision); its replies may use every id
  except the pad id and control ids other than `<|im_end|>` (the end-of-sequence id is
  allowed and ends the reply). `warm()` decodes and discards one step to make a lazily
  filling worker compute the system prompt.
- **The JS stub** defines `globalThis.createPilotStubWorker(options)` (so
  `self.createPilotStubWorker` in a worker), options `{moves, replies, stepMs, positionMs,
  reads, blockSize, terminator, chunk}`. It also carries `meta.tokenizerSize` (its vocabulary size) and `backend` (`"stub"`), which
  `serveChannel` reads for the channel's `pieces` and `backend` requests. It is a
  model-side worker: `decodeStep` is async
  and honours `shouldStop` and `maxChunk`; begin/poll/cancel come from the channel.
