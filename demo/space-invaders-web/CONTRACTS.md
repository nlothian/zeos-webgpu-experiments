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
chunk (`maxChunk`, 128 positions for a pilot step). Stop reaches the loop through a
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

### As implemented (`si/channel`)

Where the code settles what the text above leaves open, or differs from it:

- **`stats`.** `positions` and `chunks` count every run of the graph the step made, the
  final one-position decode included; `fillMs` is the time spent in those runs except a
  final one-position decode. A step stopped before any run reports zeros.
- **`maxChunk`.** `null`/absent means the worker's own chunk. A larger value is clamped
  to it; a value that is not a positive integer is a `RangeError` (an error reply).
- **Finished replies carry `resident`** as well as `stats` (the context's full length
  for the step's cache), so `pollDecode`'s `DecodeDone` has a `resident` key too.
- **`pollDecode(timeoutMs = 0)`.** `timeoutMs` defaults to 0 (a non-blocking check that
  still takes a reply that has landed). `resident` and `stats` are read from the reply as
  they are; all three workers report them.
- **Bit-identical resumption needs the same `maxChunk`.** Chunks are cut by where they
  start and by `maxChunk`; a step resumed with another `maxChunk` cuts the rest
  differently and agrees only to float16 rounding.
- **Stale replies.** Every reply's `id` is checked against the request waited on. In
  `SyncModelWorker` a mismatch, or any call that timed out (whose reply would land
  later), makes the channel unusable: every later call throws `model channel unusable:
  ...`, since the one reply buffer cannot hold a stale reply aside. `NodeWorker` reads an
  ordered pipe, so it discards a reply to an earlier request and keeps waiting.
- **`beginDecodeStep`** sets the slots before posting (a thread may answer before `post`
  returns) and clears them if `post` throws.
- **`shouldStop` is asked before every run of the graph, not before a skipped hidden
  run** (`OptZeosWorker` carries a hidden run past without a run; it costs one copy).
  `TransformersWorker` asks it before each run of `decodeStep`'s replay and before its
  final decode; its `append` prefills eagerly and is not interruptible.
- **The stub's tape.** A tape advances per step, unlike a model, which is a function of
  its context. So a begun step (one with a `shouldStop`) leaves the tape where it was and
  records the advance; the next step commits it only if the chosen token was appended
  since. A dropped result is thus chosen again, as with a model. Synchronous steps
  advance at once, as before. The stub also has `meta.tokenizerSize`, so it can be served
  over a channel (`startNodeModel({stub: {tapes, options}})`).
- **CPython (`NodeWorker`).** `beginDecodeStep`, `pollDecode(timeoutMs)`,
  `cancelDecode()` and the `inFlight` property, over the existing pipe to
  `web/node_bridge.mjs`: `pollDecode` waits with `select` on the raw pipe; a cancel is a
  control frame `{"cancel": id}` that the bridge reads while the step awaits the graph and
  never answers. `pollDecode` returns a plain dict (`attention` a list of floats or
  `None`). Calls while in flight raise `RuntimeError(CHANNEL_BUSY)`. `close()` with a
  step in flight cancels and drains it, and kills a child that has not exited 30 s after
  its stdin closed. `maxChunk` is passed through unchanged; the worker validates it. `NodeWorker(stub=path)`
  serves the stub over a JSON file of `{tapes, options}`. This touched
  `node_bridge.mjs`, which the ownership table does not list.

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
open, and the changes it made to `contracts.py`:

- **`PilotMachine.prewarm`** is `prewarm(descriptor="pilot", commands=PREWARM_COMMANDS)
  -> float`: it returns the milliseconds the walk took, so the page can report it. It was
  `prewarm() -> None`; a call with no arguments is unchanged.
- **`DecodeDone` carries `resident`**, as the channel's reply does; the machine tracks
  the positions the worker holds from it. `normalise_poll` takes a Python dict as given
  and reads a JsProxy field by field, with no defaults: a missing field is an error.
- **`PilotMachine.step_log`** is a `deque` (newest `step_log_size` entries, default
  4096); `step_totals` (per outcome), `positions_total` and `fill_ms_total` count every
  decode.
- **Optional extras.** `PilotJsMachine` also takes `tokenize_cache` (an LRU of words,
  default 4096, 0 for none: a board is tokenized a word at a time, and each tokenize is a
  round trip to the model thread), `payloads`, `step_log_size` and `settle_timeout_s`.
  `BrowserPromptPlayer` also takes `bridge`, `settle_timeout_s` and `max_chunk` (`None`, the
  default, leaves `maxChunk` out, so the worker fills in its own chunk). Both still match
  their `*Factory` Protocols.
- **Defaults from the benchmark** (`bench/RESULTS.md`): `DEFAULT_MAX_CHUNK` is 256 (it
  was 64; below about 800 positions a run's fixed overhead made chunk 64 read a board
  1.65x slower), and `stall_ms` defaults to `DEFAULT_STALL_MS`, 5 (it was 1), both in
  `contracts.py`. Every pilot step is capped at `max_chunk` (the integration notes
  below set 128). An `uncapped_above` option that let long fills run at the worker's own
  chunk was removed in the review round: nothing used it, and a step so sent could be
  cancelled only between 2048-position runs.
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
  narrowing `set_mask`, block padding and `invalidate` cancel the job's step. A decode of
  another served job cancels the step in flight only when it needs the channel for a step
  of its own; a native decode, or a served job's automatic `read stdin`, needs no channel
  and cancels nothing. A preemption therefore drops the pilot's step only through
  `ZeosDriver`'s `invalidate`. Inject, trunc, fork, a narrowing `set_mask`, `invalidate`,
  and a splice that reaches the current command also reset the job's parser and grammar
  round and drop a pending automatic `read stdin` (the native `_cancel`); an upstream
  splice renumbers the command instead. A worker that reports a cancel nobody asked for,
  or answers a poll with an error, ends the step: the machine drops it and raises.
- **`forbid_verbs`** cannot include `read` while `turn_ends_at_call` is on, since the
  machine ends each turn with one.
- **Settling.** Only operations that call the worker synchronously wait for a cancelled
  step to drain (create/destroy a served context, inject, trunc, splice, fork, `raw`,
  `prewarm`, `close`). The rest post the cancel and let a later `decode` drain it, polling
  for at most `stall_ms` and stalling meanwhile; native jobs never touch the worker.
- **Step log.** A `decode` that drains a cancelled step and then begins the next logs a
  `cancelled` entry (under the job whose step it was) and then the entry for its own
  outcome. The automatic `read stdin`
  is logged as `native` (no worker call).
- **The prompt arm** prefills the system prompt into `prompt-prefix:prompt`, keeps it,
  and forks each decision into `prompt-arm:prompt` (a truncate would rewind to the KV
  snapshot below the prefix and replay ~170 positions every decision); its replies may
  use every id except the pad id and control ids other than `<|im_end|>` (the end-of-sequence id is
  allowed and ends the reply). `warm()` decodes and discards one step to make a lazily
  filling worker compute the system prompt.
- **The JS stub** defines `globalThis.createPilotStubWorker(options)` (so
  `self.createPilotStubWorker` in a worker), options `{moves, replies, stepMs, positionMs,
  reads, blockSize, terminator, chunk}`. It also carries `meta.tokenizerSize` (its
  vocabulary size) and `backend` (`"stub"`), which `serveChannel` reads for the
  channel's `pieces` and `backend` requests. It is a model-side worker: `decodeStep` is
  async and honours `shouldStop` and `maxChunk`; begin/poll/cancel come from the channel.

## Runner notes

Written by the runner workstream (`runner.py`); none changes a type in `contracts.py`.

- **The zeos runner's clock tells `time.monotonic()` time.** `ZeosDriver._pump`
  compares its deadline against `time.monotonic()`, and the runner passes the next
  tick's due time on its `Clock` as that deadline. `MonotonicClock` qualifies, and so
  does a Pyodide clock whose `now()` is `time.monotonic()` and whose `sleep()` is an
  `Atomics.wait` on the control buffer. Under Pyodide the page must inject that clock:
  `time.sleep` does not wake on a stop. The zeos runner raises `ValueError` at
  construction if `clock.now()` is more than `CLOCK_TOLERANCE_S` (0.5 s) away from
  `time.monotonic()`.
- **The constructors take more than the factories pin**, all keyword-only with
  defaults: `max_ticks` (default the board's `max_steps`), `max_seconds`, and for the
  zeos runner `warm_timeout_s` (default 120 s). The zeos runner's `driver` is typed as
  `runner.ZeosDriverLike`, the part of `ZeosDriver` it uses, so the factory
  (`driver: ZeosDriver`) is still satisfied. Both runners have `warm()`, which
  `run()` calls if it has not been called.
- **Warm-up ends when the kernel has nothing to run**: a batch that returns before its
  deadline with no write, i.e. `Kernel.tick()` found no runnable job. In this case
  that is the pilot blocked on `game.state`. A stop during warm-up ends it without
  setting `warmed`. Load the model and create the worker context before `warm()`:
  `warm_timeout_s` bounds the prefill and the first syscall, not a model download.
  A pilot that writes during warm-up moves the ship before tick 0; that move is
  recorded at tick 0 with no lag and goes out on the first frame.
- **`lag_ticks` for a pilot move is measured against the board the pilot read**, found
  by the runner in the journal (`pipe.read` on `game.state`), not against
  `Decision.tick`. The driver stamps a move with the newest board delivered, which runs
  ahead of the one being answered whenever a board arrives mid-completion. An `evade`
  move is measured against the threat its handler was dispatched for: the runner notes
  the tick of each threat it hands over and takes the newest at each `vector.fired`.
  The driver stamps a reflex with the tick of the latest reading of either kind, which
  every later `sense` overwrites.
- **`RunResult.reflexes` and `Frame.reflexes` count moves `evade` made**, not threats
  delivered (`ZeosDriver.reflexes`). `cancellations` is `DriverMachine.cancellations`,
  part of the native protocol.
- **Frames.** One frame per batch of ticks applied, and a closing frame when the loop
  exits if the last frame does not already carry every decision and the final
  totals; its `tick` repeats the last one and its `catchup` is 0. Every
  `DecisionRecord` reaches exactly one frame. Nothing is handed to the driver or the
  prompt arm on the tick that ends the run.
- **`FrameSink.__call__` is not positional-only**, so `frames.append` does not
  type-check as a sink; wrap it. Making the parameter positional-only (`frame: Frame, /`)
  in `contracts.py` would let it.
- `build_driver(machine, spec)` registers the reflex on the machine and builds a
  `ZeosDriver` told the board's rules and view; `page.open_run` can use it for the
  zeos arm.

## Page notes

Written by the page workstream; each is the smallest change the page needed.

- **`open_run` takes an optional `clock`.** `page.open_run(arm, board, seed, worker, *,
  stub, on_frame, stop, clock=None)`: the extra keyword is optional, so the function
  still satisfies `OpenRun`. Under Pyodide `si_worker.js` passes
  `{now: performance.now() / 1000, sleep: Atomics.wait(control, CONTROL_STOP, 0, ms)}`,
  so a sleeping loop wakes the moment Stop is written; `None` is `MonotonicClock`.
  `stop` is `{is_set: Atomics.load(control, CONTROL_STOP) !== 0}`.
- **The stub machine gets a channel like the model's.** The page starts
  `stub_thread.js`, which imports `web/stub/pilot_stub_worker.js`, calls
  `self.createPilotStubWorker({stepMs, positionMs})` and serves it with `serveChannel`;
  the page attaches it as backend `stub` (`attachModel`), and `si_worker.js` hands
  `open_run` that `SyncModelWorker`. The model machine hands over the one attached for
  backend `webgpu`. `manifest.json` says whether the build has the stub (`stub`); without
  it `worker` is `None`, which only a `FakeRun` accepts.
- **Stop is cleared by the page**, just before it starts loading a thread for a run, not
  by `start`; a Stop pressed while a thread loads cancels the run before `start` is sent.
- **`frame` and `decision` bodies are spread.** `{type: "frame", ...Frame}` and
  `{type: "decision", ...DecisionRecord}`, the `{type, ...body}` convention; neither
  shape has a `type` field. `started` and `finished` also carry `stopped` (whether Stop
  was set).
- **`ready`** carries `{pyodide, python, isolated, model, stub, describe}`: `model` is the
  manifest's export name or `null`, `stub` whether the build has the stub, `describe` is
  `page.describe_json()` (boards, arms, criteria ids, the registered builders, and the
  debugger's wiring-only payload).
- **`finished.result`** is `RunResult.to_json()` less `journal` and `verdicts`, which ride
  beside it once each.
- **The integration seam is `page.RUN_BUILDERS: dict[Arm, RunBuilder]`**, where a
  `RunBuilder` takes a `page.RunContext(arm, spec, worker, stub, on_frame, stop, clock)`
  and returns a `Run`. An arm with no builder runs as `page.FakeRun`.
- **Board size** is not a `Frame` field; `board.js` reads it off `Frame.text`
  (`Game.render`: one line a row, four characters a cell).

## Integration notes

Written by the integrator after merging the five branches; where the notes above
disagree, this section says which holds.

- **The zeos runner's clock.** The page notes say `si_worker.js` passes a clock whose
  `now()` is `performance.now() / 1000`; the runner notes require `time.monotonic()`.
  Both hold: `page.ZeosRun` and `page.PromptRun` wrap whatever clock `open_run` is given
  in `page.LoopClock`, whose `now()` is `time.monotonic()` and whose `sleep()` is the
  given clock's (under Pyodide the `Atomics.wait` on the control buffer). The JS clock's
  `now` is therefore never read by a real run, only by a `FakeRun`.
- **`RUN_BUILDERS` is filled**: `"zeos"` is `page.ZeosRun` (`PilotJsMachine` over the
  channel, `runner.build_driver`, `WallClockZeosRunner`), `"prompt"` is `page.PromptRun`
  (`BrowserPromptPlayer`, `WallClockPromptRunner`). `open_run` still returns a `FakeRun`
  for a run with no worker (a build without the stub). A JavaScript `SyncModelWorker`
  reaches the builders as a JsProxy and is wrapped with `pyodide_glue.attach`
  (`page.channel`); a Python worker is used as it is.
- **Warm-up order** (`ZeosRun.warm`): `machine.prewarm()`, then `driver.start()` (which
  creates the pilot's context), then `runner.warm()`, all before the clock starts.
  `close()` closes the machine (`ZeosDriver.close`) or the prompt player, which cancel
  and drain a step in flight and destroy every context.
- **`RunResult.extras`** (new, `dict[str, object]`, default `{}`) carries what the table
  above does not: warm-up and prewarm time, overrun and cancel percentiles, step totals,
  the biggest step, pilot moves and how many were valid, `exit`s, kernel faults, splices
  (counted by the machine and in the journal), context sizes, and the longest gap
  between pilot moves with the splices inside it. `finished.result.extras` carries it to
  the page.
- **Tuning without a rebuild.** `page.configure_json(text)` replaces
  `ZEOS_MACHINE_OPTIONS`, `PROMPT_OPTIONS` and `ZEOS_KERNEL_OPTIONS` from
  `{"zeos": {...}, "prompt": {...}, "kernel": {...}}`; `start` carries an optional
  `tune` string (the page's `?tune=` query parameter) and `si_worker.js` applies it
  before `open_run`. `kernel` overrides `KernelConfig` fields on the zeos arm's kernel,
  laid over `page.DEFAULT_ZEOS_KERNEL_OPTIONS`, which is empty: the kernel is the
  native one.
- **Starvation.** The zeos arm's kernel runs at its own `starvation_limit` (8). The
  kernel counts a job's preemptions since it last made progress, a request carried out
  (`docs/notes/starvation-progress.md`); the pilot makes progress with each move and
  each read of a board, so the reflex's preemptions do not retire it.
- **The prompt arm's reply** opens with an empty think block
  (`prompt_player.ASSISTANT_OPEN`), and a newline ends a reply only once it holds more
  than a code fence (see the machine notes' prompt-arm item, which predates this).
  Both stubs count only known tokens of the open assistant turn, so the block does not
  advance their script.
- **A synchronous call behind a step in flight waits up to one chunk.** The kernel's
  inject of a resume notice and the pager's splice call the worker synchronously, so
  they settle first; on WebGPU at chunk 128 that is up to about 0.7-1.5 s (`cancel_ms`,
  rising with the context), and
  the loop stalls for it.
- **Wave 3 defaults** (measured on WebGPU with no other GPU user, README *Measured*):
  `DEFAULT_MAX_CHUNK` is 128, superseding the machine notes' 256; the pilot's context
  window is 32,768 (`page.DEFAULT_PILOT_CONTEXT`, laid over the native `goals/pilot.md`'s
  4096 on the built kernel, so the native case is unchanged and there is no copy of
  it), because a pager splice makes the pilot recompute everything after the splice
  point; `?tune={"context":{...}}` changes it. `stall_ms` stays 5 and the mask prewarm
  stays on.
- **The prompt arm renders each turn once** (`BrowserPromptPlayer.render_turn` caches
  the request's rendering): the native `record` rendered it a second time, and the
  `lead` view's aim search behind it takes up to ~360 ms under Pyodide.
- **A worker that stops answering** (review round). `_settle` and the prompt arm's
  `close` wait at most `settle_timeout_s` (`contracts.DEFAULT_SETTLE_TIMEOUT_S`, 30 s)
  for a cancelled step. On expiry they raise `contracts.ChannelBroken` and mark the
  channel broken (`contracts.mark_broken`: `PyodideAsyncWorker.markBroken` sets the
  JavaScript `SyncModelWorker.broken`, so later calls and later runs throw `model channel
  unusable`, as the channel's own call timeout does). The machine then refuses every
  worker call, and its `close` forgets the contexts without waiting again. A lost WebGPU
  device therefore ends Stop with an error on the page rather than a hang.
- **`invalidate` spares a command begun after the preemption** (review round). The
  driver invalidates at the end of a batch for each preemption in it; a pilot preempted
  and resumed inside one batch may already be saying a fresh command. The machine keeps
  an epoch that advances whenever `decode` switches job, notes the epoch at which each
  job was last switched away from, and the epoch at which its current command began; an
  `invalidate` whose job's command (in flight, or finished with its read pending) began
  after that is a no-op, counted in `invalidations_skipped`. A job never descheduled is
  invalidated as before.
- **Pump slices** (review round). The zeos loop pumps the kernel at most
  `runner.PUMP_SLICE_S` (20 ms) at a time, and the prompt loop polls for at most that
  long, so a Stop lands within a slice rather than within a tick. A batch that ends
  before its slice with nothing written still means every job is blocked, and the loop
  sleeps to the tick.
- **Automation hooks.** `app.js` keeps `window.siRuns` (each `finished`, less journal
  and payload), `window.siErrors` and `window.siPhase`, which `tests/si_webgpu.mjs`
  reads.
