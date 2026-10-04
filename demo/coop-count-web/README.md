# coop-count-web — the coop-count demo in a static web page

The ZEOS kernel and the coop-count demo, running inside [Pyodide](https://pyodide.org) in a
browser, with no server beyond a file host. A member of the repository's uv workspace.

The page loads Pyodide, installs the `zeos` wheel and this package's wheel, writes the
coop-count cases into Pyodide's in-memory filesystem, lints the chosen case and runs it.
The run's output panel has two tabs, both filled as the run streams. *user view*, the
default, shows the transcript `zeos-count run` prints to a terminal (`counter-a  say 1`,
`reset-count ──▶ count.progress_a 51`, `counter-a  ... waiting on count.b2a`), collected
from the same seat callbacks and journal events the CLI prints from
(`zeos_coop_count_web.transcript`), so it never alters the journal. *journal* shows the
journal itself, one JSON line per event. The ZEOS debugger draws the
case's wiring and then the finished run, and the journal can be downloaded as the
`.jsonl` file `zeos-count run --journal` would have written. A keypress on the page is the
console's interrupt.

```bash
uv sync --all-packages                       # from the repository root
uv run python demo/coop-count-web/build.py   # builds the wheels into web/dist/
uv run python demo/coop-count-web/serve.py   # web/dist/ on port 8765, cross-origin isolated
# open http://localhost:8765/
```

Any static file host serves `web/dist/` as well; the page fetches Pyodide from the
jsDelivr CDN, so the browser needs a network connection the first time. The model
machine (see *The model machine* below) needs the page cross-origin isolated: `serve.py`
sends the two headers that takes, and on a host that cannot send them
`coi_serviceworker.js` adds them in the browser after one reload. The scripted and stub
machines run on any host, `python -m http.server` included.

## What runs

The kernel is pure Python over the standard library and PyYAML, which Pyodide ships. It
reads no clock, starts no thread and opens no socket, and the driver injects time as one
virtual millisecond per tick, so a run in the browser writes the same journal bytes as
the same run under CPython. `tests/test_pyodide_determinism.py` checks exactly that.

Two machines are offered for a case:

| machine | what answers each decode |
|---|---|
| scripted | `CommandSeat` over `TapeSource`: each descriptor's `script:` tape, one word per decode. This is `zeos-count run --machine scripted`. |
| JsMachine | `JsMachine` over the stub worker in `web/stub_worker.js`, which plays the same tapes through every method of the JavaScript seam. |
| JsMachine — Qwen2.5-0.5B | `JsMachine` over the model worker in `web/transformers_worker.js`: a language model decoding under the kernel, with measured attention. Offered when the build found an export. |

On `coop-count-scripted` the two decode the same words and their journals hold the same
events in the same order, with two exceptions, both about attention. The seat's mask
hides the blocks a job has written since the kernel's last mask refresh, so the kernel
journals the job's attention to its own open segment as denied (`mask.denied`, 30 times
in this run) and leaves that segment out of the working set; `JsMachine` treats those
blocks as visible (see *the mask's horizon* below), so it has no `mask.denied` events
and two `vm.working_set` events count one more segment. The page's table of finished
runs shows each journal's SHA-256, so two runs of the same machine with the same inputs
show the same digest. `coop-count-pipe` and
`coop-count-vector` carry no tapes, so they need a model: on the scripted and stub
machines the page lints them and draws their wiring and does not offer to run them; the
model machine runs them, for 400 ticks, since a live counter counts for ever.

**The console.** Space writes `attention` to `keys.interrupt`, which the case's vector
table binds to `reset-count` at priority 5, so the handler preempts whichever counter is
running; then a number and Enter write to `keys.number`, which the parked handler reads.
As in the terminal, once space has sent an interrupt, further presses only move to the
number field until a number is sent; and an interrupt is withheld if, at the turn that
would deliver it, `reset-count` is already parked on `keys.number`
(`LiveRun.press(..., unless_waiting_on="keys.number")`, decided from the kernel). The handler's tape writes 51 whatever number is typed, because
a tape cannot read. Untick *play `events.jsonl`* to press the key yourself; with it
ticked, the case's own schedule presses it at 39 ms.

**The run loop.** `zeos_coop_count_web.live.LiveRun` is the `zeos-count` run loop cut
into single turns: deliver what the schedule has due, deliver what the page queued,
tick, reap, add one virtual millisecond. With no presses its journal is the CLI's,
which `tests/test_js_machine_case.py` checks against `zeos_coop_count.cli` when
llama-cpp-python is installed. A press is delivered at the start of the next turn, the
place the CLI delivers a scheduled event, so pressing at the turns `events.jsonl` names
reproduces the scheduled run exactly. While a job is parked on a device pipe the run
stays open, idling, until a press arrives or *stop* is pressed.

**Responsiveness.** Pyodide runs in a Web Worker (`web/pyodide_worker.js`), so loading
it, building the debugger's payload or a slow machine never freezes the page. The worker
steps the run one turn per timer tick, and its event loop is free between turns, which is
when a keypress posted from the page arrives. The speed control sets the delay between
turns; it has no effect on the journal, which counts virtual time only.

**The debugger** is `src/zeos/debugger/static/`, copied into `web/dist/debugger/` by
`build.py`, fed a payload from `zeos.debugger.payload.build_payload` in Pyodide. The page
assembles it as `zeos.debugger.server.page` does, substituting the stylesheet, the
script and the data into the shell, and shows it in an iframe; no server is involved.

## Pyodide

Pinned to **Pyodide 314.0.7** (Python 3.14.2), the current stable release when this was
written, in two places that `tests/test_page.py` holds equal: the CDN URL in
`web/pyodide_worker.js` and the npm dependency in `package.json`. This release refuses to
load in a classic Web Worker, so the worker is a module worker.

## Why a separate package

`zeos-coop-count` declares `llama-cpp-python`, `huggingface-hub` and `anthropic` as hard
dependencies, none of which installs under Pyodide, and its `machine.py`, `boot.py` and
`cli.py` import llama.cpp at module level. Rather than install that wheel without its
dependencies and import around it, this package depends on `zeos` alone and carries what
the browser needs. The scripted seat is `zeos.machine.seat`, the cases are read from
`demo/coop-count/cases/`, and the terminal-only `keyboard.py` is replaced by the page's
console. `JsMachine`'s word-to-token bookkeeping and ChatML framing are `LlamaMachine`'s,
written again over the worker interface, since the module that holds `LlamaMachine`
imports llama.cpp. `demo/coop-count` is unchanged.

## The JavaScript machine seam

`zeos_coop_count_web.js_machine.JsMachine` is a `SyscallSeat` and a full
`MachineBackend`. It keeps, per job, the kernel's words, the model token ids behind them
and the chat framing folded into their spans, and hands every token-level operation to a
worker object passed to its constructor: a JavaScript object reached through Pyodide in
the browser, or a Python object with the same methods under CPython.

```python
from zeos_coop_count_web.js_machine import JsMachine

JsMachine(
    worker,  # a ZeosModelWorker
    bridge=None,  # PythonBridge() by default; PyodideBridge() for a JS object
    descriptors=...,
    valued=...,  # from zeos.machine.seat.seat_maps(...)
    block_size=16,  # the kernel's block size, in kernel words
    chat_template="chatml",  # or None
)
```

The worker implements exactly this interface:

```
interface ZeosModelWorker {
  // Model identity and reserved IDs. Call once.
  info(): { blockSize: number; padId: number; controlIds: number[]; eosId: number;
            vocabSize: number };
  tokenize(text: string): Int32Array;        // no BOS, no special-token parsing
  piece(tokenId: number): string;            // the text of one token
  createContext(jobId: string): void;
  destroyContext(jobId: string): void;
  length(jobId: string): number;             // model tokens currently resident
  // Prefill. Appends ids to the context and runs the forward pass for them.
  append(jobId: string, ids: Int32Array): void;
  // Drop every token at position >= n and the KV behind it.
  truncate(jobId: string, n: number): void;
  // Copy parent's tokens and KV into a fresh context for child.
  fork(parentId: string, childId: string): void;
  // One greedy decode step. Returns the chosen id and measured attention mass per
  // KV block for this step, summed over layers and heads and normalised so the
  // values sum to 1.0 over blocks that received attention. A backend that cannot
  // measure returns attention = null.
  decodeStep(jobId: string, opts: {
    allowedBlocks: Uint8Array | null;       // 1 = may attend, indexed by block; null = all
    allowedTokens: Uint8Array | null;       // 1 = may emit, indexed by token id; null = all
  }): { tokenId: number; attention: Float32Array | null };
}
```

How `JsMachine` uses it, which is what an implementation has to get right:

- **Context ids** are `"<job>:<descriptor>"`, opaque to a worker (the stub reads the
  descriptor from them to pick a tape). `createContext` and `fork` are given only an id
  the worker does not hold; every other method only one it does.
- **Residency.** The worker's tokens are always a prefix of `JsMachine`'s ids. Before
  each `decodeStep` it appends whatever lies past `length()`, so a step never meets an
  empty context; `trunc`, `splice` and chat framing inserted mid-context `truncate`
  first and let the next decode re-append the tail.
- **A decode step leaves the context unchanged.** The chosen id is appended by the
  `append` before the next step. The step computes the next-token distribution at the
  last resident position, attending only the allowed blocks, and reports that query's
  attention.
- **Blocks** in `allowedBlocks` and `attention` are the worker's: `info().blockSize`
  model tokens each, one entry per block of the resident context. They are not the
  kernel's blocks, which count kernel words, and a word can be several tokens. A worker
  block is allowed only if every word with a token in it lies in a kernel block the
  kernel's mask allows, so a block straddling a hidden segment is hidden whole. Measured
  mass on a worker block is shared among the kernel blocks with tokens in it, in
  proportion to how many. Both are exact when every word is one token, as with the stub.
  `JsMachine` refuses attention that is negative, does not sum to 1.0 within 1e-3, or
  has the wrong number of entries (`WorkerViolation`), the unit check the
  `DecodeResult` docstring calls normative; and attention on a block the step was sent
  as 0 (`MaskViolation`).
- **The mask's horizon.** The kernel builds its mask from the segments that exist when
  it installs it — at each block boundary, inject, fork and splice — and decodes on
  until the next refresh, so blocks the job has written into since are not in it. Those
  blocks, at or past the kernel block count when the mask was installed, can only hold
  the job's own decoded tokens and padding, because every foreign arrival refreshes the
  mask first. `JsMachine` allows them: hiding them would hide the position doing the
  attending. `visible_blocks` reports the same set, so the attention the kernel sums is
  the attention the model could have paid. A `trunc` below the horizon lowers it.
- **Reserved ids.** `allowedTokens` has one entry per vocabulary id. The pad and
  end-of-sequence ids are always 0, the `controlIds` are 0 unless the kernel enabled
  control tokens for the step, and the grammar mask below zeroes the rest it forbids.
  An id the mask refused raises (`ControlTokenViolation` for a control id,
  `WorkerViolation` otherwise) rather than reaching the kernel.
- **Vocabulary size** is `info().vocabSize`. `allowedTokens` has that many entries, and
  `piece` is called once for every id below it when `JsMachine` is constructed. The pad,
  end-of-sequence and control ids must lie below it (`WorkerViolation` otherwise).
- **Padding** is `info().padId`, one per kernel pad word.
- **Chat framing.** With `chat_template="chatml"` (the default, as for `LlamaMachine`)
  the prompt opens a user turn, the first decode closes it and opens the assistant's,
  and each arrival mid-turn closes the assistant's turn and opens a user turn. Since
  `tokenize` parses no special tokens, the turn markers are found among `controlIds`:
  the id whose `piece` is exactly `<|im_start|>` and the one whose piece is
  `<|im_end|>`. The text between them (`user\n`, `assistant\n`) goes through
  `tokenize`. **A worker for a ChatML model, such as Qwen2.5-Instruct, must list both
  markers in `controlIds` and return their literal text from `piece`**; `JsMachine`
  refuses to start otherwise.

### The grammar mask

The llama seat compiles the syscall ABI to GBNF and lets llama.cpp's sampler walk it.
`zeos_coop_count_web.token_mask` walks the same language in Python and sends the result
as `allowedTokens`: before each step it marks every vocabulary id whose piece keeps the
current round a prefix of some valid round. The language is the one `build_grammar`
renders — request-free verbs, then one call; only the aliases the descriptor binds; a
number with no leading zeros for an actuator; one to `max_text` characters for a
payload — except that a round may open with one space and the space after a terminator
opens the next command, because the seat speaks every word after a job's first with a
leading space. A terminator longer than one character is matched as a literal, and a
payload excludes each of its characters, as the GBNF character class does. A piece that
completes two terminators is refused: the seat's parser splits at the first terminator
only, so the second command would never reach the kernel. It is a small automaton over
characters whose states are integer tuples.

Cost: a mask is computed once per (descriptor, round state, control flag) by one pass
over the vocabulary, at about one automaton step per character of each piece, and
cached; every later step from a seen state is a dictionary lookup plus one copy of a
vocabulary-sized `Uint8Array` across the FFI. A counting run revisits a few dozen
states. Reading the vocabulary costs one `piece` call per id when `JsMachine` is
constructed.

### The stub worker and the Python fake

`web/stub_worker.js` implements `ZeosModelWorker` with no model, to exercise every
method of the seam from Python under Pyodide. `zeos_coop_count_web.fake_worker.FakeWorker`
is the same worker in Python, for tests under CPython. Both have a fixed vocabulary (five
reserved tokens, then every word of the tapes and of the ChatML headers, with and
without a leading space), a whitespace tokenizer, a tape per descriptor taken from the
case's `emit` steps, and no attention. Tapes must be ASCII and both refuse any other
text, since Python and JavaScript split and sort non-ASCII text differently. Each step gives the next word of the current
command, split as `zeos.machine.seat.words_of` splits it, checks that `allowedBlocks`
and `allowedTokens` are sized to the context and the vocabulary, and refuses a word the
token mask forbids.

The Pyodide determinism test runs `coop-count-scripted` through `JsMachine` over the
stub in Pyodide and over the fake in CPython and requires the same bytes, so the two
cannot drift apart unnoticed. To run `JsMachine` over the fake:

```python
from zeos.descriptor.loader import load_case
from zeos.machine.seat import seat_maps
from zeos_coop_count_web.fake_worker import FakeWorker, tapes_from_scripts
from zeos_coop_count_web.js_machine import JsMachine

bundle = load_case(path)
descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
machine = JsMachine(
    FakeWorker(tapes_from_scripts(bundle.scripts)), descriptors=descriptors, valued=valued
)
```

In the browser the page builds the stub with `createStubWorker(tapes)` and passes it to
`zeos_coop_count_web.page.open_run(case_dir, "js", worker=stub)`, which wraps it in
`PyodideBridge`; any other object implementing the interface goes the same way.

## The model machine

The third machine on the page is `JsMachine` over a real language model:
Qwen2.5-0.5B-Instruct, exported to ONNX by `export/export_model.py` and run by ONNX
Runtime Web in `web/transformers_worker.js`, with the tokenizer Transformers.js uses. It
supplies all four things ZEOS asks of a serving stack: the allowed-block mask is applied
inside attention, before the softmax, in every layer; each decode step's attention is
measured, summed over layers and heads and normalised, per KV position; the control ids
are reserved in the sampler; and the syscall grammar constrains every step. The kernel
therefore receives measured attention, `DecodeResult.attention`, where every other
backend in the repository gives it a hint, and integrity demotes on it.

### The model

**Qwen2.5-0.5B-Instruct.** It speaks ChatML, which is what `JsMachine` frames prompts in
(`chat_template="chatml"`, as for `LlamaMachine`), its architecture is a plain
transformer whose KV cache can be cut at any position, which paging needs, and at int8 it
is a 632 MB download that a browser's WebAssembly heap holds twice over. One decode step
takes about 70 ms on onnxruntime-web's WebAssembly backend under Node and about 90 ms in
Chrome, one thread each. Qwen2.5-1.5B-Instruct exports with the same script
(`--model Qwen/Qwen2.5-1.5B-Instruct`, 1.8 GB) and runs from Node at about 210 ms a step;
it is not what the page loads.

Neither follows the coop-count procedure as `Qwen3.5-4B` does under llama.cpp. On
`coop-count-pipe` the 0.5B counter-a says 1 to 10, records 11 rather than 10, wakes its
peer and sleeps; counter-b, woken, says `1000000000000000` over and over until the run is
cut off; and the handler, preempting it at the keypress, says `"attention" and` and exits
instead of reading the console. The 1.5B keeps the turn structure for 400 ticks --
counter-a records 0, 10, 20, ..., wakes its peer and sleeps, and counter-b wakes it back
-- but neither says a number, and its handler does not read the console either. What the
kernel gets from either is real: measured attention, enforced masks, and commands that
only the grammar allows.

### The export

`export/export_model.py` writes, into `models/<name>-zeos-<quant>/` (gitignored):

- `prefill.onnx` -- new token ids, the past KV and a per-position key mask in; the new
  positions' KV out. The library's default attention (`scaled_dot_product_attention`),
  which the exporter lowers to plain operators. No logits: the worker never needs them
  from a prefill.
- `decode.onnx` -- one token id, the past KV and an allowed mask per KV block in; the
  logits, the new position's KV and the step's attention per block out. Attention is
  computed explicitly (scores, mask, softmax) so the probabilities can be summed over
  heads and layers and divided by their product inside the graph; a masked key gets
  `-inf` before the softmax and so exactly zero after it. Block size is baked in at
  export and reported by `info().blockSize`; it is 1, so the attention vector has one
  entry per position and `JsMachine` sums it onto the kernel's blocks exactly, which a
  fixed multi-token block cannot do when a kernel word is several tokens.
- `weights.bin` -- every large initialiser of both graphs once, so the weights download
  once; `meta.json` -- shapes, reserved ids and file digests; the tokenizer's two files.

The graphs are a rewrite of Qwen2 in PyTorch over a position-major KV layout
(`[positions, layers, 2, kv heads, head dim]`), so that appending, truncating and
copying a job's cache are contiguous copies in JavaScript; the script holds the rewrite
to `transformers`' own logits before it exports (drift 7e-5 at full precision), and
checks the exported decode graph under ONNX Runtime CPU: a masked block receives exactly
zero, the rest sum to one. `tests/test_exported_graph.py` repeats that check.

**Quantisation.** The default, `int8`, is ONNX Runtime's dynamic quantisation of every
weight matmul (per-channel int8 weights, int8 activations computed per step), plus a
per-row int8 embedding table. It was chosen by measurement: on onnxruntime-web 1.30's
WebAssembly backend a decode step takes 68 ms at int8 and 920 ms with 4-bit
`MatMulNBits` weights (`--quant q4`), which that backend dequantises on every run;
`q4a8` is no faster. The int8 embedding moves the logits by at most 0.08 against full
precision, and the int8 matmuls change which token wins on free text: asked to count
from 1 to 10 in a chat turn, the q4 export answers with the digits and the int8 one with
a sentence about how it would. The coop-count behaviour above is the int8 export's.

The export needs PyTorch, `transformers`, `onnx` and ONNX Runtime, which are a
non-default dependency group of this package, so the root `uv sync --all-packages` never
installs them:

```bash
uv sync --all-packages --group export                     # from the repository root
uv run python demo/coop-count-web/export/export_model.py  # about a minute; 632 MB
uv sync --all-packages                                    # drop the export group again
```

The script downloads the model from the Hugging Face Hub into `models/<name>/` first.
Two exports from the same source are byte-identical.

### The worker

`web/transformers_worker.js` implements `ZeosModelWorker` over the two graphs, driving
them with onnxruntime-web directly rather than through Transformers.js: Transformers.js'
model classes run their own exports with their own cache, and these graphs have inputs
and outputs no model class knows (the masks, the attention vector) and a cache the
worker owns. The tokenizer is `@huggingface/tokenizers`, the library Transformers.js 4
tokenizes with, loaded from the export's `tokenizer.json`. `tokenize` runs its
normaliser, pre-tokeniser and BPE without the added-token splitter, so a literal
`<|im_end|>` in text stays text; it matches Hugging Face's `split_special_tokens=True`
token for token.

A job's cache is one growable `Float32Array`; `fork` copies it, `truncate` shortens it.
Its last token is pending, with no KV behind it, until a decode step feeds it through the
decode graph, so every step is exactly one forward pass of one token, and `append`
prefills everything before it in chunks of 256. A decode step does not append the id it
chooses (`JsMachine` does, before the next). The mask of a job's latest step is applied
to the keys of its next prefill too, so content appended later never attends a block
the kernel had hidden. The worker refuses an `allowedBlocks` shorter than the context, a
mask that hides every block, and an `allowedTokens` that allows nothing.

`info()` is `{blockSize: 1, padId: <|endoftext|>, controlIds: [<|im_start|>,
<|im_end|>, <|endoftext|>], eosId: <|im_end|>, vocabSize: 151665}`. `vocabSize` is the
tokenizer's vocabulary; the logits are 151,936 wide because the embedding is padded, and
no id past the tokenizer's is ever chosen.

The worker imports nothing: ONNX Runtime and the tokenizer class are handed to it, so the
same file runs in the page and under Node (`web/node_load.mjs`).

### Calling an asynchronous model synchronously

`JsMachine.decode` is synchronous, the kernel loop that calls it stays synchronous, and
ONNX Runtime Web's `session.run` returns a promise. The page resolves that with a second
thread and a `SharedArrayBuffer`:

- the page starts the model in its own module Web Worker (`web/model_thread.js`, started
  by `web/model_host.js`) and hands the Pyodide worker a `MessagePort` to it and the
  shared buffer;
- in the Pyodide worker, `SyncModelWorker` (`web/model_channel.js`) is the
  `ZeosModelWorker` `JsMachine` calls: each method posts the call on the port and blocks
  in `Atomics.wait` until the model thread has awaited the session and written the reply
  into the buffer (`web/frames.js` is the encoding, typed arrays as raw bytes).

Pyodide's own `run_sync` would have needed the kernel entered through an async call and
JavaScript Promise Integration, which not every browser ships; `Atomics.wait` works in
every current browser, inside a worker. It needs the page cross-origin isolated, which is
why `serve.py` and `coi_serviceworker.js` exist. The page starts the model thread rather
than the Pyodide worker because a worker started from inside a worker failed to start in
the Chromium this was tested in. Under Node the same channel runs with a
`worker_threads` thread (`web/node_model_thread.mjs`), where `Atomics.wait` is allowed on
the main thread too.

**Backends.** The page offers WebAssembly and WebGPU and shows the one in use beside the
clock. WebAssembly runs one thread: a run's arithmetic is then a function of its inputs
alone (see *Determinism* below); with more threads the model thread did not finish
loading in the browser this was tested in. WebGPU is offered with the caveat in its
label. ONNX Runtime's WebGPU backend has no kernel for the int8 export's integer
matmuls, so they run on WebAssembly with a copy each way, about 1.2 s a step; the q4
export (`--quant q4`, 446 MB) runs on WebGPU at about 54 ms a step, and `build.py --model`
puts it on the page.

### The grammar

`JsMachine`'s Python token mask (`token_mask`) is kept. Against the Qwen vocabulary
(151,665 pieces) it is correct -- `tests/test_transformers_grammar.py` puts the model
after a context of injected text written to talk it out of the command language
(forged chat markers, a fake status line, prose demands) and every completed line parses
as a command, no control id is ever chosen and no `<` is ever emitted -- and it is fast
enough: building the mask for a round state the run has not met costs 0.11 s on average
under Pyodide (0.81 s at most, 0.06 s under CPython) and a cached state costs nothing,
so a counting round pays a few seconds once, against 90 ms a decode step for as long as
the run lasts.

### Running it

In the page: build with an export present and `npm install` done, serve with `serve.py`,
choose *JsMachine -- Qwen2.5-0.5B* and a case, and run. A run stops at 400 ticks, as
`demo/coop-count/run_all.py` stops a live seat.

From a shell, the same `JsMachine`, `LiveRun` and worker run under CPython with the model
in a Node child process (`zeos_coop_count_web.node_worker.NodeWorker`, over
`web/node_bridge.mjs`):

```bash
cd demo/coop-count-web
npm install
uv run python -m zeos_coop_count_web.node_run ../coop-count/cases/coop-count-pipe \
    --events ../coop-count/cases/coop-count-pipe/events.jsonl --journal pipe.jsonl
uv run python -m zeos_coop_count_web.evidence pipe.jsonl
```

`node_run` writes the journal and, beside it, `pipe.attention.jsonl`: one line per decode
step, keyed by the sequence number of its `machine.decode` event, with the measured mass
per kernel block and per segment. The journal has no field for attention -- its events
are the kernel's -- so the measurement is kept beside it, as `zeos.trace` keeps the raw
trace. `--theta-read` sets `KernelConfig.theta_read`, `--ring PIPE=RING` declares a pipe
at another ring for the run, and `--hide JOB:BLOCK` takes a kernel block out of every
mask a job is given. `evidence` reports what the measured attention did to integrity.

### Measured attention

`docs/evidence/README.md` is the write-up, with the journals it cites beside it: a
demotion whose `because` names an untrusted segment and the measured mass that crossed
`theta_read`; the same run at a `theta_read` above that mass, with no demotion; and a
block hidden through `set_mask` that measures exactly zero on every step. In short: the
handler of `coop-count-pipe`, with the console declared untrusted, put 0.36 of its first
step's attention on the keypress and was demoted for it; and the block a job is writing
into gets 0.17 of each step's attention on average.

### Determinism

On the WebAssembly backend the page's journal is byte for byte the one the same run
writes under CPython with the worker in Node, because both run the same onnxruntime-web
WebAssembly binary on one thread and the kernel reads no clock. In Chrome 152 (the
desktop app's browser pane, Apple silicon) and under Node 22, 400 ticks of each case:

| case | SHA-256 of the journal, page and Node alike |
|---|---|
| `coop-count-scripted` (with its schedule) | `fcd01b7ac8c159def1946b8f37e46060cdd8c34b8f2a31a8290c5910f46baa68` |
| `coop-count-pipe` (with its schedule) | `7d851f720562c396db3c4ee5bee10dc21453f8587405946c430f5cae068de6f9` |
| `coop-count-vector` | `f8bf68a3e554f155528cdce373618b95666ad5808de3da813734d9eed6b90578` |

`coop-count-scripted` ran twice in the page, in two page loads, with the same digest.
The same holds with Pyodide in Node over the model thread: `tests/test_pyodide_model.py`
runs 40 ticks of `coop-count-scripted` that way and requires CPython's bytes.

WebGPU was checked on one device only, with the q4 export, on twenty greedy steps of a
short chat prompt: three runs in the same browser chose the same tokens and reported
bit-identical attention, and the tokens were WebAssembly's, but the attention was not
bit-identical to WebAssembly's on the same export. Its floating-point order is the
GPU's, so a journal made on WebGPU is not claimed to match one made on another device,
or on WebAssembly; the determinism claim is WebAssembly's.

## Tests

```bash
cd demo/coop-count-web
npm install        # once: Pyodide for Node, and PyYAML cached beside it
uv run pytest      # or, from the root: uv run pytest demo/coop-count-web/tests
```

- `test_js_machine_contract.py` — the coop-count machine contract suite, run against
  the scripted backend and `JsMachine` over the fake, and again over a fake that
  measures attention, so the two attention clauses bind.
- `test_js_machine_case.py` — `coop-count-scripted` through `JsMachine` and `LiveRun`:
  the interrupt preempts a counter and it resumes dirty, the journal differs from the
  seat's only in attention, a press lands where the scheduled event would, a parked
  handler holds the run open, and a second interrupt while it is parked is withheld.
- `test_js_machine_seam.py` and `test_token_mask.py` — the seam's obligations and
  refusals, and the language the mask admits.
- `test_page.py` — the page's Python half, and that `build.py` assembles every file the
  page fetches.
- `test_exported_graph.py` — the exported decode graph under ONNX Runtime CPU: a masked
  block receives exactly zero, the rest sum to one, and masking changes the logits.
  Needs the export and the `export` dependency group.
- `test_transformers_grammar.py` — the token mask against the real model after
  adversarial injected text: only commands, never a control id.
- `test_node_run.py` — a kernel run over the real model: one attention line per decode
  step, each summing to one, and a block hidden through `set_mask` at exactly zero.
- `tests/js/transformers_worker.test.mjs` (`node --test tests/js/*.test.mjs`) — the
  worker's interface under Node, directly and through the synchronous channel.
- `test_js_machine_contract.py` also runs the contract suite over the real model worker
  (`js-transformers`), driven from Node.
- `test_pyodide_model.py` — the page's arrangement under Node (Pyodide, and the model
  on a thread of its own behind `SyncModelWorker`) writes CPython's journal.
- These five and the `js-transformers` backend skip without Node, `npm install` or the
  export.
- `test_pyodide_determinism.py` — the smoke fixture, the seat and the stub, each under
  Pyodide in Node and under CPython, compared byte for byte. It builds both wheels with
  `uv build`, installs them by unpacking as Pyodide installs a pure wheel, and copies the
  cases into the in-memory filesystem as the page does. It skips when Node, the npm
  package or the cached PyYAML is missing; `ZEOS_PYODIDE_DIR` points it at a Pyodide
  installed elsewhere.

`uv build` (used by `build.py`, the determinism test and `test_page.py`) fetches the
hatchling build backend from PyPI the first time it runs, so those need a network once.
