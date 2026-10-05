# coop-count-web — the coop-count demo in a static web page

The ZEOS kernel and the coop-count demo, running inside [Pyodide](https://pyodide.org) in a
browser, with no server beyond a file host. A member of the repository's uv workspace.

The page loads Pyodide, installs the `zeos` wheel and this package's wheel, writes the
coop-count cases into Pyodide's in-memory filesystem, lints the chosen case and runs it.
The run's output panel has two tabs, both filled as the run streams. *transcript*, the
default, shows the transcript `zeos-count run` prints to a terminal (`counter-a  say 1`,
`reset-count ──▶ count.progress_a 51`, `counter-a  ... waiting on count.b2a`), collected
from the same seat callbacks and journal events the CLI prints from
(`zeos_coop_count_web.transcript`), so it never alters the journal. *event log (JSON)* shows the
journal itself, one JSON line per event. The ZEOS debugger draws the
case's wiring and then the finished run, and the journal can be downloaded as the
`.jsonl` file `zeos-count run --journal` would have written. The *interrupt* button on the page is the
console's interrupt.

## Getting it running

**You need** [uv](https://docs.astral.sh/uv/), Node.js with npm (tested with Node 22),
and a browser with WebGPU (tested in Chrome on an Apple M1 Max). Exporting the model
once needs about 20 GB of free memory, about 30 GB of free disk while it runs, and a
network connection to download 8.7 GB of weights from the Hugging Face Hub.

Every command runs from the repository root.

```bash
# 1. Python packages, including the export group (PyTorch, transformers, onnx).
uv sync --all-packages --group export

# 2. ONNX Runtime Web and the tokenizer, which build.py copies into the page.
npm install --prefix demo/coop-count-web

# 3. Export Qwen3.5-4B to the graph the page runs, quantised to q4.
#    Downloads to demo/coop-count-web/models/Qwen3.5-4B/ and writes
#    demo/coop-count-web/models/Qwen3.5-4B-zeos-q4/ (3.3 GB). About five minutes
#    after the download.
uv run python demo/coop-count-web/export/export_model.py

# 4. Optional: drop PyTorch and the rest of the export group again.
uv sync --all-packages

# 5. Assemble the page into demo/coop-count-web/web/dist/.
uv run python demo/coop-count-web/build.py

# 6. Serve it, cross-origin isolated, on port 8765.
uv run python demo/coop-count-web/serve.py
```

Open <http://localhost:8765/> and press **run**. The defaults are the language model on
the GPU, the scenario `coop-count-scripted`, and *press the keys automatically*, which
sends the interrupt at 39 ms of virtual time. The first run downloads the 3.3 GB of
weights into the browser and prefills the first prompt; on the machine above the first
command came about 35 s after pressing run, and each decode step took about 220 ms.
The *transcript* tab shows the counters counting and the interrupt handler resetting
the count.

`build.py` prints the model it put on the page. If it says `model: none`, step 2 or 3
is missing. The page then still runs every scenario that has recorded answers, and
offers the language model as *(not in this build)*. To offer another export, pass
`--model`, for example `--model demo/coop-count-web/models/Qwen3.5-2B-zeos-q4`.

**Serving it elsewhere.** `web/dist/models/` is a symbolic link to the export, so copy
the export in with `build.py --copy-model` before uploading `web/dist/` to a file host.
The page fetches Pyodide from the jsDelivr CDN, so the browser needs a network
connection the first time. The language model needs the page cross-origin isolated:
`serve.py` sends the two headers that takes, and on a host that cannot send them
`coi_serviceworker.js` adds them in the browser after one reload. The recorded-answer
machines run on any host, `python -m http.server` included.

## What runs

The kernel is pure Python over the standard library and PyYAML, which Pyodide ships. It
reads no clock, starts no thread and opens no socket, and the driver injects time as one
virtual millisecond per tick, so a run in the browser writes the same journal bytes as
the same run under CPython. `tests/test_pyodide_determinism.py` checks exactly that.

Three machines are offered for a case, under *answered by*:

| on the page | machine | what answers each decode |
|---|---|---|
| Qwen3.5-4B language model, in your browser | `transformers` | `JsMachine` over the model worker in `web/transformers_worker.js`: a language model decoding under the kernel, with measured attention. The default; offered when the build found an export. |
| recorded answers (no model) | `scripted` | `CommandSeat` over `TapeSource`: each descriptor's `script:` tape, one word per decode. This is `zeos-count run --machine scripted`. |
| recorded answers, via the JavaScript model interface | `stub` | `JsMachine` over the stub worker in `web/stub_worker.js`, which plays the same tapes through every method of the JavaScript seam. |

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

**The console.** The *interrupt* button writes `attention` to `keys.interrupt`, which the case's vector
table binds to `reset-count` at priority 5, so the handler preempts whichever counter is
running; then a number and Enter write to `keys.number`, which the parked handler reads.
As in the terminal, once an interrupt has been sent, further presses only move to the
number field until a number is sent; and an interrupt is withheld if, at the turn that
would deliver it, `reset-count` is already parked on `keys.number`
(`LiveRun.press(..., unless_waiting_on="keys.number")`, decided from the kernel). The handler's tape writes 51 whatever number is typed, because
a tape cannot read. Untick *press the keys automatically* (which plays the case's `events.jsonl`) to press the button yourself; with it
ticked, the case's own schedule presses it at 39 ms.

**The run loop.** `zeos_coop_count_web.live.LiveRun` is the `zeos-count` run loop cut
into single turns: deliver what the schedule has due, deliver what the page queued,
tick, reap, add one virtual millisecond. With no presses its journal is the CLI's,
which `tests/test_js_machine_case.py` checks against `zeos_coop_count.cli` when
llama-cpp-python is installed. A press is delivered at the start of the next turn, the
place the CLI delivers a scheduled event, so pressing at the turns `events.jsonl` names
reproduces the scheduled run exactly. While a job is parked on a device pipe the run
stays open, idling, until a press arrives or *stop* is pressed.

**What the model is doing.** A *model* line under the run buttons follows the model
thread: the download as a whole (the export's `meta.json` lists every file's size), the
session ONNX Runtime builds once the files are in, and then every run of the graph --
which prompt is being prefilled and how far, with an estimate of the time left, or which
decode step is in flight -- with the mean step time once steps have run. The kernel is
blocked while a step runs, so this line is the one that moves when the journal does not.
The thread reports each run of the graph through `TransformersWorker`'s `onActivity`
callback, which the page sets and Node leaves unset.

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
  `tokenize`. **A worker for a ChatML model, such as Qwen3.5, must list both
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

The page's default machine is `JsMachine` over a real language model: Qwen3.5-4B at q4
(Qwen3.5-2B with `build.py --model`), exported to ONNX by `export/export_model.py` and run by ONNX Runtime Web in
`web/transformers_worker.js`, with the tokenizer Transformers.js uses. It supplies all
four things ZEOS asks of a serving stack: the allowed-block mask is applied inside the
forward pass, in every layer (before the softmax in the softmax layers; see *The
architecture* for the linear-attention ones); each decode step's attention is measured,
averaged over the softmax layers' heads and normalised, per position; the control ids
are reserved in the sampler; and the syscall grammar constrains every step. The kernel
therefore receives measured attention, `DecodeResult.attention`, where every other
backend in the repository gives it a hint, and integrity demotes on it.

### The model

The page offers Qwen3.5-4B at q4 by default; *Getting it running* gives its timings.
In its first run on `coop-count-scripted` the handler did what the procedure says: it
read 51 from the console, wrote it to both counters' progress, and exited, and the
preempted counter resumed. The counters still stray from the procedure: counter-a
started again from 1 after the reset and counted past 10 without waking its peer. The
measurements and runs below are of the 2B, mostly at int8 on WebAssembly.

**Qwen3.5-2B** (`Qwen/Qwen3.5-2B`, the post-trained model; its base is
`Qwen3.5-2B-Base`). It speaks ChatML, which is what `JsMachine` frames prompts in
(`chat_template="chatml"`, as for `LlamaMachine`), and it is the smaller sibling of
Qwen3.5-4B, which follows the coop-count procedure under llama.cpp. Its architecture is a
hybrid of linear and softmax attention, which the export handles as described in *The
architecture* below. At int8 it is a 2.4 GB download in three files; one decode step
takes about 316 ms on onnxruntime-web's WebAssembly backend under Node, one thread, and
a run in Chrome takes within about 10% of the same run under Node. A prefill takes about
83 ms a token, and `JsMachine` prefills a job's prompt (about 1,400 tokens) before its
first decode, so the first command comes about two minutes after *run*, and each job's
first turn pays the same again.

On `coop-count-pipe`, with its `events.jsonl`, it gets further than the Qwen2.5 models
did and then deadlocks. Counter-a does its first turn exactly as the procedure says:
`say 1` to `say 10`, `write tools 10`, `write stdout go`, `read stdin`. Counter-b, woken
with the status line reading 10, says `11` and then records it -- `write tools 11`,
`write stdout go`, `read stdin` -- instead of counting on to 20. Counter-a, woken with a
RESUME notice (`count.b: 0 -> 11`), goes straight back to `read stdin` instead of
counting. The handler, preempting at the keypress, does exactly its four commands: `read
stdin`, then `write tools 500`, `write peer 500` with the number that arrived, and `exit`.
Then both counters are asleep on each other's pipe, nothing is left to deliver, and the
run ends quiescent after 99 ticks (95 decode steps), well short of the 400-tick cap.
Counter-b's mistake is the model's, not the quantisation's: at full precision it too
prefers `write` to `say` after `say 11;`.

The Qwen2.5 models this page ran before did worse. The 0.5B counter-a said 1 to 10,
recorded 11 rather than 10, woke its peer and slept; counter-b, woken, said
`1000000000000000` over and over until the run was cut off; and the handler said
`"attention" and` and exited instead of reading the console. The 1.5B kept the turn
structure for 400 ticks, but neither counter said a number, and its handler did not read
the console either. Those exports were in the earlier two-graph format; the script still
exports Qwen2 models in the current format (`--model Qwen/Qwen2.5-0.5B-Instruct`: 632 MB,
72 ms a step under Node). What the kernel gets from any of them is real: measured
attention, enforced masks, and commands that only the grammar allows.

### The architecture

Qwen3.5-2B (`model_type` `qwen3_5`; `Qwen3_5ForConditionalGeneration` on the Hub, of
which only the text model is exported) is not a plain transformer. Against Qwen2.5:

- **Hybrid layers.** Of its 24 layers, 18 are Gated DeltaNet layers (linear attention:
  a 16-head recurrent state of 128 x 128 per head, updated by the gated delta rule) and
  6, every fourth, are softmax attention (8 query heads, 2 KV heads, head dim 256).
- **A short causal convolution** (kernel 4) over each DeltaNet layer's q, k and v
  projections, so a position's keys depend on the three positions before it.
- **Gated softmax attention.** `q_proj` also produces a per-head output gate
  (`sigmoid`), q and k are RMS-normalised per head, and rotary embedding covers only the
  first 64 of the 256 channels (`partial_rotary_factor` 0.25, theta 1e7; the multimodal
  rope sections coincide for text). No q/k/v biases.
- **Zero-centred RMSNorm**: every norm but the DeltaNet's gated one scales by
  `1 + weight`.
- Tied embeddings over a 248,320-row table (the tokenizer has 248,070 ids), and a
  multi-token-prediction head the export leaves out.

The DeltaNet state is not a per-position cache and cannot be cut at a position, which
paging needs; it can be *re-derived*. The rewrite therefore keeps the contract with three
cache parts: the softmax layers' KV, position-major as before (24 KB a position); each
DeltaNet layer's state (19 MB a context, whatever its length); and each DeltaNet layer's
last three convolution inputs (1.3 MB). The worker cuts by going back to a snapshot of
the state and running the tokens after it again (see *The worker*).

**The mask in a DeltaNet layer.** A hidden position is *skipped*: its write strength
(beta) and its decay are zeroed, so the state never takes it in, and its convolution taps
are zeroed, so no later position's keys read it. This is exact -- the layer's output is
what it would be with that position absent from its recurrence -- but it is a property of
the state, not of one query: the state a step reads must have been built under that
step's mask. The worker guarantees that by running the positions again from the first
one whose visibility changed. A softmax layer applies the mask before its softmax, as
before.

**Measured attention covers the six softmax layers.** A DeltaNet layer has no attention
distribution to measure: its output is the state read with the query. `attention` is the
last position's softmax probabilities averaged over the six softmax layers' 48 heads --
so a hidden position still gets exactly zero and the rest still sum to one -- and says
nothing of what the 18 DeltaNet layers read. Those layers are masked exactly, but their
reading is not in `DecodeResult.attention`; integrity demotion on this model rests on the
softmax layers alone.

### The export

`export/export_model.py` writes, into `models/<name>-zeos-<quant>/` (gitignored):

- `model.onnx` -- one graph for prefill and decode. Up to 16 new token ids, the cache (KV
  of the earlier positions, the DeltaNet state and convolution window) and a key mask
  over every position, past and new, in; the logits at the last position, the new
  positions' KV, the state and window after them, and the last position's attention per
  position out. Attention is computed explicitly (scores, mask, softmax); a hidden key
  gets `-inf` before the softmax and so exactly zero after it, and sees itself only when
  nothing else is visible to it, so its softmax is defined. Block size is 1, so the
  attention vector has one entry per position and `JsMachine` sums it onto the kernel's
  blocks exactly, which a fixed multi-token block cannot do when a kernel word is several
  tokens. One graph rather than a prefill and a decode graph keeps one copy of the
  weights in the WebAssembly heap.
- `weights-0.bin`, `weights-1.bin`, ... -- every large initialiser, in files of at most
  1 GiB: Node reads no file over 2 GiB whole, and the browser holds smaller buffers more
  readily. `meta.json` -- shapes, layer types, reserved ids, the weight files and every
  file's digest; the tokenizer's two files.

Over a chunk, a DeltaNet layer's pseudo-values solve a unit lower-triangular system. The
graph solves it by repeated squaring, (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8),
which is exact for 16 positions and, in float32, accurate: at 16 positions the logits
stay within 8e-5 of `transformers`, where at 32 they already drift by 1.5. Hence the
16-position chunk (`maxChunk`).

The script holds the rewrite to `transformers`' own logits before it exports, prefilling
in chunks of 1, 7 and 16 so that the state and convolution window are carried across
runs (drift 4e-5 to 8e-5 at full precision, 0.05 to 0.07 with the int8 embedding), and
checks the exported graph under ONNX Runtime CPU: a hidden position receives exactly zero,
the rest sum to one, and hiding it changes the logits. `tests/test_exported_graph.py`
repeats that check. The script still exports Qwen2 instruct models (`model_type`
`qwen2`: every layer softmax, an empty state); the worker then cuts its cache directly.

**Quantisation.** The default, `q4`, is 4-bit weight-only `MatMulNBits`, the form
ONNX Runtime's WebGPU backend runs: 3.27 GB of weights for the 4B, 1.69 GB for the 2B.
`int8` is ONNX Runtime's dynamic quantisation of every
weight matmul (per-channel int8 weights, int8 activations computed per run), plus a
per-row int8 embedding table and a separately quantised output projection: 2.40 GB of
weights. It was chosen by measurement, under Node on onnxruntime-web 1.30's WebAssembly
backend, one thread, Apple M1 Max: for the 2B a decode step takes about 316 ms and a prefill about
83 ms a token; the export loads in 4.6 s and the process settles at 3.3 GB resident
(6.6 GB at its peak while loading, when the weight files and the WebAssembly heap both
hold the weights). The q4 export is many times slower than int8
per step on the WebAssembly backend (see *Backends*). fp32 (7.5 GB) does not fit a 4 GB WebAssembly
heap.

The chunk also bounds the int8 error. Dynamic quantisation gives a matmul's activations
one scale for the whole chunk, and over 64 positions of counter-a's prompt that flipped
the model's first choice after a wake from `say` to `read` (`say` minus `read`: +0.16 at
full precision, -0.97 at int8 in chunks of 64, +0.12 in chunks of 16, +0.22 one position
at a time), which deadlocked the run. In chunks of 16 the int8 export agrees with full
precision on that choice, at 15% more prefill time than chunks of 64.

The export needs PyTorch, `transformers` (the locked 5.17 loads `qwen3_5`, so the group
needed no upgrade), `onnx` and ONNX Runtime, which are a non-default dependency group of this package,
so the root `uv sync --all-packages` never installs them:

```bash
uv sync --all-packages --group export                     # from the repository root
uv run python demo/coop-count-web/export/export_model.py  # Qwen3.5-4B at q4; 3.3 GB
uv sync --all-packages                                    # drop the export group again
```

The script downloads the model from the Hugging Face Hub (8.7 GB of bf16 safetensors for
Qwen3.5-4B) into `models/<name>/` first. It holds the model in float32 while it traces
the graph, and releases it before quantising, so the 4B export peaks at roughly 20 GB
of memory; on a 34 GB machine it took about five minutes. `--model Qwen/Qwen3.5-2B`
(4.3 GB download) and `--quant int8` give the 2B and int8 exports described below. Two
exports from the same source are byte-identical.

### The worker

`web/transformers_worker.js` implements `ZeosModelWorker` over the graph, driving it with
onnxruntime-web directly rather than through Transformers.js: Transformers.js' model
classes run their own exports with their own cache, and this graph has inputs and outputs
no model class knows (the masks, the attention vector) and a cache the worker owns. The
tokenizer is `@huggingface/tokenizers`, the library Transformers.js 4 tokenizes with,
loaded from the export's `tokenizer.json`. `tokenize` runs its normaliser, pre-tokeniser
and BPE without the added-token splitter, so a literal `<|im_end|>` in text stays text; it
matches Hugging Face's `split_special_tokens=True` token for token.

A job's KV is one growable `Float32Array`; its DeltaNet state and convolution window are
the graph's last outputs, never written to, so `fork` shares them and copies only the KV.
Its last token is pending, with no cache behind it, until a decode step feeds it through
the graph, so every step is exactly one forward pass of one token, and `append` prefills
everything before it in chunks of 16. A decode step does not append the id it chooses
(`JsMachine` does, before the next). The mask of a job's latest step is applied to old
positions when new tokens are prefilled, so content appended later never reads a position
the kernel had hidden.

A context keeps a snapshot of its state every 256 positions (`SNAPSHOT_EVERY`; 20 MB
each), the state from before its latest decode step, and, per cached position, whether
the state took that position in. `truncate`, and a step whose mask disagrees with that
record, go back to the latest snapshot at or before the position and run the tokens after
it again: at most 255 positions, about 20 s at 83 ms a token. In a `coop-count-pipe` run
this happens when the kernel rewrites a status line in place (`machine.splice`); masks
that hide nothing new cost nothing. The worker refuses an `allowedBlocks` shorter than the
context, a mask that hides every block, and an `allowedTokens` that allows nothing.

`info()` is `{blockSize: 1, padId: <|endoftext|> (248044), controlIds: [<|im_start|>,
<|im_end|>, <|endoftext|>] (248045, 248046, 248044), eosId: <|im_end|>, vocabSize:
248070}`. `vocabSize` is the tokenizer's vocabulary; the logits are 248,320 wide because
the embedding is padded, and no id past the tokenizer's is ever chosen. The config's
`eos_token_id` is `<|endoftext|>`, but the chat template ends a turn with `<|im_end|>`,
which is the end of sequence `JsMachine` reserves.

**Thinking mode.** Qwen3.5's chat template opens an assistant turn with `<think>\n` (thinking)
or an empty `<think>\n\n</think>\n\n` (not thinking), and without either the model's
first choice after `<|im_start|>assistant\n` is `<think>`. `JsMachine` frames turns as
`LlamaMachine` does, with neither: `<think>` and `</think>` (248068, 248069) are added
tokens whose pieces begin with `<`, which the grammar never admits, so a job cannot think
out loud in tags and decodes commands from its first token. Inserting the empty block
was tried at the decision where counter-b goes wrong (below) and did not change it: at
full precision the model prefers `write` over `say` after `say 11;` with the block (18.9
against 17.1) and without it (18.8 against 17.7).

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
loading in the browser this was tested in. The page's default is the Qwen model on WebGPU, falling back to the recorded answers when
the page is not cross-origin isolated and to WebAssembly when the browser has no WebGPU.
The build's default export is Qwen3.5-4B at q4 (`models/Qwen3.5-4B-zeos-q4`, 3.27 GB of
weights, which `export/export_model.py` writes by default), which
WebGPU runs: in Chrome on an Apple M1 Max its first prompt prefilled within about 35 s of
pressing run and a decode step took about 220 ms. Pass `--model` to offer another export,
such as `models/Qwen3.5-2B-zeos-q4` or the int8 export measured below. With int8 on WebAssembly, in Chrome (the desktop app's browser pane, Apple M1 Max) the model thread
was ready within 10 s of pressing run, the first command came after about two minutes of
prefill, and the 99-tick `coop-count-pipe` run below took 329 s in all, against 304 s for
the same run under Node -- a step in the page costs within about 10% of one under Node.
The heap holds the weights once (one graph, one session), well inside WebAssembly's
4 GB.

WebGPU is offered with the caveat in its label. ONNX Runtime's WebGPU backend has no
kernel for the int8 export's integer matmuls (with the 0.5B they ran on WebAssembly with a
copy each way, about 1.2 s a step; not measured with the 2B). The q4 export
(`--quant q4`, 1.69 GB) runs on WebGPU, and `build.py --model
models/Qwen3.5-2B-zeos-q4` puts it on the page: on the same machine its prompts
prefilled in about 20 s and `coop-count-pipe` ran its 400 ticks in 122 s, about 0.2 s a
tick including Pyodide's side. It decodes differently from the int8 export on
WebAssembly (4-bit weights, and the GPU's arithmetic): counter-a counted from 1 to 50
without stopping at 10, recorded 50 and woke its peer; counter-b said 10 and recorded 10;
counter-a, woken, started again from 1 and was at 25 when the run was cut off; the
handler read the 500 and then waited on the console again.

### The grammar

`JsMachine`'s Python token mask (`token_mask`) is kept. Against the Qwen3.5 vocabulary
(248,070 pieces) it is correct -- `tests/test_transformers_grammar.py` puts the model
after a context of injected text written to talk it out of the command language
(forged chat markers, a fake status line, prose demands) and every completed line parses
as a command, no control id is ever chosen and no `<` is ever emitted -- and it is fast
enough: with Qwen2.5's 151,665 pieces, building the mask for a round state the run had
not met cost 0.11 s on average under Pyodide (0.81 s at most, 0.06 s under CPython), and
the cost grows with the vocabulary, here 1.6 times larger; a cached state costs nothing,
so a counting round pays a few seconds once, against about 0.3 s a decode step for as
long as the run lasts.

### Running it

In the page: follow *Getting it running*, leave *answered by* on the language model,
choose a scenario, and run. A run stops at 400 ticks, as
`demo/coop-count/run_all.py` stops a live seat, or earlier when no job can run: quiescent
when every job is asleep on another, and, in the page, held open while a job waits on
the console.

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
Nothing presses a key under `node_run`, so it ends a run that is parked on the console
with nothing left in the schedule (`LiveRun.waiting_for_a_press`) rather than idling.

### Measured attention

The write-up and its journals were made with the Qwen2.5-0.5B-Instruct int8 export in
the earlier two-graph format, whose every layer was softmax attention; its numbers are
that model's. `docs/evidence/run.sh` now runs the default export, Qwen3.5-2B, whose
measured attention covers only its six softmax layers (see *The architecture*). On
`coop-count-pipe` with its schedule the 2B's 95 decode steps all carried measured
attention, no segment less trusted than its reader received any, and the newest, not yet
masked block got 0.05 of a step's attention on average (0.28 at most).

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
WebAssembly binary on one thread and the kernel reads no clock. With the Qwen3.5-2B int8
export, under Node 22 and, for `coop-count-pipe`, in Chrome (the desktop app's browser
pane, Apple M1 Max):

| case | how the run ended | SHA-256 of the journal |
|---|---|---|
| `coop-count-scripted` (with its schedule) | parked on the console after 85 ticks | `ab0673e0c59513338154c8a08ee0d8b6e681c0fe2305785ee562a5eb41a75fe2` (Node) |
| `coop-count-pipe` (with its schedule) | quiescent after 99 ticks | `e62e5cb405c8e2ac996bee80a35bc691ba384772ad0b0de9fb826b39c5494e8b` (page and Node alike) |
| `coop-count-vector` | 400 ticks | `cc5610c9cd5aafd4c43a06cc4f3c9954c0e73619ad4328b623327894564f8d83` (Node) |

With the Qwen2.5-0.5B export all three ran 400 ticks with the same digest in the page and
under Node, and `coop-count-scripted` twice in the page, in two page loads, with the same
digest. The same holds with Pyodide in Node over the model thread: `tests/test_pyodide_model.py`
runs 40 ticks of `coop-count-scripted` that way and requires CPython's bytes.

WebGPU was checked on one device only, with the 0.5B's q4 export, on twenty greedy steps
of a short chat prompt: three runs in the same browser chose the same tokens and reported
bit-identical attention, and the tokens were WebAssembly's, but the attention was not
bit-identical to WebAssembly's on the same export. Its floating-point order is the
GPU's, so a journal made on WebGPU is not claimed to match one made on another device,
or on WebAssembly; the determinism claim is WebAssembly's.

## The chat machine

`zeos_coop_count_web.chat_machine.ChatToolMachine` runs a chat agent rather than a counter:
the model writes prose and Qwen3.5 tool calls, and the tools run in the host page, outside
the kernel. It is a `JsMachine` with the two ABI-shaped halves replaced -- what a step may
emit, and what a finished piece asks the kernel for -- so it runs on any
`ZeosModelWorker`: the model worker, the stub worker, or a Python one.

**The case** is `src/zeos_coop_count_web/cases/chat-agent/`, shipped in the wheel
(`zeos_coop_count_web.chat.CHAT_CASE`). One pinned job, `chat-agent`, holds the whole
conversation, and its descriptor body is a placeholder the host replaces with its system
prompt and tool declarations:

| pipe | kind | ring | what it carries |
|---|---|---|---|
| `chat.user` | device, principal `user` | TRUSTED (2) | what the user types |
| `tools.results` | device, principal `tool` | EXTERNAL (3) | what a tool returned |
| `tools.read` | sink, capability `min_integrity: 3` | EXTERNAL | calls that only read |
| `tools.effect` | sink, capability `min_integrity: 2` | TRUSTED | calls with side effects |
| `chat.out` | sink, capability `min_integrity: 3` | EXTERNAL | the reply that ends a turn |

It declares `on_fault: retry` and `integrity.dynamics: low-watermark` (without dynamics
the confused-deputy lint refuses a job holding `tools.effect` and reading
`tools.results`), and a 16,384-token window.

**A turn.** The machine frames the context as Qwen3.5's chat template does: the body is
the `system` turn, a message a `user` turn, a tool result a `user` turn holding
`<tool_response>\n...\n</tool_response>`, and the model's turn opens with
`<|im_start|>assistant\n` and the thinking prefix (`<think>\n`, or the empty think block
with thinking off). The model decodes freely, except that it may not emit the pad, a
control id other than `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<think>` and
`</think>`, or a piece that completes `</?(KERNEL|RESUME|FAULT|STATUS|STUB)` or
`</?tool_response` across the turn so far (`FrameGuard`, one cached mask per guard state).
On `</tool_call>` closing a call that parses as the app's Qwen parser parses it, the
machine asks for a `WRITE_READ`: `{"name", "arguments"}` as JSON to `tools.read` or
`tools.effect` by the host's tool-class table (a tool it does not name is an effect),
then a read of `tools.results`. When the model chooses `<|im_end|>` the machine does not
append it: it writes the turn's text to `chat.out` and reads `chat.user`, and the marker
becomes framing in front of the next message. A write the kernel refuses is answered with
a `FAULT` notice; the machine's next step reads `tools.results` again rather than
decoding, so the job waits there until the host settles the refused call.

**Whitespace.** The run sets `KernelConfig.preserve_whitespace`, under which the kernel
tokenises a delivery and the body keeping each word's leading whitespace, so a CSV, a
SQL result or a file reaches the model with its line breaks. With the flag off, the
default, the kernel tokenises as it always has and every journal is unchanged. A frame
tag after a line break is still an imitation, and still alarms.

**Sampling.** Greedy by default. With `sampling=Sampling(temperature=0.7, top_k=20)` every
step sends `opts.sample = {temperature, topK, u}`, with `u` from a `random.Random` seeded
from the run's seed and the job, and the worker takes the id `sample_index` defines;
`web/transformers_worker.js` implements it as `sampleToken`, and
`tests/test_chat_machine.py` checks the two agree. The stub worker checks the options and
plays its tape.

**The host's API** is `zeos_coop_count_web.chat`:

```python
run = open_chat(
    worker,                         # a ZeosModelWorker
    tool_classes={"ReadLines": "read", "WriteLines": "effect"},
    system_prompt=prompt,           # replaces the case's body
    thinking=False, theta_read=0.2, seed=0, sampling=None,
    param_types={"RunSQL": {"sql": "string"}},
)
run.send_user(text)                 # deliver on chat.user before the next tick
events = run.step(16)               # up to 16 ticks; stops when only a delivery helps
run.drain("tools.read")             # one string per write since the last drain
run.deliver_tool_result(text)       # on tools.results, ring 3
run.deliver_refusal()               # the same, with a refusal
run.waiting_on()                    # "chat.user", "tools.results" or None
run.state()                         # integrity, session floor, segments, demotions
run.journal_bytes()
```

`step` returns plain dicts: `token`, `tool_call`, `approval_required`, `tool_refused`,
`reply`, `arrived`, `demoted`, `spoof`, `fault` and `waiting`, with the fields the module
docstring lists. A tool call on `tools.effect` is refused for privilege when the job's
watermark has fallen to 3 -- it attended a tool result past `theta_read` -- or when its
session floor is 3 because the last thing it read was a tool result (MP's confused-deputy
rule), which holds until the next user message. `approval_required` carries the call,
both integrities and the demotion history; on approval the host runs the call itself and
delivers the result, and on denial it delivers a refusal.

**Padding.** A kernel block is 16 words (`block_size`), and every injection begins on a
block boundary, so the context holds up to 15 pad ids in front of each turn marker. The
model sees them; `block_size` is a parameter of `open_chat`.

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
- `test_chat_machine.py` — the chat machine and `ChatRun` on scripted workers
  (`tests/chat_workers.py`): a multi-turn chat whose `<|im_end|>` is intercepted, a tool
  call round trip whose result keeps its line breaks, demotion after attending a ring-3
  result and the privilege fault on `tools.effect` that follows while `tools.read` still
  lands, the session floor's refusal without demotion, a frame tag in a tool result
  raising the spoof alarm, the frame guard, the parser, the sampler against the
  JavaScript one, and the same journal bytes for the same seed.
- `test_page.py` — the page's Python half, and that `build.py` assembles every file the
  page fetches.
- `test_exported_graph.py` — the exported graph under ONNX Runtime CPU, prefilled in the
  worker's chunks with the state carried between them: a hidden position receives
  exactly zero, the rest sum to one, and hiding one changes the logits. Needs the export
  and the `export` dependency group.
- `test_transformers_grammar.py` — the token mask against the real model after
  adversarial injected text: only commands, never a control id.
- `test_node_run.py` — a kernel run over the real model: one attention line per decode
  step, each summing to one, and a block hidden through `set_mask` at exactly zero.
- `tests/js/transformers_worker.test.mjs` (`node --test tests/js/*.test.mjs`) — the
  worker's interface under Node, directly and through the synchronous channel, including
  a cut past a snapshot of the recurrent state and a mask that hides what the state had
  taken in, each of which must compute what a fresh context would.
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
