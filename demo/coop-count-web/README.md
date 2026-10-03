# coop-count-web — the coop-count demo in a static web page

The ZEOS kernel and the coop-count demo, running inside [Pyodide](https://pyodide.org) in a
browser, with no server beyond a file host. A member of the repository's uv workspace.

The page loads Pyodide, installs the `zeos` wheel and this package's wheel, writes the
coop-count cases into Pyodide's in-memory filesystem, lints the chosen case and runs it.
The journal streams onto the page one JSON line per event, the ZEOS debugger draws the
case's wiring and then the finished run, and the journal can be downloaded as the
`.jsonl` file `zeos-count run --journal` would have written. A keypress on the page is the
console's interrupt.

```bash
uv sync --all-packages                       # from the repository root
uv run python demo/coop-count-web/build.py   # builds the wheels into web/dist/
python -m http.server 8765 -d demo/coop-count-web/web/dist
# open http://localhost:8765/
```

Any static file host serves `web/dist/` as well; the page fetches Pyodide from the
jsDelivr CDN, so the browser needs a network connection the first time.

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

On `coop-count-scripted` the two decode the same words and their journals hold the same
events in the same order, with two exceptions, both about attention. The seat's mask
hides the blocks a job has written since the kernel's last mask refresh, so the kernel
journals the job's attention to its own open segment as denied (`mask.denied`, 30 times
in this run) and leaves that segment out of the working set; `JsMachine` treats those
blocks as visible (see *the mask's horizon* below), so it has no `mask.denied` events
and two `vm.working_set` events count one more segment. The page's table of finished
runs shows each journal's SHA-256, so two runs of the same machine with the same inputs
show the same digest. `coop-count-pipe` and
`coop-count-vector` carry no tapes, so they need a model: the page lints them and draws
their wiring, and does not offer to run them.

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
- `test_pyodide_determinism.py` — the smoke fixture, the seat and the stub, each under
  Pyodide in Node and under CPython, compared byte for byte. It builds both wheels with
  `uv build`, installs them by unpacking as Pyodide installs a pure wheel, and copies the
  cases into the in-memory filesystem as the page does. It skips when Node, the npm
  package or the cached PyYAML is missing; `ZEOS_PYODIDE_DIR` points it at a Pyodide
  installed elsewhere.

`uv build` (used by `build.py`, the determinism test and `test_page.py`) fetches the
hatchling build backend from PyPI the first time it runs, so those need a network once.
