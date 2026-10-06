# zeos-browser — ZEOS in the browser

The browser layer of ZEOS, as a library the browser demos build on: `JsMachine`, the
machine backend that hands every token-level operation to a JavaScript model worker, with
its grammar mask and the bridge it crosses Pyodide's FFI by; the run loop a page steps one
turn at a time (`live.LiveRun`) and the functions a page's Web Worker calls (`page`);
the model workers that run Qwen3.5-4B on WebGPU, the channel that makes them synchronous
for the Python side, and the stub worker; and the tooling that exports the model. A member
of the repository's uv workspace that depends on `zeos` alone; like every workspace member
it never ships in the `zeos` wheel.

[`demo/coop-count-web`](../../demo/coop-count-web/README.md) and
[`demo/space-invaders-web`](../../demo/space-invaders-web/README.md) are built on it: each
installs the `zeos-browser` wheel under Pyodide and its `build.py` copies the JavaScript in
`web/` beside its page. The chat agent built on `JsMachine` is
[`packages/zeos-chat`](../zeos-chat/README.md).

| | |
|---|---|
| `src/zeos_browser/` | `js_machine` (`JsMachine`, the worker interface), `token_mask`, `pyodide_bridge`, `fake_worker` (the Python twin of the stub), `live` (`LiveRun`) and `transcript`, `page` (what a page's worker calls), `node_worker` (the stub in a Node child process, for tests), `model_source` (the model a page's `build.py` names in its manifest) |
| `web/` | `opt_zeos_worker.js`, `transformers_worker.js`, `stub_worker.js`, `model_channel.js`, `frames.js`, `model_host.js`, `model_thread.js`, `model_cache.js`, `opfs_store.js`, `sha256.js`, `coi_serviceworker.js`; `node_bridge.mjs` and `node_model_thread.mjs` serve the stub under Node for the tests |
| `export/` | `opt_zeos_surgery.py` (the OPT+ZEOS graph), `export_model.py`, `opt_zeos_reference.py`, and `bench/`, the WebGPU check and timing pages |
| `models/` | gitignored: where the exports land |
| `package.json` | ONNX Runtime Web, the tokenizer, and Pyodide for the Node tests |

## Getting the model

A page needs nothing prepared: it downloads the OPT+ZEOS export from the Hugging Face Hub,
[`nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16`](https://huggingface.co/nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16),
at a pinned commit, the first time the model machine runs (about 2.4 GB of the files are
read), and keeps it in the browser's storage, so a reload, or the browser started again,
reads it from disk instead (*Loading and caching the model*, below). `npm install` here
is still needed, for ONNX Runtime Web and the tokenizer each page's `build.py` copies in.

To run a page on a local export instead (to try a change to the graph, say), make the
export, then build the page with `--model`, which links it into the page and loads it by
default; `?model=huggingface` in the page's URL still loads the Hub's. Making the export
needs a network connection to download 2.8 GB from the Hub and about 6 GB of free disk.
Every command runs from the repository root:

```bash
uv sync --all-packages --group export     # PyTorch, transformers, onnx
npm install --prefix packages/zeos-browser
uv run hf download onnx-community/Qwen3.5-4B-ONNX-OPT \
    --include "*.json" "chat_template.jinja" \
      "onnx/embed_tokens_q4f16.onnx*" "onnx/decoder_model_merged_q4f16.onnx*" \
    --local-dir packages/zeos-browser/models/Qwen3.5-4B-ONNX-OPT
uv run python packages/zeos-browser/export/opt_zeos_surgery.py \
    --src packages/zeos-browser/models/Qwen3.5-4B-ONNX-OPT \
    --out packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT
uv sync --all-packages                    # optional: drop the export group again
uv run python demo/coop-count-web/build.py --model   # or --model DIR for another export
```

`--model` with no directory means `packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT`;
`--copy-model` copies the export instead of linking it, for a host that does not follow
links. The WebGPU bench pages under `export/bench/` and `tests/opt_zeos_webgpu.mjs` read
the local export only.

### Loading and caching the model

`src/zeos_browser/model_source.py` writes the manifest's model fields for both pages'
`build.py`: `model_sources` (`huggingface`: `{name, repo, revision}`, always; `local`:
`{name, path}` with `--model`), `model_source`, the default, and `model`, its name.
`--hf-repo` and `--hf-revision` change the Hub source; the revision must be a full
40-digit commit (`HF_REVISION`), never a branch, because the cache keys files by it. To
publish a new export, upload it to the repo and set `HF_REVISION` to the new commit (from
`https://huggingface.co/api/models/<repo>`). In the page, `model_host.js`'s
`modelSource(manifest)` picks the source (a `?model=` query parameter overrides the
default) and `startBrowserModel({model})` hands it to the model thread, whose `read`
callback is `model_cache.js`'s `modelFiles(...).read`, for either worker.

**Cross-origin isolation.** The pages are served with `Cross-Origin-Embedder-Policy:
require-corp` (serve.py, or `coi_serviceworker.js`). Under it a cross-origin `fetch` must
be a CORS request whose every response carries `Access-Control-Allow-Origin`; the
cache's fetches are CORS requests (the default mode) without credentials. Checked with
curl and in Chrome 154 from the model thread of a `require-corp` page: `huggingface.co/
<repo>/resolve/<commit>/<file>` answers 307 (small files, to `/api/resolve-cache/...`) or
302 (LFS files, to `us.aws.cdn.hf.co/xet-bridge-us/...`) with `Access-Control-Allow-Origin:
<origin>`, and the CDN answers with `Access-Control-Allow-Origin: *` and `Access-Control-
Expose-Headers: *`; every hop passed, including `Range: bytes=N-` requests (a
CORS-safelisted header in that form, so no preflight), which the CDN answers 206 with
`Content-Range`. So the isolation stays as it is: no `credentialless`, and the service
worker is not involved.

**Storage: the Origin Private File System (`opfs_store.js`).** Measured in Chrome 154 on
an M1 Max with a 2.07 GB file (the size of the decoder's first data file): OPFS, written
through a synchronous access handle in 8 MB pieces, took 1.4 s and streamed back in
1.1 s (0.4 s through the access handle); Cache Storage (`cache.put` of a streamed
`Response`) took 5.3 s, its read 1.4 s, and after `caches.delete` the origin's usage
stayed at 2.07 GB, where removing the OPFS file freed it at once. `Blob.arrayBuffer()` of
a 2.2 GB OPFS file did not return in ten minutes, so files are read as a stream into one
preallocated `Uint8Array`. An OPFS file can also be resumed (a `.part` file plus a
`Range` request), where a cache entry is all or nothing. Files live at
`zeos-model-cache/<repo>/<revision>/<path>` (each part URI-encoded): the key is repo,
revision and path, so a new commit never reads a file an older one stored. Once a load
has every file of its revision, other revisions of the same repo are removed.

**Memory.** A file is downloaded straight into the one `Uint8Array` the worker gets
(sized from `meta.json`), and written to the `.part` and hashed chunk by chunk as it
arrives; nothing holds a second copy of it (WebCrypto's `digest` would, which is why
`sha256.js` exists: a streaming SHA-256 at about 175 MB/s in V8, faster than the
download). The HTTP cache is bypassed (`cache: "no-store"`), or Chrome would keep part of
the model a second time on disk. A cache hit streams the stored file into the same kind
of array. ONNX Runtime then copies the bytes into its own memory, as it always has.

**Integrity and interrupted downloads.** `meta.json` lists every other file's size and
SHA-256, which are the Hub's LFS object ids too (`?blobs=true` on the API shows the same
values), so no extra request is needed to check a file; `meta.json` itself comes from the
pinned commit's immutable URL. A download becomes a stored file only after its length and
hash match: the `.part` is renamed then. A file that does not match is deleted and the
load fails with the reason ("nothing was stored, reload to try again"). A load that
stops part-way (a network error, the tab closed) leaves the `.part`; the next load hashes
what it holds ("verifying"), asks for the rest with a `Range` request, and if the
finished file does not match, downloads it once more from the start. A server that
answers the `Range` request with the whole file restarts it. The tokenizer and the
OPT+ZEOS embedding share their bytes with nothing, but the embedding's data and the
decoder's second data file are the same matrix (the same SHA-256): `OptZeosWorker.load`
reads it once, so only the embedding's copy is downloaded and stored.

**Quota and persistence.** Before downloading, the model thread compares what is still
missing with `navigator.storage.estimate()` and fails with a message naming both figures
if the origin cannot hold it; a `QuotaExceededError` mid-download deletes the `.part` and
fails the same way. Chrome allows an origin a share of the disk's free space (10.7 GB on
a disk with 18 GB free when this was measured; enough for the 2.4 GB). `startBrowserModel`
asks `navigator.storage.persist()` first, so a granted origin is not evicted under storage
pressure; Chrome decides without a prompt, by how much the site is used, and refused it
to a fresh profile on localhost, so there the cache is "best effort" and the page says so.
Two tabs downloading at once: the second cannot open the `.part` (an access handle is
exclusive) and says another tab is downloading.

**On the page.** Progress goes through the same bar and line as before, with the phase in
the words: "downloading", "loading from the browser's cache", "verifying", then ONNX
Runtime building the session (`describeModelProgress`). A line under it says where the
model comes from, how much is cached, whether it is persistent, and the origin's usage and
quota (`modelCacheStatus`), beside a "clear cached model" button (`modelCacheControls`,
which removes the whole `zeos-model-cache` directory; a model already loaded stays in use
until the page reloads). `coi_serviceworker.js` relays same-origin requests only and stores
nothing: the Hub's files never pass through it, and a local export's pass through without
being kept.

## The JavaScript machine seam

`zeos_browser.js_machine.JsMachine` is a `SyscallSeat` and a full
`MachineBackend`. It keeps, per job, the kernel's words, the model token ids behind them
and the chat framing folded into their spans, and hands every token-level operation to a
worker object passed to its constructor: a JavaScript object reached through Pyodide in
the browser, or a Python object with the same methods under CPython.

```python
from zeos_browser.js_machine import JsMachine

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
`zeos_browser.token_mask` walks the same language in Python and sends the result
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
method of the seam from Python under Pyodide. `zeos_browser.fake_worker.FakeWorker`
is the same worker in Python, for tests under CPython. Both have a fixed vocabulary (five
reserved tokens, then every word of the tapes and of the ChatML headers, with and
without a leading space), a whitespace tokenizer, a tape per descriptor taken from the
case's `emit` steps, and no attention. Tapes must be ASCII and both refuse any other
text, since Python and JavaScript split and sort non-ASCII text differently. Each step gives the next word of the current
command, split as `zeos.machine.seat.words_of` splits it, checks that `allowedBlocks`
and `allowedTokens` are sized to the context and the vocabulary, and refuses a word the
token mask forbids.

`demo/coop-count-web`'s Pyodide determinism test runs `coop-count-scripted` through
`JsMachine` over the stub in Pyodide and over the fake in CPython and requires the same
bytes, so the two cannot drift apart unnoticed. To run `JsMachine` over the fake:

```python
from zeos.descriptor.loader import load_case
from zeos.machine.seat import seat_maps
from zeos_browser.fake_worker import FakeWorker, tapes_from_scripts
from zeos_browser.js_machine import JsMachine

bundle = load_case(path)
descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
machine = JsMachine(
    FakeWorker(tapes_from_scripts(bundle.scripts)), descriptors=descriptors, valued=valued
)
```

In the browser a page builds the stub with `createStubWorker(tapes)` and passes it to
`zeos_browser.page.open_run(case_dir, "js", worker=stub)`, which wraps it in
`PyodideBridge`; any other object implementing the interface goes the same way.

## The model machine

The pages' model machine is `JsMachine` over a real language model: Qwen3.5-4B as the
OPT+ZEOS graph (`export/opt_zeos_surgery.py`, run by `web/opt_zeos_worker.js`; see *The
OPT+ZEOS graph* and *The OPT+ZEOS worker*), or, with a page's `build.py --model`, Qwen3.5-2B or a
Qwen2.5 model exported to ONNX by `export/export_model.py` and run by ONNX Runtime Web in
`web/transformers_worker.js`, with the tokenizer Transformers.js uses. It supplies all
four things ZEOS asks of a serving stack: the allowed-block mask is applied inside the
forward pass, in every layer (before the softmax in the softmax layers; see *The
architecture* for the linear-attention ones); each decode step's attention is measured,
averaged over the softmax layers' heads and normalised, per position; the control ids
are reserved in the sampler; and the syscall grammar constrains every step. The kernel
therefore receives measured attention, `DecodeResult.attention`, where every other
backend in the repository gives it a hint, and integrity demotes on it.

### The model

The pages offer Qwen3.5-4B as the OPT+ZEOS graph by default; *The OPT+ZEOS worker*
gives its timings. The runs below are of the 2B, exported by `export/export_model.py`,
and were made at int8 on the WebAssembly backend the page then had; the model now runs
on WebGPU only, where the int8 export does not run (see *Backends*).

**Qwen3.5-2B** (`Qwen/Qwen3.5-2B`, the post-trained model; its base is
`Qwen3.5-2B-Base`). It speaks ChatML, which is what `JsMachine` frames prompts in
(`chat_template="chatml"`, as for `LlamaMachine`), and it is the smaller sibling of
Qwen3.5-4B, which follows the coop-count procedure under llama.cpp. Its architecture is a
hybrid of linear and softmax attention, which the export handles as described in *The
architecture* below.

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
exports Qwen2 models in the current format (`--model Qwen/Qwen2.5-0.5B-Instruct`). What the kernel gets from any of them is real: measured
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
  weights.
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
the rest sum to one, and hiding it changes the logits. The script still exports Qwen2 instruct models (`model_type`
`qwen2`: every layer softmax, an empty state); the worker then cuts its cache directly.

**Quantisation.** The default, `q4`, is 4-bit weight-only `MatMulNBits`, the form
ONNX Runtime's WebGPU backend runs: 1.69 GB of weights for the 2B.
`int8` is ONNX Runtime's dynamic quantisation of every
weight matmul (per-channel int8 weights, int8 activations computed per run), plus a
per-row int8 embedding table and a separately quantised output projection: 2.40 GB of
weights. WebGPU has no kernel for its integer matmuls, so the page cannot run it; it was
the fast path on the WebAssembly backend the page used to offer.

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
uv run python packages/zeos-browser/export/export_model.py  # Qwen3.5-2B at q4; 1.7 GB
uv sync --all-packages                                    # drop the export group again
```

The script downloads the model from the Hugging Face Hub (4.3 GB of bf16 safetensors for
Qwen3.5-2B) into `models/<name>/` first. It holds the model in float32 while it traces
the graph, and releases it before quantising, since the quantiser holds the whole
float32 graph as well. `--quant int8` gives the int8 export described above. Two
exports from the same source are byte-identical. The page's Qwen3.5-4B is not made by
this script but by `export/opt_zeos_surgery.py`, below.

### The OPT+ZEOS graph

`export/opt_zeos_surgery.py` writes a second graph for Qwen3.5-4B. It is not a re-export:
it edits `onnx-community/Qwen3.5-4B-ONNX-OPT`, whose decoder runs every DeltaNet layer as
the fused `com.microsoft` operators `LinearAttention` and `CausalConvWithState` and every
softmax layer as `GroupQueryAttention`, all with 4-bit `MatMulNBits` weights, which ONNX
Runtime Web runs on WebGPU. Every one of those nodes, and every weight byte, is kept; only
the way the mask reaches them changes, and the measured attention is added.

```bash
uv run python packages/zeos-browser/export/opt_zeos_surgery.py \
    --src <the -OPT download> --out packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT
```

It takes seconds and loads no weights. The output (2.8 GB) is:

- `onnx/decoder_zeos_q4f16.onnx`, plus `onnx/decoder_zeos_q4f16.onnx_data` (1.9 GiB) and
  `onnx/decoder_zeos_q4f16.onnx_data_1`, the `-OPT` decoder's two data files copied byte
  for byte under the new name.
- `onnx/embed_tokens_q4f16.onnx` and its data, copied unchanged: the token ids go through
  this graph first and the decoder takes its `inputs_embeds`.
- `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
  `chat_template.jinja` from `-OPT`.
- `meta.json`: the decoder's inputs and outputs with their shapes and types, the layer
  types, the cache shapes, `eosId` (`<|im_end|>`, 248046), `padId` (`<|endoftext|>`,
  248044), `controlIds` (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`), `vocabSize`
  (248,077: the `-OPT` tokenizer's ids, added tokens included; the logits are 248,320
  wide), and the size and SHA-256 of every file.

**The decoder's contract** (batch 1; `S` new positions, `P` past positions, `T = P + S`):

| input | type | shape | |
|---|---|---|---|
| `inputs_embeds` | float32 | `[1, S, 2560]` | the embedding graph's output for the new ids |
| `key_mask` | bool | `[1, T]` | true = visible, over every position, past and new |
| `position_ids` | int64 | `[3, 1, S]` | the absolute positions, the same in all three rows (text) |
| `num_logits_to_keep` | int64 | scalar | 1 for the last position only |
| `past_key_values.N.key`, `.value` | float16 | `[1, 4, P, 256]` | softmax layers N = 3, 7, ..., 31 (BNSH) |
| `past_conv.N` | float16 | `[1, 8192, 3]` | DeltaNet layers N = 0, 1, 2, 4, ..., 30: the convolution's carried inputs |
| `past_recurrent.N` | float16 | `[1, 32, 128, 128]` | DeltaNet layer N: the recurrent state |

| output | type | shape | |
|---|---|---|---|
| `logits` | float16 | `[1, num_logits_to_keep, 248320]` | |
| `present.N.key`, `.value` | float16 | `[1, 4, T, 256]` | the whole cache, past and new |
| `present_conv.N`, `present_recurrent.N` | float16 | as the inputs | after the last new position |
| `attention` | float32 | `[T]` | the last position's attention, per position |

`attention_mask` is gone: the graph casts `key_mask` back to it internally, so the
lengths `GroupQueryAttention` reads (`seqlens_k`, `total_sequence_length`) are still the
full length. A run takes any number of new positions; `LinearAttention` solves a prefill
chunk-parallel, so there is no 16-position bound, and `meta.json` gives `maxChunk` 2048,
the size the tests and benchmark run.

**The mask.** In the softmax layers `GroupQueryAttention`'s `attention_bias` input,
which `-OPT` filled from its padding mask, is now 0 at a visible key and -65504 at a
hidden one, on top of the operator's own causal mask, so a hidden key gets exactly zero
weight; a position whose every causal key is hidden sees itself, as in `ZeosQwen`, so its
softmax is defined. No operator is replaced: the bias input can hide an arbitrary
position, where `seqlens_k` cannot. In the DeltaNet layers, a hidden *new* position has
its write strength (beta) and its log decay set to zero, so the state passes it by
unchanged; its projection enters `CausalConvWithState` as zeros, so neither later
positions' taps nor the carried `present_conv` window hold it; and it still reads its
own projection through the last tap, which the graph adds back before the SiLU (the
convolution runs with `activation` none for that). The `-OPT` graph's padding mask on the
output gate and on the inputs of beta and the decay is removed. With every position
visible the logits are `-OPT`'s up to float16 rounding (2e-2 at most, KL under 1e-5).

A position that was visible when the state took it in cannot be hidden by the graph, as
with the `ZeosQwen` export: the state is not per position. Whoever drives the graph
rewinds `past_recurrent` and `past_conv` to a snapshot from before the position and runs
the positions after it again under the new mask, as `web/transformers_worker.js` does
for that export. The softmax layers' cache needs no replay; the mask acts on it at every
run.

**Measured attention.** `attention` is recomputed beside each `GroupQueryAttention`
from the tensors it reads, the rotated query of the last new position and
`present.N.key`: the scaled scores, the hidden keys set to -1e9, a softmax, summed over
the 16 heads, then averaged over the 8 layers and multiplied by the visibility. A hidden
position is exactly zero and the vector sums to one. As with `ZeosQwen`, the 24 DeltaNet
layers contribute nothing to it.

**Checks.** `export/bench/checks.html` runs the decoder on WebGPU through
`OptZeosWorker` (`tests/opt_zeos_webgpu.mjs` drives it; see *The OPT+ZEOS worker*).
Prefilled in chunks of 1 and 7 (an 88-token chat turn) and of 16 and 512 (a 634-token
one), with the cache carried, then one decode step, its logits choose what
`transformers` (float32) chooses and stay within KL 0.1 of it. Hiding the three tokens
of a password the last position must recall moves its distribution as far as `ZeosQwen`
in float32 moves it under the same mask, and to the same first choice. The hidden
positions receive exactly zero and the rest sums to one. The prompts are
`export/bench/reference_prompts.json`, and the reference logits
`models/.reference/opt-zeos-<key>.npz` (the key is a digest of the prompts' ids), which
`export/opt_zeos_reference.py` writes once from the bf16 weights at `models/Qwen3.5-4B`,
keyed with the export's tokenizer (the `export` group, about 20 GB of memory); without
that file the check fails. Measured
on ONNX Runtime CPU when the graph was written: KL 0.003 on the short turn and 0.016 on
the long one, for both this graph and `-OPT`, and a password-hiding KL of 9.4 against
9.3 for `ZeosQwen`.

**Speed.** `export/bench/` is a page that times the decoder on WebGPU with ONNX
Runtime Web 1.31.0-dev.20260914, the build Transformers.js 4.3 runs `-OPT` with, and the
cache kept on the GPU between runs. Open it with the checks' driver (see *The OPT+ZEOS
worker*), which serves this directory:

```bash
node packages/zeos-browser/tests/opt_zeos_webgpu.mjs --page index.html \
    --query "opt=/models/Qwen3.5-4B-ONNX-OPT/"
```

In Chrome 154 on an Apple M1 Max, three runs each, with nothing else on the GPU (the
runs agreed within 3%; with another WebGPU page running they halved):

| | OPT+ZEOS | `-OPT` |
|---|---|---|
| prefill, 512 from empty | 324 tok/s | 299 tok/s |
| prefill, 2048 from empty | 326 tok/s | 270 tok/s |
| prefill, 4 x 512 with the cache carried | 313 tok/s | 270 tok/s |
| prefill, 2048 after 2048 | 302 tok/s | 236 tok/s |
| decode step, ~64 positions | 45 ms (22 tok/s) | 40 ms (25 tok/s) |
| decode step, ~4,100 positions | 56 ms (18 tok/s) | 46 ms (22 tok/s) |

The decode times include reading back the logits and, for OPT+ZEOS, the attention
vector. Without the attention output and the convolution's own-tap correction a step
took 42 ms and 49 ms, so most of the decode cost is the attention output, which reads
every layer's whole key cache once per step.

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
it again: at most 255 positions. In a `coop-count-pipe` run
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

The worker imports nothing: ONNX Runtime and the tokenizer class are handed to it.

### The OPT+ZEOS worker

`web/opt_zeos_worker.js` (`OptZeosWorker`) implements `ZeosModelWorker` over the OPT+ZEOS
graph. It has the same interface, `info()`, tokenizer (`encodePlain`, so a literal
`<|im_end|>` in content stays text) and `sample` rule (`sampleToken`) as
`TransformersWorker`, and it imports nothing else: ONNX Runtime and the `Tokenizer` class
are handed in. `model_thread.js` chooses it when `meta.json` names a
`decoder` and an `embedTokens` graph (`isOptZeosMeta`), and `TransformersWorker`
otherwise, so a page's `build.py --model` offers either kind of export with no other
change. `info()` is `{blockSize: 1, padId: 248044, controlIds: [248045, 248046,
248044], eosId: 248046, vocabSize: 248077}`, and `meta.tokenizerSize` (what the `pieces`
call reads) is `vocabSize`.

**Two sessions.** The embedding graph turns the new ids into `inputs_embeds` (float32,
left on the GPU) and the decoder runs them against the cache, with `key_mask` built from
the step's `allowedBlocks`, `position_ids` the absolute positions in all three rows, and
`num_logits_to_keep` 1. The tied embedding matrix is both the embedding graph's data and
the decoder's second shard; files with the same SHA-256 in `meta.json` are read once.

**Pending tokens.** `append` records ids and runs nothing. The next `decodeStep` runs
every id without a cache behind it under that step's mask, in chunks of at most
`maxChunk` (2048) cut at the snapshot positions, and reads the logits and the attention of
the last position only. In a run of steps each step is one forward pass of one token,
since `JsMachine` appends the chosen id before the next step.

**The cache stays on the GPU.** On WebGPU every `present*` output, the logits and the
attention are `gpu-buffer` outputs. The presents are fed back as the next run's inputs,
and only the last chunk's logits (0.5 MB of float16) and attention are read back. A run's
outputs are new tensors that are never written to, so a context, its snapshots and its
forks share one set of them, with a reference count, and `dispose` it when the count
reaches zero. `fork` copies nothing on the GPU. Cutting the softmax layers' keys and
values to `n` positions copies the first `n` positions of each of the four heads into a
new buffer with `copyBufferToBuffer`, without a round trip through the CPU.

**Snapshots.** The DeltaNet state is not per position. A cache keeps the state after
every `SNAPSHOT_EVERY` (256) positions, and the state from before its latest run, as
references to the graph's own output tensors. About 25 MB of GPU memory each (24 layers,
a 1 MiB recurrent state and a 48 KiB window per layer). There is no copy and no
download. Downloaded snapshots would cost 25 MB of read-back for every snapshot taken and
25 MB of upload for every rewind. That traffic happens on every prefill, and the memory
is bounded, so the snapshots stay on the GPU. At most `MAX_SNAPSHOTS` (16, so 400 MB) are
kept per cache. Past that, the one whose neighbours are closest is dropped, so old
snapshots thin out and recent ones stay 256 apart.

A cut goes back to the latest snapshot (or the pre-run state) at or before the position,
and the next step replays from there. A cut is a `truncate`, or a step whose mask
disagrees, at some past position, with the mask the cache was built under. Masks that
hide nothing new cost nothing. Replay and a fresh prefill cut their chunks at the same
positions, so a replay under a new mask is bit for bit the fresh prefill under that mask.

Prefill speed hardly depends on the chunk size. With chunks of 256, 512, 1,024 and 2,048,
a 2,048-token prefill ran at 250–305 tok/s, and the order changed from run to run. So
256 costs nothing measurable and bounds a replay to 255 positions.

**Two caches.** A context keeps the cache of its latest step's mask and the cache of the
mask before it (`maxTracks`, 2 by default; each with its own softmax keys and values, 32
KB a position, and snapshots). A step whose mask agrees with the current cache over the
positions it holds runs there. One that agrees with the kept cache switches to it and runs
only the positions that cache has not seen. One that agrees with neither starts a new
cache, a copy of whichever of the two shares more of its history rewound to the latest
snapshot before they part, keeps the current one and drops the other. So a mask that hides
past positions for a few steps and then shows them again -- `ChatToolMachine`'s masked
tool name -- costs a replay from a snapshot the first time it narrows and a catch-up after
that, and the cache of the wide mask is never rewound. With `maxTracks` 1 a disagreeing
mask rewinds the one cache, as it always did.

**Hidden runs.** A hidden position leaves the DeltaNet state as it was (beta and the log
decay are zero) and enters the convolution as zeros, and no later query can attend its
keys. So a run of at least 3 hidden positions (`convShape[2]`), not including the last
position, is not run through the graph (`skipHidden`, on by default): the recurrent state
is carried as it is, the convolution window becomes zeros, and the softmax cache grows by
zeros there in one copy, which the key mask hides. Chunks are also cut where such a run
starts. A fresh prefill and a replay under the same mask cut and skip alike, so they
still agree bit for bit, and a skipped run aligned with the chunks is bit for bit the run
it stands in for. `export/bench/checks.html` checks that hidden tokens swapped for others
change no logit and no attention weight, with hidden runs carried past and with every
hidden position run through the graph.

**Mask on demand, measured** by `export/bench/mask.html` (`tests/opt_zeos_webgpu.mjs
--page mask.html`) in Chrome on an Apple M1 Max: a chat-agent context of a 6,739-token
prompt and three tool calls, the first two answered by 518- and 728-token results, 8,178
positions in all, stepped as `JsMachine` steps it, with the results hidden for the 7 steps
of each tool's name. The extra time of the name's steps and the step after them, against
the same schedule unmasked:

| | second call (one result to hide) | third call (two) |
|---|---|---|
| the call unmasked: its text, name and arguments, and its result's prefill | 9.7 s | 5.7 s |
| masked: two caches, hidden runs carried past (the default) | +2.2 s (22%) | +1.8 s (31%) |
| masked: two caches, hidden runs run | +4.7 s (48%) | +6.1 s (114%) |
| masked: one cache, a rewind and replay each way | +7.7 s (80%) | +17.1 s (321%) |

The first call, with nothing to hide, costs nothing extra. In the default the extra time
is the second cache's catch-up and the first cache's run of the name's 8 tokens. The
second cache, first narrowed at the second call, replays 136 positions from the snapshot
at 6,656 (0.5 s) and carries the 490 result positions past; at the third it runs the 51
positions it had not seen (0.4 s), carries 700 past, and runs the 42 after them (1.2 s).
The first cache's 8-token run takes 0.33 s against 0.06 s for the decode step it replaces.
Short prefill runs are what cost: at 7,000 positions a run of 8 new tokens takes 0.32 s,
of 40 tokens 0.76 s, against 0.065 s for one, and a skip's copy of the cache adds about
0.15 s to the run after it. The name's own steps cost what they would unmasked.

**WebGPU only.** `OptZeosWorker.load` runs both sessions on WebGPU. onnxruntime-web's
WebAssembly backend loads the graph, but a run fails with `std::bad_alloc`: a 4 GiB heap
cannot hold 2.4 GB of weights as well as the activations.

**Measured** in Chrome on an Apple M1 Max, through the interface, with nothing else on the
GPU, by `export/bench/worker.html` with ONNX Runtime Web 1.31.0-dev.20260914 (1.30.0,
this demo's pinned version, gave the same results):

| | |
|---|---|
| load (files in the HTTP cache) | 3.8–4.2 s |
| prefill, a 1,020-token prompt, until the first step's choice | 3.5 s (289 tok/s) |
| decode step at ~1,050 positions | 47–48 ms (21 tok/s) |
| step after hiding 16 positions at 600 in a 1,052-token context (540 positions replayed from the snapshot at 512, the 16 carried past) | 2.4–2.5 s |

**The checks on WebGPU.** The model runs on WebGPU only, so its checks run there, in
headed Chrome, driven by `tests/opt_zeos_webgpu.mjs`. They are opt-in -- they need a GPU,
the export, `npm install`, Playwright and Chrome -- and no default test suite runs them:

```bash
PLAYWRIGHT_MODULE=<a node_modules/playwright> node packages/zeos-browser/tests/opt_zeos_webgpu.mjs
```

It serves this directory cross-origin isolated and runs three pages, loading the model
afresh in each (about two minutes in all on an Apple M1 Max). It exits 1 if a check
fails and 2 if a page errors; `--page NAME` runs one page, and with no export it prints
a skip and exits 0. Run nothing else heavy on the GPU at the same time.

- `export/bench/checks.html`: the graph's logits against `transformers`' reference (see
  *The OPT+ZEOS graph*); the hidden tokens of a context swapped for others, with the same
  mask, leave every logit and every attention weight the same bits -- after a prefill,
  after a single-token decode step over that cache, and with every position, the hidden
  ones too, run as a single-token step; both with hidden runs carried past and with every
  hidden position run through the graph, while the same swap with nothing hidden does
  move the logits; and a step stopped before a run of the graph, resumed with the same
  `maxChunk`, chooses the same token with the same logits and attention as the
  uninterrupted step, with and without a mask. Then the workers' interface clauses
  (`export/bench/unit_clauses.js`). For `OptZeosWorker`, with snapshots every 16 positions
  and at most 4 kept: tokenisation and pieces; greedy steps against cache-free runs, and
  the old snapshots thinning out; a replay after a past position is hidden, equal bit for
  bit to a fresh prefill; a skipped hidden run aligned with the chunks bit for bit the run
  executed; a narrowed mask on a second cache, switching back without a rewind, catching
  up, and a third mask; one cache rewinding instead; a repeated step, `truncate` and
  `fork`; `allowedTokens`, `sample` and the refusals; `maxChunk` cut at the snapshot
  positions, to the same token (within KL 1e-2: two chunkings differ by about 4e-3 on
  WebGPU); a stopped step resumed bit for bit; and `load` refusing a backend other than
  WebGPU or an option it does not know. For `TransformersWorker`, over the
  `export_model.py` q4 export at `models/Qwen3.5-2B-zeos-q4` (the `transformers` query
  parameter; skipped, and said so, without it): pieces and tokenisation, normalised
  attention, a masked position at zero, `allowedTokens` and the refusals, a truncate
  past a snapshot, a mask that hides what the recurrent state saw, a stopped step
  resumed, `fork`, and the same calls giving the same bits.
- `export/bench/grammar.html`: `JsMachine` under Pyodide, in a worker, over the model on
  its own thread through `SyncModelWorker`, as the page runs them. After injected text
  written to talk the model out of the command language (forged chat markers, a fake
  status line, prose demands), every line it completes parses as a command on a pipe the
  job may use, no control id is chosen and no `<` is emitted; and a context one token
  short of a chat turn gets no control id out. It runs on the `zeos` and
  `zeos-browser` wheels, which the driver builds into `export/bench/wheels/` first.
- `export/bench/worker.html`: the 32 greedy steps choose what one run without a cache
  chooses for the same tokens; the first step is bit for bit a cache-free run cut into the
  same chunks (against one cut into a single 2,048 chunk its KL is 4e-3: the recurrent
  state crosses a chunk boundary in float16, and two chunkings of the graph alone differ
  by that much); hidden positions receive exactly zero and the rest sums to one; the
  replay after a mask change equals a fresh prefill under the new mask, logit for logit;
  `truncate` back to the prompt reproduces the first step exactly; a fork taken before
  the cut keeps the longer context; and the timings above.

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
why each page's `serve.py` and `coi_serviceworker.js` exist. The page starts the model thread rather
than the Pyodide worker because a worker started from inside a worker failed to start in
the Chromium this was tested in. Under Node, for the tests, the same channel serves the
stub worker on a `worker_threads` thread (`web/node_model_thread.mjs`), where
`Atomics.wait` is allowed on the main thread too.

**A decode step that does not block.** `SyncModelWorker` also has
`beginDecodeStep(jobId, opts)`, which posts a step and returns its request id at once;
`pollDecode(timeoutMs)`, an `Atomics.wait` on the reply for at most `timeoutMs` that
returns null while the step runs and otherwise
`{tokenId, attention, cancelled: false, resident, stats}`; and `cancelDecode()`, which
stores the step's id in an abort slot of the buffer (int32 slot 2; slot 3 marks the step
in flight). `serveChannel` hands the worker a `shouldStop` that reads that slot, and the
workers ask it before every run of the graph, so a cancel lands within one chunk; `opts`
may carry `maxChunk` to make the chunks of that step smaller (they are still cut at the
snapshot positions). A cancelled step answers `{cancelled: true, resident, stats}`: its
token is dropped even if it had finished, the `resident` positions it ran stay cached,
and the next step for the context resumes from them and, given the same `maxChunk`,
chooses what an uninterrupted step would have, bit for bit (with another `maxChunk` the
later chunks are cut differently, so it agrees only to float16 rounding). While a step is in flight every other call that reaches
the model thread throws `channel busy: decode in flight`. Every reply names its request;
a reply to another request (possible only after a call timed out) makes the channel
unusable, and every later call throws. Without the new options every
worker computes what it did before; replies only gain `resident` and `stats`
(`positions` run, `chunks` runs, `fillMs` spent in them but a final one-position decode).
`NodeWorker` has the same four members for CPython, over a pipe to the stub worker in a
Node child process (`node_bridge.mjs`): it waits with `select`, and a cancel is a control
frame `{cancel: id}` that the bridge reads while the step runs. `stub_worker.js` takes
`{stepMs, positionMs}` of simulated latency so tests can poll, time out and cancel
(`startNodeModel({stub})`, `NodeWorker(stub=...)`); neither serves a model.

What smaller chunks cost on WebGPU, measured by `export/bench/chunks.html`
(`tests/opt_zeos_webgpu.mjs --page chunks.html`) with the 4B in Chrome on an Apple M1
Max, while other jobs loaded the machine (median of five fresh prefills; a second
session agreed within 10% except the 250-position rows at 256 and the graph's chunk,
which it ran in about 835 ms):

| positions | graph's chunk | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|
| 250 | 1132 ms (1 run) | 1103 ms (1) | 1098 ms (2) | 1315 ms (4) | 2039 ms (8) |
| 1000 | 4328 ms (4) | 4416 ms (4) | 4560 ms (8) | 5619 ms (16) | 8825 ms (32) |

So each extra run costs roughly 60 to 160 ms at 64 and 130 to 160 ms at 32; at 1000
positions 128 is within 5% of the graph's chunk and 64 about 30% slower. A stop lands
at the next run's start, so its latency is at most one run: at 64, about 260 ms for each
of the first two runs of a 1000-position step and about 350 ms on average over all 16.
A step stopped before its third run (128 positions resident) returned at once, and,
resumed with the same `maxChunk`, matched the uninterrupted step bit for bit.

**Backends.** The model runs on WebGPU only, and the coop-count page shows it beside the
clock. That page's default is the Qwen model, falling back to the recorded answers when the page is
not cross-origin isolated or the browser has no WebGPU; in either case the model option
says why it is unavailable. A browser that has `navigator.gpu` but offers no adapter
gets the reason from the model thread when *run* is pressed, and loads nothing. The
build's default export is Qwen3.5-4B as the OPT+ZEOS graph
(`models/Qwen3.5-4B-ZEOS-OPT`, 2.8 GB, which `export/opt_zeos_surgery.py` writes); *The
OPT+ZEOS worker* gives its timings. Pass `--model` to offer another export, such as
`models/Qwen3.5-2B-zeos-q4`.

ONNX Runtime's WebGPU backend has no kernel for the int8 export's integer matmuls, so an
int8 export does not run on the page. The q4 export (`--quant q4`, 1.69 GB) runs on
WebGPU, and `demo/coop-count-web/build.py --model packages/zeos-browser/models/Qwen3.5-2B-zeos-q4`
puts it on the coop-count page: on an
Apple M1 Max its prompts prefilled in about 20 s and `coop-count-pipe` ran its 400 ticks
in 122 s, about 0.2 s a tick including Pyodide's side. It decodes differently from the
int8 export the 2B runs above were made with (4-bit weights, and the GPU's arithmetic):
counter-a counted from 1 to 50 without stopping at 10, recorded 50 and woke its peer;
counter-b said 10 and recorded 10; counter-a, woken, started again from 1 and was at 25
when the run was cut off; the handler read the 500 and then waited on the console again.

### The grammar

`JsMachine`'s Python token mask (`token_mask`) is kept. Against the Qwen3.5 vocabulary
(248,077 pieces) it is correct -- `export/bench/grammar.html` puts the 4B, on WebGPU,
after a context of injected text written to talk it out of the command language
(forged chat markers, a fake status line, prose demands) and every completed line parses
as a command, no control id is ever chosen and no `<` is ever emitted -- and it is fast
enough: with Qwen2.5's 151,665 pieces, building the mask for a round state the run had
not met cost 0.11 s on average under Pyodide (0.81 s at most, 0.06 s under CPython), and
the cost grows with the vocabulary, here 1.6 times larger; a cached state costs nothing,
so a counting round pays a few seconds once, against about 0.3 s a decode step for as
long as the run lasts.

## Tests

```bash
npm install --prefix packages/zeos-browser   # once: Pyodide for Node, and PyYAML cached beside it
uv run pytest packages/zeos-browser          # or, from packages/zeos-browser: uv run pytest
node --test packages/zeos-browser/tests/js/*.test.mjs
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
- `tests/js/channel_async.test.mjs` and `test_node_worker_async.py` — the decode step
  that does not block, over the stub with simulated latency: begin never waits, a poll
  times out and then returns the result, calls refuse while a step is in flight, a
  cancel lands between chunks and the step resumes from what it ran, a result that raced
  the cancel is dropped, and cancelled-then-resumed steps say what uncancelled ones do.
  `tests/js/model_channel.test.mjs` — what `SyncModelWorker` forwards. They need Node
  and skip without it.
- `tests/js/model_cache.test.mjs` — the model cache over a fake Hub and an in-memory store
  with `opfs_store.js`'s interface: a cache hit fetches nothing; the key holds the
  revision, so another revision downloads its own file; an interrupted download is never
  complete and resumes with a `Range` request (and restarts when the server ignores it);
  a corrupt part, a wrong hash or a wrong length stores nothing; quota is checked before
  downloading and a `QuotaExceededError` leaves no part; `sha256.js` against node:crypto.
  `test_model_source.py` — the manifest fields `build.py` writes.

No test in these suites runs the model, so none needs a GPU or the export. The model runs
on WebGPU only, and its checks are the opt-in `tests/opt_zeos_webgpu.mjs` (see *The
OPT+ZEOS worker*). No test, default or opt-in, downloads the model from the Hugging Face
Hub; the cache's tests are the Node ones above, with a fake server and store.
