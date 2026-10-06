# coop-count-web — the coop-count demo in a static web page

The ZEOS kernel and the coop-count demo, running inside [Pyodide](https://pyodide.org) in a
browser, with no server beyond a file host. A member of the repository's uv workspace.

The page loads Pyodide, installs the `zeos` and `zeos-browser` wheels, writes the
coop-count cases into Pyodide's in-memory filesystem, lints the chosen case and runs it.
The run's output panel has two tabs, both filled as the run streams. *transcript*, the
default, shows the transcript `zeos-count run` prints to a terminal (`counter-a  say 1`,
`reset-count ──▶ count.progress_a 51`, `counter-a  ... waiting on count.b2a`), collected
from the same seat callbacks and journal events the CLI prints from
(`zeos_browser.transcript`), so it never alters the journal. *event log (JSON)* shows the
journal itself, one JSON line per event. The ZEOS debugger draws the
case's wiring and then the finished run, and the journal can be downloaded as the
`.jsonl` file `zeos-count run --journal` would have written. The *interrupt* button on the page is the
console's interrupt.

## Getting it running

**You need** [uv](https://docs.astral.sh/uv/), Node.js with npm (tested with Node 22),
and a browser with WebGPU (tested in Chrome on an Apple M1 Max). The first time the
language model runs, the page downloads it from the Hugging Face Hub (about 2.4 GB) and
keeps it in the browser's storage, so the browser needs that much room for the site.

Every command runs from the repository root.

```bash
# 1. Python packages.
uv sync --all-packages

# 2. The tokenizer, which build.py copies into the page, and the ONNX Runtime Web
#    version it names, which the page loads from jsDelivr.
npm install --prefix packages/zeos-browser

# 3. Assemble the page into demo/coop-count-web/web/dist/.
uv run python demo/coop-count-web/build.py

# 4. Serve it, cross-origin isolated, on port 8765.
uv run python demo/coop-count-web/serve.py
```

Open <http://localhost:8765/> and press **run**. The defaults are the language model on
the GPU, the scenario `coop-count-scripted`, and *press the keys automatically*, which
sends the interrupt at 39 ms of virtual time. The first run downloads the model from
[`nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16`](https://huggingface.co/nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16)
at the commit `build.py` pins, checks every file's size and SHA-256, and prefills the
first prompt; a reload, or the browser started again, loads it from the browser's
storage instead (in Chrome on an M1 Max, the model was ready about 4–5 s after **run**
from the browser's storage, against about 20 s for a download from a local server). *Loading and caching the model* and *The OPT+ZEOS worker* in
[`packages/zeos-browser`](../../packages/zeos-browser/README.md) say how, and give the
model worker's load, prefill and decode timings.
The *transcript* tab shows the counters counting and the interrupt handler resetting
the count.

Under the *model* line, a line says where the model comes from, how much of it the
browser has stored (and whether it may evict it when short of space), with a **clear
cached model** button; after it, the next run downloads the model again. A download cut
short (the tab closed, the network gone) resumes where it stopped on the next run.

`build.py` prints the model it put on the page. If it says `model: none`, step 2 is
missing. The page then still runs every scenario that has recorded answers, and offers
the language model as *(not in this build)*.

**A local export.** For working on the model, `build.py --model` links the export at
`packages/zeos-browser/models/Qwen3.5-4B-ZEOS-OPT` (made as *Getting the model* in
`packages/zeos-browser` describes) into the page and loads it by default, from this
server and without the browser's cache; `--model DIR` names another export, for example
`--model packages/zeos-browser/models/Qwen3.5-2B-zeos-q4`. Either source can be chosen per
load with `?model=huggingface` or `?model=local` in the URL. `--hf-repo` and
`--hf-revision` (a full commit) change the Hub source; `manifest.json` records the
sources (`model_sources`) and the default (`model_source`).

**Serving it elsewhere.** `web/dist/` is a static site: upload it to any file host. The
page fetches Pyodide and ONNX Runtime Web from the jsDelivr CDN and the model from the
Hub, so the browser needs a network connection the first time. No file in it is over
25 MiB, the most Cloudflare Pages takes. The language model needs the page
cross-origin isolated: `serve.py` sends the two headers that takes, and on a host that
cannot send them `coi_serviceworker.js` adds them in the browser after one reload; the
Hub's responses are CORS responses, which the isolation admits. With `--model`,
`web/dist/models/` is a symbolic link to the export, so copy the export in with
`build.py --model --copy-model` before uploading. The recorded-answer machines run on any
host, `python -m http.server` included.

## What runs

The kernel is pure Python over the standard library and PyYAML, which Pyodide ships. It
reads no clock, starts no thread and opens no socket, and the driver injects time as one
virtual millisecond per tick, so a run in the browser writes the same journal bytes as
the same run under CPython. `tests/test_pyodide_determinism.py` checks exactly that.

Three machines are offered for a case, under *answered by*:

| on the page | machine | what answers each decode |
|---|---|---|
| Qwen3.5-4B language model, in your browser | `transformers` | `JsMachine` over a model worker: zeos-browser's `web/opt_zeos_worker.js` for the default OPT+ZEOS export, `web/transformers_worker.js` for an `export_model.py` one. A language model decoding under the kernel, with measured attention. The default; offered when the build found an export. |
| recorded answers (no model) | `scripted` | `CommandSeat` over `TapeSource`: each descriptor's `script:` tape, one word per decode. This is `zeos-count run --machine scripted`. |
| recorded answers, via the JavaScript model interface | `stub` | `JsMachine` over zeos-browser's stub worker, `web/stub_worker.js`, which plays the same tapes through every method of the JavaScript seam. |

On `coop-count-scripted` the two decode the same words and their journals hold the same
events in the same order, with two exceptions, both about attention. The seat's mask
hides the blocks a job has written since the kernel's last mask refresh, so the kernel
journals the job's attention to its own open segment as denied (`mask.denied`, 30 times
in this run) and leaves that segment out of the working set; `JsMachine` treats those
blocks as visible (see *the mask's horizon* in zeos-browser's README), so it has no `mask.denied` events
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

**The run loop.** `zeos_browser.live.LiveRun` is the `zeos-count` run loop cut
into single turns: deliver what the schedule has due, deliver what the page queued,
tick, reap, add one virtual millisecond. With no presses its journal is the CLI's,
which zeos-browser's `tests/test_js_machine_case.py` checks against `zeos_coop_count.cli` when
llama-cpp-python is installed. A press is delivered at the start of the next turn, the
place the CLI delivers a scheduled event, so pressing at the turns `events.jsonl` names
reproduces the scheduled run exactly. While a job is parked on a device pipe the run
stays open, idling, until a press arrives or *stop* is pressed.

**What the model is doing.** A *model* line under the run buttons follows the model
thread: the download as a whole (the export's `meta.json` lists every file's size), or
the read from the browser's storage, and the check of a resumed download, then the
session ONNX Runtime builds once the files are in, and then every run of the graph --
which prompt is being prefilled and how far, with an estimate of the time left, or which
decode step is in flight -- with the mean step time once steps have run. The kernel is
blocked while a step runs, so this line is the one that moves when the journal does not.
The thread reports each run of the graph through the worker's `onActivity` callback.

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
`web/pyodide_worker.js` and the npm dependency in `packages/zeos-browser/package.json`. This release refuses to
load in a classic Web Worker, so the worker is a module worker.

## Why separate packages

`zeos-coop-count` declares `llama-cpp-python`, `huggingface-hub` and `anthropic` as hard
dependencies, none of which installs under Pyodide, and its `machine.py`, `boot.py` and
`cli.py` import llama.cpp at module level. Rather than install that wheel without its
dependencies and import around it, the page runs on `zeos` and
[`zeos-browser`](../../packages/zeos-browser/README.md), which depends on `zeos` alone and
carries what a browser needs: `JsMachine` and its grammar mask, the run loop the page
steps (`zeos_browser.live`, `zeos_browser.page`), the model workers and the channel to
them. The scripted seat is `zeos.machine.seat`, the cases are read from
`demo/coop-count/cases/`, and the terminal-only `keyboard.py` is replaced by the page's
console. `JsMachine`'s word-to-token bookkeeping and ChatML framing are `LlamaMachine`'s,
written again over the worker interface, since the module that holds `LlamaMachine`
imports llama.cpp. `demo/coop-count` is unchanged.

This demo keeps the page itself (`web/index.html`, `app.js`, `style.css`,
`pyodide_worker.js`), `build.py`, which copies zeos-browser's JavaScript beside it, and
`serve.py`, and its Python package, `zeos-coop-count-web`, holds the reader of the
measured-attention evidence (`zeos_coop_count_web.evidence`).

## The machine seam, the model and the chat agent

`JsMachine` and the worker interface it drives, the grammar mask, the stub worker and
the Python fake, the model (Qwen3.5-4B as the OPT+ZEOS graph), its export, the workers
that run it on WebGPU, the channel that makes them synchronous, and the model's checks on
WebGPU are documented in [`packages/zeos-browser`](../../packages/zeos-browser/README.md).
The chat agent the gemma-data-agent site runs, `ChatToolMachine`, is
[`packages/zeos-chat`](../../packages/zeos-chat/README.md).

## Running it

In the page: follow *Getting it running*, leave *answered by* on the language model,
choose a scenario, and run. A run stops at 400 ticks, as
`demo/coop-count/run_all.py` stops a live seat, or earlier when no job can run: quiescent
when every job is asleep on another, and, in the page, held open while a job waits on
the console.

## Measured attention

The write-up and its journals were made with the Qwen2.5-0.5B-Instruct int8 export in
the earlier two-graph format, whose every layer was softmax attention; its numbers are
that model's. They were run from a shell, with the model worker on the WebAssembly
backend under Node, a path the repository no longer has (`node_run` and
`docs/evidence/run.sh`, last at `a397805`); the journals and attention files are kept as
the record they are, and `zeos_coop_count_web.evidence` still reads them:

```bash
uv run python -m zeos_coop_count_web.evidence demo/coop-count-web/docs/evidence/pipe-untrusted.jsonl
```

A later run of the same script with Qwen3.5-2B, whose measured attention covers only its
six softmax layers (see *The architecture* in zeos-browser's README), found that on `coop-count-pipe` with its
schedule the 2B's 95 decode steps all carried measured attention, no segment less trusted
than its reader received any, and the newest, not yet masked block got 0.05 of a step's
attention on average (0.28 at most).

`docs/evidence/README.md` is the write-up, with the journals it cites beside it: a
demotion whose `because` names an untrusted segment and the measured mass that crossed
`theta_read`; the same run at a `theta_read` above that mass, with no demotion; and a
block hidden through `set_mask` that measures exactly zero on every step. In short: the
handler of `coop-count-pipe`, with the console declared untrusted, put 0.36 of its first
step's attention on the keypress and was demoted for it; and the block a job is writing
into gets 0.17 of each step's attention on average.

## Determinism

The kernel's half of a run is deterministic in the page as under CPython: with the stub
worker, a run in Pyodide writes the same journal bytes as the same run under CPython
(`tests/test_pyodide_determinism.py`). A model run is not claimed to be: WebGPU's
floating-point order is the GPU's, so a journal made on one device is not claimed to
match one made on another. On the one device checked, with the 0.5B's q4 export, three runs of
twenty greedy steps of a short chat prompt in the same browser chose the same tokens and
reported bit-identical attention.

## Tests

```bash
npm install --prefix packages/zeos-browser   # once: Pyodide for Node, and PyYAML cached beside it
uv run pytest demo/coop-count-web            # or, from demo/coop-count-web: uv run pytest
```

- `test_page.py` — the page's Python half, and that `build.py` assembles every file the
  page fetches.
- `test_pyodide_determinism.py` — the smoke fixture, the seat and the stub, each under
  Pyodide in Node and under CPython, compared byte for byte. It builds the `zeos` and
  `zeos-browser` wheels with `uv build`, installs them by unpacking as Pyodide installs a
  pure wheel, and copies the cases into the in-memory filesystem as the page does. It
  skips when Node, the npm package or the cached PyYAML is missing; `ZEOS_PYODIDE_DIR`
  points it at a Pyodide installed elsewhere.

`JsMachine`, the grammar mask, the channel, the model cache and the model have their
tests in `packages/zeos-browser`, and the chat agent in `packages/zeos-chat`. No test here
runs the model, so none needs a GPU or the export.

No test, default or opt-in, downloads the model from the Hugging Face Hub: the model
cache is tested with a fake server and an in-memory store
(`packages/zeos-browser/tests/js/model_cache.test.mjs`).

`uv build` (used by `build.py`, the determinism test and `test_page.py`) fetches the
hatchling build backend from PyPI the first time it runs, so those need a network once.
