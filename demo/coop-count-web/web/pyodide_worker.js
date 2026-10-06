// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The Web Worker that hosts Pyodide, the kernel and the run.
//
// Python runs here rather than on the page's thread so that loading Pyodide, building a
// debugger payload or a slow machine never freezes the page. The run is stepped from
// JavaScript, one turn of LiveRun per timer tick: between two turns the worker's event
// loop is free, so a keypress posted from the page is queued with run.press() and
// delivered by the next turn, at that turn's virtual time. The delay between turns is
// the page's speed control and has no effect on the journal, which counts virtual
// milliseconds, never real ones.
//
// A module worker, because Pyodide 314 refuses to load in a classic one. The stub worker
// (stub_worker.js) is imported into this same global scope, because JsMachine calls it
// synchronously across Pyodide's FFI. The model worker is synchronous here too, but its
// model runs on a thread of its own (model_thread.js), which the page starts and hands
// over; this worker calls it through a SharedArrayBuffer, blocking in Atomics.wait
// (model_channel.js). The page starts it rather than this worker because a worker
// started from inside a worker failed to start in the Chromium this was tested in.

import { loadPyodide, version as PYODIDE_VERSION } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import "./stub_worker.js";
import { SyncModelWorker } from "./model_channel.js";

const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let page = null;
let run = null;
let delay = 120;
let presses = 0;
/** The model thread, kept across runs: loading it is the slow part. */
let model = null;
/** A live model seat counts for ever, so a run of it stops here, as run_all.py stops one. */
const MODEL_MAX_TICKS = 400;

function post(type, body = {}) {
  self.postMessage({ type, ...body });
}

function caseDir(name) {
  return `/cases/${name}`;
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function boot() {
  post("status", { text: `loading Pyodide ${PYODIDE_VERSION}` });
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  pyodide.setStdout({ batched: (line) => post("log", { text: line }) });
  pyodide.setStderr({ batched: (line) => post("log", { text: line }) });

  const manifest = await (await fetch("manifest.json")).json();
  post("status", { text: "installing the zeos and zeos-coop-count-web wheels" });
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  const wheels = manifest.wheels.map((name) => new URL(`wheels/${name}`, self.location.href).href);
  await micropip.install(wheels);
  micropip.destroy();

  post("status", { text: "writing the cases into the in-memory filesystem" });
  for (const [name, files] of Object.entries(manifest.cases)) {
    for (const file of files) {
      const target = `${caseDir(name)}/${file}`;
      pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
      pyodide.FS.writeFile(target, await fetchBytes(`cases/${name}/${file}`));
    }
  }

  page = pyodide.pyimport("zeos_browser.page");
  const python = pyodide.runPython("import sys; sys.version.split()[0]");
  post("ready", {
    cases: Object.keys(manifest.cases),
    pyodide: pyodide.version,
    python,
    model: manifest.model ?? null,
    // The model thread answers through a SharedArrayBuffer, which needs isolation.
    isolated: self.crossOriginIsolated,
  });
}

function describe(name) {
  post("described", { name, info: JSON.parse(page.describe(caseDir(name))) });
}

function linesOf(proxy) {
  const lines = proxy.toJs();
  proxy.destroy();
  return lines;
}

/** Wrap the model thread the page started, whose buffer and port it has handed over. */
function attachModel({ backend, buffer, port, name }) {
  model = new SyncModelWorker(buffer, (m) => port.postMessage(m));
  post("model", { name, backend });
}

function start({ name, machine, schedule }) {
  if (run !== null) stop("replaced by a new run");
  const dir = caseDir(name);
  if (machine === "transformers") {
    const worker = model;
    if (worker === null) throw new Error("no model thread; the page starts one");
    run = page.open_run.callKwargs(dir, "js", { schedule, worker, max_ticks: MODEL_MAX_TICKS });
  } else if (machine === "stub") {
    const worker = self.createStubWorker(JSON.parse(page.tapes_json(dir)));
    run = page.open_run.callKwargs(dir, "js", { schedule, worker });
  } else {
    run = page.open_run.callKwargs(dir, "scripted", { schedule });
  }
  presses = 0;
  post("started", {
    name,
    machine,
    schedule,
    lines: linesOf(run.lines()),
    transcript: linesOf(run.transcript_lines()),
  });
  setTimeout(turn, 0);
}

function turn() {
  if (run === null) return;
  try {
    const lines = linesOf(run.step());
    post("lines", {
      lines,
      // What `zeos-count run` would have printed this turn, for the page's user view.
      transcript: linesOf(run.transcript_lines()),
      virtualNs: run.now_ns,
      ticks: run.ticks,
      awaiting: run.awaiting_input(),
      parked: run.blocked_on("keys.number"),
      withheld: run.withheld.length,
    });
  } catch (err) {
    post("error", { text: String(err) });
    run.stop("failed");
  }
  if (run.finished) finish();
  else setTimeout(turn, delay);
}

function finish() {
  const done = run;
  run = null;
  const bytes = done.journal_bytes();
  const journal = bytes.toJs();
  bytes.destroy();
  const payload = page.payload_json(done);
  post("finished", {
    reason: done.reason,
    ticks: done.ticks,
    virtualNs: done.now_ns,
    presses,
    journal,
    payload,
  });
  done.destroy();
}

function stop(reason) {
  if (run === null) return;
  run.stop(reason);
  finish();
}

const handlers = {
  describe: ({ name }) => describe(name),
  attachModel,
  start,
  press: ({ pipe, text, unlessWaitingOn }) => {
    if (run === null) return;
    run.press.callKwargs(pipe, text, { unless_waiting_on: unlessWaitingOn ?? null });
    presses += 1;
    post("pressed", { pipe, text, atNs: run.now_ns });
  },
  speed: ({ ms }) => {
    delay = ms;
  },
  stop: () => stop("stopped from the page"),
};

const ready = boot().catch((err) => post("error", { text: `boot failed: ${err}` }));

self.onmessage = async (event) => {
  await ready;
  try {
    await handlers[event.data.type](event.data);
  } catch (err) {
    post("error", { text: err?.message ? `${err.name}: ${err.message}` : String(err) });
  }
};
