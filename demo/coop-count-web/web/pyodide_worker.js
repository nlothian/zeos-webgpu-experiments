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
// synchronously across Pyodide's FFI.

import { loadPyodide, version as PYODIDE_VERSION } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import "./stub_worker.js";

const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let page = null;
let run = null;
let delay = 120;
let presses = 0;

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

  page = pyodide.pyimport("zeos_coop_count_web.page");
  const python = pyodide.runPython("import sys; sys.version.split()[0]");
  post("ready", { cases: Object.keys(manifest.cases), pyodide: pyodide.version, python });
}

function describe(name) {
  post("described", { name, info: JSON.parse(page.describe(caseDir(name))) });
}

function linesOf(proxy) {
  const lines = proxy.toJs();
  proxy.destroy();
  return lines;
}

function start({ name, machine, schedule }) {
  if (run !== null) stop("replaced by a new run");
  const dir = caseDir(name);
  if (machine === "stub") {
    const worker = self.createStubWorker(JSON.parse(page.tapes_json(dir)));
    run = page.open_run.callKwargs(dir, "js", { schedule, worker });
  } else {
    run = page.open_run.callKwargs(dir, "scripted", { schedule });
  }
  presses = 0;
  post("started", { name, machine, schedule, lines: linesOf(run.lines()) });
  setTimeout(turn, 0);
}

function turn() {
  if (run === null) return;
  try {
    const lines = linesOf(run.step());
    post("lines", {
      lines,
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
    handlers[event.data.type](event.data);
  } catch (err) {
    post("error", { text: String(err) });
  }
};
