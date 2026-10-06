// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The Web Worker that hosts Pyodide, the kernel, the game and the run loop.
//
// A run is one synchronous Python call (`run.run()`) that owns both the game clock and
// the kernel, so while it runs this worker's event loop never turns and no message
// reaches it. That is why Stop is not a message: the page writes control[CONTROL_STOP]
// of a SharedArrayBuffer handed over in `boot`, and the loop reads it every iteration.
// The same word is what the loop sleeps on (`clock.sleep` is an Atomics.wait on it), so
// Stop also wakes a sleeping loop. Frames go out with postMessage, which works from a
// busy worker: only receiving needs the event loop.
//
// The model thread, and the stub's thread (stub_thread.js) for the stub machine, are
// started by the page (a worker started from inside a worker failed in the Chromium
// coop-count-web was tested in) and handed over in `attachModel` as backends "webgpu"
// and "stub", as in coop-count-web's pyodide_worker.js. Either is reached through a
// SyncModelWorker, so the run loop begins, polls and cancels decode steps the same way. Message shapes are CONTRACTS.md section 3.

import { loadPyodide, version as PYODIDE_VERSION } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import { SyncModelWorker } from "./model_channel.js";

const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
/** contracts.CONTROL_STOP */
const CONTROL_STOP = 0;
/** The copied case, which the debugger draws. */
const CASE_ROOT = "/cases/space-invaders";

let pyodide = null;
let page = null;
let control = null;
/** Whether the build has the Space Invaders stub (manifest.stub). */
let stubBuilt = false;
/** Model threads the page has handed over, by backend. */
const models = new Map();

function post(type, body = {}) {
  self.postMessage({ type, ...body });
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function boot({ controlSab }) {
  if (!(controlSab instanceof SharedArrayBuffer)) {
    throw new Error("boot needs the control SharedArrayBuffer; the page must be cross-origin isolated");
  }
  control = new Int32Array(controlSab);
  post("status", { text: `loading Pyodide ${PYODIDE_VERSION}` });
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  pyodide.setStdout({ batched: (line) => post("log", { text: line }) });
  pyodide.setStderr({ batched: (line) => post("log", { text: line }) });

  const manifest = await (await fetch("manifest.json")).json();
  post("status", { text: `installing ${manifest.wheels.length} wheels (and PyYAML from Pyodide)` });
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(manifest.wheels.map((name) => new URL(`wheels/${name}`, self.location.href).href));
  micropip.destroy();

  for (const [name, files] of Object.entries(manifest.cases)) {
    for (const file of files) {
      const target = `/cases/${name}/${file}`;
      pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
      pyodide.FS.writeFile(target, await fetchBytes(`cases/${name}/${file}`));
    }
  }
  stubBuilt = manifest.stub === true;
  page = pyodide.pyimport("zeos_space_invaders_web.page");
  const python = pyodide.runPython("import sys; sys.version.split()[0]");
  post("ready", {
    pyodide: pyodide.version,
    python,
    isolated: self.crossOriginIsolated,
    model: manifest.model ?? null,
    stub: manifest.stub === true,
    describe: JSON.parse(page.describe_json(CASE_ROOT)),
  });
}

/** Wrap the model thread the page started, whose buffer and port it has handed over. */
function attachModel({ backend, buffer, port, name }) {
  models.set(backend, new SyncModelWorker(buffer, (m) => port.postMessage(m)));
  post("model", { name, backend });
}

const stop = { is_set: () => Atomics.load(control, CONTROL_STOP) !== 0 };
const clock = {
  now: () => performance.now() / 1000,
  // Returns at once if Stop is already set, and the moment the page sets it.
  sleep: (seconds) => {
    if (seconds > 0) Atomics.wait(control, CONTROL_STOP, 0, seconds * 1000);
  },
};

/** The channel for `machine`: the model thread or the stub thread the page attached.
 * Only a build without the stub runs the stub machine on no worker (a FakeRun). */
function workerFor(machine) {
  const backend = machine === "model" ? "webgpu" : "stub";
  const worker = models.get(backend);
  if (worker !== undefined) return worker;
  if (machine === "stub" && !stubBuilt) return null;
  throw new Error(`no ${backend} thread attached; the page starts one before "start"`);
}

// The page clears control[CONTROL_STOP] before posting `start`, so a Stop pressed while
// the model was loading is not lost.
function start({ arm, board, seed, machine, tune = "" }) {
  const worker = workerFor(machine);
  page.configure_json(tune);
  const onFrame = page.json_sink((text) => {
    const frame = JSON.parse(text);
    post("frame", frame);
    for (const decision of frame.decisions) post("decision", decision);
  });
  // The first error wins: a run that fails and then fails to close reports why it
  // failed, and every proxy is destroyed on every path.
  let failure = null;
  try {
    const run = page.open_run.callKwargs(arm, board, seed ?? null, worker, {
      stub: machine === "stub",
      on_frame: onFrame,
      stop,
      clock,
    });
    try {
      post("warming", { arm, board });
      run.warm();
      post("started", { arm, board, seed, machine, stopped: stop.is_set() });
      const result = run.run();
      try {
        const finished = JSON.parse(page.finished_json(result, CASE_ROOT));
        post("finished", { ...finished, stopped: stop.is_set() });
      } finally {
        result.destroy();
      }
    } catch (err) {
      failure = err;
    } finally {
      try {
        run.close();
      } catch (err) {
        failure ??= err;
      } finally {
        run.destroy();
      }
    }
  } catch (err) {
    failure ??= err;
  } finally {
    onFrame.destroy();
  }
  if (failure !== null) throw failure;
}

const handlers = { attachModel, start };
let ready = null;

self.onmessage = async (event) => {
  const { type } = event.data;
  try {
    if (type === "boot") {
      ready = boot(event.data);
      await ready;
      return;
    }
    if (ready === null) throw new Error(`"${type}" before "boot"`);
    await ready;
    await handlers[type](event.data);
  } catch (err) {
    post("error", { message: err?.message ? `${err.name}: ${err.message}` : String(err), stack: err?.stack ?? "" });
  }
};
