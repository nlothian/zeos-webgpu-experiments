// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Run one Python file under Pyodide in Node, with wheels installed and host
// directories copied into the in-memory filesystem, the way si_worker.js does it.
// coop-count-web's harness, for this page: Pyodide comes from zeos-browser, whose npm
// install is the one this demo uses.
//
//   node pyodide_run.mjs --wheel W.whl ... --copy HOST:PYPATH ... \
//        --fetch PYPATH:HOST ... --js FILE.js ... SCRIPT.py
//
// --copy puts a host directory into MEMFS before the script runs; --fetch copies one
// file back out afterwards; --js loads a classic script, such as web/stub_worker.js,
// into the global scope the script reaches as Pyodide's `js` module; --pilot-stub OPTS_JSON
// starts web/stub/pilot_stub_worker.js on a thread of its own (tests/pilot_stub_thread.mjs),
// as the page does, and puts its synchronous face in that scope as `zeosPilotStub`. The scope also has `zeosControl`, an Int32Array over a control
// SharedArrayBuffer like the page's (slot 0 is Stop), and `zeosStopAfter(ms)`, which sets
// Stop from another thread `ms` later -- the run loop never yields, so a timer on this
// thread could not -- and records Date.now() at that moment in `zeosStoppedAt()`. Nothing is mounted: the script sees exactly the
// filesystem the page builds, so a difference between the two runs is Pyodide's and
// not the host's directory order.
//
// ZEOS_PYODIDE_DIR overrides where the pyodide npm package is looked for; otherwise it
// is node_modules here (build.py --link-node-modules) or zeos-browser's.

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function splitAt(text, at) {
  return [text.slice(0, at), text.slice(at + 1)];
}

const args = process.argv.slice(2);
const wheels = [];
const copies = [];
const fetches = [];
const scripts = [];
let pilotStub = null;
let script = null;
for (let i = 0; i < args.length; i++) {
  const flag = args[i];
  if (flag === "--wheel") wheels.push(args[++i]);
  // Split at the colon nearest the in-Pyodide path, which never holds one, so a host
  // path with a drive letter survives.
  else if (flag === "--copy") copies.push(splitAt(args[++i], args[i].lastIndexOf(":")));
  else if (flag === "--fetch") fetches.push(splitAt(args[++i], args[i].indexOf(":")));
  else if (flag === "--js") scripts.push(args[++i]);
  else if (flag === "--pilot-stub") pilotStub = JSON.parse(args[++i]);
  else script = flag;
}
if (!script) {
  console.error(
    "usage: pyodide_run.mjs [--wheel W] [--copy HOST:PY] [--fetch PY:HOST] [--js F] SCRIPT.py",
  );
  process.exit(2);
}

const here = path.dirname(new URL(import.meta.url).pathname);
const browser = path.join(here, "..", "..", "..", "packages", "zeos-browser");
const pyodideDir =
  process.env.ZEOS_PYODIDE_DIR ||
  [path.join(here, "..", "node_modules", "pyodide"), path.join(browser, "node_modules", "pyodide")].find((dir) =>
    fs.existsSync(path.join(dir, "pyodide.mjs")),
  );
if (!pyodideDir) {
  console.error("no pyodide npm package: run npm install in packages/zeos-browser");
  process.exit(2);
}
const { loadPyodide } = await import(pathToFileURL(path.join(pyodideDir, "pyodide.mjs")).href);

const pyodide = await loadPyodide({ indexURL: pyodideDir + path.sep });
pyodide.setStdout({ batched: (line) => console.log(line) });
pyodide.setStderr({ batched: (line) => console.error(line) });
await pyodide.loadPackage(["pyyaml"], { messageCallback: () => {} });

for (const wheel of wheels) {
  // A Uint8Array rather than the Buffer readFileSync returns: Pyodide refuses Buffer.
  pyodide.unpackArchive(new Uint8Array(fs.readFileSync(wheel)), "wheel");
}

function copyTree(host, target) {
  pyodide.FS.mkdirTree(target);
  // Sorted so the in-memory directory is built in the same order on every host.
  for (const name of fs.readdirSync(host).sort()) {
    const from = path.join(host, name);
    const to = `${target}/${name}`;
    if (fs.statSync(from).isDirectory()) copyTree(from, to);
    else pyodide.FS.writeFile(to, new Uint8Array(fs.readFileSync(from)));
  }
}
for (const [host, target] of copies) copyTree(host, target);
for (const file of scripts) await import(pathToFileURL(path.resolve(file)).href);
const controlSab = new SharedArrayBuffer(16);
globalThis.zeosControl = new Int32Array(controlSab);
globalThis.zeosStoppedAt = () => new Float64Array(controlSab)[1];
globalThis.zeosStopAfter = (ms) => {
  const { Worker } = globalThis.zeosWorkerThreads;
  new Worker(
    "const { workerData } = require('node:worker_threads');" +
      "setTimeout(() => { const c = new Int32Array(workerData.sab);" +
      " new Float64Array(workerData.sab)[1] = Date.now();" +
      " Atomics.store(c, 0, 1); Atomics.notify(c, 0); }, workerData.ms);",
    { eval: true, workerData: { sab: controlSab, ms } },
  ).unref();
};
globalThis.zeosWorkerThreads = await import("node:worker_threads");
if (pilotStub !== null) {
  const { startPilotStub } = await import(pathToFileURL(path.join(here, "pilot_stub_thread.mjs")).href);
  globalThis.zeosPilotStub = await startPilotStub(pilotStub);
}

let status = 0;
try {
  await pyodide.runPythonAsync(fs.readFileSync(script, "utf-8"));
} catch (err) {
  console.error(String(err));
  status = 1;
}
for (const [source, host] of fetches) {
  fs.writeFileSync(host, pyodide.FS.readFile(source));
}
process.exit(status);
