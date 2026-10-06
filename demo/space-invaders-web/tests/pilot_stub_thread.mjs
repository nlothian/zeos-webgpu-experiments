// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The page's stub_thread.js for Node: web/stub/pilot_stub_worker.js on a worker thread
// of its own, served over zeos-browser's channel (serveChannel), reached from the main
// thread as a SyncModelWorker. pyodide_run.mjs --pilot-stub uses it, so a Python run
// under Pyodide begins, polls and cancels the stub's steps through the real channel, as
// si_worker.js does in the browser.

import path from "node:path";
import { pathToFileURL } from "node:url";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

const here = path.dirname(new URL(import.meta.url).pathname);
const browserWeb = path.join(here, "..", "..", "..", "packages", "zeos-browser", "web");
const stubFile = path.join(here, "..", "web", "stub", "pilot_stub_worker.js");

async function channel() {
  return import(pathToFileURL(path.join(browserWeb, "model_channel.js")).href);
}

/** Start the stub's thread and return a SyncModelWorker over it. */
export async function startPilotStub(opts = {}) {
  const { CHANNEL_BYTES, SyncModelWorker } = await channel();
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const thread = new Worker(new URL(import.meta.url), { workerData: { pilotStub: opts, buffer } });
  await new Promise((resolve, reject) => {
    thread.once("message", (message) => (message.ready ? resolve() : reject(new Error(message.error))));
    thread.once("error", reject);
  });
  const worker = new SyncModelWorker(buffer, (request) => thread.postMessage(request));
  worker.terminate = () => thread.terminate();
  return worker;
}

if (!isMainThread && workerData?.pilotStub) {
  const { pilotStub, buffer } = workerData;
  try {
    const { serveChannel } = await channel();
    await import(pathToFileURL(stubFile).href);
    const worker = globalThis.createPilotStubWorker(pilotStub);
    serveChannel(worker, buffer, (handle) => parentPort.on("message", handle));
    parentPort.postMessage({ ready: true });
  } catch (error) {
    parentPort.postMessage({ ready: false, error: String(error.stack ?? error) });
  }
}
