// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model thread under Node, serving the stub: a `worker_threads` Worker that answers
 * `SyncModelWorker` calls through the shared buffer it is given, with `stub_worker.js`
 * over `stub` (`{tapes, options}`). `startNodeModel` is the main-thread side, and returns
 * once the thread is ready.
 *
 * The thread serves the decode step that does not block as the page's does
 * (`serveChannel` reads the abort slot before every chunk), so `beginDecodeStep`,
 * `pollDecode` and `cancelDecode` work on the returned worker unchanged, for tests of the
 * channel that need a step that takes time (`options.stepMs`, `positionMs`).
 */

import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

import { CHANNEL_BYTES, SyncModelWorker, serveChannel } from "./model_channel.js";

/** Start a stub thread and return a synchronous worker over it. */
export async function startNodeModel({ stub, timeoutMs } = {}) {
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const thread = new Worker(new URL(import.meta.url), { workerData: { stub, buffer } });
  await new Promise((resolve, reject) => {
    thread.once("message", (message) => (message.ready ? resolve() : reject(new Error(message.error))));
    thread.once("error", reject);
  });
  const worker = new SyncModelWorker(buffer, (request) => thread.postMessage(request), { timeoutMs });
  worker.terminate = () => thread.terminate();
  return worker;
}

if (!isMainThread && workerData?.buffer) {
  const { stub, buffer } = workerData;
  try {
    await import("./stub_worker.js");
    const worker = globalThis.createStubWorker(stub.tapes, stub.options ?? {});
    serveChannel(worker, buffer, (handle) => parentPort.on("message", handle));
    parentPort.postMessage({ ready: true });
  } catch (error) {
    parentPort.postMessage({ ready: false, error: String(error.stack ?? error) });
  }
}
