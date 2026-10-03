// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model thread under Node: a `worker_threads` Worker that loads the export and
 * answers `SyncModelWorker` calls through the shared buffer it is given. `startNodeModel`
 * is the main-thread side, and returns once the model has loaded.
 */

import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

import { CHANNEL_BYTES, SyncModelWorker, serveChannel } from "./model_channel.js";
import { loadNodeWorker } from "./node_load.mjs";

/** Start a model thread and return a synchronous worker over it. */
export async function startNodeModel({ modelDir, runtime = "web", threads = 1 } = {}) {
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const thread = new Worker(new URL(import.meta.url), {
    workerData: { modelDir, runtime, threads, buffer },
  });
  await new Promise((resolve, reject) => {
    thread.once("message", (message) => (message.ready ? resolve() : reject(new Error(message.error))));
    thread.once("error", reject);
  });
  const worker = new SyncModelWorker(buffer, (request) => thread.postMessage(request));
  worker.terminate = () => thread.terminate();
  return worker;
}

if (!isMainThread && workerData?.buffer) {
  const { modelDir, runtime, threads, buffer } = workerData;
  try {
    const worker = await loadNodeWorker({ modelDir, runtime, threads });
    serveChannel(worker, buffer, (handle) => parentPort.on("message", handle));
    parentPort.postMessage({ ready: true });
  } catch (error) {
    parentPort.postMessage({ ready: false, error: String(error.stack ?? error) });
  }
}
