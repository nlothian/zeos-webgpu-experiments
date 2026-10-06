// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The stub machine's thread: model_thread.js with the Space Invaders stub in place of
 * the model. The stub's decode step is asynchronous (it simulates latency), so it must
 * run on a thread of its own and answer the Pyodide worker through the same
 * SharedArrayBuffer channel the model does (`serveChannel`); the run loop then reaches
 * it as a `SyncModelWorker`, with begin/poll/cancel, exactly as it reaches the model.
 *
 * `startStubThread` runs on the page and returns what `attachModel` takes. The first
 * message to the thread configures it; it answers `{ready: true, backend: "stub"}` or
 * `{ready: false, error}`.
 */

import { CHANNEL_BYTES, serveChannel } from "./model_channel.js";

/**
 * @param {object} options
 * @param {string} options.stubUrl web/stub/pilot_stub_worker.js, which defines
 *   `self.createPilotStubWorker(opts)`.
 * @param {{stepMs?: number, positionMs?: number}} [options.opts] simulated latency.
 */
export async function startStubThread({ stubUrl, opts = {} }) {
  if (!self.crossOriginIsolated) {
    throw new Error("the page is not cross-origin isolated, so SharedArrayBuffer is unavailable");
  }
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const channel = new MessageChannel();
  const thread = new Worker(new URL("./stub_thread.js", import.meta.url), { type: "module" });
  const ready = new Promise((resolve, reject) => {
    thread.onmessage = (event) => (event.data.ready ? resolve(event.data.backend) : reject(new Error(event.data.error)));
    thread.onerror = (event) =>
      reject(new Error(`stub thread failed to start: ${event.message ?? "no message"} (${event.filename ?? "?"}:${event.lineno ?? "?"})`));
  });
  thread.postMessage(
    { buffer, port: channel.port1, stubUrl: new URL(stubUrl, self.location.href).href, opts },
    [channel.port1],
  );
  const backend = await ready;
  return { buffer, port: channel.port2, backend, thread };
}

// In the thread: the same module, entered by the configuring message. On the page
// `self.document` exists and this listener is never installed.
if (typeof self.document === "undefined") {
  self.onmessage = async (event) => {
    const { buffer, port, stubUrl, opts } = event.data;
    self.onmessage = null;
    try {
      await import(stubUrl);
      if (typeof self.createPilotStubWorker !== "function") {
        throw new Error(`${stubUrl} did not define self.createPilotStubWorker`);
      }
      const worker = self.createPilotStubWorker(opts);
      serveChannel(worker, buffer, (handle) => {
        port.onmessage = (message) => handle(message.data);
      });
      self.postMessage({ ready: true, backend: "stub" });
    } catch (error) {
      self.postMessage({ ready: false, error: String(error?.stack ?? error) });
    }
  };
}
