// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * Start the model thread from the page. What comes back is what the thread running
 * Pyodide needs to build a `SyncModelWorker`: the shared buffer replies arrive in, and a
 * MessagePort requests go out on. Both are transferable to another worker, which is how
 * the Pyodide worker gets them.
 */

import { CHANNEL_BYTES } from "./model_channel.js";

/**
 * @param {object} options
 * @param {string} options.modelUrl the export directory, relative to the page.
 * @param {string} options.ortWebgpuUrl onnxruntime-web's `ort.webgpu.min.mjs`.
 * @param {string} options.tokenizersUrl `@huggingface/tokenizers`' `tokenizers.min.mjs`.
 * @param {(progress: object) => void} [options.onProgress] download and session progress.
 * @param {(activity: object) => void} [options.onActivity] every run of the graph, before
 *   and after, as `TransformersWorker`'s `onActivity` reports it.
 */
export async function startBrowserModel(options) {
  if (!self.crossOriginIsolated) {
    throw new Error(
      "the page is not cross-origin isolated, so SharedArrayBuffer is unavailable; serve it " +
        "with COOP/COEP headers (serve.py does) or let coi.js install its service worker",
    );
  }
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const channel = new MessageChannel();
  const thread = new Worker(new URL("./model_thread.js", import.meta.url), { type: "module" });
  const absolute = (url) => new URL(url, self.location.href).href;
  const ready = new Promise((resolve, reject) => {
    thread.onmessage = (event) => {
      const data = event.data;
      if (data.progress) options.onProgress?.(data.progress);
      else if (data.activity) options.onActivity?.(data.activity);
      else if (data.ready) resolve(data.backend);
      else reject(new Error(data.error));
    };
    thread.onerror = (event) =>
      reject(new Error(`model thread failed to start: ${event.message ?? "no message"} (${event.filename ?? "?"}:${event.lineno ?? "?"})`));
  });
  thread.postMessage(
    {
      buffer,
      port: channel.port1,
      modelUrl: absolute(options.modelUrl),
      ortWebgpuUrl: absolute(options.ortWebgpuUrl),
      tokenizersUrl: absolute(options.tokenizersUrl),
    },
    [channel.port1],
  );
  const backend = await ready;
  return { buffer, port: channel.port2, backend, thread };
}
