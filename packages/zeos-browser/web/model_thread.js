// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model thread in the page: a module Web Worker that reads the export, runs it
 * with ONNX Runtime Web, and answers `SyncModelWorker` calls through a SharedArrayBuffer
 * (`model_channel.js`). `model_host.js` starts it.
 *
 * The first message configures it, its `model` being `{url, cache}`: the export's
 * directory, and `{repo, revision}` when that is a Hugging Face revision to keep in the
 * browser's storage (`model_cache.js`, `opfs_store.js`), or null to fetch every file each
 * time. Every later message on `port` is a worker call. Progress and the outcome of
 * loading go back to the starter as
 * `{progress: {phase, file, loaded, total, files, file_index, bytes, bytes_total}}`
 * (`phase` is "download", "cache", "verify" or, once the files are read, "session") and
 * `{ready: true, backend}` or `{ready: false, error}`; after that, every run of the
 * graph is reported as `{activity: {...}}` (see `TransformersWorker`'s `onActivity`),
 * which is how the page can say what the model is doing while a step blocks the kernel.
 *
 * The export's `meta.json` picks the worker: `OptZeosWorker` for an OPT+ZEOS export
 * (`export/opt_zeos_surgery.py`), `TransformersWorker` for an `export_model.py` one.
 * Either runs on WebGPU, the only backend: without a WebGPU adapter the thread reports
 * `{ready: false, error}` and loads nothing.
 */

import { modelReader } from "./model_cache.js";
import { serveChannel } from "./model_channel.js";
import { OptZeosWorker, isOptZeosMeta } from "./opt_zeos_worker.js";
import { dropOtherRevisions, markUsed, opfsStore } from "./opfs_store.js";
import { TransformersWorker } from "./transformers_worker.js";

/** Fail with a message the page can show when the browser has no WebGPU adapter. */
async function requireWebGpu() {
  const adapter = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null;
  if (adapter === null) {
    throw new Error(
      "this browser has no WebGPU adapter, and the model runs on WebGPU only; " +
        "use a browser with WebGPU (Chrome or Edge 113+, Safari 26+), or the recorded answers",
    );
  }
}

self.onmessage = async (event) => {
  const { buffer, port, model, ortWebgpuUrl, tokenizersUrl } = event.data;
  self.onmessage = null;
  try {
    await requireWebGpu();
    const ort = await import(ortWebgpuUrl);
    ort.env.wasm.wasmPaths = new URL(".", ortWebgpuUrl).href;
    // ONNX Runtime still runs some kernels as WebAssembly beside WebGPU: one thread, so no
    // worker pool is started and those kernels run as they always have.
    ort.env.wasm.numThreads = 1;
    const { Tokenizer } = await import(tokenizersUrl);
    const reader = modelReader({
      url: new URL(model.url, self.location.href).href,
      cache: model.cache,
      fetch: self.fetch.bind(self),
      store: model.cache === null ? null : opfsStore,
      onProgress: (progress) => self.postMessage({ progress }),
    });
    // The last file a load reads; ONNX Runtime builds the session after it.
    const lastFile = () => (isOptZeosMeta(reader.meta()) ? reader.meta().decoder.file : "model.onnx");
    const read = async (name) => {
      const bytes = await reader.read(name);
      if (name === lastFile()) {
        // What follows is ONNX Runtime building the session, which has no progress to report.
        self.postMessage({ progress: { phase: "session", bytes: reader.bytesRead(), bytes_total: reader.bytesTotal() } });
      }
      return bytes;
    };
    const onActivity = (activity) => self.postMessage({ activity });
    await read("meta.json");
    if (model.cache !== null) await markUsed(model.cache).catch((error) => console.warn(`model cache: ${error}`));
    const worker = isOptZeosMeta(reader.meta())
      ? await OptZeosWorker.load({ ort, Tokenizer, read, onActivity })
      : await TransformersWorker.load({ ort, Tokenizer, read, onActivity });
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (message) => handle(message.data);
    });
    self.postMessage({ ready: true, backend: worker.backend });
    // Every file of this revision is stored now. Revisions no load has used for a while
    // are removed, best-effort: a failure here never touches the load that finished.
    if (model.cache !== null) {
      dropOtherRevisions(model.cache).catch((error) => console.warn(`model cache cleanup: ${error}`));
    }
  } catch (error) {
    self.postMessage({ ready: false, error: String(error?.stack ?? error) });
  }
};
