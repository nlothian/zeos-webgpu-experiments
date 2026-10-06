// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model thread in the page: a module Web Worker that downloads the export, runs it
 * with ONNX Runtime Web, and answers `SyncModelWorker` calls through a SharedArrayBuffer
 * (`model_channel.js`). `model_host.js` starts it.
 *
 * The first message configures it; every later message on `port` is a worker call.
 * Progress and the outcome of loading go back to the starter as
 * `{progress: {phase, file, loaded, total, files, file_index, bytes, bytes_total}}` and
 * `{ready: true, backend}` or `{ready: false, error}`; after that, every run of the
 * graph is reported as `{activity: {...}}` (see `TransformersWorker`'s `onActivity`),
 * which is how the page can say what the model is doing while a step blocks the kernel.
 *
 * The export's `meta.json` picks the worker: `OptZeosWorker` for an OPT+ZEOS export
 * (`export/opt_zeos_surgery.py`), `TransformersWorker` for an `export_model.py` one.
 * Either runs on WebGPU, the only backend: without a WebGPU adapter the thread reports
 * `{ready: false, error}` and loads nothing.
 */

import { serveChannel } from "./model_channel.js";
import { OptZeosWorker, isOptZeosMeta } from "./opt_zeos_worker.js";
import { TransformersWorker } from "./transformers_worker.js";

async function fetchBytes(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || total === 0) return new Uint8Array(await response.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = response.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, loaded);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  return out;
}

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
  const { buffer, port, modelUrl, ortWebgpuUrl, tokenizersUrl } = event.data;
  self.onmessage = null;
  try {
    await requireWebGpu();
    const ort = await import(ortWebgpuUrl);
    ort.env.wasm.wasmPaths = new URL(".", ortWebgpuUrl).href;
    // ONNX Runtime still runs some kernels as WebAssembly beside WebGPU: one thread, so no
    // worker pool is started and those kernels run as they always have.
    ort.env.wasm.numThreads = 1;
    const { Tokenizer } = await import(tokenizersUrl);
    const base = new URL(modelUrl, self.location.href);
    // meta.json lists every file's size, so the download can be reported as a whole.
    // It is read first by `load`, and the figures below are empty until then.
    let sizes = {};
    let meta = null;
    let metaBytes = null;
    let bytesBefore = 0;
    let fileIndex = 0;
    // Files with the same hash are read once (the OPT+ZEOS embedding and the decoder's
    // second shard), so they count once.
    const unique = () => {
      const seen = new Map();
      for (const [name, f] of Object.entries(sizes)) if (!seen.has(f.sha256 ?? name)) seen.set(f.sha256 ?? name, f);
      return [...seen.values()];
    };
    // The last file a load reads; ONNX Runtime builds the session after it.
    const lastFile = () => (meta !== null && isOptZeosMeta(meta) ? meta.decoder.file : "model.onnx");
    const read = async (name) => {
      if (name === "meta.json" && metaBytes !== null) return metaBytes;
      const total = sizes[name]?.bytes ?? 0;
      const sum = unique().reduce((a, f) => a + f.bytes, 0);
      const names = unique();
      const bytes = await fetchBytes(new URL(name, base), (loaded, fileTotal) =>
        self.postMessage({
          progress: {
            phase: "download",
            file: name,
            loaded,
            total: fileTotal || total,
            files: names.length + 1, // meta.json itself is not in its own list
            file_index: fileIndex,
            bytes: bytesBefore + loaded,
            bytes_total: sum,
          },
        }),
      );
      if (name === "meta.json") {
        metaBytes = bytes;
        meta = JSON.parse(new TextDecoder().decode(bytes));
        sizes = meta.files ?? {};
      }
      bytesBefore += bytes.byteLength;
      fileIndex += 1;
      if (name === lastFile()) {
        // The graph is the last file `load` reads; what follows is ONNX Runtime
        // building the session, which has no progress to report.
        self.postMessage({ progress: { phase: "session", bytes: bytesBefore, bytes_total: sum } });
      }
      return bytes;
    };
    const onActivity = (activity) => self.postMessage({ activity });
    await read("meta.json");
    const worker = isOptZeosMeta(meta)
      ? await OptZeosWorker.load({ ort, Tokenizer, read, onActivity })
      : await TransformersWorker.load({ ort, Tokenizer, read, onActivity });
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (message) => handle(message.data);
    });
    self.postMessage({ ready: true, backend: worker.backend });
  } catch (error) {
    self.postMessage({ ready: false, error: String(error?.stack ?? error) });
  }
};
