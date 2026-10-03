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
 * `{progress: {file, loaded, total}}` and `{ready: true, backend}` or
 * `{ready: false, error}`.
 */

import { serveChannel } from "./model_channel.js";
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

/** Which execution provider to run: WebGPU when asked for and an adapter exists,
 * otherwise WebAssembly. */
async function chooseBackend(wanted) {
  if (wanted === "wasm") return "wasm";
  const adapter = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null;
  if (adapter) return "webgpu";
  if (wanted === "webgpu") throw new Error("WebGPU was asked for, but this browser has no adapter");
  return "wasm";
}

self.onmessage = async (event) => {
  const { buffer, port, modelUrl, ortWasmUrl, ortWebgpuUrl, tokenizersUrl, backend, threads } =
    event.data;
  self.onmessage = null;
  try {
    const chosen = await chooseBackend(backend ?? "wasm");
    // The plain WebAssembly build for the WebAssembly backend, so the kernels are the same
    // binary Node's onnxruntime-web runs and a journal can be compared across the two.
    const ort = await import(chosen === "webgpu" ? ortWebgpuUrl : ortWasmUrl);
    ort.env.wasm.numThreads = threads ?? 1;
    ort.env.wasm.wasmPaths = new URL(".", chosen === "webgpu" ? ortWebgpuUrl : ortWasmUrl).href;
    const { Tokenizer } = await import(tokenizersUrl);
    const base = new URL(modelUrl, self.location.href);
    const read = (name) =>
      fetchBytes(new URL(name, base), (loaded, total) =>
        self.postMessage({ progress: { file: name, loaded, total } }),
      );
    const worker = await TransformersWorker.load({ ort, Tokenizer, read, backend: chosen });
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (message) => handle(message.data);
    });
    self.postMessage({ ready: true, backend: chosen });
  } catch (error) {
    self.postMessage({ ready: false, error: String(error?.stack ?? error) });
  }
};
