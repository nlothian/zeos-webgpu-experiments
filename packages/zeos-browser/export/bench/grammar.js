// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The grammar mask (`token_mask`, sent by `JsMachine` as `allowedTokens`) against the
// model on WebGPU, as the page runs them: the model on its own thread (`model_host.js`,
// `model_thread.js`), and `JsMachine` under Pyodide in a worker (`grammar_worker.js`)
// calling it through `SyncModelWorker`. The worker runs the checks and posts them here.
// The result is left in `window.benchResult` (`{checks, timings, error?}`) for
// `tests/opt_zeos_webgpu.mjs --page grammar.html`.

import { startBrowserModel } from "../../web/model_host.js";

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/";

const logEl = document.getElementById("log");
function log(line) {
  logEl.textContent += `${line}\n`;
  console.log(line);
}

try {
  const began = performance.now();
  const model = await startBrowserModel({
    model: { url: modelUrl, cache: null },
    ortWebgpuUrl: ortUrl,
    tokenizersUrl: "/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs",
  });
  const timings = { loadMs: performance.now() - began };
  log(`model loaded on ${model.backend} in ${(timings.loadMs / 1000).toFixed(1)} s`);
  const index = await (await fetch("wheels/index.json")).json();
  const worker = new Worker(new URL("./grammar_worker.js", import.meta.url), { type: "module" });
  const result = await new Promise((resolve, reject) => {
    worker.onmessage = (event) => {
      if (event.data.log !== undefined) log(event.data.log);
      else resolve(event.data);
    };
    worker.onerror = (event) => reject(new Error(`grammar worker: ${event.message}`));
    worker.postMessage(
      {
        buffer: model.buffer,
        port: model.port,
        wheels: index.wheels.map((name) => new URL(`wheels/${name}`, location.href).href),
        pyodideUrl: new URL("/node_modules/pyodide/", location.href).href,
      },
      [model.port],
    );
  });
  for (const c of result.checks ?? []) log(`${c.ok ? "ok  " : "FAIL"} ${c.name}${c.detail ? ` -- ${c.detail}` : ""}`);
  timings.runMs = performance.now() - began - timings.loadMs;
  window.benchResult = { checks: result.checks ?? [], timings, ...(result.error ? { error: result.error } : {}) };
  if (result.error) log(`error: ${result.error}`);
  model.thread.terminate();
} catch (error) {
  window.benchResult = { checks: [], error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
