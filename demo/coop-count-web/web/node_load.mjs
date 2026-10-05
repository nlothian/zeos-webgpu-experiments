// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * Node entry point for the model worker: loads an export from disk with either ONNX
 * Runtime build. `web` is onnxruntime-web's WebAssembly backend, the same kernels the
 * page runs, so a journal made under Node is comparable with one made in the browser;
 * `node` is the native CPU build, faster and numerically different. Which worker class
 * runs the export is read from its `meta.json`: `OptZeosWorker` for an OPT+ZEOS export,
 * `TransformersWorker` otherwise.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { OptZeosWorker, isOptZeosMeta } from "./opt_zeos_worker.js";
import { TransformersWorker } from "./transformers_worker.js";

export const DEFAULT_MODEL_DIR = new URL(`../models/${process.env.ZEOS_WEB_MODEL ?? "Qwen3.5-2B-zeos-int8"}/`, import.meta.url)
  .pathname;

export async function loadNodeWorker({ modelDir = DEFAULT_MODEL_DIR, runtime = "web", threads = 1, options = {} } = {}) {
  const { Tokenizer } = await import("@huggingface/tokenizers");
  const read = async (name) => new Uint8Array(await readFile(join(modelDir, name)));
  const opt = isOptZeosMeta(JSON.parse(await readFile(join(modelDir, "meta.json"), "utf8")));
  // The OPT+ZEOS graph's weights are 2.4 GB: ONNX Runtime reads them from disk itself.
  const source = async (name) => join(modelDir, name);
  if (runtime === "web") {
    const ort = await import("onnxruntime-web");
    // One thread, so a run's arithmetic is a function of its inputs alone.
    ort.env.wasm.numThreads = threads;
    if (opt) return OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "wasm", numThreads: threads, ...options });
    return TransformersWorker.load({ ort, Tokenizer, read, backend: "wasm" });
  }
  if (runtime === "node") {
    const ort = await import("onnxruntime-node");
    const sessionOptions = { intraOpNumThreads: threads, interOpNumThreads: 1 };
    if (opt) return OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "cpu", sessionOptions, ...options });
    return TransformersWorker.load({ ort, Tokenizer, read, backend: "cpu", sessionOptions });
  }
  throw new Error(`unknown runtime ${runtime}; use web or node`);
}
