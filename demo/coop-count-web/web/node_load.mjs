// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * Node entry point for the model worker: loads an export from disk with either ONNX
 * Runtime build. `web` is onnxruntime-web's WebAssembly backend, the same kernels the
 * page runs, so a journal made under Node is comparable with one made in the browser;
 * `node` is the native CPU build, faster and numerically different.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { TransformersWorker } from "./transformers_worker.js";

export const DEFAULT_MODEL_DIR = new URL(`../models/${process.env.ZEOS_WEB_MODEL ?? "Qwen2.5-0.5B-Instruct-zeos-int8"}/`, import.meta.url)
  .pathname;

export async function loadNodeWorker({ modelDir = DEFAULT_MODEL_DIR, runtime = "web", threads = 1 } = {}) {
  const { Tokenizer } = await import("@huggingface/tokenizers");
  const read = async (name) => new Uint8Array(await readFile(join(modelDir, name)));
  if (runtime === "web") {
    const ort = await import("onnxruntime-web");
    // One thread, so a run's arithmetic is a function of its inputs alone.
    ort.env.wasm.numThreads = threads;
    return TransformersWorker.load({ ort, Tokenizer, read, backend: "wasm" });
  }
  if (runtime === "node") {
    const ort = await import("onnxruntime-node");
    return TransformersWorker.load({
      ort,
      Tokenizer,
      read,
      backend: "cpu",
      sessionOptions: { intraOpNumThreads: threads, interOpNumThreads: 1 },
    });
  }
  throw new Error(`unknown runtime ${runtime}; use web or node`);
}
