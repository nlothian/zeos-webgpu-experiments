// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model worker as a child process, for a CPython parent
 * (`zeos_coop_count_web.node_worker.NodeWorker`). One request at a time on stdin, one
 * reply per request on stdout, in the frames of `frames.js`, so each of the parent's
 * calls returns only when the worker has finished.
 *
 *   node web/node_bridge.mjs [--model DIR] [--runtime web|node] [--threads N]
 */

import { parseArgs } from "node:util";

import { decodeFrame, encodeFrame, frameLength, serveRequest } from "./frames.js";
import { DEFAULT_MODEL_DIR, loadNodeWorker } from "./node_load.mjs";

async function main() {
  // stdout carries frames, so anything a library logs goes to stderr instead.
  console.log = console.info = console.warn = console.debug = console.error;
  const { values } = parseArgs({
    options: {
      model: { type: "string", default: DEFAULT_MODEL_DIR },
      runtime: { type: "string", default: "web" },
      threads: { type: "string", default: "1" },
    },
  });
  const worker = await loadNodeWorker({
    modelDir: values.model,
    runtime: values.runtime,
    threads: Number(values.threads),
  });
  process.stdout.write(encodeFrame({ ready: true, backend: worker.backend }));

  let buffer = new Uint8Array(0);
  let chain = Promise.resolve();
  process.stdin.on("data", (chunk) => {
    const joined = new Uint8Array(buffer.byteLength + chunk.byteLength);
    joined.set(buffer);
    joined.set(chunk, buffer.byteLength);
    buffer = joined;
    for (let length = frameLength(buffer); length > 0; length = frameLength(buffer)) {
      const request = decodeFrame(buffer.subarray(0, length));
      buffer = buffer.slice(length);
      chain = chain.then(async () => {
        process.stdout.write(encodeFrame(await serveRequest(worker, request)));
      });
    }
  });
  process.stdin.on("end", () => chain.then(() => process.exit(0)));
}

main().catch((error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exit(1);
});
