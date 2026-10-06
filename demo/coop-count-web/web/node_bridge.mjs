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
 * A decode step the parent *began* (`begun: true`) can be cancelled while it runs: the
 * parent writes a control frame `{cancel: id}`, which is read as soon as it arrives (the
 * step is awaiting a run of the graph) and is not answered, and the step asks for it
 * before every chunk. `--stub FILE` serves `stub_worker.js` over a JSON file of
 * `{tapes, options}` instead of a model.
 *
 *   node web/node_bridge.mjs [--model DIR] [--runtime web|node] [--threads N] [--stub FILE]
 */

import { readFile } from "node:fs/promises";
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
      stub: { type: "string" },
    },
  });
  let worker;
  if (values.stub === undefined) {
    worker = await loadNodeWorker({
      modelDir: values.model,
      runtime: values.runtime,
      threads: Number(values.threads),
    });
  } else {
    await import("./stub_worker.js");
    const { tapes, options = {} } = JSON.parse(await readFile(values.stub, "utf8"));
    worker = globalThis.createStubWorker(tapes, options);
    worker.backend = "stub";
  }
  process.stdout.write(encodeFrame({ ready: true, backend: worker.backend }));

  let buffer = new Uint8Array(0);
  let chain = Promise.resolve();
  // The id of the latest step the parent cancelled.
  let cancelled = -1;
  process.stdin.on("data", (chunk) => {
    const joined = new Uint8Array(buffer.byteLength + chunk.byteLength);
    joined.set(buffer);
    joined.set(chunk, buffer.byteLength);
    buffer = joined;
    for (let length = frameLength(buffer); length > 0; length = frameLength(buffer)) {
      const request = decodeFrame(buffer.subarray(0, length));
      buffer = buffer.slice(length);
      if (request.cancel !== undefined) {
        cancelled = request.cancel;
        continue;
      }
      const shouldStop = request.begun ? () => cancelled === request.id : null;
      chain = chain.then(async () => {
        process.stdout.write(encodeFrame(await serveRequest(worker, request, { shouldStop })));
      });
    }
  });
  process.stdin.on("end", () => chain.then(() => process.exit(0)));
}

main().catch((error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exit(1);
});
