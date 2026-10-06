// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// SyncModelWorker forwards every decode option across the shared buffer, `sample`
// included, to a worker on another thread. No model needed: the far side echoes what
// it received.
//
//   node --test tests/js/model_channel.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";
import { Worker } from "node:worker_threads";

import { CHANNEL_BYTES, SyncModelWorker } from "../../web/model_channel.js";

const ECHO = `
import { parentPort, workerData } from "node:worker_threads";
const { serveChannel } = await import(workerData.channel);
const echo = {
  meta: { tokenizerSize: 2 },
  backend: "echo",
  piece: (id) => String(id),
  decodeStep: (jobId, opts) => ({
    tokenId: 1,
    attention: null,
    seen: {
      jobId,
      allowedBlocks: opts.allowedBlocks === null ? null : Array.from(opts.allowedBlocks),
      allowedTokens: opts.allowedTokens === null ? null : Array.from(opts.allowedTokens),
      sample: opts.sample ?? null,
    },
  }),
};
serveChannel(echo, workerData.buffer, (handle) => parentPort.on("message", handle));
parentPort.postMessage("ready");
`;

test("decodeStep forwards allowedBlocks, allowedTokens and sample", async () => {
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const thread = new Worker(new URL(`data:text/javascript,${encodeURIComponent(ECHO)}`), {
    workerData: { buffer, channel: new URL("../../web/model_channel.js", import.meta.url).href },
  });
  try {
    await new Promise((resolve, reject) => {
      thread.once("message", resolve);
      thread.once("error", reject);
    });
    const sync = new SyncModelWorker(buffer, (m) => thread.postMessage(m));
    assert.equal(sync.backend, "echo");
    const sample = { temperature: 0.7, topK: 20, u: 0.25 };
    const sampled = sync.decodeStep("j", { allowedBlocks: [1, 0, 1], allowedTokens: [0, 1], sample });
    assert.deepEqual(sampled.seen, { jobId: "j", allowedBlocks: [1, 0, 1], allowedTokens: [0, 1], sample });
    const greedy = sync.decodeStep("j", { allowedBlocks: null, allowedTokens: null });
    assert.equal(greedy.seen.sample, null, "greedy sends no sample");
  } finally {
    await thread.terminate();
  }
});
