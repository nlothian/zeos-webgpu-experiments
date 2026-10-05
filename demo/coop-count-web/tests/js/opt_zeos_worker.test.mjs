// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// OptZeosWorker, the ZeosModelWorker over the OPT+ZEOS graph, exercised directly under
// Node with onnxruntime-node's CPU provider, which has kernels for the fused operators
// (LinearAttention, CausalConvWithState). onnxruntime-web's WebAssembly backend loads
// the graph too, but its 4 GiB heap cannot hold the 2.4 GB of weights and a run's
// buffers (std::bad_alloc on the first run), so it is not tested here; the WebGPU path
// is `tests/opt_zeos_webgpu.mjs`. Skipped when the export is absent.
//
//   node --test tests/js/opt_zeos_worker.test.mjs
//
// Snapshots are taken every 16 positions and at most 4 kept, so short prompts cross
// snapshot boundaries and the thinning runs.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import { loadNodeWorker } from "../../web/node_load.mjs";
import { argmax } from "../../web/opt_zeos_worker.js";
import { sampleToken } from "../../web/transformers_worker.js";
import { startNodeModel } from "../../web/node_model_thread.mjs";

const MODEL =
  process.env.ZEOS_OPT_MODEL_DIR ?? fileURLToPath(new URL("../../models/Qwen3.5-4B-ZEOS-OPT/", import.meta.url));
const present = existsSync(join(MODEL, "meta.json"));
const skip = present ? false : `no export at ${MODEL}; run export/opt_zeos_surgery.py`;
const EVERY = 16;
const THREADS = Number(process.env.ZEOS_OPT_THREADS ?? 8);

const ALL = { allowedBlocks: null, allowedTokens: null };

function prompt(w, text) {
  const [imStart, imEnd] = w.info().controlIds;
  return Int32Array.from([
    imStart,
    ...w.tokenize(`user\n${text}`),
    imEnd,
    ...w.tokenize("\n"),
    imStart,
    ...w.tokenize("assistant\n<think>\n\n</think>\n\n"),
  ]);
}

async function step(w, jobId, opts = ALL) {
  const result = await w.decodeStep(jobId, opts);
  w.append(jobId, Int32Array.from([result.tokenId]));
  return result;
}

function bits(floats) {
  return Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength).toString("hex");
}

function maxDiff(a, b) {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe("OptZeosWorker on onnxruntime-node", { skip }, () => {
  let w;
  before(async () => {
    w = await loadNodeWorker({
      modelDir: MODEL,
      runtime: "node",
      threads: THREADS,
      options: { snapshotEvery: EVERY, maxSnapshots: 4 },
    });
  });
  after(async () => {
    await w?.release();
  });

  test("info, plain tokenisation and pieces", () => {
    const info = w.info();
    assert.deepEqual(info, {
      blockSize: 1,
      padId: 248044,
      controlIds: [248045, 248046, 248044],
      eosId: 248046,
      vocabSize: 248077,
    });
    assert.equal(w.meta.tokenizerSize, info.vocabSize);
    assert.deepEqual(info.controlIds.map((id) => w.piece(id)), ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]);
    for (const text of ["<|im_start|>user", " <|im_end|>", "<|endoftext|>", "<tool_call>"]) {
      const ids = Array.from(w.tokenize(text));
      for (const id of info.controlIds) assert.equal(ids.includes(id), false, `${text} -> ${ids}`);
      assert.equal(ids.map((id) => w.piece(id)).join(""), text);
    }
    assert.equal(w.tokenize("").length, 0);
    assert.equal(typeof w.piece(info.vocabSize - 1), "string");
    assert.throws(() => w.piece(info.vocabSize));
  });

  test("greedy decoding through the cache matches a from-scratch run at every step", async () => {
    const ids = Array.from(prompt(w, "The branch line leaves the main station, crosses the river on an iron bridge built in 1887, and climbs through three tunnels to the village of Ashby Cross. Write one sentence about the bridge."));
    assert.ok(ids.length > 2 * EVERY);
    w.createContext("greedy");
    w.append("greedy", ids);
    for (let i = 0; i < 8; i++) {
      const got = await step(w, "greedy");
      const ref = await w.reference(ids, null, EVERY);
      assert.equal(got.tokenId, argmax(ref.logits, null, w.info().vocabSize), `step ${i}`);
      assert.equal(got.attention.length, ids.length);
      assert.ok(maxDiff(got.attention, ref.attention) < 1e-2, `attention at step ${i}`);
      const total = got.attention.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(total - 1) < 1e-3, `attention sums to ${total}`);
      ids.push(got.tokenId);
    }
    assert.ok(w.contexts.get("greedy").snapshots.length <= 4);
    w.destroyContext("greedy");
  });

  test("hiding a past position replays from a snapshot and equals a fresh prefill under that mask", async () => {
    const ids = prompt(w, "The password is swordfish. Remember it, then tell me a colour.");
    w.createContext("seen");
    w.append("seen", ids);
    for (let i = 0; i < 3; i++) await step(w, "seen");
    const n = w.length("seen");
    const hide = [9, 10, 11, 12, 13];
    const mask = new Uint8Array(n).fill(1);
    for (const p of hide) mask[p] = 0;
    const before = { ...w.stats };
    const got = await w.decodeStep("seen", { allowedBlocks: mask, allowedTokens: null });
    assert.equal(w.stats.reruns - before.reruns, 1);
    assert.ok(w.stats.rerunPositions - before.rerunPositions < EVERY);
    for (const p of hide) assert.equal(got.attention[p], 0, `position ${p}`);

    w.createContext("fresh");
    w.append("fresh", w.contexts.get("seen").tokens);
    const want = await w.decodeStep("fresh", { allowedBlocks: mask, allowedTokens: null });
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention));
    // And a mask that changes nothing costs nothing.
    const again = { ...w.stats };
    w.append("seen", [got.tokenId]);
    const next = new Uint8Array(n + 1).fill(1);
    for (const p of hide) next[p] = 0;
    await w.decodeStep("seen", { allowedBlocks: next, allowedTokens: null });
    assert.equal(w.stats.reruns, again.reruns);
    w.destroyContext("seen");
    w.destroyContext("fresh");
  });

  test("a repeated step, truncate and fork each recompute what a fresh prefix would", async () => {
    const ids = prompt(w, "Name three colours, one per line, and nothing else please.");
    w.createContext("fresh");
    w.append("fresh", ids);
    const want = await w.decodeStep("fresh", ALL);
    const again = await w.decodeStep("fresh", ALL);
    assert.equal(again.tokenId, want.tokenId);
    assert.equal(bits(again.attention), bits(want.attention));

    w.createContext("cut");
    w.append("cut", ids);
    for (let i = 0; i < 4; i++) await step(w, "cut");
    w.truncate("cut", ids.length);
    assert.equal(w.length("cut"), ids.length);
    const got = await w.decodeStep("cut", ALL);
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention));
    assert.throws(() => w.truncate("cut", w.length("cut") + 1));

    w.fork("cut", "child");
    const fromChild = await w.decodeStep("child", ALL);
    assert.equal(bits(fromChild.attention), bits(want.attention));
    w.append("child", [w.tokenize(" banana")[0]]);
    await w.decodeStep("child", ALL);
    w.destroyContext("child");
    const parent = await w.decodeStep("cut", ALL);
    assert.equal(bits(parent.attention), bits(want.attention), "the fork left the parent alone");
    for (const id of ["fresh", "cut"]) w.destroyContext(id);
  });

  test("allowedTokens, sample, and the refusals", async () => {
    w.createContext("vocab");
    w.append("vocab", prompt(w, "Write a poem about the sea."));
    const digits = [..."0123456789"].map((d) => w.tokenize(d)[0]);
    const allowedTokens = new Uint8Array(w.info().vocabSize);
    for (const id of digits) allowedTokens[id] = 1;
    for (let i = 0; i < 2; i++) {
      const { tokenId } = await step(w, "vocab", { allowedBlocks: null, allowedTokens });
      assert.ok(digits.includes(tokenId), `chose ${tokenId} (${w.piece(tokenId)})`);
    }
    const greedy = await w.decodeStep("vocab", ALL);
    const one = await w.decodeStep("vocab", { ...ALL, sample: { temperature: 0.7, topK: 1, u: 0.5 } });
    assert.equal(one.tokenId, greedy.tokenId);
    const sampled = await w.decodeStep("vocab", { allowedBlocks: null, allowedTokens, sample: { temperature: 1, topK: 5, u: 0.99 } });
    assert.ok(digits.includes(sampled.tokenId));
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: null, allowedTokens: new Uint8Array(w.info().vocabSize) }));
    const n = w.length("vocab");
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n - 1).fill(1), allowedTokens: null }));
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n), allowedTokens: null }));
    w.destroyContext("vocab");
    assert.throws(() => w.length("vocab"));
    w.createContext("empty");
    await assert.rejects(w.decodeStep("empty", ALL));
    w.destroyContext("empty");
    assert.equal(typeof sampleToken, "function");
  });
});

describe("OptZeosWorker behind SyncModelWorker", { skip }, () => {
  test("one synchronous step over the shared buffer", async () => {
    const sync = await startNodeModel({ modelDir: MODEL, runtime: "node", threads: THREADS });
    try {
      assert.equal(sync.backend, "cpu");
      assert.equal(sync.pieces.length, 248077);
      sync.createContext("s");
      const [imStart, imEnd] = sync.info().controlIds;
      const ids = [imStart, ...sync.tokenize("user\nWhat is the capital of France? One word."), imEnd,
        ...sync.tokenize("\n"), imStart, ...sync.tokenize("assistant\n<think>\n\n</think>\n\n")];
      sync.append("s", ids);
      const { tokenId, attention } = sync.decodeStep("s", ALL);
      assert.equal(sync.piece(tokenId), "Paris");
      assert.equal(attention.length, ids.length);
    } finally {
      await sync.terminate();
    }
  });
});
