// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The ZeosModelWorker interface, exercised directly against the exported model under
// Node, before any Python is involved. Skipped when the export is absent.
//
//   node --test tests/js/

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";

import { DEFAULT_MODEL_DIR, loadNodeWorker } from "../../web/node_load.mjs";
import { SNAPSHOT_EVERY } from "../../web/transformers_worker.js";
import { startNodeModel } from "../../web/node_model_thread.mjs";

const MODEL = process.env.ZEOS_WEB_MODEL_DIR ?? DEFAULT_MODEL_DIR;
const present = existsSync(join(MODEL, "meta.json"));
const skip = present ? false : `no export at ${MODEL}; run export/export_model.py --model Qwen/Qwen3.5-2B --quant int8`;

const ALL = { allowedBlocks: null, allowedTokens: null };

/** A short ChatML prompt, built the way JsMachine frames one. */
function prompt(w, text) {
  const [imStart, imEnd] = w.info().controlIds;
  return Int32Array.from([
    imStart,
    ...w.tokenize(`user\n${text}`),
    imEnd,
    ...w.tokenize("\n"),
    imStart,
    ...w.tokenize("assistant\n"),
  ]);
}

/** One decode step, then the chosen id appended, as JsMachine drives the worker. */
async function step(w, jobId, opts = ALL) {
  const result = await w.decodeStep(jobId, opts);
  await w.append(jobId, Int32Array.from([result.tokenId]));
  return result;
}

function syncStep(w, jobId, opts = ALL) {
  const result = w.decodeStep(jobId, opts);
  w.append(jobId, Int32Array.from([result.tokenId]));
  return result;
}

function bits(floats) {
  return Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength).toString("hex");
}

describe("TransformersWorker", { skip }, () => {
  let w;
  before(async () => {
    w = await loadNodeWorker({ modelDir: MODEL });
  });
  after(async () => {
    await w?.release();
  });

  test("info names a pad the tokenizer cannot produce and the ChatML markers", () => {
    const info = w.info();
    assert.equal(info.blockSize >= 1, true);
    assert.deepEqual(
      info.controlIds.slice(0, 2).map((id) => w.piece(id)),
      ["<|im_start|>", "<|im_end|>"],
    );
    assert.equal(info.controlIds.includes(info.padId), true);
    for (const text of [w.piece(info.padId), "<|im_start|>user", " <|im_end|>", "<|endoftext|>"]) {
      const ids = Array.from(w.tokenize(text));
      for (const id of info.controlIds) assert.equal(ids.includes(id), false, `${text} -> ${ids}`);
    }
  });

  test("tokenize adds no BOS and round-trips through piece", () => {
    const text = "say 1; write tools 10; read stdin;";
    const ids = w.tokenize(text);
    assert.ok(ids instanceof Int32Array);
    assert.equal(Array.from(ids).map((id) => w.piece(id)).join(""), text);
    assert.equal(w.tokenize("").length, 0);
  });

  test("info().vocabSize ids all have a piece, and the first id past them throws", () => {
    const { vocabSize } = w.info();
    assert.equal(vocabSize, w.meta.tokenizerSize);
    assert.equal(typeof w.piece(vocabSize - 1), "string");
    assert.throws(() => w.piece(vocabSize));
  });

  test("a decode step leaves the context as it was and reports normalised attention", async () => {
    w.createContext("step");
    const ids = prompt(w, "Count to three.");
    await w.append("step", ids);
    assert.equal(w.length("step"), ids.length);
    const { tokenId, attention } = await w.decodeStep("step", ALL);
    assert.equal(w.length("step"), ids.length);
    const again = await w.decodeStep("step", ALL);
    assert.equal(again.tokenId, tokenId, "a repeated step recomputes the same position");
    assert.equal(bits(again.attention), bits(attention));
    await w.append("step", Int32Array.from([tokenId]));
    assert.equal(w.length("step"), ids.length + 1);
    assert.ok(attention instanceof Float32Array);
    assert.equal(attention.length, Math.ceil(ids.length / w.info().blockSize));
    const total = attention.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(total - 1) < 1e-4, `attention sums to ${total}`);
    assert.ok(attention.every((v) => v >= 0));
    w.destroyContext("step");
    assert.throws(() => w.length("step"));
  });

  test("a masked block receives exactly zero on every step, and the rest still sum to one", async () => {
    w.createContext("mask");
    await w.append("mask", prompt(w, "The secret word is pineapple. Say a fruit."));
    const blockSize = w.info().blockSize;
    const hidden = new Set([3, 4, 5, 6, 7, 8].map((p) => Math.floor(p / blockSize)));
    for (let n = 0; n < 6; n++) {
      const blocks = Math.ceil(w.length("mask") / blockSize);
      const allowedBlocks = new Uint8Array(blocks).map((_, b) => (hidden.has(b) ? 0 : 1));
      const { attention } = await step(w, "mask", { allowedBlocks, allowedTokens: null });
      for (const b of hidden) assert.equal(attention[b], 0, `block ${b} on step ${n}`);
      const total = attention.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(total - 1) < 1e-4);
    }
    w.destroyContext("mask");
  });

  test("a mask that does not cover the context, or hides all of it, is refused", async () => {
    w.createContext("short");
    await w.append("short", prompt(w, "Hello."));
    const blocks = Math.ceil(w.length("short") / w.info().blockSize);
    await assert.rejects(
      w.decodeStep("short", { allowedBlocks: new Uint8Array(blocks - 1).fill(1), allowedTokens: null }),
    );
    await assert.rejects(
      w.decodeStep("short", { allowedBlocks: new Uint8Array(blocks), allowedTokens: null }),
    );
    w.destroyContext("short");
  });

  test("allowedTokens is the only vocabulary a step may choose from", async () => {
    w.createContext("vocab");
    await w.append("vocab", prompt(w, "Write a poem about the sea."));
    const digits = [..."0123456789"].map((d) => w.tokenize(d)[0]);
    const allowedTokens = new Uint8Array(w.meta.tokenizerSize);
    for (const id of digits) allowedTokens[id] = 1;
    for (let i = 0; i < 4; i++) {
      const { tokenId } = await step(w, "vocab", { allowedBlocks: null, allowedTokens });
      assert.ok(digits.includes(tokenId), `chose ${tokenId} (${w.piece(tokenId)})`);
    }
    await assert.rejects(
      w.decodeStep("vocab", { allowedBlocks: null, allowedTokens: new Uint8Array(10) }),
    );
    w.destroyContext("vocab");
  });

  test("truncate discards the cache past n, so decoding resumes as from a fresh prefix", async () => {
    const ids = prompt(w, "Name three colours.");
    w.createContext("fresh");
    await w.append("fresh", ids);
    const want = await w.decodeStep("fresh", ALL);

    w.createContext("cut");
    await w.append("cut", ids);
    for (let i = 0; i < 5; i++) await step(w, "cut");
    w.truncate("cut", ids.length);
    assert.equal(w.length("cut"), ids.length);
    const got = await w.decodeStep("cut", ALL);
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention));
    assert.throws(() => w.truncate("cut", w.length("cut") + 1));
    w.destroyContext("fresh");
    w.destroyContext("cut");
  });

  test("truncate past a snapshot re-runs from the snapshot, as a fresh prefix would", async () => {
    // Long enough that the cut lands after the first snapshot of the recurrent state.
    const ids = prompt(w, "Count from one to ten, then back down again. ".repeat(30));
    assert.ok(ids.length > SNAPSHOT_EVERY + 1);
    w.createContext("fresh");
    await w.append("fresh", ids);
    const want = await w.decodeStep("fresh", ALL);

    w.createContext("cut");
    await w.append("cut", ids);
    for (let i = 0; i < 3; i++) await step(w, "cut");
    const before = { ...w.stats };
    w.truncate("cut", ids.length);
    const got = await w.decodeStep("cut", ALL);
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention));
    if (w.meta.stateShape[0] > 0) {
      assert.equal(w.stats.rerunPositions - before.rerunPositions, ids.length - 1 - SNAPSHOT_EVERY);
    }
    w.destroyContext("fresh");
    w.destroyContext("cut");
  });

  test("a step whose mask hides what the recurrent state saw re-runs it without that", async () => {
    const ids = prompt(w, "The secret word is pineapple. Say a fruit.");
    const hidden = 5;
    const allowedBlocks = (n) => new Uint8Array(n).map((_, b) => (b === hidden ? 0 : 1));

    w.createContext("seen");
    await w.append("seen", ids);
    const first = await step(w, "seen");
    const before = { ...w.stats };
    const masked = await w.decodeStep("seen", { allowedBlocks: allowedBlocks(ids.length + 1), allowedTokens: null });
    assert.equal(masked.attention[hidden], 0);
    if (w.meta.stateShape[0] > 0) assert.equal(w.stats.reruns, before.reruns + 1);

    // A context that met the mask with the same tokens resident computes the same step.
    w.createContext("never");
    await w.append("never", Int32Array.from([...ids, first.tokenId]));
    const again = await w.decodeStep("never", { allowedBlocks: allowedBlocks(ids.length + 1), allowedTokens: null });
    assert.equal(again.tokenId, masked.tokenId);
    assert.equal(bits(again.attention), bits(masked.attention));

    // Showing the position again re-runs the state once more, back to the open step.
    const open = await w.decodeStep("seen", ALL);
    w.createContext("open");
    await w.append("open", Int32Array.from([...ids, first.tokenId]));
    const want = await w.decodeStep("open", ALL);
    assert.equal(open.tokenId, want.tokenId);
    assert.ok(open.attention[hidden] > 0);
    for (const name of ["seen", "never", "open"]) w.destroyContext(name);
  });

  test("a step told to stop keeps what it ran, and resumes to the same bits", async () => {
    const ids = prompt(w, "The secret word is pineapple. Say a fruit, then say a vegetable.");
    const hidden = 1;
    const mask = new Uint8Array(ids.length).map((_, b) => (b === hidden ? 0 : 1));
    const opts = { allowedBlocks: mask, allowedTokens: null, maxChunk: 4 };
    w.createContext("twin");
    await w.append("twin", ids);
    const want = await w.decodeStep("twin", opts);
    assert.equal(want.resident, ids.length);

    // The mask disagrees with the prefill from position 1, so the step replays from 0 in
    // runs of 4; stopped before the third run, then before the final decode.
    w.createContext("stopped");
    await w.append("stopped", ids);
    let asked = 0;
    const first = await w.decodeStep("stopped", { ...opts, shouldStop: () => ++asked > 2 });
    assert.deepEqual({ ...first, stats: { ...first.stats, fillMs: 0 } }, {
      cancelled: true,
      resident: 8,
      stats: { positions: 8, chunks: 2, fillMs: 0 },
    });
    const second = await w.decodeStep("stopped", { ...opts, shouldStop: () => w.contexts.get("stopped").kvLength === ids.length - 1 });
    assert.equal(second.cancelled, true);
    assert.equal(second.resident, ids.length - 1);
    assert.equal(second.stats.positions, ids.length - 1 - 8);
    const done = await w.decodeStep("stopped", opts);
    assert.equal(done.stats.positions, 1, "only the final decode was left");
    assert.equal(done.tokenId, want.tokenId);
    assert.equal(bits(done.attention), bits(want.attention));
    for (const name of ["twin", "stopped"]) w.destroyContext(name);
  });

  test("fork deep-copies tokens and cache: the two contexts then decode independently", async () => {
    w.createContext("parent");
    await w.append("parent", prompt(w, "Count to five."));
    await step(w, "parent");
    w.fork("parent", "child");
    const a = await step(w, "parent");
    const b = await step(w, "child");
    assert.equal(a.tokenId, b.tokenId);
    assert.equal(bits(a.attention), bits(b.attention));
    const before = w.length("parent");
    w.truncate("child", 3);
    assert.equal(w.length("child"), 3);
    assert.equal(w.length("parent"), before);
    const c = await w.decodeStep("parent", ALL);
    assert.equal(c.attention.length, Math.ceil(w.length("parent") / w.info().blockSize));
    w.destroyContext("parent");
    w.destroyContext("child");
  });

  test("the same calls give byte-identical tokens and attention", async () => {
    const run = async (name) => {
      w.createContext(name);
      await w.append(name, prompt(w, "Say the numbers from one to four."));
      const out = [];
      for (let i = 0; i < 8; i++) {
        const { tokenId, attention } = await step(w, name);
        out.push(`${tokenId}:${bits(attention)}`);
      }
      w.destroyContext(name);
      return out;
    };
    assert.deepEqual(await run("first"), await run("second"));
  });
});

describe("SyncModelWorker over a model thread", { skip }, () => {
  test("answers synchronously, and as the worker does in-thread", async () => {
    const direct = await loadNodeWorker({ modelDir: MODEL });
    const sync = await startNodeModel({ modelDir: MODEL });
    try {
      const ids = prompt(direct, "Count to three.");
      direct.createContext("j");
      await direct.append("j", ids);
      sync.createContext("j");
      sync.append("j", ids);
      assert.equal(sync.length("j"), ids.length);
      for (let i = 0; i < 4; i++) {
        const want = await step(direct, "j");
        const got = syncStep(sync, "j");
        assert.equal(got.tokenId, want.tokenId);
        assert.equal(bits(got.attention), bits(want.attention));
      }
      assert.deepEqual(Array.from(sync.tokenize("say 7;")), Array.from(direct.tokenize("say 7;")));
      assert.equal(sync.piece(42), direct.piece(42));
      assert.throws(() => sync.truncate("j", 10_000));
      assert.throws(() => sync.piece(sync.pieces.length));
    } finally {
      await direct.release();
      await sync.terminate();
    }
  });
});
