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
import { startNodeModel } from "../../web/node_model_thread.mjs";

const MODEL = process.env.ZEOS_WEB_MODEL_DIR ?? DEFAULT_MODEL_DIR;
const present = existsSync(join(MODEL, "meta.json"));
const skip = present ? false : `no export at ${MODEL}; run export/export_model.py`;

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

  test("piece throws for the first id past the vocabulary", () => {
    assert.equal(typeof w.piece(w.meta.tokenizerSize - 1), "string");
    assert.throws(() => w.piece(w.meta.tokenizerSize));
  });

  test("a decode step appends one token and reports normalised per-block attention", async () => {
    w.createContext("step");
    const ids = prompt(w, "Count to three.");
    await w.append("step", ids);
    assert.equal(w.length("step"), ids.length);
    const { tokenId, attention } = await w.decodeStep("step", ALL);
    assert.equal(w.length("step"), ids.length + 1);
    assert.equal(w.tokens("step").at(-1), tokenId);
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
    for (let step = 0; step < 6; step++) {
      const blocks = Math.ceil(w.length("mask") / blockSize);
      const allowedBlocks = new Uint8Array(blocks).map((_, b) => (hidden.has(b) ? 0 : 1));
      const { attention } = await w.decodeStep("mask", { allowedBlocks, allowedTokens: null });
      for (const b of hidden) assert.equal(attention[b], 0, `block ${b} on step ${step}`);
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
      const { tokenId } = await w.decodeStep("vocab", { allowedBlocks: null, allowedTokens });
      assert.ok(digits.includes(tokenId), `chose ${tokenId} (${w.piece(tokenId)})`);
    }
    await assert.rejects(
      w.decodeStep("vocab", { allowedBlocks: null, allowedTokens: new Uint8Array(10) }),
    );
    w.destroyContext("vocab");
  });

  test("truncate discards the KV past n, so decoding resumes as from a fresh prefix", async () => {
    const ids = prompt(w, "Name three colours.");
    w.createContext("fresh");
    await w.append("fresh", ids);
    const want = await w.decodeStep("fresh", ALL);

    w.createContext("cut");
    await w.append("cut", ids);
    for (let i = 0; i < 5; i++) await w.decodeStep("cut", ALL);
    w.truncate("cut", ids.length);
    assert.equal(w.length("cut"), ids.length);
    const got = await w.decodeStep("cut", ALL);
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention));
    assert.throws(() => w.truncate("cut", w.length("cut") + 1));
    w.destroyContext("fresh");
    w.destroyContext("cut");
  });

  test("fork deep-copies tokens and KV: the two contexts then decode independently", async () => {
    w.createContext("parent");
    await w.append("parent", prompt(w, "Count to five."));
    await w.decodeStep("parent", ALL);
    w.fork("parent", "child");
    const a = await w.decodeStep("parent", ALL);
    const b = await w.decodeStep("child", ALL);
    assert.equal(a.tokenId, b.tokenId);
    assert.equal(bits(a.attention), bits(b.attention));
    const before = w.length("parent");
    w.truncate("child", 3);
    assert.equal(w.length("child"), 3);
    assert.equal(w.length("parent"), before);
    const c = await w.decodeStep("parent", ALL);
    assert.equal(c.attention.length, Math.ceil((w.length("parent") - 1) / w.info().blockSize));
    w.destroyContext("parent");
    w.destroyContext("child");
  });

  test("the same calls give byte-identical tokens and attention", async () => {
    const run = async (name) => {
      w.createContext(name);
      await w.append(name, prompt(w, "Say the numbers from one to four."));
      const out = [];
      for (let i = 0; i < 8; i++) {
        const { tokenId, attention } = await w.decodeStep(name, ALL);
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
        const want = await direct.decodeStep("j", ALL);
        const got = sync.decodeStep("j", ALL);
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
