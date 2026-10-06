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

  test("a skipped hidden run equals running it through the graph", async () => {
    const ids = prompt(w, `Here are notes from the station.\n${"The night train leaves at ten past eleven from platform four, and the first train at six. ".repeat(4)}\nWhat time does the night train leave?`);
    assert.ok(ids.length > 3 * EVERY);
    // A run cut at the snapshot positions, so both ways chunk alike; and one that is not.
    for (const [from, to] of [[EVERY, 3 * EVERY], [EVERY + 5, 3 * EVERY - 4]]) {
      const mask = new Uint8Array(ids.length).fill(1);
      mask.fill(0, from, to);
      const opts = { allowedBlocks: mask, allowedTokens: null };
      const results = [];
      for (const skipHidden of [true, false]) {
        w.skipHidden = skipHidden;
        const before = w.stats.skipped;
        w.createContext("skip");
        w.append("skip", ids);
        results.push({ step: await w.decodeStep("skip", opts), logits: w.lastLogits, skipped: w.stats.skipped - before });
        w.destroyContext("skip");
      }
      w.skipHidden = true;
      const [skipped, ran] = results;
      assert.equal(skipped.skipped, to - from, `[${from}, ${to}) skipped`);
      assert.equal(ran.skipped, 0);
      for (let p = from; p < to; p++) assert.equal(skipped.step.attention[p], 0, `position ${p}`);
      if (from % EVERY === 0 && to % EVERY === 0) {
        assert.equal(bits(skipped.logits), bits(ran.logits), "aligned: bit for bit");
        assert.equal(bits(skipped.step.attention), bits(ran.step.attention));
      } else {
        // Cut elsewhere, so only float16 rounding apart.
        const diff = maxDiff(skipped.logits, ran.logits);
        assert.ok(diff < 0.5, `max |logit diff| ${diff}`);
      }
    }
  });

  test("a mask narrowed for a few steps runs on a second cache and leaves the first alone", async () => {
    // A prompt whose middle stands for a tool result, hidden while a tool's name is chosen.
    const head = Array.from(prompt(w, "Summarise the result below in one sentence.\n\nRESULT:"));
    const result = Array.from(w.tokenize(" The night train leaves at ten past eleven from platform four, and the first train at six."));
    const done = Array.from(w.tokenize(" Done."));
    const ids = [...head, ...result, ...w.tokenize("\nThat was the result.")];
    const hidden = (n) => {
      const mask = new Uint8Array(n).fill(1);
      mask.fill(0, head.length, head.length + result.length);
      return mask;
    };
    const narrow = (n) => ({ allowedBlocks: hidden(n), allowedTokens: null });

    // Twins, one per mask, stepped at the same points, for bit-for-bit comparisons.
    for (const id of ["two", "wide", "narrow"]) {
      w.createContext(id);
      w.append(id, ids);
    }
    const wide1 = await w.decodeStep("two", ALL);
    assert.equal(bits(wide1.attention), bits((await w.decodeStep("wide", ALL)).attention));
    for (const id of ["two", "wide", "narrow"]) w.append(id, [wide1.tokenId]);

    // Narrowed: a second cache from the latest snapshot before the hidden run.
    let before = { ...w.stats };
    const masked = await w.decodeStep("two", narrow(ids.length + 1));
    assert.equal(w.stats.tracks - before.tracks, 1);
    assert.ok(w.stats.rerunPositions - before.rerunPositions < EVERY);
    assert.ok(w.stats.skipped > before.skipped);
    const fresh = await w.decodeStep("narrow", narrow(ids.length + 1));
    assert.equal(masked.tokenId, fresh.tokenId);
    assert.equal(bits(masked.attention), bits(fresh.attention), "equals a fresh prefill under the mask");
    for (let p = head.length; p < head.length + result.length; p++) assert.equal(masked.attention[p], 0);

    // Two more narrowed steps, then the wide mask again: back to the first cache, which
    // runs only the four tokens it has not seen, with no rewind.
    const named = [masked.tokenId];
    for (const id of ["two", "narrow"]) w.append(id, [masked.tokenId]);
    for (let i = 0; i < 2; i++) {
      const a = await w.decodeStep("two", narrow(w.length("two")));
      const b = await w.decodeStep("narrow", narrow(w.length("narrow")));
      assert.equal(bits(a.attention), bits(b.attention));
      named.push(a.tokenId);
      for (const id of ["two", "narrow"]) w.append(id, [a.tokenId]);
    }
    w.append("wide", named);
    before = { ...w.stats };
    const back = await w.decodeStep("two", ALL);
    assert.equal(w.stats.switches - before.switches, 1);
    assert.equal(w.stats.reruns, before.reruns, "no rewind");
    assert.equal(w.stats.positions - before.positions, named.length + 1);
    const want = await w.decodeStep("wide", ALL);
    assert.equal(bits(back.attention), bits(want.attention), "equals the wide twin");

    // More content, a second hidden run, and the narrow mask again: the second cache
    // catches up from where it stopped, skipping the new hidden run.
    const more = [back.tokenId, ...w.tokenize(" Next result:"), ...result, ...done];
    for (const id of ["two", "wide", "narrow"]) w.append(id, more);
    const second = (n) => {
      const mask = hidden(n);
      const at = n - result.length - done.length;
      mask.fill(0, at, at + result.length);
      return { allowedBlocks: mask, allowedTokens: null };
    };
    before = { ...w.stats };
    const again = await w.decodeStep("two", second(w.length("two")));
    assert.equal(w.stats.switches - before.switches, 1);
    assert.equal(w.stats.reruns, before.reruns, "caught up, not replayed");
    assert.ok(w.stats.skipped > before.skipped);
    const twin = await w.decodeStep("narrow", second(w.length("narrow")));
    assert.equal(bits(again.attention), bits(twin.attention), "equals the narrow twin");

    // A third mask starts a cache from the closer of the two and drops the older one.
    before = { ...w.stats };
    const third = narrow(w.length("two"));
    third.allowedBlocks[1] = 0;
    await w.decodeStep("two", third);
    assert.equal(w.stats.tracks - before.tracks, 1);
    assert.notEqual(w.contexts.get("two").other, null);
    for (const id of ["two", "wide", "narrow"]) w.destroyContext(id);
  });

  test("with one cache a narrowed mask rewinds it", async () => {
    w.maxTracks = 1;
    try {
      w.createContext("one");
      w.append("one", prompt(w, "The password is swordfish. Remember it, then tell me a colour."));
      await step(w, "one");
      const mask = new Uint8Array(w.length("one")).fill(1);
      mask.fill(0, 9, 14);
      const before = { ...w.stats };
      await w.decodeStep("one", { allowedBlocks: mask, allowedTokens: null });
      await w.decodeStep("one", ALL);
      assert.equal(w.stats.tracks, before.tracks);
      assert.equal(w.stats.reruns - before.reruns, 2);
      assert.equal(w.contexts.get("one").other, null);
      w.destroyContext("one");
    } finally {
      w.maxTracks = 2;
    }
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

  test("maxChunk runs smaller chunks, still cut at the snapshot positions, to the same token", async () => {
    const ids = prompt(w, "The branch line climbs through three tunnels to the village of Ashby Cross. Name the village.");
    assert.ok(ids.length > 2 * EVERY);
    w.createContext("whole");
    w.append("whole", ids);
    const whole = await w.decodeStep("whole", ALL);
    const wholeLogits = w.lastLogits;
    assert.deepEqual(whole.stats.chunks, Math.ceil(ids.length / EVERY), "the graph's chunk, cut every 16");
    assert.equal(whole.stats.positions, ids.length);
    assert.equal(whole.resident, ids.length);

    w.createContext("small");
    w.append("small", ids);
    const before = w.stats.runs;
    const small = await w.decodeStep("small", { ...ALL, maxChunk: 5 });
    // Per 16 positions between snapshots: 5, 5, 5, 1.
    let chunks = 0;
    for (let at = 0; at < ids.length; at += EVERY) chunks += Math.ceil(Math.min(EVERY, ids.length - at) / 5);
    assert.equal(small.stats.chunks, chunks);
    assert.equal(w.stats.runs - before, chunks);
    assert.equal(small.tokenId, whole.tokenId);
    assert.ok(maxDiff(w.lastLogits, wholeLogits) < 0.5, "only float16 rounding apart");
    assert.ok(maxDiff(small.attention, whole.attention) < 1e-2);
    // Larger than the graph's own chunk is the graph's own chunk.
    w.createContext("large");
    w.append("large", ids);
    const large = await w.decodeStep("large", { ...ALL, maxChunk: 1 << 20 });
    assert.equal(large.stats.chunks, whole.stats.chunks);
    assert.equal(bits(large.attention), bits(whole.attention));
    await assert.rejects(w.decodeStep("large", { ...ALL, maxChunk: 0 }), RangeError);
    for (const id of ["whole", "small", "large"]) w.destroyContext(id);
  });

  test("a step stopped before a chunk resumes from what it ran, to the same bits", async () => {
    const ids = prompt(w, `Here are notes from the station.\n${"The night train leaves at ten past eleven from platform four. ".repeat(3)}\nWhen does it leave?`);
    assert.ok(ids.length > 3 * EVERY);
    const opts = { ...ALL, maxChunk: 6 };
    w.createContext("twin");
    w.append("twin", ids);
    const want = await w.decodeStep("twin", opts);
    const wantLogits = w.lastLogits;

    w.createContext("stopped");
    w.append("stopped", ids);
    let asked = 0;
    const before = { ...w.stats };
    const stopped = await w.decodeStep("stopped", { ...opts, shouldStop: () => ++asked > 3 });
    assert.equal(stopped.cancelled, true);
    assert.equal(stopped.stats.chunks, 3);
    assert.equal(stopped.resident, 6 + 6 + 4, "two runs of 6, then 4 to the snapshot at 16");
    assert.equal(stopped.stats.positions, stopped.resident);
    assert.equal(w.stats.runs - before.runs, 3);
    assert.equal(w.contexts.get("stopped").track.kvLength, stopped.resident);
    assert.equal(w.length("stopped"), ids.length, "a step never changes the tokens");

    // Stopped again further on, then let finish: nothing recomputed, nothing rewound.
    asked = 0;
    const again = await w.decodeStep("stopped", { ...opts, shouldStop: () => ++asked > 2 });
    assert.equal(again.cancelled, true);
    assert.equal(again.resident, stopped.resident + 6 + 6);
    const done = await w.decodeStep("stopped", { ...opts, shouldStop: () => false });
    assert.equal(done.cancelled, undefined);
    assert.equal(done.stats.positions, ids.length - again.resident);
    assert.equal(w.stats.reruns, before.reruns, "no rewind");
    assert.equal(w.stats.positions - before.positions, ids.length, "every position run once");
    assert.equal(done.tokenId, want.tokenId);
    assert.equal(bits(w.lastLogits), bits(wantLogits), "bit for bit the uninterrupted step");
    assert.equal(bits(done.attention), bits(want.attention));

    // A stop asked before the final one-position decode leaves that position pending.
    for (const id of ["twin", "stopped"]) w.append(id, [want.tokenId]);
    const next = await w.decodeStep("twin", opts);
    const atDecode = await w.decodeStep("stopped", { ...opts, shouldStop: () => true });
    assert.deepEqual(atDecode, {
      cancelled: true,
      resident: ids.length,
      stats: { positions: 0, chunks: 0, fillMs: 0 },
    });
    const resumed = await w.decodeStep("stopped", opts);
    assert.equal(resumed.stats.positions, 1);
    assert.equal(bits(resumed.attention), bits(next.attention));
    for (const id of ["twin", "stopped"]) w.destroyContext(id);
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

      // The same step begun, cancelled at once, and begun again: the token is dropped,
      // the positions run are kept, and the resumed step gives the same answer.
      sync.createContext("b");
      sync.append("b", ids);
      sync.beginDecodeStep("b", { ...ALL, maxChunk: 8 });
      assert.throws(() => sync.length("b"), /channel busy/);
      sync.cancelDecode();
      const cancelled = sync.pollDecode(60_000);
      assert.equal(cancelled.cancelled, true);
      assert.ok(cancelled.resident < ids.length, `resident ${cancelled.resident}`);
      sync.beginDecodeStep("b", { ...ALL, maxChunk: 8 });
      let done = null;
      while (done === null) done = sync.pollDecode(5);
      assert.equal(done.cancelled, false);
      assert.equal(sync.piece(done.tokenId), "Paris");
      assert.equal(done.stats.positions, ids.length - cancelled.resident);
      assert.equal(done.resident, ids.length);
    } finally {
      await sync.terminate();
    }
  });
});
