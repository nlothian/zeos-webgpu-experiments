// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The interface clauses of OptZeosWorker and TransformersWorker, on WebGPU, for
// checks.html. Each clause is a function that throws on the first broken expectation;
// `runClauses` turns each into one check. OptZeosWorker's run with snapshots every 16
// positions and at most 4 kept (set on the worker for the section, then restored), so
// short prompts cross snapshot boundaries and the thinning runs.

import { argmax, OptZeosWorker } from "../../web/opt_zeos_worker.js";
import { SNAPSHOT_EVERY as TW_SNAPSHOT_EVERY, sampleToken } from "../../web/transformers_worker.js";

const ALL = { allowedBlocks: null, allowedTokens: null };

class Failed extends Error {}

const assert = {
  ok(value, message = "expected a true value") {
    if (!value) throw new Failed(message);
  },
  equal(a, b, message = "") {
    if (a !== b) throw new Failed(`${message ? `${message}: ` : ""}${String(a).slice(0, 80)} !== ${String(b).slice(0, 80)}`);
  },
  notEqual(a, b, message = "") {
    if (a === b) throw new Failed(`${message ? `${message}: ` : ""}${String(a).slice(0, 80)} === ${String(b).slice(0, 80)}`);
  },
  deepEqual(a, b, message = "") {
    const sa = JSON.stringify(a);
    const sb = JSON.stringify(b);
    if (sa !== sb) throw new Failed(`${message ? `${message}: ` : ""}${sa} != ${sb}`);
  },
  throws(fn, message = "expected a throw") {
    try {
      fn();
    } catch {
      return;
    }
    throw new Failed(message);
  },
  async rejects(promise, type = null, message = "expected a rejection") {
    try {
      await promise;
    } catch (error) {
      if (type !== null && !(error instanceof type)) throw new Failed(`${message}: ${error}`);
      return;
    }
    throw new Failed(message);
  },
};

function bits(f) {
  return Array.from(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)).join(",");
}

function maxDiff(a, b) {
  assert.equal(a.length, b.length, "lengths");
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** KL(p || q) of the softmaxes of two logit vectors. */
function kl(p, q) {
  const lse = (x) => {
    let m = -Infinity;
    for (const v of x) m = Math.max(m, v);
    let z = 0;
    for (const v of x) z += Math.exp(v - m);
    return m + Math.log(z);
  };
  const a = lse(p);
  const b = lse(q);
  let total = 0;
  for (let i = 0; i < p.length; i++) total += Math.exp(p[i] - a) * (p[i] - a - (q[i] - b));
  return total;
}

/** A ChatML prompt; `think` opens the reply with the empty think block, as the 4B's runs do. */
function prompt(w, text, think = true) {
  const [imStart, imEnd] = w.info().controlIds;
  return [
    imStart,
    ...w.tokenize(`user\n${text}`),
    imEnd,
    ...w.tokenize("\n"),
    imStart,
    ...w.tokenize(think ? "assistant\n<think>\n\n</think>\n\n" : "assistant\n"),
  ];
}

async function step(w, jobId, opts = ALL) {
  const result = await w.decodeStep(jobId, opts);
  await w.append(jobId, [result.tokenId]);
  return result;
}

/** Run each clause, one check apiece; a thrown error fails that clause only. Contexts a
 * failed clause left behind are destroyed. */
export async function runClauses(w, clauses, check, prefix) {
  for (const [name, clause] of clauses) {
    const before = new Set(w.contexts.keys());
    try {
      await clause(w);
      check(`${prefix}: ${name}`, true);
    } catch (error) {
      check(`${prefix}: ${name}`, false, String(error?.message ?? error));
    }
    for (const id of [...w.contexts.keys()]) if (!before.has(id)) w.destroyContext(id);
  }
}

const EVERY = 16;

/** OptZeosWorker's clauses, with snapshots every EVERY positions and at most 4 kept. */
export async function optZeosClauses(w, check) {
  const saved = { snapshotEvery: w.snapshotEvery, maxSnapshots: w.maxSnapshots, maxTracks: w.maxTracks, skipHidden: w.skipHidden };
  w.snapshotEvery = EVERY;
  w.maxSnapshots = 4;
  try {
    await runClauses(w, OPT_ZEOS, check, "OptZeosWorker");
  } finally {
    Object.assign(w, saved);
  }
}

const OPT_ZEOS = [
  ["info, plain tokenisation and pieces", (w) => {
    const info = w.info();
    assert.deepEqual(info, { blockSize: 1, padId: 248044, controlIds: [248045, 248046, 248044], eosId: 248046, vocabSize: 248077 });
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
  }],
  ["greedy steps through the cache choose what a from-scratch run does, and old snapshots thin out", async (w) => {
    const ids = prompt(w, "The branch line leaves the main station, crosses the river on an iron bridge built in 1887, and climbs through three tunnels to the village of Ashby Cross. Write one sentence about the bridge.");
    assert.ok(ids.length > 2 * EVERY);
    w.createContext("greedy");
    w.append("greedy", ids);
    for (let i = 0; i < 8; i++) {
      const got = await step(w, "greedy");
      const ref = await w.reference(ids, null, EVERY);
      assert.equal(got.tokenId, argmax(ref.logits, null, w.info().vocabSize), `step ${i}`);
      assert.equal(got.attention.length, ids.length);
      assert.ok(maxDiff(got.attention, ref.attention) < 1e-2, `attention at step ${i}`);
      assert.ok(Math.abs(got.attention.reduce((a, b) => a + b, 0) - 1) < 1e-3, "attention sums to one");
      ids.push(got.tokenId);
    }
    assert.ok(w.contexts.get("greedy").snapshots.length <= 4, "at most maxSnapshots kept");
  }],
  ["hiding a past position replays from a snapshot and equals a fresh prefill under that mask", async (w) => {
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
    assert.equal(w.stats.reruns - before.reruns, 1, "one replay");
    assert.ok(w.stats.rerunPositions - before.rerunPositions < EVERY, "from the latest snapshot");
    for (const p of hide) assert.equal(got.attention[p], 0, `position ${p}`);
    w.createContext("fresh");
    w.append("fresh", w.contexts.get("seen").tokens);
    const want = await w.decodeStep("fresh", { allowedBlocks: mask, allowedTokens: null });
    assert.equal(got.tokenId, want.tokenId);
    assert.equal(bits(got.attention), bits(want.attention), "bit for bit");
    const again = { ...w.stats };
    w.append("seen", [got.tokenId]);
    const next = new Uint8Array(n + 1).fill(1);
    for (const p of hide) next[p] = 0;
    await w.decodeStep("seen", { allowedBlocks: next, allowedTokens: null });
    assert.equal(w.stats.reruns, again.reruns, "a mask that changes nothing costs nothing");
  }],
  ["a skipped hidden run aligned with the chunks is bit for bit the run executed", async (w) => {
    const ids = prompt(w, `Here are notes from the station.\n${"The night train leaves at ten past eleven from platform four, and the first train at six. ".repeat(4)}\nWhat time does the night train leave?`);
    assert.ok(ids.length > 3 * EVERY);
    for (const [from, to] of [[EVERY, 3 * EVERY], [EVERY + 5, 3 * EVERY - 4]]) {
      const mask = new Uint8Array(ids.length).fill(1);
      mask.fill(0, from, to);
      const results = [];
      for (const skipHidden of [true, false]) {
        w.skipHidden = skipHidden;
        const before = w.stats.skipped;
        w.createContext("skip");
        w.append("skip", ids);
        results.push({ step: await w.decodeStep("skip", { allowedBlocks: mask, allowedTokens: null }), logits: w.lastLogits, skipped: w.stats.skipped - before });
        w.destroyContext("skip");
      }
      w.skipHidden = true;
      const [skipped, ran] = results;
      assert.equal(skipped.skipped, to - from, `[${from}, ${to}) carried past`);
      assert.equal(ran.skipped, 0);
      for (let p = from; p < to; p++) assert.equal(skipped.step.attention[p], 0, `position ${p}`);
      if (from % EVERY === 0 && to % EVERY === 0) {
        assert.equal(bits(skipped.logits), bits(ran.logits), "aligned: logits bit for bit");
        assert.equal(bits(skipped.step.attention), bits(ran.step.attention), "aligned: attention bit for bit");
      } else {
        assert.ok(maxDiff(skipped.logits, ran.logits) < 0.5, "not aligned: float16 rounding apart");
      }
    }
  }],
  ["a mask narrowed for a few steps runs on a second cache, comes back to the first, and catches up", async (w) => {
    const head = prompt(w, "Summarise the result below in one sentence.\n\nRESULT:");
    const result = Array.from(w.tokenize(" The night train leaves at ten past eleven from platform four, and the first train at six."));
    const done = Array.from(w.tokenize(" Done."));
    const ids = [...head, ...result, ...w.tokenize("\nThat was the result.")];
    const hidden = (n) => {
      const mask = new Uint8Array(n).fill(1);
      mask.fill(0, head.length, head.length + result.length);
      return mask;
    };
    const narrow = (n) => ({ allowedBlocks: hidden(n), allowedTokens: null });
    for (const id of ["two", "wide", "narrow"]) {
      w.createContext(id);
      w.append(id, ids);
    }
    const wide1 = await w.decodeStep("two", ALL);
    assert.equal(bits(wide1.attention), bits((await w.decodeStep("wide", ALL)).attention));
    for (const id of ["two", "wide", "narrow"]) w.append(id, [wide1.tokenId]);
    let before = { ...w.stats };
    const masked = await w.decodeStep("two", narrow(ids.length + 1));
    assert.equal(w.stats.tracks - before.tracks, 1, "a second cache");
    assert.ok(w.stats.rerunPositions - before.rerunPositions < EVERY);
    assert.ok(w.stats.skipped > before.skipped);
    const fresh = await w.decodeStep("narrow", narrow(ids.length + 1));
    assert.equal(masked.tokenId, fresh.tokenId);
    assert.equal(bits(masked.attention), bits(fresh.attention), "equals a fresh prefill under the mask");
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
    assert.equal(w.stats.switches - before.switches, 1, "switched back");
    assert.equal(w.stats.reruns, before.reruns, "no rewind");
    assert.equal(w.stats.positions - before.positions, named.length + 1, "ran only what it had not seen");
    const want = await w.decodeStep("wide", ALL);
    assert.equal(bits(back.attention), bits(want.attention), "equals the wide twin");
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
    before = { ...w.stats };
    const third = narrow(w.length("two"));
    third.allowedBlocks[1] = 0;
    await w.decodeStep("two", third);
    assert.equal(w.stats.tracks - before.tracks, 1, "a third mask starts a cache");
    assert.notEqual(w.contexts.get("two").other, null);
  }],
  ["with one cache a narrowed mask rewinds it", async (w) => {
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
    } finally {
      w.maxTracks = 2;
    }
  }],
  ["a repeated step, truncate and fork each recompute what a fresh prefix would", async (w) => {
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
  }],
  ["allowedTokens, sample, and the refusals", async (w) => {
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
    assert.equal(one.tokenId, greedy.tokenId, "topK 1 is greedy");
    const sampled = await w.decodeStep("vocab", { allowedBlocks: null, allowedTokens, sample: { temperature: 1, topK: 5, u: 0.99 } });
    assert.ok(digits.includes(sampled.tokenId));
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: null, allowedTokens: new Uint8Array(w.info().vocabSize) }), null, "no token allowed");
    const n = w.length("vocab");
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n - 1).fill(1), allowedTokens: null }), null, "a short mask");
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n), allowedTokens: null }), null, "a mask hiding everything");
    w.destroyContext("vocab");
    assert.throws(() => w.length("vocab"));
    w.createContext("empty");
    await assert.rejects(w.decodeStep("empty", ALL), null, "an empty context");
    assert.equal(typeof sampleToken, "function");
  }],
  ["maxChunk runs smaller chunks, still cut at the snapshot positions, to the same token", async (w) => {
    const ids = prompt(w, "The branch line climbs through three tunnels to the village of Ashby Cross. Name the village.");
    assert.ok(ids.length > 2 * EVERY);
    w.createContext("whole");
    w.append("whole", ids);
    const whole = await w.decodeStep("whole", ALL);
    const wholeLogits = w.lastLogits;
    assert.equal(whole.stats.chunks, Math.ceil(ids.length / EVERY), "the graph's chunk, cut every 16");
    assert.equal(whole.stats.positions, ids.length);
    assert.equal(whole.resident, ids.length);
    w.createContext("small");
    w.append("small", ids);
    const before = w.stats.runs;
    const small = await w.decodeStep("small", { ...ALL, maxChunk: 5 });
    let chunks = 0;
    for (let at = 0; at < ids.length; at += EVERY) chunks += Math.ceil(Math.min(EVERY, ids.length - at) / 5);
    assert.equal(small.stats.chunks, chunks, "5, 5, 5, 1 per 16 positions");
    assert.equal(w.stats.runs - before, chunks);
    assert.equal(small.tokenId, whole.tokenId);
    // Another chunking moves float16 rounding in the recurrent state: on WebGPU two
    // chunkings of one prompt differ by a KL of about 4e-3 (worker.html), with single
    // logits up to about 0.7 apart, so the bound is on the distribution.
    const divergence = kl(wholeLogits, w.lastLogits);
    assert.ok(divergence < 1e-2, `KL ${divergence.toExponential(2)}, max |logit diff| ${maxDiff(w.lastLogits, wholeLogits).toFixed(3)}`);
    assert.ok(maxDiff(small.attention, whole.attention) < 1e-2);
    w.createContext("large");
    w.append("large", ids);
    const large = await w.decodeStep("large", { ...ALL, maxChunk: 1 << 20 });
    assert.equal(large.stats.chunks, whole.stats.chunks, "larger than the graph's chunk is the graph's chunk");
    assert.equal(bits(large.attention), bits(whole.attention));
    await assert.rejects(w.decodeStep("large", { ...ALL, maxChunk: 0 }), RangeError, "maxChunk 0");
  }],
  ["a step stopped before a chunk resumes from what it ran, to the same bits", async (w) => {
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
    assert.equal(w.contexts.get("stopped").track.kvLength, stopped.resident);
    assert.equal(w.length("stopped"), ids.length, "a step never changes the tokens");
    asked = 0;
    const again = await w.decodeStep("stopped", { ...opts, shouldStop: () => ++asked > 2 });
    assert.equal(again.resident, stopped.resident + 6 + 6);
    const done = await w.decodeStep("stopped", { ...opts, shouldStop: () => false });
    assert.equal(done.cancelled, undefined);
    assert.equal(w.stats.reruns, before.reruns, "no rewind");
    assert.equal(w.stats.positions - before.positions, ids.length, "every position run once");
    assert.equal(done.tokenId, want.tokenId);
    assert.equal(bits(w.lastLogits), bits(wantLogits), "logits bit for bit");
    assert.equal(bits(done.attention), bits(want.attention), "attention bit for bit");
    for (const id of ["twin", "stopped"]) w.append(id, [want.tokenId]);
    const next = await w.decodeStep("twin", opts);
    const atDecode = await w.decodeStep("stopped", { ...opts, shouldStop: () => true });
    assert.deepEqual(atDecode, { cancelled: true, resident: ids.length, stats: { positions: 0, chunks: 0, fillMs: 0 } });
    const resumed = await w.decodeStep("stopped", opts);
    assert.equal(resumed.stats.positions, 1);
    assert.equal(bits(resumed.attention), bits(next.attention));
  }],
  ["load refuses a backend other than WebGPU and options it does not know", async () => {
    const stubOrt = { env: {} };
    await assert.rejects(OptZeosWorker.load({ ort: stubOrt, Tokenizer: null, read: null, backend: "wasm" }), null, "backend wasm");
    await assert.rejects(OptZeosWorker.load({ ort: stubOrt, Tokenizer: null, read: null, threads: 4 }), null, "an unknown option");
  }],
];

/** TransformersWorker's clauses (an `export_model.py` export, run on WebGPU). */
export async function transformersClauses(w, check) {
  await runClauses(w, TRANSFORMERS, check, "TransformersWorker");
}

const TRANSFORMERS = [
  ["info names a pad the tokenizer cannot produce and the ChatML markers", (w) => {
    const info = w.info();
    assert.deepEqual(info.controlIds.slice(0, 2).map((id) => w.piece(id)), ["<|im_start|>", "<|im_end|>"]);
    assert.equal(info.controlIds.includes(info.padId), true);
    for (const text of [w.piece(info.padId), "<|im_start|>user", " <|im_end|>", "<|endoftext|>"]) {
      const ids = Array.from(w.tokenize(text));
      for (const id of info.controlIds) assert.equal(ids.includes(id), false, `${text} -> ${ids}`);
    }
    assert.equal(info.vocabSize, w.meta.tokenizerSize);
    assert.equal(typeof w.piece(info.vocabSize - 1), "string");
    assert.throws(() => w.piece(info.vocabSize));
  }],
  ["tokenize adds no BOS and round-trips through piece", (w) => {
    const text = "say 1; write tools 10; read stdin;";
    const ids = w.tokenize(text);
    assert.ok(ids instanceof Int32Array);
    assert.equal(Array.from(ids).map((id) => w.piece(id)).join(""), text);
    assert.equal(w.tokenize("").length, 0);
  }],
  ["a decode step leaves the context as it was and reports normalised attention", async (w) => {
    w.createContext("step");
    const ids = prompt(w, "Count to three.", false);
    await w.append("step", ids);
    const { tokenId, attention } = await w.decodeStep("step", ALL);
    assert.equal(w.length("step"), ids.length);
    const again = await w.decodeStep("step", ALL);
    assert.equal(again.tokenId, tokenId, "a repeated step recomputes the same position");
    assert.equal(bits(again.attention), bits(attention));
    assert.equal(attention.length, ids.length);
    assert.ok(Math.abs(attention.reduce((a, b) => a + b, 0) - 1) < 1e-3);
    assert.ok(attention.every((v) => v >= 0));
  }],
  ["a masked position receives exactly zero on every step, and the rest still sum to one", async (w) => {
    w.createContext("mask");
    await w.append("mask", prompt(w, "The secret word is pineapple. Say a fruit.", false));
    const hidden = [3, 4, 5, 6, 7, 8];
    for (let n = 0; n < 4; n++) {
      const allowedBlocks = new Uint8Array(w.length("mask")).fill(1);
      for (const p of hidden) allowedBlocks[p] = 0;
      const { attention } = await step(w, "mask", { allowedBlocks, allowedTokens: null });
      for (const p of hidden) assert.equal(attention[p], 0, `position ${p} on step ${n}`);
      assert.ok(Math.abs(attention.reduce((a, b) => a + b, 0) - 1) < 1e-3);
    }
  }],
  ["allowedTokens, and the refusals", async (w) => {
    w.createContext("vocab");
    await w.append("vocab", prompt(w, "Write a poem about the sea.", false));
    const digits = [..."0123456789"].map((d) => w.tokenize(d)[0]);
    const allowedTokens = new Uint8Array(w.meta.tokenizerSize);
    for (const id of digits) allowedTokens[id] = 1;
    for (let i = 0; i < 2; i++) {
      const { tokenId } = await step(w, "vocab", { allowedBlocks: null, allowedTokens });
      assert.ok(digits.includes(tokenId), `chose ${tokenId}`);
    }
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: null, allowedTokens: new Uint8Array(10) }), null, "a short allowedTokens");
    const n = w.length("vocab");
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n - 1).fill(1), allowedTokens: null }), null, "a short mask");
    await assert.rejects(w.decodeStep("vocab", { allowedBlocks: new Uint8Array(n), allowedTokens: null }), null, "a mask hiding everything");
  }],
  ["truncate past a snapshot re-runs from it, as a fresh prefix would", async (w) => {
    const ids = prompt(w, "Count from one to ten, then back down again. ".repeat(30), false);
    assert.ok(ids.length > TW_SNAPSHOT_EVERY + 1);
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
    assert.equal(w.stats.rerunPositions - before.rerunPositions, ids.length - 1 - TW_SNAPSHOT_EVERY);
  }],
  ["a step whose mask hides what the recurrent state saw re-runs it without that", async (w) => {
    const ids = prompt(w, "The secret word is pineapple. Say a fruit.", false);
    const hidden = 5;
    const allowedBlocks = (n) => new Uint8Array(n).map((_, b) => (b === hidden ? 0 : 1));
    w.createContext("seen");
    await w.append("seen", ids);
    const first = await step(w, "seen");
    const before = { ...w.stats };
    const masked = await w.decodeStep("seen", { allowedBlocks: allowedBlocks(ids.length + 1), allowedTokens: null });
    assert.equal(masked.attention[hidden], 0);
    assert.equal(w.stats.reruns, before.reruns + 1);
    w.createContext("never");
    await w.append("never", [...ids, first.tokenId]);
    const again = await w.decodeStep("never", { allowedBlocks: allowedBlocks(ids.length + 1), allowedTokens: null });
    assert.equal(again.tokenId, masked.tokenId);
    assert.equal(bits(again.attention), bits(masked.attention));
  }],
  ["a step told to stop keeps what it ran, and resumes to the same bits", async (w) => {
    const ids = prompt(w, "The secret word is pineapple. Say a fruit, then say a vegetable.", false);
    const mask = new Uint8Array(ids.length).map((_, b) => (b === 1 ? 0 : 1));
    const opts = { allowedBlocks: mask, allowedTokens: null, maxChunk: 4 };
    w.createContext("twin");
    await w.append("twin", ids);
    const want = await w.decodeStep("twin", opts);
    w.createContext("stopped");
    await w.append("stopped", ids);
    let asked = 0;
    const first = await w.decodeStep("stopped", { ...opts, shouldStop: () => ++asked > 2 });
    assert.deepEqual({ cancelled: first.cancelled, resident: first.resident, chunks: first.stats.chunks }, { cancelled: true, resident: 8, chunks: 2 });
    const done = await w.decodeStep("stopped", opts);
    assert.equal(done.tokenId, want.tokenId);
    assert.equal(bits(done.attention), bits(want.attention));
  }],
  ["fork deep-copies tokens and cache: the two then decode independently", async (w) => {
    w.createContext("parent");
    await w.append("parent", prompt(w, "Count to five.", false));
    await step(w, "parent");
    w.fork("parent", "child");
    const a = await step(w, "parent");
    const b = await step(w, "child");
    assert.equal(a.tokenId, b.tokenId);
    assert.equal(bits(a.attention), bits(b.attention));
    const before = w.length("parent");
    w.truncate("child", 3);
    assert.equal(w.length("parent"), before);
  }],
  ["the same calls give the same tokens and attention bits", async (w) => {
    const run = async (name) => {
      w.createContext(name);
      await w.append(name, prompt(w, "Say the numbers from one to four.", false));
      const out = [];
      for (let i = 0; i < 6; i++) {
        const { tokenId, attention } = await step(w, name);
        out.push(`${tokenId}:${bits(attention)}`);
      }
      w.destroyContext(name);
      return out;
    };
    assert.deepEqual(await run("first"), await run("second"));
  }],
];
