// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The OPT+ZEOS graph and OptZeosWorker (web/opt_zeos_worker.js) on ONNX Runtime Web's
// WebGPU backend, checked where it matters:
//
// - **Against the reference.** Prefilled in chunks of 1 and 7 (an 88-token chat turn) and
//   16 and 512 (a 634-token one), then one decode step, the logits choose what
//   `transformers`' Qwen3.5-4B (float32) chooses and stay within KL 0.1 of it. Hiding the
//   three tokens of a password the last position must recall moves the distribution as
//   `export_model.ZeosQwen` moves it under the same mask. The prompts are
//   `reference_prompts.json`; the reference logits are what `export/opt_zeos_reference.py`
//   writes (`models/.reference/opt-zeos-<key>.npz`, keyed by the prompts' ids). Without
//   that file the check fails: the comparison is the point. And one run of 2048
//   positions after a cache, the graph's largest chunk, leaves a hidden position at
//   exactly zero.
// - **The mask hides tokens.** The hidden tokens of a context are swapped for others,
//   with the same mask: every logit and every attention weight is the same bit for bit,
//   after a prefill, after a single-token decode step over that cache, and when every
//   position (the hidden ones too) runs as a single-token step. Both with hidden runs
//   carried past (`skipHidden`, the default) and with every hidden position run through
//   the graph. With nothing hidden the same swap does change the logits.
// - **A cancelled step resumes to the same bits.** A step stopped before a run of the
//   graph and resumed with the same `maxChunk` chooses the same token with the same
//   logits and attention as the uninterrupted step, with and without a mask.
// - **The interface clauses** (`unit_clauses.js`), with snapshots every 16 positions and
//   at most 4 kept: tokenisation and pieces, greedy steps against cache-free runs and the
//   snapshot thinning, a replay after a past position is hidden, skipped hidden runs bit
//   for bit the runs executed when aligned, a narrowed mask on a second cache (switching
//   back, catching up, a third mask), one cache rewinding, repeated steps, truncate and
//   fork, allowedTokens, sample and the refusals, maxChunk cut at the snapshots, a stopped
//   step resumed, and `load` refusing what it cannot do. Then the same kind of clauses for
//   TransformersWorker over an `export_model.py` q4 export (`transformers`, default
//   `/models/Qwen3.5-2B-zeos-q4/`), skipped and said so when that export is absent.
//
// The result is printed and left in `window.benchResult` (`{checks: [{name, ok,
// detail}], skipped, timings, error?}`) for `tests/opt_zeos_webgpu.mjs --page checks.html`.

import { OptZeosWorker, argmax } from "../../web/opt_zeos_worker.js";
import { TransformersWorker } from "../../web/transformers_worker.js";
import { optZeosClauses, transformersClauses } from "./unit_clauses.js";

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = new URL(params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/", location.href);
const referenceUrl = new URL(params.get("reference") ?? "/models/.reference/", location.href);
/** An `export_model.py` export that runs on WebGPU (q4), for TransformersWorker's clauses. */
const transformersUrl = new URL(params.get("transformers") ?? "/models/Qwen3.5-2B-zeos-q4/", location.href);

const logEl = document.getElementById("log");
function log(line) {
  logEl.textContent += `${line}\n`;
  console.log(line);
}

const checks = [];
const skipped = [];
function check(name, ok, detail = "") {
  checks.push({ name, ok: Boolean(ok), detail });
  log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` -- ${detail}` : ""}`);
}
function skip(name, why) {
  skipped.push({ name, why });
  log(`skip ${name} -- ${why}`);
}

const SPECIAL = /(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<think>|<\/think>)/;

/** `text` as ids, with the special tokens it spells taken as those tokens. */
function encode(w, specials, text) {
  const ids = [];
  for (const part of text.split(SPECIAL)) {
    if (part === "") continue;
    if (specials.has(part)) ids.push(specials.get(part));
    else ids.push(...w.tokenize(part));
  }
  return ids;
}

/** The ids of the added tokens SPECIAL names, read from their pieces. */
function specialIds(w) {
  const specials = new Map();
  const { vocabSize } = w.info();
  for (let id = vocabSize - 1; id >= vocabSize - 64; id--) {
    const piece = w.piece(id);
    if (SPECIAL.test(piece) && piece.replace(SPECIAL, "") === "") specials.set(piece, id);
  }
  for (const name of ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]) {
    if (!specials.has(name)) throw new Error(`no added token ${name}`);
  }
  return specials;
}

/** The arrays of an `np.savez` file (stored, not compressed), as Float32Arrays. */
function readNpz(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const arrays = {};
  for (let at = 0; at + 30 <= bytes.length; ) {
    if (view.getUint32(at, true) !== 0x04034b50) break;
    const method = view.getUint16(at + 8, true);
    const nameLength = view.getUint16(at + 26, true);
    const extraLength = view.getUint16(at + 28, true);
    const name = new TextDecoder().decode(bytes.subarray(at + 30, at + 30 + nameLength));
    if (method !== 0) throw new Error(`${name} is compressed; expected np.savez's stored entries`);
    const start = at + 30 + nameLength + extraLength;
    const headerLength = view.getUint16(start + 8, true);
    const header = new TextDecoder().decode(bytes.subarray(start + 10, start + 10 + headerLength));
    if (!header.includes("'<f4'")) throw new Error(`${name}: ${header}`);
    const count = Number(/'shape': \((\d+),\)/.exec(header)[1]);
    const data = start + 10 + headerLength;
    arrays[name.replace(/\.npy$/, "")] = new Float32Array(buffer.slice(data, data + 4 * count));
    at = data + 4 * count;
    // A data descriptor follows an entry written without its sizes up front.
    if (view.getUint32(at, true) === 0x08074b50) at += view.getUint32(at + 8, true) === 0xffffffff ? 24 : 16;
    while (at + 4 <= bytes.length && view.getUint32(at, true) !== 0x04034b50 && view.getUint32(at, true) !== 0x02014b50) at++;
  }
  return arrays;
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** KL(p || q) of the softmaxes of the first `n` entries of two logit vectors. */
function kl(p, q, n = Math.min(p.length, q.length)) {
  const lse = (x) => {
    let m = -Infinity;
    for (let i = 0; i < n; i++) m = Math.max(m, x[i]);
    let z = 0;
    for (let i = 0; i < n; i++) z += Math.exp(x[i] - m);
    return m + Math.log(z);
  };
  const a = lse(p);
  const b = lse(q);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const lp = p[i] - a;
    total += Math.exp(lp) * (lp - (q[i] - b));
  }
  return total;
}

function corr(x, y) {
  const n = x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function bits(f) {
  return Array.from(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)).join(",");
}

function sum(xs) {
  return xs.reduce((a, b) => a + b, 0);
}

function maskOf(n, hidden) {
  const mask = new Uint8Array(n).fill(1);
  for (const p of hidden) mask[p] = 0;
  return mask;
}

let jobs = 0;

/** Prefill all but the last id in runs of at most `chunk`, then one decode step, under
 * `mask` (all visible by default): the logits and attention of that step. */
async function lastStep(w, ids, chunk, mask = null) {
  const job = `last-${jobs++}`;
  w.createContext(job);
  try {
    const opts = { allowedBlocks: mask, allowedTokens: null, maxChunk: chunk };
    w.append(job, ids.slice(0, -1));
    if (ids.length > 1) await w.decodeStep(job, opts);
    w.append(job, ids.slice(-1));
    const step = await w.decodeStep(job, opts);
    return { tokenId: step.tokenId, logits: w.lastLogits, attention: step.attention };
  } finally {
    w.destroyContext(job);
  }
}

/** Another token for each hidden position: an ordinary id, never the one there. */
function swapped(ids, hidden, vocabSize) {
  const out = ids.slice();
  for (const p of hidden) {
    let id = 1000 + ((ids[p] * 7919 + p * 104729) % 200000);
    if (id === ids[p]) id += 1;
    if (id >= vocabSize) throw new Error("swap id outside the vocabulary");
    out[p] = id;
  }
  return out;
}

/** Each context's step results: after `ids` under `mask` (in runs of at most `chunk`),
 * then after one more token. */
async function twoSteps(w, ids, mask, chunk, next) {
  const job = `swap-${jobs++}`;
  w.createContext(job);
  try {
    w.append(job, ids);
    const first = await w.decodeStep(job, { allowedBlocks: mask, allowedTokens: null, maxChunk: chunk });
    const firstLogits = w.lastLogits;
    w.append(job, [next ?? first.tokenId]);
    const wider = new Uint8Array(ids.length + 1);
    wider.set(mask);
    wider[ids.length] = 1;
    const second = await w.decodeStep(job, { allowedBlocks: wider, allowedTokens: null });
    return {
      first: { tokenId: first.tokenId, logits: firstLogits, attention: first.attention },
      second: { tokenId: second.tokenId, logits: w.lastLogits, attention: second.attention },
    };
  } finally {
    w.destroyContext(job);
  }
}

function same(a, b) {
  return a.tokenId === b.tokenId && bits(a.logits) === bits(b.logits) && bits(a.attention) === bits(b.attention);
}

const ALL = { allowedBlocks: null, allowedTokens: null };

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const ort = await import(ortUrl);
  ort.env.wasm.wasmPaths = ortUrl.replace(/[^/]*$/, "");
  const { Tokenizer } = await import("/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs");
  log(`ort ${ort.env.versions?.web ?? "?"}; adapter ${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""}`);
  const read = async (name) => {
    const response = await fetch(new URL(name, modelUrl));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const source = async (name) => new URL(name, modelUrl).href;
  const began = performance.now();
  const w = await OptZeosWorker.load({ ort, Tokenizer, read, source });
  const timings = { loadMs: performance.now() - began };
  log(`loaded in ${(timings.loadMs / 1000).toFixed(1)} s on ${w.backend}`);
  const { vocabSize } = w.info();
  const specials = specialIds(w);
  const prompts = await (await fetch("reference_prompts.json")).json();
  const short = encode(w, specials, prompts.short);
  const long = encode(w, specials, prompts.long);
  const secret = encode(w, specials, prompts.secret);
  const SECRET_HIDDEN = prompts.secretHidden;
  log(`prompts: ${short.length}, ${long.length} and ${secret.length} tokens`);

  // -- against the reference ---------------------------------------------------------
  const t0 = performance.now();
  const key = await sha256Hex(`[${[short, long, secret, SECRET_HIDDEN].map((a) => `[${a.join(", ")}]`).join(", ")}]`);
  const npz = new URL(`opt-zeos-${key.slice(0, 16)}.npz`, referenceUrl);
  const response = await fetch(npz);
  if (!response.ok) {
    check("the reference logits exist", false, `no reference at ${npz.pathname} (HTTP ${response.status}); run export/opt_zeos_reference.py`);
  } else {
    const ref = readNpz(await response.arrayBuffer());
    for (const [name, ids, chunk] of [
      ["short", short, 1],
      ["short", short, 7],
      ["long", long, 16],
      ["long", long, 512],
    ]) {
      const got = await lastStep(w, ids, chunk);
      const want = ref[name];
      const divergence = kl(want, got.logits, vocabSize);
      check(
        `${name} prompt in chunks of ${chunk}: transformers' first choice, within KL 0.1`,
        argmax(got.logits, null, vocabSize) === argmax(want, null, vocabSize) && divergence < 0.1,
        `KL ${divergence.toExponential(2)}`,
      );
      check(
        `${name} prompt in chunks of ${chunk}: attention covers the context and sums to one`,
        got.attention.length === ids.length && Math.abs(sum(got.attention) - 1) < 1e-3 && got.attention.every((a) => a >= 0),
      );
    }
    check("the password is \" tangerine\"", SECRET_HIDDEN.map((p) => w.piece(secret[p])).join("") === " tangerine");
    const hiddenMask = maskOf(secret.length, SECRET_HIDDEN);
    const moved = kl(ref.zeos_open, ref.zeos_hidden, vocabSize);
    const want = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) want[i] = ref.zeos_hidden[i] - ref.zeos_open[i];
    for (const [chunk, skipHidden] of [
      [1, true],
      [16, true],
      [16, false],
    ]) {
      w.skipHidden = skipHidden;
      const open = await lastStep(w, secret, chunk);
      const hidden = await lastStep(w, secret, chunk, hiddenMask);
      const label = `password hidden, chunks of ${chunk}, hidden runs ${skipHidden ? "carried past" : "run"}`;
      const shift = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) shift[i] = hidden.logits[i] - open.logits[i];
      check(`${label}: the hidden tokens receive exactly zero, and are attended when open`,
        SECRET_HIDDEN.every((p) => hidden.attention[p] === 0 && open.attention[p] > 0));
      check(`${label}: hiding moves the distribution as ZeosQwen's mask does`,
        moved > 1 &&
          kl(open.logits, hidden.logits, vocabSize) > 0.5 * moved &&
          kl(ref.zeos_hidden, hidden.logits, vocabSize) < 0.1 * moved &&
          corr(shift, want) > 0.8,
        `moved ${moved.toFixed(2)} (reference), ${kl(open.logits, hidden.logits, vocabSize).toFixed(2)} (here); ` +
          `KL to the reference ${kl(ref.zeos_hidden, hidden.logits, vocabSize).toFixed(3)}; corr ${corr(shift, want).toFixed(3)}`);
      check(`${label}: the same first choices as the reference`,
        argmax(hidden.logits, null, vocabSize) === argmax(ref.zeos_hidden, null, vocabSize) &&
          argmax(open.logits, null, vocabSize) === argmax(ref.zeos_open, null, vocabSize));
    }
    w.skipHidden = true;
  }
  {
    // One run of 2048 positions after a cache of 50, as the graph's maxChunk allows.
    const ids = [];
    while (ids.length < 2099) ids.push(...long);
    ids.length = 2099;
    const mask = maskOf(ids.length, [100]);
    const every = w.snapshotEvery;
    w.snapshotEvery = 4096;
    w.createContext("wide");
    w.append("wide", ids.slice(0, 51));
    await w.decodeStep("wide", { allowedBlocks: mask, allowedTokens: null });
    w.append("wide", ids.slice(51));
    const step = await w.decodeStep("wide", { allowedBlocks: mask, allowedTokens: null });
    w.destroyContext("wide");
    w.snapshotEvery = every;
    check("one run of 2048 positions after a cache: the hidden one gets zero, the rest sums to one",
      step.stats.chunks === 1 && step.stats.positions === 2048 && step.attention.length === ids.length &&
        step.attention[100] === 0 && Math.abs(sum(step.attention) - 1) < 1e-3 && w.lastLogits.every(Number.isFinite),
      JSON.stringify(step.stats));
  }
  timings.referenceMs = performance.now() - t0;

  // -- swapping hidden tokens changes nothing ---------------------------------------
  const t1 = performance.now();
  // A hidden run long enough to carry past (8), a single position, and a pair.
  const longHidden = [40, 41, 42, 43, 44, 45, 46, 47, 300, 450, 451];
  const longMask = maskOf(long.length, longHidden);
  const longSwapped = swapped(long, longHidden, vocabSize);
  for (const skipHidden of [true, false]) {
    w.skipHidden = skipHidden;
    const how = skipHidden ? "hidden runs carried past" : "every hidden position run";
    const a = await twoSteps(w, long, longMask, null, null);
    const b = await twoSteps(w, longSwapped, longMask, null, a.first.tokenId);
    check(`hidden tokens swapped (${how}): the prefill's logits and attention are the same bits`, same(a.first, b.first),
      `token ${a.first.tokenId}/${b.first.tokenId}, max |logit diff| ${maxDiff(a.first.logits, b.first.logits)}, attention ${maxDiff(a.first.attention, b.first.attention)}`);
    check(`hidden tokens swapped (${how}): the decode step over that cache is the same bits`, same(a.second, b.second),
      `token ${a.second.tokenId}/${b.second.tokenId}, max |logit diff| ${maxDiff(a.second.logits, b.second.logits)}, attention ${maxDiff(a.second.attention, b.second.attention)}`);
    check(`hidden tokens swapped (${how}): they receive exactly zero`,
      longHidden.every((p) => a.first.attention[p] === 0 && a.second.attention[p] === 0));
  }
  // Every position a single-token step, the hidden ones too.
  const shortHidden = [10, 11, 12, 30, 60, 61];
  const shortMask = maskOf(short.length, shortHidden);
  const shortSwapped = swapped(short, shortHidden, vocabSize);
  for (const skipHidden of [true, false]) {
    w.skipHidden = skipHidden;
    const how = skipHidden ? "hidden runs carried past" : "every hidden position run";
    const a = await twoSteps(w, short, shortMask, 1, null);
    const b = await twoSteps(w, shortSwapped, shortMask, 1, a.first.tokenId);
    check(`hidden tokens swapped, one position a run (${how}): the same bits`, same(a.first, b.first) && same(a.second, b.second),
      `max |logit diff| ${maxDiff(a.first.logits, b.first.logits)} then ${maxDiff(a.second.logits, b.second.logits)}`);
  }
  w.skipHidden = true;
  {
    const a = await lastStep(w, long, null);
    const b = await lastStep(w, longSwapped, null);
    check("with nothing hidden the same swap changes the logits", maxDiff(a.logits, b.logits) > 0,
      `max |logit diff| ${maxDiff(a.logits, b.logits).toFixed(3)}`);
  }
  timings.swapMs = performance.now() - t1;

  // -- a cancelled step resumes to the same bits ------------------------------------
  const t2 = performance.now();
  for (const [stopAt, mask] of [
    [3, null],
    [5, null],
    [3, longMask],
  ]) {
    const opts = { allowedBlocks: mask, allowedTokens: null, maxChunk: 64 };
    const label = `stopped before run ${stopAt}${mask === null ? "" : ", under a mask"}`;
    w.createContext("twin");
    w.append("twin", long);
    const want = await w.decodeStep("twin", opts);
    const wantLogits = w.lastLogits;
    w.destroyContext("twin");
    w.createContext("cut");
    w.append("cut", long);
    let asked = 0;
    const cancelled = await w.decodeStep("cut", { ...opts, shouldStop: () => ++asked >= stopAt });
    check(`${label}: the step reports itself cancelled with what it ran resident`,
      cancelled.cancelled === true && cancelled.resident > 0 && cancelled.resident < long.length && cancelled.stats.chunks === stopAt - 1,
      JSON.stringify({ resident: cancelled.resident, chunks: cancelled.stats.chunks }));
    const done = await w.decodeStep("cut", opts);
    check(`${label}: resumed, it is the uninterrupted step bit for bit`,
      done.tokenId === want.tokenId && bits(w.lastLogits) === bits(wantLogits) && bits(done.attention) === bits(want.attention));
    check(`${label}: resumed, it ran only the rest`, done.stats.positions === long.length - cancelled.resident,
      `${done.stats.positions} of ${long.length}`);
    w.destroyContext("cut");
  }
  timings.cancelMs = performance.now() - t2;

  // -- the interface clauses, on a worker with small snapshot spacing ----------------
  const t3 = performance.now();
  await optZeosClauses(w, check);
  timings.clausesMs = performance.now() - t3;
  const stats = w.stats;
  await w.release();

  // -- TransformersWorker, on an export_model.py export that runs on WebGPU ---------
  const t4 = performance.now();
  const transformersRead = async (name) => {
    const response = await fetch(new URL(name, transformersUrl));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  if (!(await fetch(new URL("meta.json", transformersUrl))).ok) {
    skip("TransformersWorker's clauses", `no export at ${transformersUrl.pathname}; export_model.py --quant q4 writes one`);
  } else {
    const tw = await TransformersWorker.load({ ort, Tokenizer, read: transformersRead });
    log(`TransformersWorker loaded ${transformersUrl.pathname} on ${tw.backend}`);
    await transformersClauses(tw, check);
    await tw.release();
  }
  timings.transformersMs = performance.now() - t4;

  window.benchResult = { checks, skipped, timings, stats };
  log(JSON.stringify({ timings, skipped }, null, 2));
} catch (error) {
  window.benchResult = { checks, skipped, error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
