// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// OptZeosWorker (web/opt_zeos_worker.js) on ONNX Runtime Web's WebGPU backend, driven
// through the ZeosModelWorker interface as JsMachine drives it, checked against runs
// with no cache, and timed. The result is printed and left in `window.benchResult`
// (`{checks: [{name, ok, detail}], timings, error?}`) for `tests/opt_zeos_webgpu.mjs`.

import { OptZeosWorker, argmax } from "../../web/opt_zeos_worker.js";

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = new URL(params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/", location.href);
const STEPS = Number(params.get("steps") ?? 32);

const logEl = document.getElementById("log");
function log(line) {
  logEl.textContent += `${line}\n`;
  console.log(line);
}

const checks = [];
function check(name, ok, detail = "") {
  checks.push({ name, ok: Boolean(ok), detail });
  log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` -- ${detail}` : ""}`);
}

const NOTES = [
  "The branch line leaves the main station at the north end of the town, crosses the river on an iron bridge built in 1887, and climbs through three tunnels to the village of Ashby Cross.",
  "Trains run every forty minutes on weekdays, starting at six fifteen in the morning, and every hour on Saturdays. There is no service on Sundays except during the summer festival.",
  "The first train of the day waits at the junction for the connection from the coast, which is often late in winter, so passengers for the eight o'clock school bus should take the second train.",
  "The station at Millford Halt is a request stop: tell the guard before the train leaves the bridge, or wave from the platform. Bicycles are carried free of charge, but only four on each train.",
  "The ticket office at the main station opens at a quarter to six and closes at nine in the evening; outside those hours tickets are bought from the guard, who takes cash and cards.",
  "Return tickets are valid for a month, and a family ticket covers two adults and up to three children. Season tickets can be bought for a week, a month or a year.",
  "The line was closed for six months in 1962 after a landslide in the second tunnel, and reopened with a new signal box at the junction, which is now a museum open on Saturday afternoons.",
  "The steam train that runs during the festival is a tank engine built in 1924, restored by volunteers over eleven years. It pulls four carriages, one of them a dining car serving tea.",
  "Ashby Cross has a bakery, a post office, two public houses and a church with a square tower. The walk from the station to the reservoir takes forty minutes along a marked footpath.",
  "In heavy snow the last train of the evening may be cancelled; a notice is posted on the board by the ticket office by four o'clock, and a bus replaces it from the station forecourt.",
  "Dogs travel free if they stay on the floor. Luggage larger than a suitcase must go in the guard's van, and the guard's van also carries the newspapers for the village shop each morning.",
  "The timetable changes twice a year, in May and in October, and the new times are printed in the parish magazine as well as on the posters at every station on the line.",
];

function chat(w, text) {
  const [imStart, imEnd] = w.info().controlIds;
  return [
    imStart,
    ...w.tokenize(`user\n${text}`),
    imEnd,
    ...w.tokenize("\n"),
    imStart,
    ...w.tokenize("assistant\n<think>\n\n</think>\n\n"),
  ];
}

/** KL(p || q) of the softmaxes of two logit vectors. */
function kl(p, q) {
  const soft = (x) => {
    let m = -Infinity;
    for (const v of x) m = Math.max(m, v);
    let z = 0;
    for (const v of x) z += Math.exp(v - m);
    return { m, lz: Math.log(z) };
  };
  const a = soft(p);
  const b = soft(q);
  let total = 0;
  for (let i = 0; i < p.length; i++) {
    const lp = p[i] - a.m - a.lz;
    const lq = q[i] - b.m - b.lz;
    total += Math.exp(lp) * (lp - lq);
  }
  return total;
}

function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** The best and second-best logit, for how close a choice was. */
function margin(logits) {
  let a = -Infinity;
  let b = -Infinity;
  for (const v of logits) {
    if (v > a) {
      b = a;
      a = v;
    } else if (v > b) b = v;
  }
  return a - b;
}

async function timed(fn) {
  const t0 = performance.now();
  const value = await fn();
  return { ms: performance.now() - t0, value };
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
  const load = await timed(() => OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "webgpu" }));
  const w = load.value;
  log(`loaded in ${(load.ms / 1000).toFixed(1)} s on ${w.backend}`);
  const timings = { loadMs: load.ms, ort: ort.env.versions?.web, adapter: adapter.info?.architecture };

  // Shader compilation and first-use allocation, at the chunk sizes the checks run.
  w.createContext("warm");
  w.append("warm", chat(w, NOTES.join(" ").repeat(3)).slice(0, 600));
  await w.decodeStep("warm", ALL);
  w.append("warm", [w.tokenize(" The")[0]]);
  await w.decodeStep("warm", ALL);
  w.destroyContext("warm");

  // -- a ~1k-token prompt, then greedy steps ----------------------------------------
  const prompt = chat(
    w,
    `Here are my notes on the branch line.\n\n${NOTES.join("\n\n")}\n\n${NOTES.slice().reverse().join("\n\n")}\n\nIn two sentences, when should I travel to Ashby Cross for the school bus, and why?`,
  );
  const P = prompt.length;
  log(`prompt: ${P} tokens`);
  w.createContext("main");
  w.append("main", prompt);
  const gen = [];
  const first = await timed(() => w.decodeStep("main", ALL));
  timings.prefill = { tokens: P, ms: first.ms, tokPerS: (P * 1000) / first.ms };
  log(`prefill ${P} + first step: ${first.ms.toFixed(0)} ms, ${timings.prefill.tokPerS.toFixed(0)} tok/s`);
  const firstLogits = w.lastLogits;
  const firstAttention = first.value.attention;
  gen.push(first.value.tokenId);
  check("attention covers the context and sums to one",
    firstAttention.length === P && Math.abs(firstAttention.reduce((a, b) => a + b, 0) - 1) < 1e-3,
    `length ${firstAttention.length}`);
  const steps = [];
  for (let i = 1; i < STEPS; i++) {
    w.append("main", [gen[gen.length - 1]]);
    const s = await timed(() => w.decodeStep("main", ALL));
    steps.push(s.ms);
    gen.push(s.value.tokenId);
  }
  const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
  timings.decode = { steps: steps.length, meanMs: mean, tokPerS: 1000 / mean, atLength: P + STEPS };
  log(`decode: ${mean.toFixed(1)} ms/step (${(1000 / mean).toFixed(1)} tok/s) over ${steps.length} steps`);
  const text = gen.map((id) => w.piece(id)).join("");
  log(`reply: ${JSON.stringify(text)}`);
  timings.reply = text;

  // The same tokens in one run with no cache, the logits of every generated position.
  const full = [...prompt, ...gen.slice(0, -1)];
  const ref = await timed(() => w.reference(full, null, 2048, STEPS));
  // The cache path runs the prompt in other chunks (cut at the snapshots) and then one
  // token per run, so float16 rounding differs: a choice may only differ from the
  // no-cache run's where that run's best two logits are within NEAR_TIE.
  const NEAR_TIE = 0.125;
  const mismatches = [];
  let worstKl = 0;
  for (let k = 0; k < STEPS; k++) {
    const want = argmax(ref.value.rows[k], null, w.info().vocabSize);
    if (want !== gen[k]) mismatches.push({ step: k, got: gen[k], want, margin: margin(ref.value.rows[k]) });
  }
  worstKl = kl(ref.value.rows[0], firstLogits);
  timings.agreement = { mismatches, firstStepKl: worstKl, firstStepMaxDiff: maxDiff(firstLogits, ref.value.rows[0]) };
  check(`${STEPS} greedy steps equal a no-cache run of the same tokens, but for float16 near-ties`,
    mismatches.every((m) => m.margin < NEAR_TIE),
    `${STEPS - mismatches.length}/${STEPS} identical; ${JSON.stringify(mismatches)}; reference ${ref.ms.toFixed(0)} ms`);
  // Between two chunkings of the same prompt the logits differ by about KL 4e-3: the
  // recurrent state crosses a chunk boundary in float16. Not the cache's doing; the
  // bit-for-bit check below holds the chunking fixed.
  check("the first step's distribution matches the no-cache run's", worstKl < 1e-2,
    `KL ${worstKl.toExponential(2)}, max |logit diff| ${maxDiff(firstLogits, ref.value.rows[0]).toFixed(3)}`);
  const aligned = await w.reference(prompt, null, 256);
  check("and is bit for bit a no-cache run cut into the same chunks", maxDiff(aligned.logits, firstLogits) === 0 &&
    maxDiff(aligned.attention, firstAttention) === 0);

  // -- hiding past positions --------------------------------------------------------
  w.append("main", [gen[gen.length - 1]]);
  const n = w.length("main");
  const hidden = [];
  for (let p = 600; p < 616; p++) hidden.push(p);
  const mask = new Uint8Array(n).fill(1);
  for (const p of hidden) mask[p] = 0;
  const before = { ...w.stats };
  const masked = await timed(() => w.decodeStep("main", { allowedBlocks: mask, allowedTokens: null }));
  const replayed = w.stats.rerunPositions - before.rerunPositions;
  timings.maskReplay = { ms: masked.ms, fromSnapshot: replayed, positionsRun: n - (600 - replayed) };
  log(`mask over 600..615 at length ${n}: ${masked.ms.toFixed(0)} ms, ${n - (600 - replayed)} positions run from the snapshot at ${600 - replayed}`);
  const maskedLogits = w.lastLogits;
  check("hidden positions receive exactly zero attention", hidden.every((p) => masked.value.attention[p] === 0));
  check("the rest still sums to one", Math.abs(masked.value.attention.reduce((a, b) => a + b, 0) - 1) < 1e-3);
  check("the replay started at the latest snapshot before the change", replayed === 600 - 512);

  w.createContext("fresh");
  w.append("fresh", [...full, gen[gen.length - 1]]);
  const fresh = await w.decodeStep("fresh", { allowedBlocks: mask, allowedTokens: null });
  const freshLogits = w.lastLogits;
  check("replay under the new mask equals a fresh prefill under it",
    fresh.tokenId === masked.value.tokenId && maxDiff(freshLogits, maskedLogits) === 0,
    `max |logit diff| ${maxDiff(freshLogits, maskedLogits)}, attention ${maxDiff(fresh.attention, masked.value.attention)}`);
  const unmasked = await w.reference([...full, gen[gen.length - 1]], null, 256);
  check("the mask changed the logits", maxDiff(unmasked.logits, maskedLogits) > 0,
    `max |diff| ${maxDiff(unmasked.logits, maskedLogits).toFixed(3)}`);
  w.destroyContext("fresh");

  // -- truncate and fork ------------------------------------------------------------
  w.fork("main", "child");
  w.truncate("main", P);
  const cut = await w.decodeStep("main", ALL);
  check("truncate to the prompt reproduces the first step",
    cut.tokenId === gen[0] && maxDiff(w.lastLogits, firstLogits) === 0,
    `max |diff| ${maxDiff(w.lastLogits, firstLogits)}`);
  const child = await w.decodeStep("child", { allowedBlocks: mask, allowedTokens: null });
  check("a fork made before the cut keeps the longer context",
    w.length("child") === n && child.tokenId === masked.value.tokenId && maxDiff(w.lastLogits, maskedLogits) === 0);
  w.destroyContext("child");
  w.destroyContext("main");

  // -- prefill speed by chunk size --------------------------------------------------
  const long = [];
  while (long.length < 2048) long.push(...prompt);
  long.length = 2048;
  for (const every of [256, 512, 1024, 2048]) {
    w.snapshotEvery = every;
    w.createContext("speed");
    w.append("speed", long);
    const t = await timed(() => w.decodeStep("speed", ALL));
    timings[`prefill2048_chunk${every}`] = { ms: t.ms, tokPerS: (2048 * 1000) / t.ms };
    log(`prefill 2048 in chunks of ${every}: ${t.ms.toFixed(0)} ms, ${((2048 * 1000) / t.ms).toFixed(0)} tok/s`);
    w.destroyContext("speed");
  }
  w.snapshotEvery = 256;

  window.benchResult = { checks, timings, stats: w.stats };
  log(JSON.stringify(window.benchResult, null, 2));
  await w.release();
} catch (error) {
  window.benchResult = { checks, error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
