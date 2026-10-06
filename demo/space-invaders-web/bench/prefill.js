// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// How long OptZeosWorker on WebGPU takes to read one Space Invaders board after the
// pilot's real prefix, by prefill chunk size; decode steps at 3-4k positions; and the
// replay a splice of one old board costs. The prefix and the boards are the real ids
// (`prompt_sizes.py` writes them to `.cache/webgpu_inputs.json`): the kernel's injection
// of pilot.md and the rules, then framed boards, each with the arrival framing, the
// board's words, the reopened turn and a move.
//
// The chunk size is `worker.chunk`, the field `fill()` cuts runs at (it also cuts at
// every multiple of `snapshotEvery`, 256), which is what `maxChunk` will set per step.
// Each trial forks a base context that already holds the prefix, so the prefix is run
// once per board. Result in `window.benchResult` for `prefill_webgpu.mjs`.

import { OptZeosWorker } from "/coop-count-web/web/opt_zeos_worker.js";

const params = new URLSearchParams(location.search);
const ortUrl = params.get("ort") ?? "/coop-count-web/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs";
const modelUrl = new URL(params.get("model") ?? "/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT/", location.href);
const CHUNKS = (params.get("chunks") ?? "32,64,128,256,2048").split(",").map(Number);
const REPS = Number(params.get("reps") ?? 3);
const DECODE = Number(params.get("decode") ?? 24);
const BOARDS = (params.get("boards") ?? "default,ablation").split(",").filter(Boolean);

const logEl = document.getElementById("log");
function log(line) {
  logEl.textContent += `${line}\n`;
  console.log(line);
}

const ALL = { allowedBlocks: null, allowedTokens: null };
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const mean = (xs) => (xs.length ? sum(xs) / xs.length : 0);
const median = (xs) => {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

/** Runs of the graph during `fn`, from the worker's activity hook. */
async function recorded(w, fn) {
  const runs = [];
  w.onActivity = (a) => {
    if (a.ms !== undefined && a.phase !== "skip") runs.push({ phase: a.phase, start: a.start, count: a.count, ms: a.ms });
  };
  const t0 = performance.now();
  const value = await fn();
  const ms = performance.now() - t0;
  w.onActivity = null;
  return { ms, value, runs };
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const ort = await import(ortUrl);
  ort.env.wasm.wasmPaths = ortUrl.replace(/[^/]*$/, "");
  const { Tokenizer } = await import("/coop-count-web/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs");
  const inputs = await (await fetch("/space-invaders-web/bench/.cache/webgpu_inputs.json")).json();
  const read = async (name) => {
    const response = await fetch(new URL(name, modelUrl));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const source = async (name) => new URL(name, modelUrl).href;
  const t0 = performance.now();
  const w = await OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "webgpu" });
  const timings = {
    loadMs: performance.now() - t0,
    ort: ort.env.versions?.web,
    adapter: `${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""}`,
    boards: {},
  };
  log(`loaded in ${(timings.loadMs / 1000).toFixed(1)} s; ort ${timings.ort}; ${timings.adapter}`);
  const vocabMask = new Uint8Array(w.info().vocabSize).fill(1);

  for (const name of BOARDS) {
    const { prefix: pilot, boards } = inputs[name];
    const prefix = [...pilot, ...boards[0], ...boards[1]];
    const out = { prefixTokens: prefix.length, chunks: {}, decode: null, splice: {} };
    timings.boards[name] = out;

    w.chunk = 2048;
    w.createContext("base");
    w.append("base", prefix);
    const base = await recorded(w, () => w.decodeStep("base", ALL));
    out.prefixPrefill = { ms: base.ms, tokPerS: (prefix.length * 1000) / base.ms, runs: base.runs.length };
    log(`${name}: prefix ${prefix.length} tokens in ${base.ms.toFixed(0)} ms (${out.prefixPrefill.tokPerS.toFixed(0)} tok/s)`);

    // One untimed pass per chunk size: the first run at a new input length compiles.
    for (const chunk of CHUNKS) {
      w.chunk = chunk;
      w.fork("base", "warm");
      w.append("warm", boards[2]);
      await w.decodeStep("warm", ALL);
      w.destroyContext("warm");
    }

    for (const chunk of CHUNKS) {
      const trials = [];
      for (let rep = 0; rep < REPS; rep++) {
        const board = boards[2 + (rep % (boards.length - 2))];
        w.chunk = chunk;
        w.fork("base", "t");
        w.append("t", board);
        const step = await recorded(w, () => w.decodeStep("t", { allowedBlocks: null, allowedTokens: vocabMask }));
        trials.push({
          tokens: board.length,
          ms: step.ms,
          runs: step.runs.length,
          runMs: step.runs.map((r) => Math.round(r.ms * 10) / 10),
          runCounts: step.runs.map((r) => r.count),
        });
        w.destroyContext("t");
      }
      const tokens = mean(trials.map((t) => t.tokens));
      const ms = median(trials.map((t) => t.ms));
      out.chunks[chunk] = {
        tokens,
        msMedian: ms,
        msMean: mean(trials.map((t) => t.ms)),
        tokPerS: (tokens * 1000) / ms,
        runs: mean(trials.map((t) => t.runs)),
        trials,
      };
      log(`${name}: board ~${tokens.toFixed(0)} tokens, chunk ${chunk}: ${ms.toFixed(0)} ms median, ${out.chunks[chunk].runs.toFixed(1)} runs, ${out.chunks[chunk].tokPerS.toFixed(0)} tok/s`);
    }

    // Decode steps with the context at prefix + 6 boards (~4k positions).
    w.chunk = 2048;
    w.fork("base", "d");
    for (const b of boards.slice(2, 6)) w.append("d", b);
    let step = await w.decodeStep("d", ALL);
    const steps = [];
    for (let i = 0; i < DECODE; i++) {
      w.append("d", [step.tokenId]);
      const t = await recorded(w, () => w.decodeStep("d", { allowedBlocks: null, allowedTokens: vocabMask }));
      steps.push(t.ms);
      step = t.value;
    }
    out.decode = { atLength: w.length("d"), steps: steps.length, msMean: mean(steps), msMedian: median(steps), msMax: Math.max(...steps) };
    log(`${name}: decode at ${out.decode.atLength}: ${out.decode.msMean.toFixed(1)} ms mean, ${out.decode.msMedian.toFixed(1)} median`);

    // Splice of the first board after the pilot prefix, as JsMachine.splice does it:
    // truncate to the board's start, append the replacement and everything after.
    const at = pilot.length;
    const tail = w.ctx("d").tokens.slice(at + boards[0].length);
    const stub = boards[0].slice(0, 8);
    for (const chunk of [2048, 256, 64]) {
      w.chunk = 2048;
      w.fork("d", "s");
      w.chunk = chunk;
      const before = { ...w.stats };
      const t = await recorded(w, async () => {
        w.truncate("s", at);
        w.append("s", [...stub, ...tail]);
        return w.decodeStep("s", ALL);
      });
      out.splice[chunk] = {
        length: w.length("s"),
        spliceAt: at,
        positionsRun: w.stats.positions - before.positions,
        rerunPositions: w.stats.rerunPositions - before.rerunPositions,
        ms: t.ms,
        runs: t.runs.length,
      };
      log(`${name}: splice at ${at} of ${w.length("s")}, chunk ${chunk}: ${t.ms.toFixed(0)} ms, ${out.splice[chunk].positionsRun} positions run`);
      w.destroyContext("s");
    }
    w.destroyContext("d");
    w.destroyContext("base");
  }
  // What one run costs by how much context is behind it: a run of `count` positions
  // after `past` cached ones, the pilot's own ids. `scaling=` empty skips it.
  const SCALING = (params.get("scaling") ?? "0,512,1024,2048,3072").split(",").filter(Boolean).map(Number);
  const COUNTS = (params.get("counts") ?? "1,8,32,64,128,256").split(",").map(Number);
  if (SCALING.length) {
    const { prefix: pilot, boards } = inputs.default;
    const ids = [...pilot, ...boards.flat()];
    timings.scaling = [];
    for (const past of SCALING) {
      w.chunk = 2048;
      w.createContext("p");
      if (past > 0) {
        w.append("p", ids.slice(0, past));
        await w.decodeStep("p", ALL);
      }
      for (const count of COUNTS) {
        const ms = [];
        for (let rep = 0; rep < 4; rep++) {
          w.fork("p", "q");
          w.chunk = count;
          w.append("q", ids.slice(past, past + count));
          const t = await recorded(w, () => w.decodeStep("q", ALL));
          // The first rep at a new shape compiles; it is kept but the median outvotes it.
          ms.push(...t.runs.map((r) => r.ms));
          w.destroyContext("q");
        }
        const row = { past, count, msMedian: median(ms), runs: ms.length, ms: ms.map((v) => Math.round(v)) };
        timings.scaling.push(row);
        log(`scaling: past ${past}, run of ${count}: ${row.msMedian.toFixed(0)} ms median`);
      }
      w.destroyContext("p");
    }
  }
  window.benchResult = { timings, stats: w.stats };
  log(JSON.stringify(window.benchResult, null, 2));
  await w.release();
} catch (error) {
  window.benchResult = { error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
