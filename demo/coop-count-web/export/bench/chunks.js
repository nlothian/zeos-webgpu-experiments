// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// OptZeosWorker's decode step that can stop (web/opt_zeos_worker.js) on ONNX Runtime
// Web's WebGPU backend: the time of a fresh prefill of 250 and 1000 positions at
// maxChunk 32, 64, 128, 256 and the graph's own, with the overhead of each extra chunk;
// and a step stopped mid-fill by `shouldStop`, resumed, and checked bit for bit against
// the uninterrupted step. The result is printed and left in `window.benchResult`
// (`{checks: [{name, ok, detail}], timings, error?}`) for `tests/opt_zeos_webgpu.mjs`:
//
//   node tests/opt_zeos_webgpu.mjs --page chunks.html

import { OptZeosWorker } from "../../web/opt_zeos_worker.js";

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = new URL(params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/", location.href);

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

const ALL = { allowedBlocks: null, allowedTokens: null };

function bits(f) {
  return Array.from(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)).join(",");
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const ort = await import(ortUrl);
  ort.env.wasm.wasmPaths = ortUrl.replace(/[^/]*$/, "");
  const { Tokenizer } = await import("/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs");
  const read = async (name) => {
    const response = await fetch(new URL(name, modelUrl));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const source = async (name) => new URL(name, modelUrl).href;
  const w = await OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "webgpu" });
  log(`loaded on ${w.backend}, ${adapter.info?.architecture ?? ""}`);
  const timings = {};
  const all = chat(w, NOTES.join(" ").repeat(3));
  // Warm every chunk size used below.
  for (const mc of [32, 64, 128, 256, null]) {
    w.createContext("warm");
    w.append("warm", all.slice(0, 300));
    await w.decodeStep("warm", { ...ALL, maxChunk: mc });
    w.destroyContext("warm");
  }
  for (const length of [250, 1000]) {
    const ids = all.slice(0, length);
    const rows = {};
    for (const mc of [32, 64, 128, 256, null]) {
      const ms = [];
      let chunks = 0;
      let token = null;
      for (let rep = 0; rep < 5; rep++) {
        w.createContext("t");
        w.append("t", ids);
        const t0 = performance.now();
        const r = await w.decodeStep("t", { ...ALL, maxChunk: mc });
        ms.push(performance.now() - t0);
        chunks = r.stats.chunks;
        token = r.tokenId;
        w.destroyContext("t");
      }
      ms.sort((a, b) => a - b);
      rows[mc ?? "default"] = { medianMs: ms[2], chunks, token };
      log(`len ${length} maxChunk ${mc}: ${ms[2].toFixed(1)} ms, ${chunks} chunks, token ${token}`);
    }
    const base = rows.default;
    for (const [k, r] of Object.entries(rows)) {
      if (k === "default" || r.chunks === base.chunks) continue;
      r.overheadPerChunkMs = (r.medianMs - base.medianMs) / (r.chunks - base.chunks);
      r.overheadPct = (100 * (r.medianMs - base.medianMs)) / base.medianMs;
    }
    timings[`len${length}`] = rows;
  }
  // Cancel and resume: bits equal the uninterrupted step at the same maxChunk.
  const ids = all.slice(0, 1000);
  w.createContext("twin");
  w.append("twin", ids);
  const want = await w.decodeStep("twin", { ...ALL, maxChunk: 64 });
  w.createContext("cut");
  w.append("cut", ids);
  let stop = false;
  let stopAt = 0;
  setTimeout(() => {
    stop = true;
    stopAt = performance.now();
  }, 60);
  const cancelled = await w.decodeStep("cut", { ...ALL, maxChunk: 64, shouldStop: () => stop });
  const cancelMs = performance.now() - stopAt;
  check("cancelled mid-fill", cancelled.cancelled === true && cancelled.resident > 0 && cancelled.resident < 1000, JSON.stringify(cancelled));
  const done = await w.decodeStep("cut", { ...ALL, maxChunk: 64 });
  check("resumed step equals the uninterrupted one bit for bit", done.tokenId === want.tokenId && bits(done.attention) === bits(want.attention));
  check("resumed only the rest", done.stats.positions === 1000 - cancelled.resident, `${done.stats.positions}`);
  timings.cancel = { cancelMs, resident: cancelled.resident };
  log(`cancel landed ${cancelMs.toFixed(1)} ms after the flag, resident ${cancelled.resident}`);
  window.benchResult = { checks, timings };
} catch (error) {
  log(String(error.stack ?? error));
  window.benchResult = { checks, timings: {}, error: String(error) };
}
