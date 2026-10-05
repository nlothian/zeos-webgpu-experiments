// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// What a tool's name chosen masked costs OptZeosWorker on WebGPU (ChatToolMachine's
// `mask_tool_choice`): a chat-agent context of about 7,000 tokens with tool results in it,
// stepped as JsMachine steps it -- a delivery appended and run by the next step, then one
// token per step -- with the results hidden while each tool's name is decoded. The same
// schedule runs unmasked, masked with the worker's defaults (two caches, hidden runs
// carried past), masked with two caches and every hidden position run, and masked with one
// cache (a rewind and replay each way). The tokens are fixed, not the model's choices, so
// every configuration runs the same positions. The result is printed and left in
// `window.benchResult` (`{checks, timings, error?}`) for `tests/opt_zeos_webgpu.mjs
// --page mask.html`.

import { OptZeosWorker } from "../../web/opt_zeos_worker.js";

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = new URL(params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/", location.href);
/** Tokens in the system turn, about what the app's agent prompt and tool schemas take. */
const SYSTEM_TOKENS = Number(params.get("system") ?? 6700);

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

const FILLER =
  "Tools: ListFiles lists the sandbox, ReadLines reads a file, RunSQL runs one DuckDB statement over the loaded tables, " +
  "WriteLines writes a file under /scratchpad, and RunPython runs a script with pandas. Prefer one call at a time, " +
  "say what you found in a sentence or two, and never invent a column that the schema does not list. ";
const RESULT_ROW = "2024-03-14,Ashby Cross,northbound,07:42,on time,4 carriages,212 passengers\n";

/**
 * The schedule: what JsMachine appends and when it steps, as a list of turns. Each tool
 * call is model text with the call in it; its result is a delivery whose content is
 * hidden while a later name is chosen.
 */
function build(w) {
  const [imStart, imEnd] = w.info().controlIds;
  const t = (text) => Array.from(w.tokenize(text));
  const system = [];
  while (system.length < SYSTEM_TOKENS) system.push(...t(FILLER));
  system.length = SYSTEM_TOKENS;
  const prompt = [
    imStart, ...t("system\n"), ...system, imEnd, ...t("\n"),
    imStart, ...t("user\nLoad trains.csv, summarise the busiest service, and save the summary to a note."), imEnd, ...t("\n"),
    imStart, ...t("assistant\n<think>\n\n</think>\n\n"),
  ];
  const calls = [
    { text: "I'll look at the file first.\n", name: "ReadLines", args: "<parameter=path>\n/input/trains.csv\n</parameter>\n", rows: 14 },
    { text: "The file has a header and service rows. Let me count passengers per service.\n", name: "RunSQL", args: "<parameter=sql>\nSELECT service, sum(passengers) AS total FROM trains GROUP BY service ORDER BY total DESC LIMIT 5\n</parameter>\n", rows: 20 },
    { text: "The northbound 07:42 is the busiest. Saving the note now.\n", name: "WriteLines", args: "<parameter=path>\n/scratchpad/busiest.txt\n</parameter>\n<parameter=lines>\n[\"The 07:42 northbound carries the most passengers.\"]\n</parameter>\n", rows: 2 },
  ];
  return { prompt, calls: calls.map((c, i) => ({
    before: t(c.text),
    open: t("<tool_call>"),
    name: t(`\n<function=${c.name}>`),
    rest: t(`\n${c.args}</function>\n</tool_call>`),
    // The delivery: the turn change and framing, the result, and the next turn's opening.
    head: [imEnd, ...t("\n"), imStart, ...t("user\n<tool_response>\n")],
    result: t(RESULT_ROW.repeat(c.rows)),
    tail: [...t("\n</tool_response>"), imEnd, ...t("\n"), imStart, ...t("assistant\n<think>\n\n</think>\n\n")],
    last: i === calls.length - 1,
  })) };
}

async function timed(fn) {
  const t0 = performance.now();
  const value = await fn();
  return { ms: performance.now() - t0, value };
}

/**
 * Run the schedule on a fresh context, masking each name when `masked`; the time of every
 * step by what it was, per call.
 */
let activity = [];

async function play(w, plan, masked, label) {
  const job = `mask-${label}`;
  w.createContext(job);
  const hidden = [];
  const mask = (narrow) => {
    const n = w.length(job);
    const m = new Uint8Array(n).fill(1);
    if (narrow) for (const [a, b] of hidden) m.fill(0, a, b);
    return { allowedBlocks: m, allowedTokens: null };
  };
  const before = { ...w.stats };
  const perCall = [];
  w.append(job, plan.prompt);
  const first = await timed(() => w.decodeStep(job, mask(false)));
  for (const call of plan.calls) {
    const row = { text: 0, open: 0, span: 0, spanSteps: 0, afterSpan: 0, args: 0, prefill: 0, total: 0, runs: [] };
    const step = async (kind, narrow) => {
      const from = activity.length;
      const { ms } = await timed(() => w.decodeStep(job, mask(narrow)));
      row[kind] += ms;
      row.total += ms;
      // The graph runs behind the span and the step after it, for where the time goes.
      if (kind === "span" || kind === "afterSpan") {
        for (const a of activity.slice(from)) row.runs.push(`${kind} ${a.phase} track ${a.track} ${a.count}@${a.start} ${Math.round(a.ms)}ms`);
      }
    };
    // Each decoded token is appended before the next step, as JsMachine does.
    for (const id of call.before) {
      w.append(job, [id]);
      await step("text", false);
    }
    for (const id of call.open) {
      w.append(job, [id]);
      await step("open", false);
    }
    // From the step after <tool_call> to the one that closes the name: masked.
    for (const id of call.name) {
      w.append(job, [id]);
      await step("span", masked && hidden.length > 0);
      row.spanSteps += 1;
    }
    let firstAfter = true;
    for (const id of call.rest) {
      w.append(job, [id]);
      await step(firstAfter ? "afterSpan" : "args", false);
      firstAfter = false;
    }
    if (!call.last) {
      // The delivery, run by the next step with the first token of the next turn.
      w.append(job, call.head);
      const at = w.length(job);
      w.append(job, call.result);
      hidden.push([at, w.length(job)]);
      w.append(job, call.tail);
      row.resultTokens = call.head.length + call.result.length + call.tail.length;
      const next = plan.calls[plan.calls.indexOf(call) + 1];
      w.append(job, [next.before[0]]);
      const { ms } = await timed(() => w.decodeStep(job, mask(false)));
      row.prefill = ms;
      row.total += ms;
      next.before = next.before.slice(1);
    }
    perCall.push(row);
  }
  const stats = Object.fromEntries(Object.entries(w.stats).map(([k, v]) => [k, v - before[k]]));
  const length = w.length(job);
  w.destroyContext(job);
  return { label, firstMs: first.ms, promptTokens: plan.prompt.length, length, perCall, stats };
}

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
  const onActivity = (a) => {
    if (typeof a.ms === "number") activity.push(a);
  };
  const w = await OptZeosWorker.load({ ort, Tokenizer, read, source, backend: "webgpu", onActivity });
  const timings = { ort: ort.env.versions?.web, adapter: adapter.info?.architecture };

  // Shader compilation and first-use allocation.
  w.createContext("warm");
  w.append("warm", Array.from(w.tokenize(FILLER.repeat(8))));
  await w.decodeStep("warm", { allowedBlocks: null, allowedTokens: null });
  w.append("warm", [w.tokenize(" The")[0]]);
  await w.decodeStep("warm", { allowedBlocks: null, allowedTokens: null });
  w.destroyContext("warm");

  const configs = [
    ["unmasked", false, 2, true],
    ["masked", true, 2, true],
    ["masked, hidden runs run", true, 2, false],
    ["masked, one cache", true, 1, false],
  ].slice(0, Number(params.get("configs") ?? 4));
  const runs = [];
  for (const [label, masked, tracks, skip] of configs) {
    w.maxTracks = tracks;
    w.skipHidden = skip;
    const result = await play(w, build(w), masked, label);
    runs.push(result);
    log(`${label}: ${JSON.stringify(result)}`);
  }
  w.maxTracks = 2;
  w.skipHidden = true;
  const [base, ...others] = runs;
  timings.length = base.length;
  timings.promptTokens = base.promptTokens;
  timings.prefillMs = base.firstMs;
  // The calls after a result: the first has nothing to hide.
  const names = base.perCall.map((_, i) => `call ${i + 1}`);
  timings.baseline = base.perCall;
  timings.overhead = {};
  for (const run of others) {
    timings.overhead[run.label] = run.perCall.map((row, i) => {
      const ref = base.perCall[i];
      const extra = row.span + row.afterSpan - (ref.span + ref.afterSpan);
      return {
        call: names[i],
        extraMs: Math.round(extra),
        ofCall: Number((extra / ref.total).toFixed(3)),
        spanMs: Math.round(row.span),
        baselineSpanMs: Math.round(ref.span),
        afterSpanMs: Math.round(row.afterSpan),
        baselineCallMs: Math.round(ref.total),
      };
    });
    timings.overhead[run.label].stats = run.stats;
  }
  log(JSON.stringify(timings, null, 2));
  check("masking adds no cost to a call with nothing to hide",
    timings.overhead.masked[0].extraMs < 0.2 * base.perCall[0].total, JSON.stringify(timings.overhead.masked[0]));
  check("the second cache is caught up, not replayed, after the first narrowing",
    others[0].stats.tracks === 1 && others[0].stats.switches >= 3, JSON.stringify(others[0].stats));
  window.benchResult = { checks, timings };
  await w.release();
} catch (error) {
  window.benchResult = { checks, error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
