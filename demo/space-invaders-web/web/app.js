// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The page: controls, the model's load bar, the board as it plays, the result and the
// debugger. All Python runs in si_worker.js; this file posts requests to it, writes the
// Stop word of the control buffer, and draws what comes back. Message shapes are
// CONTRACTS.md section 3.

import { BoardView, boardSize, lagStats, markerFor } from "./board.js";
import { describeModelProgress, modelCacheControls, modelSource, startBrowserModel } from "./model_host.js";

/** Simulated latency of the stub machine, per decode step and per position filled. */
const STUB_LATENCY = { stepMs: 20, positionMs: 1 };

const $ = (id) => document.getElementById(id);
/** contracts.CONTROL_STOP and CONTROL_BYTES */
const CONTROL_STOP = 0;
const CONTROL_BYTES = 16;
const MOVES_KEPT = 60;

const ARM_NAMES = { zeos: "ZEOS kernel", prompt: "prompt loop" };
const MACHINE_NAMES = { model: "model", stub: "stub" };

const state = {
  booted: false,
  running: false,
  isolated: self.crossOriginIsolated === true,
  gpu: null, // null unknown, else {ok, why}
  model: null, // the export's name, from the manifest
  source: null, // where the model loads from (model_host.js's modelSource), or why it cannot
  ortUrl: null, // onnxruntime-web's ort.webgpu.min.mjs, from the manifest
  refreshCache: () => {}, // redraws the model cache line
  stub: false, // whether the build has the Space Invaders stub (web/stub/)
  describe: null,
  started: null, // {arm, board, seed, machine}
  lags: [],
  playerByTick: new Map(),
  last: null, // the finished message
};

const board = new BoardView($("board-canvas"));
// What tests/si_webgpu.mjs reads: each finished run (less its journal and payload), the
// errors shown, and the phase the page is in with when it was entered.
window.siRuns = [];
window.siErrors = [];
window.siPhase = { phase: null, at: 0 };
const control = state.isolated ? new Int32Array(new SharedArrayBuffer(CONTROL_BYTES)) : null;
const worker = new Worker("si_worker.js", { type: "module" });
const send = (type, body = {}, transfer = []) => worker.postMessage({ type, ...body }, transfer);

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function log(text) {
  const pre = $("log");
  pre.textContent += text + "\n";
  pre.scrollTop = pre.scrollHeight;
}

// -- capabilities -------------------------------------------------------------

async function checkGpu() {
  if (!navigator.gpu) return { ok: false, why: "this browser has no WebGPU (navigator.gpu is missing)" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? { ok: true, why: "" } : { ok: false, why: "WebGPU is present but offered no adapter" };
  } catch (err) {
    return { ok: false, why: `WebGPU adapter request failed: ${err.message}` };
  }
}

const TUNE = new URLSearchParams(location.search).get("tune");

function renderChecks() {
  const parts = [];
  if (TUNE) parts.push(`<span class="tuned">tuned: ${TUNE.replace(/[<>&]/g, "")}</span>`);
  const item = (ok, text) => `<span class="${ok ? "good" : "bad"}">${text}: ${ok ? "yes" : "no"}</span>`;
  parts.push(item(state.isolated, "cross-origin isolated"));
  if (state.gpu !== null) parts.push(item(state.gpu.ok, "WebGPU adapter"));
  if (state.booted) parts.push(item(state.model !== null, "model in this build"));
  $("checks").innerHTML = parts.join(" · ");
}

/** Why the model machine cannot run here, or null when it can. The model runs on
 * WebGPU only. */
function modelBlocker() {
  if (!state.isolated) return "the page is not cross-origin isolated (serve it with serve.py)";
  if (state.model === null) return "this build has no model (see the README: npm install, then build.py)";
  if (state.source instanceof Error) return state.source.message;
  if (state.gpu === null) return "checking WebGPU…";
  if (!state.gpu.ok) return `the model needs WebGPU, and ${state.gpu.why}; choose the stub`;
  return null;
}

// -- controls -----------------------------------------------------------------

function refreshControls() {
  const idle = !state.running;
  const ready = state.booted && state.isolated;
  for (const id of ["arm", "board", "seed", "machine"]) $(id).disabled = !idle || !ready;
  const blocker = modelBlocker();
  const option = $("model-option");
  option.disabled = blocker !== null;
  option.textContent = `${state.model ?? "language model"} on WebGPU${blocker ? " (unavailable)" : ""}`;
  option.title = blocker ?? "";
  if (blocker !== null && $("machine").value === "model") $("machine").value = "stub";
  $("run").disabled = !idle || !ready;
  $("stop").disabled = idle;
  $("download").disabled = !(state.last && state.last.journal.length > 0);
  $("download-result").disabled = state.last === null;
}

const PHASES = ["loading", "warming", "playing", "finished"];

/** The phases a run on `machine` has none of: no thread to load for a stub the build
 * does not have. */
function skippedFor(machine) {
  return machine === "stub" && !state.stub ? ["loading"] : [];
}

/** Mark `phase` as the current one; `skipped` names phases this run has none of. */
function setPhase(phase, skipped = []) {
  state.phase = phase;
  window.siPhase = { phase, at: performance.now() };
  const at = PHASES.indexOf(phase);
  for (const li of $("phases").children) {
    const index = PHASES.indexOf(li.dataset.phase);
    li.className = skipped.includes(li.dataset.phase)
      ? "skipped"
      : index < at
        ? "done"
        : index === at
          ? phase === "finished"
            ? "done"
            : "active"
          : "";
  }
}

/** Mark the phase a run was in as the one it failed in. */
function failPhase() {
  for (const li of $("phases").children) {
    if (li.dataset.phase === state.phase) li.className = "failed";
  }
  state.phase = "failed";
}

// -- model --------------------------------------------------------------------

function setModelStatus(text, fraction = null) {
  $("model-status-row").hidden = false;
  $("model-status").textContent = text;
  const bar = $("model-progress");
  bar.hidden = fraction === null;
  if (fraction !== null) bar.value = fraction;
}

function onModelProgress(progress) {
  const { text, fraction } = describeModelProgress(progress);
  setModelStatus(text, progress.phase === "session" ? 1 : fraction);
}

let modelThread = null;

/** Start the model thread on the page's thread and hand its buffer and port to the
 * Pyodide worker, once; it stays loaded across runs. */
async function ensureModel() {
  if (modelThread !== null) return;
  setModelStatus(`loading ${state.source.name} on WebGPU`, 0);
  const started = performance.now();
  const model = await startBrowserModel({
    model: state.source,
    ortWebgpuUrl: state.ortUrl,
    tokenizersUrl: "vendor/tokenizers/tokenizers.min.mjs",
    onProgress: onModelProgress,
    onPersisted: () => state.refreshCache(),
  });
  modelThread = model;
  setModelStatus(`loaded on ${model.backend} in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  state.refreshCache();
  send("attachModel", { backend: "webgpu", buffer: model.buffer, port: model.port, name: state.source.name }, [model.port]);
}

let stubThread = null;

/** Start the stub's thread on the page's thread and attach it as backend "stub", once:
 * the stub answers through the same channel as the model (stub_thread.js). */
async function ensureStub() {
  if (stubThread !== null) return;
  const { startStubThread } = await import("./stub_thread.js");
  setModelStatus("starting the stub thread");
  stubThread = await startStubThread({ stubUrl: "stub/pilot_stub_worker.js", opts: STUB_LATENCY });
  setModelStatus(`stub ready: ${STUB_LATENCY.stepMs} ms a step + ${STUB_LATENCY.positionMs} ms a position`);
  send("attachModel", { backend: "stub", buffer: stubThread.buffer, port: stubThread.port, name: "pilot stub" }, [
    stubThread.port,
  ]);
}

// -- the board and its live numbers -------------------------------------------------

function resetLive() {
  state.lags = [];
  state.playerByTick = new Map();
  $("moves").replaceChildren();
  $("marker").textContent = " ";
  $("marker").className = "marker";
  for (const id of ["lives", "kills", "ticks", "preemptions", "cancellations", "reflexes", "lag", "catchup"]) {
    $(`s-${id}`).textContent = "–";
  }
  state.catchup = 0;
  state.lastFrame = null;
  $("measures").replaceChildren();
  $("measures-h").hidden = true;
}

function showFrame(frame) {
  state.lastFrame = frame;
  state.playerByTick.set(frame.tick, frame.player);
  state.catchup += frame.catchup;
  let ghost = null;
  for (const d of frame.decisions) {
    if (d.by !== "evade" && d.lag_ticks > 0) ghost = state.playerByTick.get(d.tick) ?? null;
  }
  board.show(frame, ghost);
  const { w, h } = boardSize(frame);
  $("board-label").textContent = `${w}×${h} · tick ${frame.tick}${frame.over ? (frame.won ? " · won" : " · over") : ""}`;
  $("s-lives").textContent = frame.lives;
  $("s-kills").textContent = frame.kills;
  $("s-ticks").textContent = frame.tick;
  $("s-preemptions").textContent = frame.preemptions;
  $("s-cancellations").textContent = frame.cancellations;
  $("s-reflexes").textContent = frame.reflexes;
  $("s-catchup").textContent = state.catchup;
}

function showDecision(decision) {
  if (decision.by !== "evade") state.lags.push(decision.lag_ticks);
  const lag = lagStats(state.lags);
  $("s-lag").textContent = lag.count ? `${lag.mean.toFixed(1)} / ${lag.p95}` : "–";
  const { text, tone } = markerFor(decision);
  $("marker").textContent = text;
  $("marker").className = `marker ${tone}`;
  const item = document.createElement("li");
  item.textContent = `t${String(decision.tick_applied).padStart(4, " ")}  ${text}`;
  item.className = tone;
  const list = $("moves");
  list.prepend(item);
  while (list.children.length > MOVES_KEPT) list.lastChild.remove();
}

// -- the result -----------------------------------------------------------------

function renderVerdicts(verdicts, arm) {
  const list = $("verdicts");
  list.replaceChildren();
  if (verdicts.length === 0) {
    const li = document.createElement("li");
    li.className = "unjudged";
    li.textContent =
      arm === "prompt" ? "The prompt loop runs no kernel, so there is no journal to judge." : "No criteria were judged.";
    list.append(li);
    return;
  }
  for (const v of verdicts) {
    const li = document.createElement("li");
    li.className = v.passed === true ? "pass" : v.passed === false ? "fail" : "unjudged";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = v.passed === true ? "PASS" : v.passed === false ? "FAIL" : "n/a";
    const id = document.createElement("strong");
    id.textContent = v.id;
    const detail = document.createElement("span");
    detail.className = "detail";
    detail.textContent = `${v.kind}${v.detail ? ` — ${v.detail}` : ""}`;
    li.append(tag, id, detail);
    list.append(li);
  }
}

/** "1 kill", "3 kills". */
function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** The run's own measurements (RunResult.extras), beside the verdicts. */
function renderMeasures(result) {
  const x = result.extras ?? {};
  const rows = [];
  const add = (label, value, bad = false) => value !== null && value !== undefined && rows.push([label, value, bad]);
  const ms = (o) => (o ? `${Math.round(o.mean)} mean / ${Math.round(o.p95 ?? o.max)} p95 / ${Math.round(o.max)} max ms` : null);
  if (x.warm_s !== undefined) add("warm-up", `${fmt(x.warm_s)} s${x.prewarm_ms ? ` (mask prewarm ${fmt(x.prewarm_ms / 1000)} s)` : ""}`);
  add("loop overrun", ms(x.overrun_ms));
  add("catch-up ticks", String(result.catchup_ticks));
  if (x.pilot_moves !== undefined) {
    add("pilot moves", `${x.pilot_valid_moves} of ${x.pilot_moves} valid${x.pilot_exits ? `, ${x.pilot_exits} exits` : ""}`, x.pilot_valid_moves !== x.pilot_moves);
    add("pilot request to move", x.roundtrip_s ? `${fmt(x.roundtrip_s.mean)} mean / ${fmt(x.roundtrip_s.max)} max s` : null);
    add("cancel latency", ms(x.cancel_ms));
    if (x.longest_pilot_gap) {
      const g = x.longest_pilot_gap;
      add("longest gap between pilot moves", `${fmt(g.seconds)} s (ticks ${g.from_tick}–${g.to_tick})`);
    }
    add("pager splices", String(x.journal_splices ?? x.splices ?? 0), (x.journal_splices ?? 0) > 0);
    const faults = x.faults ?? [];
    add("kernel faults", faults.length ? faults.map((f) => `${f.fault} (${f.job})`).join(", ") : "none", faults.length > 0);
  }
  if (x.replies !== undefined) {
    add("replies", `${x.replies}, ${x.reply_latency_s ? `${fmt(x.reply_latency_s.mean)} s mean` : ""}`);
    if (x.unparsed?.length) add("unparsed", x.unparsed.slice(0, 3).map((t) => JSON.stringify(t)).join(", "), true);
  }
  const list = $("measures");
  list.replaceChildren();
  for (const [label, value, bad] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (bad) dd.className = "bad";
    list.append(dt, dd);
  }
  $("measures-h").hidden = rows.length === 0;
}

function fmt(n, digits = 1) {
  return n === null || n === undefined ? "–" : Number(n).toFixed(digits);
}

function recordRun(run, result, verdicts) {
  const judged = verdicts.filter((v) => v.passed !== null && v.passed !== undefined);
  const passed = judged.filter((v) => v.passed).length;
  const cells = [
    [ARM_NAMES[run.arm], false],
    [run.board, false],
    [String(result.seed ?? "–"), true],
    [MACHINE_NAMES[run.machine], false],
    [String(result.lives), true],
    [String(result.kills), true],
    [String(result.ticks), true],
    [fmt(result.lag_ticks.mean), true],
    [fmt(result.lag_ticks.p95, 0), true],
    [String(result.preemptions), true],
    [String(result.cancellations), true],
    [String(result.extras?.pilot_moves ?? result.extras?.replies ?? "–"), true],
    [result.extras?.warm_s === undefined ? "–" : `${fmt(result.extras.warm_s)} s`, true],
    [result.parse_rate === null ? "–" : `${Math.round(result.parse_rate * 100)}%`, true],
    [judged.length ? `${passed}/${judged.length}` : verdicts.length ? "not judged" : "–", false],
  ];
  const row = document.createElement("tr");
  for (const [text, num] of cells) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (num) cell.className = "num";
    row.append(cell);
  }
  $("runs").tBodies[0].append(row);
}

// -- debugger -------------------------------------------------------------------

/** The debugger's three files, fetched once; every caller awaits the same promise, so
 * two payloads arriving close together are drawn in the order they were asked for. */
let assets = null;

function fetchText(name) {
  return fetch(name).then((r) => {
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.text();
  });
}

// server.page() in src/zeos/debugger, done here because the page has no server: each
// marker must appear exactly once, and the data goes in last (as coop-count-web does).
function inline(built, marker, text) {
  const parts = built.split(marker);
  if (parts.length !== 2) throw new Error(`${marker} appears ${parts.length - 1} times in the debugger page, not once`);
  return parts[0] + text + parts[1];
}

async function showDebugger(dataJson) {
  try {
    if (assets === null) {
      assets = Promise.all(["debugger/index.html", "debugger/debugger.css", "debugger/debugger.js"].map(fetchText));
    }
    const [shell, css, script] = await assets;
    let built = inline(shell, "/*CSS*/", css);
    built = inline(built, "/*JS*/", script);
    built = inline(built, "__DATA__", dataJson);
    $("debugger").srcdoc = built;
  } catch (err) {
    assets = null;
    log(`the debugger could not be drawn: ${err.message}`);
  }
}

function download(name, text, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  // After the click has been handled: a browser may still be starting the download.
  setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
}

const runName = (run) => `space-invaders-${run.arm}-${run.board}-seed${run.seed ?? "none"}-${run.machine}`;

// -- the worker's messages ------------------------------------------------------

const handlers = {
  status: ({ text }) => setStatus(text),
  log: ({ text }) => log(text),
  error: ({ message, stack }) => {
    window.siErrors.push(message);
    setStatus(message, true);
    log(stack || message);
    if (state.running) {
      state.running = false;
      failPhase();
      refreshControls();
    }
  },
  ready: async ({ pyodide, python, model, stub, describe }) => {
    state.booted = true;
    state.model = model;
    if (model !== null) {
      try {
        const manifest = await (await fetch("manifest.json")).json();
        state.source = modelSource(manifest);
        state.ortUrl = manifest.ort_url;
        $("model-cache-row").hidden = false;
        state.refreshCache = modelCacheControls({
          status: $("model-cache"),
          button: $("clear-model-cache"),
          source: state.source,
        });
      } catch (err) {
        state.source = err;
      }
    }
    state.stub = stub;
    state.describe = describe;
    renderChecks();
    refreshControls();
    setStatus(`ready — Python ${python} (Pyodide ${pyodide}); choose a player and press run`);
    if (!stub) log("this build has no web/stub/pilot_stub_worker.js; the stub machine runs on no worker");
    if (describe.builders.length < 2) {
      log(`page.RUN_BUILDERS has ${JSON.stringify(describe.builders)}; other arms run as a FakeRun (random moves, simulated latency)`);
    }
    showDebugger(describe.payload);
  },
  model: ({ name, backend }) => log(`${backend === "stub" ? "stub" : "model"} thread attached: ${name} on ${backend}`),
  warming: ({ arm }) => {
    setPhase("warming", skippedFor(state.started.machine));
    setStatus(`warming: prefilling the ${ARM_NAMES[arm]}'s system prompt before the clock starts`);
  },
  started: ({ arm, board: name, seed, machine }) => {
    setPhase("playing", skippedFor(machine));
    const spec = state.describe.boards[name];
    setStatus(`playing: ${ARM_NAMES[arm]} on the ${name} board (${spec.w}×${spec.h}, ${spec.tick} s tick), seed ${seed}, ${MACHINE_NAMES[machine]}`);
  },
  frame: (frame) => showFrame(frame),
  decision: (decision) => showDecision(decision),
  finished: (finished) => {
    const run = state.started;
    state.running = false;
    state.last = { ...finished, run };
    window.siRuns.push({ run, result: finished.result, verdicts: finished.verdicts, stopped: finished.stopped });
    setPhase("finished", skippedFor(run.machine));
    const r = finished.result;
    const end = state.lastFrame;
    const how = finished.stopped
      ? "stopped"
      : end?.won
        ? "won"
        : r.lives === 0
          ? "lost"
          : end?.over
            ? "overrun by the invaders"
            : "out of ticks";
    $("summary").textContent =
      `${ARM_NAMES[run.arm]} on ${run.board}, seed ${r.seed ?? "none"}: ${how} after ${r.ticks} ticks — ` +
      `${r.lives === 1 ? "1 life" : `${r.lives} lives`}, ${plural(r.kills, "kill")}, ${plural(r.decisions, "move")} (${r.reflexes} by the reflex), ` +
      `lag ${fmt(r.lag_ticks.mean)} mean / ${fmt(r.lag_ticks.p95, 0)} p95 ticks, ` +
      `${r.preemptions} preemptions, ${r.cancellations} cancellations, overrun ${Math.round(r.overrun_ms)} ms` +
      (r.parse_rate === null ? "." : `, ${Math.round(r.parse_rate * 100)}% of replies parsed.`);
    renderVerdicts(finished.verdicts, run.arm);
    renderMeasures(r);
    recordRun(run, r, finished.verdicts);
    setStatus(`${how}: ${plural(r.kills, "kill")}, ${r.lives === 1 ? "1 life" : `${r.lives} lives`} left after ${r.ticks} ticks`);
    refreshControls();
    showDebugger(finished.payload);
  },
};

worker.onmessage = (event) => {
  const handler = handlers[event.data.type];
  if (handler) handler(event.data);
  else log(`unknown message from the worker: ${event.data.type}`);
};
worker.onerror = (event) => setStatus(`worker error: ${event.message}`, true);

// -- the page's own events --------------------------------------------------------

$("run").addEventListener("click", async () => {
  const seedText = $("seed").value.trim();
  state.started = {
    arm: $("arm").value,
    board: $("board").value,
    seed: seedText === "" ? null : Number.parseInt(seedText, 10),
    machine: $("machine").value,
  };
  state.running = true;
  resetLive();
  refreshControls();
  // Cleared here, not by the worker, so a Stop pressed while a thread loads is kept.
  Atomics.store(control, CONTROL_STOP, 0);
  const { machine } = state.started;
  if (machine === "model" || state.stub) {
    setPhase("loading");
    const blocker = machine === "model" ? modelBlocker() : null;
    try {
      if (blocker !== null) throw new Error(blocker);
      await (machine === "model" ? ensureModel() : ensureStub());
    } catch (err) {
      state.running = false;
      failPhase();
      refreshControls();
      setModelStatus(`failed: ${err.message}`);
      setStatus(`the ${machine} did not load: ${err.message}`, true);
      return;
    }
  }
  if (Atomics.load(control, CONTROL_STOP) !== 0) {
    state.running = false;
    setPhase("finished", [...skippedFor(machine), "warming", "playing"]);
    refreshControls();
    setStatus(`stopped before the run started (the ${machine} is loaded and kept)`);
    return;
  }
  setPhase("warming", skippedFor(machine));
  // `?tune={"zeos": {...}, "prompt": {...}}` overrides the arms' options for this page
  // load (page.configure_json); the tuning runs use it.
  send("start", { ...state.started, tune: new URLSearchParams(location.search).get("tune") ?? "" });
});

$("stop").addEventListener("click", () => {
  // Not a message: the run loop never yields to the worker's event loop. It reads this
  // word every iteration, and its sleeps wait on it.
  Atomics.store(control, CONTROL_STOP, 1);
  Atomics.notify(control, CONTROL_STOP);
  setStatus("stopping");
});

$("download").addEventListener("click", () => {
  const { run, journal } = state.last;
  download(`${runName(run)}.jsonl`, journal.map((r) => JSON.stringify(r)).join("\n") + "\n", "application/x-ndjson");
});

$("download-result").addEventListener("click", () => {
  const { run, result, verdicts } = state.last;
  download(`${runName(run)}.json`, JSON.stringify({ run, result, verdicts }, null, 2), "application/json");
});

// -- boot -----------------------------------------------------------------------

renderChecks();
refreshControls();
if (control === null) {
  setStatus(
    "this page is not cross-origin isolated, so there is no SharedArrayBuffer: neither the model nor Stop can work. " +
      "Serve it with serve.py (or reload once to let coi_serviceworker.js install).",
    true,
  );
} else {
  send("boot", { controlSab: control.buffer });
}
checkGpu().then((gpu) => {
  state.gpu = gpu;
  renderChecks();
  refreshControls();
});
