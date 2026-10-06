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
  describe: null,
  started: null, // {arm, board, seed, machine}
  lags: [],
  playerByTick: new Map(),
  last: null, // the finished message
};

const board = new BoardView($("board-canvas"));
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

function renderChecks() {
  const parts = [];
  const item = (ok, text) => `<span class="${ok ? "good" : "bad"}">${text}: ${ok ? "yes" : "no"}</span>`;
  parts.push(item(state.isolated, "cross-origin isolated"));
  if (state.gpu !== null) parts.push(item(state.gpu.ok, "WebGPU adapter"));
  if (state.booted) parts.push(item(state.model !== null, "model in this build"));
  $("checks").innerHTML = parts.join(" · ");
}

/** Why the model machine cannot run here, or null when it can. WebGPU only: the 4B
 * does not fit the WebAssembly backend, so there is no fallback. */
function modelBlocker() {
  if (!state.isolated) return "the page is not cross-origin isolated (serve it with serve.py)";
  if (state.model === null) return "this build has no model (see the README: npm install, then build.py)";
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

/** Mark `phase` as the current one; `skipped` names phases this run has none of. */
function setPhase(phase, skipped = []) {
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

// -- model --------------------------------------------------------------------

const mb = (n) => `${Math.round(n / 1e6)} MB`;

function setModelStatus(text, fraction = null) {
  $("model-status-row").hidden = false;
  $("model-status").textContent = text;
  const bar = $("model-progress");
  bar.hidden = fraction === null;
  if (fraction !== null) bar.value = fraction;
}

function onModelProgress(progress) {
  if (progress.phase === "session") {
    setModelStatus(`downloaded ${mb(progress.bytes)}; ONNX Runtime is building the WebGPU session`, 1);
    return;
  }
  const { file, loaded, total, files, file_index, bytes, bytes_total } = progress;
  const whole = bytes_total > 0 ? `${mb(bytes)} of ${mb(bytes_total)}` : `${mb(loaded)} of ${mb(total)}`;
  const which = files > 0 ? ` (file ${file_index + 1} of ${files}: ${file})` : ` (${file})`;
  const fraction = bytes_total > 0 ? bytes / bytes_total : total > 0 ? loaded / total : null;
  setModelStatus(`downloading ${whole}${which}`, fraction);
}

let modelThread = null;

/** Start the model thread on the page's thread and hand its buffer and port to the
 * Pyodide worker, once; it stays loaded across runs. */
async function ensureModel() {
  if (modelThread !== null) return;
  const { startBrowserModel } = await import("./model_host.js");
  setModelStatus(`loading ${state.model} on WebGPU`, 0);
  const started = performance.now();
  const model = await startBrowserModel({
    modelUrl: `models/${state.model}/`,
    ortWasmUrl: "vendor/onnxruntime-web/ort.wasm.min.mjs",
    ortWebgpuUrl: "vendor/onnxruntime-web/ort.webgpu.min.mjs",
    tokenizersUrl: "vendor/tokenizers/tokenizers.min.mjs",
    backend: "webgpu",
    onProgress: onModelProgress,
  });
  modelThread = model;
  setModelStatus(`loaded on ${model.backend} in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  send("attachModel", { backend: "webgpu", buffer: model.buffer, port: model.port, name: state.model }, [model.port]);
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

let assets = null;

// server.page() in src/zeos/debugger, done here because the page has no server: each
// marker must appear exactly once, and the data goes in last (as coop-count-web does).
function inline(built, marker, text) {
  const parts = built.split(marker);
  if (parts.length !== 2) throw new Error(`${marker} appears ${parts.length - 1} times in the debugger page, not once`);
  return parts[0] + text + parts[1];
}

async function showDebugger(dataJson) {
  if (assets === null) {
    const names = ["debugger/index.html", "debugger/debugger.css", "debugger/debugger.js"];
    assets = await Promise.all(names.map((name) => fetch(name).then((r) => r.text())));
  }
  const [shell, css, script] = assets;
  let built = inline(shell, "/*CSS*/", css);
  built = inline(built, "/*JS*/", script);
  built = inline(built, "__DATA__", dataJson);
  $("debugger").srcdoc = built;
}

function download(name, text, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

const runName = (run) => `space-invaders-${run.arm}-${run.board}-seed${run.seed ?? "none"}-${run.machine}`;

// -- the worker's messages ------------------------------------------------------

const handlers = {
  status: ({ text }) => setStatus(text),
  log: ({ text }) => log(text),
  error: ({ message, stack }) => {
    setStatus(message, true);
    log(stack || message);
    if (state.running) {
      state.running = false;
      setPhase("finished");
      refreshControls();
    }
  },
  ready: ({ pyodide, python, model, describe }) => {
    state.booted = true;
    state.model = model;
    state.describe = describe;
    renderChecks();
    refreshControls();
    setStatus(`ready — Python ${python} (Pyodide ${pyodide}); choose a player and press run`);
    if (describe.builders.length < 2) {
      log(`page.RUN_BUILDERS has ${JSON.stringify(describe.builders)}; other arms run as a FakeRun (random moves, simulated latency)`);
    }
    showDebugger(describe.payload);
  },
  model: ({ name, backend }) => log(`model thread attached: ${name} on ${backend}`),
  warming: ({ arm }) => {
    setPhase("warming", state.started.machine === "model" ? [] : ["loading"]);
    setStatus(`warming: prefilling the ${ARM_NAMES[arm]}'s system prompt before the clock starts`);
  },
  started: ({ arm, board: name, seed, machine }) => {
    setPhase("playing", machine === "model" ? [] : ["loading"]);
    const spec = state.describe.boards[name];
    setStatus(`playing: ${ARM_NAMES[arm]} on the ${name} board (${spec.w}×${spec.h}, ${spec.tick} s tick), seed ${seed}, ${MACHINE_NAMES[machine]}`);
  },
  frame: (frame) => showFrame(frame),
  decision: (decision) => showDecision(decision),
  finished: (finished) => {
    const run = state.started;
    state.running = false;
    state.last = { ...finished, run };
    setPhase("finished", run.machine === "model" ? [] : ["loading"]);
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
      `${r.lives} lives, ${r.kills} kills, ${r.decisions} moves (${r.reflexes} by the reflex), ` +
      `lag ${fmt(r.lag_ticks.mean)} mean / ${fmt(r.lag_ticks.p95, 0)} p95 ticks, ` +
      `${r.preemptions} preemptions, ${r.cancellations} cancellations, overrun ${Math.round(r.overrun_ms)} ms` +
      (r.parse_rate === null ? "." : `, ${Math.round(r.parse_rate * 100)}% of replies parsed.`);
    renderVerdicts(finished.verdicts, run.arm);
    recordRun(run, r, finished.verdicts);
    setStatus(`${how}: ${r.kills} kills, ${r.lives} lives left after ${r.ticks} ticks`);
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
  if (state.started.machine === "model") {
    setPhase("loading");
    const blocker = modelBlocker();
    try {
      if (blocker !== null) throw new Error(blocker);
      await ensureModel();
    } catch (err) {
      state.running = false;
      setPhase("finished");
      refreshControls();
      setModelStatus(`failed: ${err.message}`);
      setStatus(`the model did not load: ${err.message}`, true);
      return;
    }
  }
  setPhase("warming", state.started.machine === "model" ? [] : ["loading"]);
  send("start", state.started);
});

$("stop").addEventListener("click", () => {
  // Not a message: the run loop never yields to the worker's event loop. It reads this
  // word every iteration, and its sleeps wait on it.
  Atomics.store(control, CONTROL_STOP, 1);
  Atomics.notify(control, CONTROL_STOP);
  setStatus("stopping at the next tick");
});

$("download").addEventListener("click", () => {
  const { run, journal } = state.last;
  download(`${runName(run)}.jsonl`, journal.map((r) => JSON.stringify(r)).join("\n") + "\n", "application/x-ndjson");
});

$("download-result").addEventListener("click", () => {
  const { run, result, verdicts } = state.last;
  download(`${runName(run)}.json`, JSON.stringify({ run, result: { ...result, journal: undefined }, verdicts }, null, 2), "application/json");
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
