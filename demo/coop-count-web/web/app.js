// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The page: controls, the console, the journal as it streams, and the debugger. All
// Python runs in pyodide_worker.js; this file only posts requests to it and draws what
// comes back.

"use strict";

const $ = (id) => document.getElementById(id);
const INTERRUPT_PIPE = "keys.interrupt";
const NUMBER_PIPE = "keys.number";

const state = {
  info: null, // the selected case, as page.describe() reports it
  running: false,
  collecting: false, // between an interrupt and the number that answers it, as the CLI's console
  withheld: 0,
  started: null, // {name, machine, schedule} of the current run
  journal: null, // the finished run's bytes
  count: 0,
  digests: new Map(), // digest -> rows showing it
};

const worker = new Worker("pyodide_worker.js", { type: "module" });
const send = (type, body = {}) => worker.postMessage({ type, ...body });

/** What the page calls each machine, case and finish reason; the worker uses the ids. */
const MACHINE_NAMES = {
  transformers: "the language model",
  scripted: "recorded answers",
  stub: "recorded answers (JS)",
};
const CASE_NAMES = {
  "coop-count-scripted": "counters + interrupt (recorded answers available)",
  "coop-count-pipe": "counters taking turns over pipes (model only)",
  "coop-count-vector": "counters starting each other (model only)",
};
const REASONS = { quiescent: "finished — every job is idle", stopped: "stopped" };
const machineName = (id) => MACHINE_NAMES[id] || id;

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function log(text) {
  const pre = $("log");
  pre.textContent += text + "\n";
  pre.scrollTop = pre.scrollHeight;
}

// -- controls ---------------------------------------------------------------

function consoleWired() {
  const devices = state.info ? state.info.devices : [];
  return devices.includes(INTERRUPT_PIPE) && devices.includes(NUMBER_PIPE);
}

function refreshControls() {
  const idle = !state.running;
  const model = $("machine").value === "transformers";
  // A case with no tapes needs a model; the model machine runs any case that lints.
  const runnable =
    state.info !== null && (state.info.runnable || model) && state.info.errors === 0;
  // The lint line's note on a case with no tapes depends on the machine chosen.
  if (state.info !== null) {
    $("lint-tapes").textContent = state.info.runnable
      ? ""
      : model
        ? " — no recorded answers: the model decides everything"
        : " — no recorded answers, so only the Qwen model can run this scenario; choose it under “answered by”";
  }
  $("case").disabled = !idle;
  $("machine").disabled = !idle;
  $("schedule").disabled = !idle || !(state.info && state.info.schedule);
  $("run").disabled = !idle || !runnable;
  $("stop").disabled = idle;
  const live = state.running && consoleWired();
  $("interrupt").disabled = !live;
  $("number").disabled = !live;
  $("send").disabled = !live;
  $("download").disabled = state.journal === null;
}

function renderLint(info) {
  const box = $("lint");
  box.replaceChildren();
  const head = document.createElement("div");
  const warnings = info.lint.length - info.errors;
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  head.textContent =
    `scenario check: ${plural(info.descriptors, "job")}, ` +
    `${plural(info.vectors, "interrupt handler")} — ` +
    (info.errors || warnings
      ? `${plural(info.errors, "error")}, ${plural(warnings, "warning")}`
      : "no problems");
  const tapes = document.createElement("span");
  tapes.id = "lint-tapes"; // filled by refreshControls, which knows the machine
  head.append(tapes);
  box.append(head);
  for (const line of info.lint) {
    const div = document.createElement("div");
    div.textContent = line;
    if (/\berror\b/i.test(line)) div.className = "error";
    box.append(div);
  }
}

// -- output: the user view and the journal ---------------------------------------

function classify(line) {
  if (line.includes('"kind":"vector.fired"') || line.includes('"kind":"job.preempted"')) {
    return "interrupt";
  }
  if (line.includes('"kind":"job.resumed"')) return "resume";
  if (line.includes(`"${INTERRUPT_PIPE}"`) || line.includes(`"${NUMBER_PIPE}"`)) return "press";
  return "";
}

/** Append lines to a list, each with the class `kindOf` gives it, keeping to the bottom
 * while *follow* is ticked. A hidden list has no height to scroll, so showing a tab
 * scrolls it again. */
function appendTo(list, lines, kindOf) {
  const fragment = document.createDocumentFragment();
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    const kind = kindOf(line);
    if (kind) item.className = kind;
    fragment.append(item);
  }
  list.append(fragment);
  if ($("follow").checked) list.scrollTop = list.scrollHeight;
}

/** Empty both views and what is counted from them, leaving the selected tab as it is.
 * No `lines` from an earlier run can land after this: the worker posts a run's
 * `finished` after its last `lines`, and *run* is enabled again only by `finished`. */
function clearOutput() {
  $("journal").replaceChildren();
  $("transcript").replaceChildren();
  state.count = 0;
  state.journal = null;
  $("count").textContent = "0 events";
  $("clock").textContent = "";
  $("prompt").textContent = "";
}

function appendLines(lines) {
  appendTo($("journal"), lines, classify);
  state.count += lines.length;
  $("count").textContent = `${state.count} events`;
}

// The user view: the lines `zeos-count run` prints, which the worker sends beside the
// journal lines (zeos_coop_count_web.transcript).
function classifyTranscript(line) {
  if (line.includes("<RESUME>")) return "resume";
  if (line.includes(" \u25c0\u2500\u2500 ")) return "arrive";
  if (line.includes(" ... waiting on ")) return "wait";
  return "";
}

function appendTranscript(lines) {
  appendTo($("transcript"), lines, classifyTranscript);
}

const TABS = ["tab-transcript", "tab-journal"];

function selectTab(id, focus = false) {
  for (const tabId of TABS) {
    const tab = $(tabId);
    const selected = tabId === id;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) $(id).focus();
  followVisible();
}

function followVisible() {
  if (!$("follow").checked) return;
  for (const list of [$("transcript"), $("journal")]) list.scrollTop = list.scrollHeight;
}

for (const tabId of TABS) {
  $(tabId).addEventListener("click", () => selectTab(tabId));
  $(tabId).addEventListener("keydown", (event) => {
    const at = TABS.indexOf(tabId);
    const next = {
      ArrowRight: TABS[(at + 1) % TABS.length],
      ArrowLeft: TABS[(at - 1 + TABS.length) % TABS.length],
      Home: TABS[0],
      End: TABS[TABS.length - 1],
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectTab(next, true);
  });
}
$("follow").addEventListener("change", followVisible);

// -- model status -----------------------------------------------------------

const mb = (n) => `${Math.round(n / 1e6)} MB`;

/** Show what the model thread is doing, with a bar when there is a fraction to show.
 * `busy` marks a step in flight, which is when the kernel is waiting on the model. */
function setModelStatus(text, { fraction = null, busy = false } = {}) {
  $("model-status-row").hidden = false;
  $("model-status-row").classList.toggle("busy", busy);
  $("model-status").textContent = text;
  const bar = $("model-progress");
  bar.hidden = fraction === null;
  if (fraction !== null) bar.value = fraction;
}

function onModelProgress(progress) {
  if (progress.phase === "session") {
    setModelStatus(`downloaded ${mb(progress.bytes)}; ONNX Runtime is building the session`, {
      busy: true,
    });
    return;
  }
  const { file, loaded, total, files, file_index, bytes, bytes_total } = progress;
  const whole = bytes_total > 0 ? `${mb(bytes)} of ${mb(bytes_total)}` : `${mb(loaded)} of ${mb(total)}`;
  const which = files > 0 ? ` (file ${file_index + 1} of ${files}: ${file})` : ` (${file})`;
  const fraction = bytes_total > 0 ? bytes / bytes_total : total > 0 ? loaded / total : null;
  setModelStatus(`downloading ${whole}${which}`, { fraction });
}

const modelActivity = { steps: 0, decodeMs: 0, prefillMsPerToken: null };

/** A context id is `<job>:<descriptor>` (see the README's seam section); the descriptor
 * is the name the transcript uses. */
function jobLabel(contextId) {
  if (contextId === null) return "";
  const match = /^(\d+):(.*)$/.exec(contextId);
  return match ? ` for ${match[2]} (job ${match[1]})` : ` for ${contextId}`;
}

function onModelActivity(activity) {
  const job = jobLabel(activity.job);
  if (activity.ms === undefined) {
    // A run of the graph has started; the kernel is blocked until it returns.
    if (activity.phase === "decode") {
      setModelStatus(`decoding step ${modelActivity.steps + 1}${job}, ${activity.length} tokens in context`, {
        busy: true,
      });
      return;
    }
    const end = activity.start + activity.count;
    const left = modelActivity.prefillMsPerToken === null
      ? ""
      : `, about ${Math.max(1, Math.round(((activity.length - end) * modelActivity.prefillMsPerToken) / 1000))} s left`;
    setModelStatus(`prefilling positions ${activity.start}–${end} of ${activity.length}${job}${left}`, {
      busy: true,
      fraction: activity.start / activity.length,
    });
    return;
  }
  if (activity.phase === "decode") {
    modelActivity.steps += 1;
    modelActivity.decodeMs += activity.ms;
    const mean = modelActivity.decodeMs / modelActivity.steps;
    setModelStatus(`idle after ${modelActivity.steps} decode steps, ${Math.round(mean)} ms each`);
  } else {
    // A running mean of the prefill cost per token, for the estimate above.
    const perToken = activity.ms / activity.count;
    modelActivity.prefillMsPerToken =
      modelActivity.prefillMsPerToken === null ? perToken : 0.8 * modelActivity.prefillMsPerToken + 0.2 * perToken;
    setModelStatus(`idle; prefilled to position ${activity.start + activity.count}${job}`);
  }
}

// -- debugger ---------------------------------------------------------------

let assets = null;

// server.page() in src/zeos/debugger, done here because the page has no server: each
// marker must appear exactly once, and the data goes in last.
function inline(built, marker, text) {
  const parts = built.split(marker);
  if (parts.length !== 2) {
    throw new Error(`${marker} appears ${parts.length - 1} times in the debugger page, not once`);
  }
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

// -- finished runs ----------------------------------------------------------

async function digest(bytes) {
  if (!globalThis.crypto || !crypto.subtle) return "(needs a secure context)";
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(hash.slice(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function recordRun(run, finished, events) {
  const hex = await digest(finished.journal);
  const row = document.createElement("tr");
  const cells = [
    run.name,
    machineName(run.machine),
    run.schedule ? "on" : "off",
    String(finished.presses),
    String(finished.ticks),
    String(events),
    hex,
  ];
  for (const text of cells) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.append(cell);
  }
  const cell = row.lastChild;
  cell.className = "digest";
  const same = state.digests.get(hex) || [];
  same.push(cell);
  state.digests.set(hex, same);
  if (same.length > 1) for (const c of same) c.classList.add("same");
  $("runs").tBodies[0].append(row);
}

// -- the worker's messages ------------------------------------------------------

const handlers = {
  status: ({ text }) => setStatus(text),
  log: ({ text }) => log(text),
  error: ({ text }) => {
    setStatus(text.split("\n").filter(Boolean).pop() || text, true);
    log(text);
  },
  ready: ({ cases, pyodide, python, model, isolated }) => {
    const option = $("model-option");
    // The model runs on WebGPU only.
    const gpu = Boolean(navigator.gpu);
    option.disabled = !(model && isolated && gpu);
    if (model) {
      // The export's directory name, less the `-zeos-<quant>` (or `-ZEOS-OPT`) suffix.
      MACHINE_NAMES.transformers = model.replace(/-zeos-[^-]+$/i, "");
      option.textContent = `${MACHINE_NAMES.transformers} language model, in your browser`;
    }
    if (!model) option.textContent += " (not in this build)";
    else if (!isolated) option.textContent += " (needs a cross-origin isolated page)";
    else if (!gpu) option.textContent += " (needs WebGPU, which this browser does not have)";
    const select = $("case");
    for (const name of cases) {
      select.add(new Option(CASE_NAMES[name] || name, name));
    }
    // The model is the default when this page can run it.
    $("machine").value = option.disabled ? "scripted" : "transformers";
    if (cases.includes("coop-count-scripted")) select.value = "coop-count-scripted";
    setStatus(`ready — Python ${python} (Pyodide ${pyodide}) loaded; press run`);
    send("describe", { name: select.value });
  },
  described: ({ info }) => {
    state.info = info;
    $("schedule").checked = info.schedule;
    renderLint(info);
    refreshControls();
    showDebugger(JSON.stringify(info.payload));
  },
  model: ({ name, backend }) => {
    state.model = `${name} on ${backend}`;
    $("model-backend").textContent = `model: ${state.model}`;
    // The thread is attached while a run is starting, so this must not overwrite what
    // the run has already said the model is doing.
    if (!state.running) setModelStatus("ready");
    log(`model thread ready: ${state.model}`);
  },
  started: ({ name, machine, lines, transcript }) => {
    state.collecting = false;
    state.withheld = 0;
    clearOutput();
    appendLines(lines);
    appendTranscript(transcript);
    setStatus(`running “${CASE_NAMES[name] || name}”, answered by ${machineName(machine)}`);
  },
  lines: ({ lines, transcript, virtualNs, ticks, awaiting, parked, withheld }) => {
    if (withheld > state.withheld) {
      log("interrupt withheld: reset-count was already parked on keys.number");
      state.withheld = withheld;
    }
    appendLines(lines);
    appendTranscript(transcript);
    $("clock").textContent = `virtual time ${virtualNs / 1e6} ms (turn ${ticks})`;
    $("prompt").textContent = parked
      ? "reset-count is waiting for the new count: type a number and press Enter"
      : awaiting
        ? "a job is waiting for keyboard input"
        : "";
  },
  pressed: ({ pipe, text, atNs }) =>
    log(`queued ${pipe} <- ${JSON.stringify(text)} for the turn at t = ${atNs / 1e6} ms`),
  finished: async (finished) => {
    const run = state.started;
    state.running = false;
    state.journal = finished.journal;
    $("prompt").textContent = "";
    const reason = REASONS[finished.reason] || finished.reason;
    setStatus(`${machineName(run.machine)}: ${reason} after ${finished.ticks} turns, ${state.count} events`);
    refreshControls();
    // The count is read now: *run* is enabled again above, and pressing it while the
    // digest is computed empties the views.
    await recordRun(run, finished, state.count);
    showDebugger(finished.payload);
  },
};

worker.onmessage = (event) => handlers[event.data.type](event.data);
worker.onerror = (event) => setStatus(`worker error: ${event.message}`, true);

// -- the page's own events ----------------------------------------------------

$("case").addEventListener("change", () => {
  state.info = null;
  refreshControls();
  send("describe", { name: $("case").value });
});

$("machine").addEventListener("change", refreshControls);

$("speed").addEventListener("input", () => {
  const ms = Number($("speed").value);
  $("speed-out").textContent = `${ms} ms/turn`;
  send("speed", { ms });
});

/** The model thread once handed to the Pyodide worker; it stays loaded across runs. */
let modelThread = null;

/** Start the model thread here, on the page's thread, and hand its buffer and port to the
 * Pyodide worker, which calls it synchronously from then on. */
async function ensureModel() {
  if (modelThread !== null) return;
  const { startBrowserModel } = await import("./model_host.js");
  const manifest = await (await fetch("manifest.json")).json();
  setStatus(`loading ${manifest.model} on the GPU`);
  const model = await startBrowserModel({
    modelUrl: `models/${manifest.model}/`,
    ortWebgpuUrl: "vendor/onnxruntime-web/ort.webgpu.min.mjs",
    tokenizersUrl: "vendor/tokenizers/tokenizers.min.mjs",
    onProgress: onModelProgress,
    onActivity: onModelActivity,
  });
  modelThread = model;
  worker.postMessage(
    { type: "attachModel", backend: model.backend, buffer: model.buffer, port: model.port, name: manifest.model },
    [model.port],
  );
}

$("run").addEventListener("click", async () => {
  state.started = {
    name: $("case").value,
    machine: $("machine").value,
    schedule: $("schedule").checked,
  };
  state.running = true;
  clearOutput();
  refreshControls();
  if (state.started.machine === "transformers") {
    modelActivity.steps = 0;
    modelActivity.decodeMs = 0;
    modelActivity.prefillMsPerToken = null;
    try {
      await ensureModel();
    } catch (err) {
      state.running = false;
      refreshControls();
      setModelStatus(`failed: ${err.message}`);
      setStatus(`the model did not load: ${err.message}`, true);
      return;
    }
    setModelStatus("loaded; the kernel is booting the jobs and the first prompt is being prefilled", { busy: true });
  }
  send("start", state.started);
});

$("stop").addEventListener("click", () => {
  send("stop");
  // The worker reads the message between turns, and a turn on the model machine lasts
  // as long as the model takes: the whole of a prompt's prefill, on a job's first turn.
  if (state.running) setStatus("stopping after the current turn");
});

function interrupt() {
  if ($("interrupt").disabled) return;
  // As the zeos-count console does: once an interrupt is sent, further presses only ask
  // for the number until it is given. The run also withholds the interrupt if, when it
  // is delivered, a handler is already parked on keys.number.
  if (!state.collecting) {
    send("press", { pipe: INTERRUPT_PIPE, text: "attention", unlessWaitingOn: NUMBER_PIPE });
    state.collecting = true;
  }
  $("number").focus();
}

function sendNumber() {
  const text = $("number").value.trim();
  if ($("send").disabled || !/^[0-9]+$/.test(text)) return;
  send("press", { pipe: NUMBER_PIPE, text });
  state.collecting = false;
  $("number").value = "";
  $("number").blur();
}

$("interrupt").addEventListener("click", interrupt);
$("send").addEventListener("click", sendNumber);
$("number").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    sendNumber();
  }
});

$("download").addEventListener("click", () => {
  const run = state.started;
  const blob = new Blob([state.journal], { type: "application/x-ndjson" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${run.name}-${run.machine}.jsonl`;
  link.click();
  URL.revokeObjectURL(link.href);
});
