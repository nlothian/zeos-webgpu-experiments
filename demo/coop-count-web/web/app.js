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
  $("case").disabled = !idle;
  $("machine").disabled = !idle;
  $("backend").disabled = !idle || !model;
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
  head.textContent =
    `${info.descriptors} descriptors, ${info.vectors} vectors: ${info.errors} error(s), ` +
    `${info.lint.length - info.errors} warning(s)` +
    (info.runnable ? "" : " — no tapes, so this case needs a model and cannot run here");
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
    run.machine,
    run.schedule ? "events.jsonl" : "none",
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
    option.disabled = !(model && isolated);
    if (!model) option.textContent += " (not in this build)";
    else if (!isolated) option.textContent += " (needs a cross-origin isolated page)";
    const select = $("case");
    for (const name of cases) select.add(new Option(name, name));
    if (cases.includes("coop-count-scripted")) select.value = "coop-count-scripted";
    setStatus(`Pyodide ${pyodide}, Python ${python}: ready`);
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
    log(`model thread ready: ${state.model}`);
  },
  started: ({ name, machine, lines, transcript }) => {
    state.collecting = false;
    state.withheld = 0;
    clearOutput();
    appendLines(lines);
    appendTranscript(transcript);
    setStatus(`running ${name} on ${machine}`);
  },
  lines: ({ lines, transcript, virtualNs, ticks, awaiting, parked, withheld }) => {
    if (withheld > state.withheld) {
      log("interrupt withheld: reset-count was already parked on keys.number");
      state.withheld = withheld;
    }
    appendLines(lines);
    appendTranscript(transcript);
    $("clock").textContent = `t = ${virtualNs / 1e6} ms, ${ticks} ticks`;
    $("prompt").textContent = parked
      ? "reset-count is parked on keys.number: type a number"
      : awaiting
        ? "a job waits on the console"
        : "";
  },
  pressed: ({ pipe, text, atNs }) =>
    log(`queued ${pipe} <- ${JSON.stringify(text)} for the turn at t = ${atNs / 1e6} ms`),
  finished: async (finished) => {
    const run = state.started;
    state.running = false;
    state.journal = finished.journal;
    $("prompt").textContent = "";
    setStatus(`${run.name} on ${run.machine}: ${finished.reason} after ${finished.ticks} ticks, ${state.count} events`);
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

/** Model threads already handed to the Pyodide worker, by backend. */
const modelThreads = new Map();

/** Start the model thread for a backend here, on the page's thread, and hand its buffer
 * and port to the Pyodide worker, which calls it synchronously from then on. */
async function ensureModel(backend) {
  if (modelThreads.has(backend)) return;
  const { startBrowserModel } = await import("./model_host.js");
  const manifest = await (await fetch("manifest.json")).json();
  setStatus(`loading ${manifest.model} on ${backend}`);
  const model = await startBrowserModel({
    modelUrl: `models/${manifest.model}/`,
    ortWasmUrl: "vendor/onnxruntime-web/ort.wasm.min.mjs",
    ortWebgpuUrl: "vendor/onnxruntime-web/ort.webgpu.min.mjs",
    tokenizersUrl: "vendor/tokenizers/tokenizers.min.mjs",
    backend,
    onProgress: ({ file, loaded, total }) =>
      setStatus(`downloading ${file}: ${Math.round(loaded / 1e6)} of ${Math.round(total / 1e6)} MB`),
  });
  modelThreads.set(backend, model);
  worker.postMessage(
    { type: "attachModel", backend, buffer: model.buffer, port: model.port, name: manifest.model },
    [model.port],
  );
}

$("run").addEventListener("click", async () => {
  state.started = {
    name: $("case").value,
    machine: $("machine").value,
    schedule: $("schedule").checked,
    backend: $("backend").value,
  };
  state.running = true;
  clearOutput();
  refreshControls();
  if (state.started.machine === "transformers") {
    try {
      await ensureModel(state.started.backend);
    } catch (err) {
      state.running = false;
      refreshControls();
      setStatus(`the model did not load: ${err.message}`, true);
      return;
    }
  }
  send("start", state.started);
});

$("stop").addEventListener("click", () => send("stop"));

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
