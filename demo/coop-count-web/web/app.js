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
  parked: false,
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
  const runnable = state.info !== null && state.info.runnable && state.info.errors === 0;
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

// -- journal ----------------------------------------------------------------

function classify(line) {
  if (line.includes('"kind":"vector.fired"') || line.includes('"kind":"job.preempted"')) {
    return "interrupt";
  }
  if (line.includes('"kind":"job.resumed"')) return "resume";
  if (line.includes(`"${INTERRUPT_PIPE}"`) || line.includes(`"${NUMBER_PIPE}"`)) return "press";
  return "";
}

function appendLines(lines) {
  const list = $("journal");
  const fragment = document.createDocumentFragment();
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    const kind = classify(line);
    if (kind) item.className = kind;
    fragment.append(item);
  }
  list.append(fragment);
  state.count += lines.length;
  $("count").textContent = `${state.count} events`;
  if ($("follow").checked) list.scrollTop = list.scrollHeight;
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

async function recordRun(run, finished) {
  const hex = await digest(finished.journal);
  const row = document.createElement("tr");
  const cells = [
    run.name,
    run.machine,
    run.schedule ? "events.jsonl" : "none",
    String(finished.presses),
    String(finished.ticks),
    String(state.count),
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
  ready: ({ cases, pyodide, python }) => {
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
  started: ({ name, machine, lines }) => {
    $("journal").replaceChildren();
    state.count = 0;
    state.journal = null;
    appendLines(lines);
    setStatus(`running ${name} on ${machine}`);
  },
  lines: ({ lines, virtualNs, ticks, awaiting, parked }) => {
    appendLines(lines);
    $("clock").textContent = `t = ${virtualNs / 1e6} ms, ${ticks} ticks`;
    state.parked = parked;
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
    state.parked = false;
    state.journal = finished.journal;
    $("prompt").textContent = "";
    setStatus(`${run.name} on ${run.machine}: ${finished.reason} after ${finished.ticks} ticks, ${state.count} events`);
    refreshControls();
    await recordRun(run, finished);
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

$("speed").addEventListener("input", () => {
  const ms = Number($("speed").value);
  $("speed-out").textContent = `${ms} ms/turn`;
  send("speed", { ms });
});

$("run").addEventListener("click", () => {
  state.started = {
    name: $("case").value,
    machine: $("machine").value,
    schedule: $("schedule").checked,
  };
  state.running = true;
  refreshControls();
  send("start", state.started);
});

$("stop").addEventListener("click", () => send("stop"));

function interrupt() {
  if ($("interrupt").disabled) return;
  // As the zeos-count console does: a second interrupt while the handler is already
  // parked would queue another fire to ask for a number nobody wanted to give.
  if (!state.parked) send("press", { pipe: INTERRUPT_PIPE, text: "attention" });
  $("number").focus();
}

function sendNumber() {
  const text = $("number").value.trim();
  if ($("send").disabled || !/^[0-9]+$/.test(text)) return;
  send("press", { pipe: NUMBER_PIPE, text });
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

document.addEventListener("keydown", (event) => {
  const tag = event.target.tagName;
  if (event.key !== " " || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
  event.preventDefault();
  interrupt();
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
