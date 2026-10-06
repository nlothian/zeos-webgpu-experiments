// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Play the built page (web/dist, from build.py) on the real model in headed Chrome and
// print each run's metrics as JSON: both boards x both arms by default, each stopped
// `--seconds` after its clock starts. The model loads once and stays loaded across runs.
//
//   node tests/si_webgpu.mjs [--port 8768] [--seconds 60] [--seed 7]
//     [--runs zeos:default,prompt:default,zeos:ablation,prompt:ablation]
//     [--tune '{"zeos": {"max_chunk": 128}}'] [--out results.json] [--shots DIR]
//
// Exits 2 when the page errors or offers no WebGPU adapter, and 0 with a skip message
// when the build has no model. Needs Playwright and Google Chrome; Playwright is not a
// dependency of this demo: PLAYWRIGHT_MODULE names a directory to import it from (any
// project's node_modules/playwright). WebGPU is shared: run nothing else heavy on the GPU
// at the same time.

import { execFileSync, spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const DEMO = fileURLToPath(new URL("..", import.meta.url));
const REPO = join(DEMO, "..", "..");
const DIST = join(DEMO, "web", "dist");
const { values } = parseArgs({
  options: {
    port: { type: "string", default: "8768" },
    seconds: { type: "string", default: "60" },
    seed: { type: "string", default: "7" },
    runs: { type: "string", default: "zeos:default,prompt:default,zeos:ablation,prompt:ablation" },
    tune: { type: "string", default: "" },
    out: { type: "string" },
    shots: { type: "string" },
  },
});

const manifestPath = join(DIST, "manifest.json");
if (!existsSync(manifestPath)) {
  console.error("no web/dist: run build.py first");
  process.exit(2);
}
if (!JSON.parse(readFileSync(manifestPath, "utf-8")).model) {
  console.log("skip: the build has no model (export it and npm install in coop-count-web, then build.py)");
  process.exit(0);
}

async function playwright() {
  const from = process.env.PLAYWRIGHT_MODULE;
  if (from) return createRequire(join(from, "package.json"))("playwright");
  return import("playwright");
}

/** Other GPU users: processes matching a browser GPU process, Playwright, a dev server or
 * an e2e run, that are not this script or its descendants (its server and Chrome). WebGPU
 * is shared, so a run measured beside one of these is not a quiet-GPU measurement. */
const GPU_USERS = /playwright|chrome.*--type=gpu|vite|test:e2e/i;
function otherGpuProcesses() {
  const rows = execFileSync("ps", ["-axo", "pid=,ppid=,%cpu=,command="], { encoding: "utf-8" })
    .split("\n")
    .map((line) => line.trim().match(/^(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$/))
    .filter(Boolean)
    .map(([, pid, ppid, cpu, command]) => ({ pid: Number(pid), ppid: Number(ppid), cpu: Number(cpu), command }));
  const mine = new Set([process.pid]);
  // And this script's ancestors: the shell that started it names it on its command line.
  const parent = new Map(rows.map((row) => [row.pid, row.ppid]));
  const ancestors = new Set();
  for (let pid = parent.get(process.pid); pid && pid > 1; pid = parent.get(pid)) ancestors.add(pid);
  for (let grew = true; grew; ) {
    grew = false;
    for (const row of rows) {
      if (!mine.has(row.pid) && mine.has(row.ppid)) {
        mine.add(row.pid);
        grew = true;
      }
    }
  }
  return rows
    .filter((row) => !mine.has(row.pid) && !ancestors.has(row.pid) && GPU_USERS.test(row.command))
    .map((row) => ({
      pid: row.pid,
      cpu: row.cpu,
      // An idle test server (VS Code's Playwright extension) starts no browser; a GPU
      // process or anything busy counts.
      active: /--type=gpu/.test(row.command) || row.cpu >= 1,
      command: row.command.slice(0, 160),
    }));
}

const round = (n, d = 2) => (n === null || n === undefined ? null : Math.round(n * 10 ** d) / 10 ** d);

/** The handover table's row for one finished run. */
function metrics(entry, warmWallS, contention) {
  const { run, result: r, verdicts } = entry;
  const x = r.extras ?? {};
  const row = {
    arm: run.arm,
    board: run.board,
    seed: r.seed,
    stopped: entry.stopped,
    contention,
    quiet_gpu: ![...contention.before, ...contention.after].some((p) => p.active),
    lives: r.lives,
    kills: r.kills,
    ticks: r.ticks,
    decisions: r.decisions,
    lag_ticks: { mean: round(r.lag_ticks.mean), p95: r.lag_ticks.p95, max: r.lag_ticks.max },
    preemptions: r.preemptions,
    cancellations: r.cancellations,
    reflexes: r.reflexes,
    overrun_ms: x.overrun_ms ?? null,
    catchup_ticks: r.catchup_ticks,
    warm_s: { page: round(warmWallS), python: x.warm_s ?? null, prewarm_ms: x.prewarm_ms ?? null },
    parse_rate: r.parse_rate,
    verdicts: verdicts.map((v) => `${v.id}:${v.passed}`),
    worker_calls: x.worker_calls,
    gc: x.gc,
  };
  if (run.arm === "zeos") {
    Object.assign(row, {
      cancel_ms: x.cancel_ms,
      roundtrip_s: x.roundtrip_s,
      syscall_adherence: { moves: x.pilot_moves, valid: x.pilot_valid_moves, exits: x.pilot_exits },
      step_totals: x.step_totals,
      biggest_step: x.biggest_step,
      splices: x.splices,
      journal_splices: x.journal_splices,
      context_ids: x.context_ids,
      pilot_context: x.pilot_context,
      kernel_options: x.kernel_options,
      longest_pilot_gap: x.longest_pilot_gap,
      faults: x.faults,
    });
  } else {
    Object.assign(row, {
      replies: x.replies,
      reply_latency_s: x.reply_latency_s,
      unparsed: x.unparsed,
      begin_max_ms: x.begin_max_ms,
      poll_over_max_ms: x.poll_over_max_ms,
    });
  }
  return row;
}

const { chromium } = await playwright();
const server = spawn("uv", ["run", "--frozen", "python", join(DEMO, "serve.py"), "--port", values.port], {
  cwd: REPO,
  env: { ...process.env, UV_NO_CONFIG: "1" },
  stdio: ["ignore", "pipe", "inherit"],
});
await new Promise((resolve, reject) => {
  server.stdout.on("data", (chunk) => String(chunk).includes("serving") && resolve());
  server.once("exit", (code) => reject(new Error(`serve.py exited with ${code}`)));
});

let status = 0;
const rows = [];
const browser = await chromium.launch({
  channel: "chrome",
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"],
});
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") console.error(`[page] ${message.text()}`);
  });
  const query = values.tune ? `?${new URLSearchParams({ tune: values.tune })}` : "";
  await page.goto(`http://localhost:${values.port}/${query}`);
  await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 10 * 60_000 });
  const gpu = await page.evaluate(async () => Boolean(navigator.gpu && (await navigator.gpu.requestAdapter())));
  if (!gpu) throw Object.assign(new Error("no WebGPU adapter"), { code: 2 });

  for (const spec of values.runs.split(",")) {
    const [arm, board] = spec.split(":");
    const before = await page.evaluate(() => window.siRuns.length);
    const contentionBefore = otherGpuProcesses();
    await page.selectOption("#arm", arm);
    await page.selectOption("#board", board);
    await page.selectOption("#machine", "model");
    await page.fill("#seed", values.seed);
    await page.click("#run");
    await page.waitForFunction(() => window.siPhase.phase === "warming" || window.siErrors.length > 0, null, {
      timeout: 20 * 60_000,
    });
    const warmAt = await page.evaluate(() => window.siPhase.at);
    await page.waitForFunction(
      () => window.siPhase.phase !== "warming" || window.siErrors.length > 0,
      null,
      { timeout: 20 * 60_000 },
    );
    const { startedAt, errors } = await page.evaluate(() => ({ startedAt: window.siPhase.at, errors: window.siErrors }));
    if (errors.length) throw Object.assign(new Error(`page error: ${errors.join(" | ")}`), { code: 2 });
    const warmWallS = (startedAt - warmAt) / 1000;
    console.error(`${arm}/${board}: warm ${warmWallS.toFixed(1)} s; playing ${values.seconds} s`);
    // Stop at --seconds unless the game ends first.
    const deadline = Date.now() + Number(values.seconds) * 1000;
    while (Date.now() < deadline) {
      const done = await page.evaluate((n) => window.siRuns.length > n || window.siErrors.length > 0, before);
      if (done) break;
      await page.waitForTimeout(500);
    }
    if (values.shots) await page.screenshot({ path: join(values.shots, `si-${arm}-${board}.png`), fullPage: true });
    if ((await page.evaluate(() => window.siRuns.length)) === before) await page.click("#stop");
    await page.waitForFunction((n) => window.siRuns.length > n || window.siErrors.length > 0, before, {
      timeout: 10 * 60_000,
    });
    const errs = await page.evaluate(() => window.siErrors);
    if (errs.length) throw Object.assign(new Error(`page error: ${errs.join(" | ")}`), { code: 2 });
    const entry = await page.evaluate((n) => window.siRuns[n], before);
    const contention = { before: contentionBefore, after: otherGpuProcesses() };
    const row = metrics(entry, warmWallS, {
      other_gpu_processes: [...contention.before, ...contention.after].filter((p) => p.active),
      ...contention,
    });
    rows.push(row);
    console.log(JSON.stringify(row));
  }
  if (values.shots) await page.screenshot({ path: join(values.shots, "si-final.png"), fullPage: true });
} catch (err) {
  console.error(String(err.stack ?? err));
  status = err.code ?? 2;
} finally {
  if (values.out) writeFileSync(values.out, JSON.stringify({ tune: values.tune || null, rows }, null, 2) + "\n");
  await browser.close();
  server.kill();
}
process.exit(status);
