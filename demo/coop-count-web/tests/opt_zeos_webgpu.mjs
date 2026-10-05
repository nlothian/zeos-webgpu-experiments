// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Run export/bench/worker.html (OptZeosWorker on WebGPU) in headed Chrome and report its
// checks and timings. Exits 1 if a check fails, 2 if the page errors, and 0 with a
// skip message when the export is absent.
//
//   node tests/opt_zeos_webgpu.mjs [--port 8767] [--ort URL] [--steps 32]
//
// Needs Playwright and Google Chrome. Playwright is not a dependency of this demo:
// PLAYWRIGHT_MODULE names a directory to import it from (any project's
// node_modules/playwright). WebGPU is shared: run nothing else heavy on the GPU at the
// same time, or the timings halve.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const DEMO = fileURLToPath(new URL("..", import.meta.url));
const REPO = join(DEMO, "..", "..");
const { values } = parseArgs({
  options: {
    port: { type: "string", default: "8767" },
    ort: { type: "string" },
    steps: { type: "string", default: "32" },
    model: { type: "string", default: "/models/Qwen3.5-4B-ZEOS-OPT/" },
  },
});

if (!existsSync(join(DEMO, values.model.replace(/^\//, ""), "meta.json"))) {
  console.log(`skip: no export at ${values.model}; run export/opt_zeos_surgery.py`);
  process.exit(0);
}

async function playwright() {
  const from = process.env.PLAYWRIGHT_MODULE;
  if (from) return createRequire(join(from, "package.json"))("playwright");
  return import("playwright");
}

const { chromium } = await playwright();
const server = spawn("uv", ["run", "--frozen", "python", join(DEMO, "serve.py"), "--dir", DEMO, "--port", values.port], {
  cwd: REPO,
  stdio: ["ignore", "pipe", "inherit"],
});
await new Promise((resolve, reject) => {
  server.stdout.on("data", (chunk) => String(chunk).includes("serving") && resolve());
  server.once("exit", (code) => reject(new Error(`serve.py exited with ${code}`)));
});

let status = 0;
const browser = await chromium.launch({
  channel: "chrome",
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"],
});
try {
  const page = await browser.newPage();
  page.on("console", (message) => console.log(`[page] ${message.text()}`));
  const query = new URLSearchParams({ steps: values.steps, model: values.model });
  if (values.ort) query.set("ort", values.ort);
  await page.goto(`http://localhost:${values.port}/export/bench/worker.html?${query}`);
  await page.waitForFunction(() => window.benchResult !== undefined, null, { timeout: 30 * 60_000, polling: 1000 });
  const result = await page.evaluate(() => window.benchResult);
  const failed = result.checks.filter((c) => !c.ok);
  console.log(JSON.stringify({ timings: result.timings, failed, error: result.error }, null, 2));
  status = result.error ? 2 : failed.length ? 1 : 0;
} finally {
  await browser.close();
  server.kill();
}
process.exit(status);
