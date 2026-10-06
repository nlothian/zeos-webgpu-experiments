// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Run bench/prefill.html (board prefill by chunk size on WebGPU) in headed Chrome, as
// coop-count-web's tests/opt_zeos_webgpu.mjs runs its bench page, and write
// results/prefill_webgpu.json. Serves `demo/` so the page reaches coop-count-web's
// worker, model and node_modules and this directory's inputs. Run prompt_sizes.py first.
//
//   PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
//     node demo/space-invaders-web/bench/prefill_webgpu.mjs [--port 8768] [--query "reps=5"]
//
// Exits 2 if the page reports an error (no WebGPU adapter included).

import { spawn } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const BENCH = fileURLToPath(new URL(".", import.meta.url));
const DEMO = join(BENCH, "..", "..");
const REPO = join(DEMO, "..");
const { values } = parseArgs({
  options: {
    port: { type: "string", default: "8768" },
    query: { type: "string", default: "" },
    out: { type: "string", default: join(BENCH, "results", "prefill_webgpu.json") },
  },
});

if (!existsSync(join(BENCH, ".cache", "webgpu_inputs.json"))) {
  console.error("no .cache/webgpu_inputs.json: run prompt_sizes.py first");
  process.exit(2);
}

async function playwright() {
  const from = process.env.PLAYWRIGHT_MODULE;
  if (from) return createRequire(join(from, "package.json"))("playwright");
  return import("playwright");
}

const { chromium } = await playwright();
const server = spawn(
  "uv",
  ["run", "--frozen", "python", join(DEMO, "coop-count-web", "serve.py"), "--dir", DEMO, "--port", values.port],
  { cwd: REPO, stdio: ["ignore", "pipe", "inherit"], env: { ...process.env, UV_NO_CONFIG: "1" } },
);
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
  page.on("console", (message) => console.log(`[page] ${message.text().split("\n")[0]}`));
  await page.goto(`http://localhost:${values.port}/space-invaders-web/bench/prefill.html?${values.query}`);
  await page.waitForFunction(() => window.benchResult !== undefined, null, { timeout: 30 * 60_000, polling: 1000 });
  const result = await page.evaluate(() => window.benchResult);
  if (result.error) {
    console.error(result.error);
    status = 2;
  } else {
    writeFileSync(values.out, `${JSON.stringify(result, null, 2)}\n`);
    console.log(`wrote ${values.out}`);
  }
} finally {
  await browser.close();
  server.kill();
}
process.exit(status);
