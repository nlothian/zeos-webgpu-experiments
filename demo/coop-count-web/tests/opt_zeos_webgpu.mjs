// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The model's checks on WebGPU, in headed Chrome. Opt-in: it needs a GPU, the export,
// `npm install`, Playwright and Google Chrome, so no default test suite runs it.
//
// By default it runs, one page each, with the model loaded afresh:
//
// - export/bench/checks.html: the graph's logits against transformers' reference, hidden
//   tokens swapped with nothing else moving (after a prefill, a decode step and
//   single-token runs), and a cancelled step resumed bit for bit;
// - export/bench/grammar.html: JsMachine's grammar mask under Pyodide over the model,
//   after injected text, yields only valid commands and no control token;
// - export/bench/worker.html: the cache against cache-free runs, a replay after a mask
//   change against a fresh prefill, truncate and fork, and the prefill and decode timings.
//
// `--page NAME` runs one page instead (also mask.html, the masked-tool-name timings, and
// chunks.html, the maxChunk timings). Prints each page's checks and timings; exits 1 if a
// check fails, 2 if a page errors, and 0 with a skip message when the export is absent.
//
//   PLAYWRIGHT_MODULE=<a node_modules/playwright> node tests/opt_zeos_webgpu.mjs \
//     [--page checks.html] [--port 8767] [--ort URL] [--steps 32] [--query "configs=2"]
//
// PLAYWRIGHT_MODULE names a directory to import Playwright from (any project's
// node_modules/playwright); Playwright is not a dependency here. ONNX Runtime Web is this
// directory's npm install unless --ort names another build. grammar.html runs on the
// zeos and zeos-coop-count-web wheels, which this builds into export/bench/wheels/ with
// `uv build` first. WebGPU is shared: run nothing else heavy on the GPU at the same time,
// or the timings halve.

import { spawnSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const REPO = join(ROOT, "..", "..");
const PAGES = ["checks.html", "grammar.html", "worker.html"];
const WHEELS = ["zeos", "zeos-coop-count-web"];
const { values } = parseArgs({
  options: {
    port: { type: "string", default: "8767" },
    ort: { type: "string", default: "/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs" },
    steps: { type: "string", default: "32" },
    model: { type: "string", default: "/models/Qwen3.5-4B-ZEOS-OPT/" },
    page: { type: "string" },
    // More query parameters for the page, as `a=1&b=2`.
    query: { type: "string", default: "" },
  },
});
const pages = values.page === undefined ? PAGES : [values.page];

if (!existsSync(join(ROOT, values.model.replace(/^\//, ""), "meta.json"))) {
  console.log(`skip: no export at ${values.model}; run export/opt_zeos_surgery.py`);
  process.exit(0);
}

if (pages.includes("grammar.html")) {
  const out = join(ROOT, "export", "bench", "wheels");
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  for (const name of WHEELS) {
    const built = spawnSync("uv", ["build", "--wheel", "--package", name, "--out-dir", out], {
      cwd: REPO,
      stdio: ["ignore", "ignore", "inherit"],
    });
    if (built.status !== 0) throw new Error(`uv build --package ${name} failed`);
  }
  const wheels = readdirSync(out).filter((n) => n.endsWith(".whl")).sort();
  writeFileSync(join(out, "index.json"), `${JSON.stringify({ wheels })}\n`);
}

async function playwright() {
  const from = process.env.PLAYWRIGHT_MODULE;
  if (from) return createRequire(join(from, "package.json"))("playwright");
  return import("playwright");
}

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".whl": "application/zip",
  ".zip": "application/zip",
};

// This directory, cross-origin isolated (SharedArrayBuffer, which the model thread's
// channel needs), with symbolic links followed.
const server = createServer((request, response) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(request.url, "http://x").pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || !statSync(path).isFile()) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    "Content-Type": TYPES[extname(path)] ?? "application/octet-stream",
    "Content-Length": statSync(path).size,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  });
  createReadStream(path).pipe(response);
});
await new Promise((resolve) => server.listen(Number(values.port), "127.0.0.1", resolve));

const { chromium } = await playwright();
let status = 0;
const browser = await chromium.launch({
  channel: "chrome",
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"],
});
try {
  for (const name of pages) {
    const page = await browser.newPage();
    page.on("console", (message) => console.log(`[${name}] ${message.text()}`));
    const query = new URLSearchParams({ steps: values.steps, model: values.model, ort: values.ort });
    for (const [k, v] of new URLSearchParams(values.query)) query.set(k, v);
    await page.goto(`http://localhost:${values.port}/export/bench/${name}?${query}`);
    await page.waitForFunction(() => window.benchResult !== undefined, null, { timeout: 30 * 60_000, polling: 1000 });
    const result = await page.evaluate(() => window.benchResult);
    const failed = (result.checks ?? []).filter((c) => !c.ok);
    console.log(
      JSON.stringify(
        { page: name, passed: (result.checks ?? []).length - failed.length, failed, skipped: result.skipped, timings: result.timings, error: result.error },
        null,
        2,
      ),
    );
    status = Math.max(status, result.error ? 2 : failed.length ? 1 : 0);
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(status);
