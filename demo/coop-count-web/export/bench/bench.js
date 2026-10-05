// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Prefill and decode timings of the OPT+ZEOS decoder (export/opt_zeos_surgery.py) on
// ONNX Runtime Web's WebGPU backend, with the cache kept on the GPU between runs as a
// worker keeps it. Optionally the same timings of the unmodified -OPT decoder. The
// result is printed and left in `window.benchResult` for a driver to read.

const ORT_VERSION = "1.31.0-dev.20260914-8d85527a0";
const params = new URLSearchParams(location.search);
const ortUrl =
  params.get("ort") ??
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const modelUrl = new URL(params.get("model") ?? "/models/Qwen3.5-4B-ZEOS-OPT/", location.href);
const optUrl = params.get("opt") ? new URL(params.get("opt"), location.href) : null;
// `decoder` picks another graph file in the export's onnx/ directory (same data files).
// `only=opt` skips the OPT+ZEOS decoder and times the -OPT one alone.
const DECODE_STEPS = Number(params.get("steps") ?? 32);

const logEl = document.getElementById("log");
function log(line) {
  logEl.textContent += `${line}\n`;
  console.log(line);
}

const ort = await import(ortUrl);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = ortUrl.replace(/[^/]*$/, "");

function ids(n, offset = 0) {
  const out = new BigInt64Array(n);
  for (let i = 0; i < n; i++) out[i] = BigInt((((i + offset) * 7919) % 20000) + 100);
  return out;
}

class Runner {
  static async create(base, decoder, shards, zeos) {
    const embed = await ort.InferenceSession.create(
      new URL("onnx/embed_tokens_q4f16.onnx", base).href,
      {
        executionProviders: ["webgpu"],
        externalData: [
          {
            path: "embed_tokens_q4f16.onnx_data",
            data: new URL("onnx/embed_tokens_q4f16.onnx_data", base).href,
          },
        ],
        preferredOutputLocation: { inputs_embeds: "gpu-buffer" },
      },
    );
    // The present outputs' names, read from a session without them pinned first would
    // cost a second load; they follow the -OPT naming, so build them.
    const probe = { full: [], linear: [] };
    const meta = await (await fetch(new URL("config.json", base))).json();
    const types = (meta.text_config ?? meta).layer_types;
    types.forEach((t, i) => (t === "full_attention" ? probe.full : probe.linear).push(i));
    const present = [];
    for (const l of probe.full) present.push(`present.${l}.key`, `present.${l}.value`);
    for (const i of probe.linear) present.push(`present_conv.${i}`, `present_recurrent.${i}`);
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(new URL(`onnx/${decoder}`, base).href, {
      executionProviders: ["webgpu"],
      externalData: shards.map((s) => ({ path: s, data: new URL(`onnx/${s}`, base).href })),
      preferredOutputLocation: Object.fromEntries(present.map((n) => [n, "gpu-buffer"])),
    });
    log(`loaded ${decoder} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    return new Runner(embed, session, probe, zeos);
  }

  constructor(embed, session, probe, zeos) {
    this.embed = embed;
    this.session = session;
    this.probe = probe;
    this.zeos = zeos;
  }

  empty() {
    const cache = {};
    for (const l of this.probe.full) {
      for (const kv of ["key", "value"]) {
        cache[`past_key_values.${l}.${kv}`] = new ort.Tensor("float16", new Uint16Array(0), [1, 4, 0, 256]);
      }
    }
    for (const i of this.probe.linear) {
      cache[`past_conv.${i}`] = new ort.Tensor("float16", new Uint16Array(8192 * 3), [1, 8192, 3]);
      cache[`past_recurrent.${i}`] = new ort.Tensor("float16", new Uint16Array(32 * 128 * 128), [1, 32, 128, 128]);
    }
    return { tensors: cache, length: 0 };
  }

  release(cache) {
    for (const t of Object.values(cache.tensors)) if (t.location === "gpu-buffer") t.dispose();
  }

  // One run of `n` new tokens after `cache`; replaces the cache with the presents.
  async run(cache, n) {
    const start = cache.length;
    const total = start + n;
    const { inputs_embeds } = await this.embed.run({
      input_ids: new ort.Tensor("int64", ids(n, start), [1, n]),
    });
    const positions = new BigInt64Array(3 * n);
    for (let r = 0; r < 3; r++) for (let i = 0; i < n; i++) positions[r * n + i] = BigInt(start + i);
    const feed = {
      ...cache.tensors,
      inputs_embeds,
      position_ids: new ort.Tensor("int64", positions, [3, 1, n]),
      num_logits_to_keep: new ort.Tensor("int64", new BigInt64Array([1n]), []),
    };
    if (this.zeos) {
      const mask = new Uint8Array(total).fill(1);
      if (total > 10) mask[10] = 0;
      feed.key_mask = new ort.Tensor("bool", mask, [1, total]);
    } else {
      feed.attention_mask = new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]);
    }
    const out = await this.session.run(feed);
    inputs_embeds.dispose();
    const logits = await out.logits.getData();
    let attention = null;
    if (out.attention) attention = await out.attention.getData();
    this.release(cache);
    const next = {};
    for (const [name, t] of Object.entries(out)) {
      if (name.startsWith("present")) {
        next[name.replace("present_", "past_").replace("present.", "past_key_values.")] = t;
      }
    }
    cache.tensors = next;
    cache.length = total;
    return { logits, attention };
  }
}

async function timed(fn) {
  const t0 = performance.now();
  const value = await fn();
  return { ms: performance.now() - t0, value };
}

async function measure(runner, label) {
  const result = { label };
  // Shader compilation and first-use allocation, at each chunk size measured.
  for (const n of [2048, 512, 1]) {
    const warm = runner.empty();
    await runner.run(warm, n);
    await runner.run(warm, 1);
    runner.release(warm);
  }
  for (const n of [512, 2048]) {
    const cache = runner.empty();
    const { ms, value } = await timed(() => runner.run(cache, n));
    result[`prefill${n}`] = { ms, tokPerS: (n * 1000) / ms };
    if (value.attention) {
      let sum = 0;
      for (const v of value.attention) sum += v;
      result[`prefill${n}`].attentionSum = sum;
      result[`prefill${n}`].hiddenAttention = value.attention[10];
    }
    log(`${label}: prefill ${n} from empty: ${ms.toFixed(0)} ms, ${result[`prefill${n}`].tokPerS.toFixed(0)} tok/s`);
    runner.release(cache);
  }
  {
    const cache = runner.empty();
    const t0 = performance.now();
    for (let i = 0; i < 4; i++) await runner.run(cache, 512);
    const ms = performance.now() - t0;
    result.prefill4x512 = { ms, tokPerS: (2048 * 1000) / ms };
    log(`${label}: prefill 4 x 512 carried: ${ms.toFixed(0)} ms, ${result.prefill4x512.tokPerS.toFixed(0)} tok/s`);
    const after = await timed(() => runner.run(cache, 2048));
    result.prefill2048after2048 = { ms: after.ms, tokPerS: (2048 * 1000) / after.ms };
    log(`${label}: prefill 2048 after 2048: ${after.ms.toFixed(0)} ms, ${result.prefill2048after2048.tokPerS.toFixed(0)} tok/s`);
    const steps = [];
    for (let i = 0; i < DECODE_STEPS; i++) steps.push((await timed(() => runner.run(cache, 1))).ms);
    const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
    result.decodeAt4096 = { meanMs: mean, tokPerS: 1000 / mean, steps: steps.length };
    log(`${label}: decode at ~4k context: ${mean.toFixed(1)} ms/step, ${(1000 / mean).toFixed(1)} tok/s`);
    runner.release(cache);
  }
  {
    const cache = runner.empty();
    await runner.run(cache, 64);
    const steps = [];
    for (let i = 0; i < DECODE_STEPS; i++) steps.push((await timed(() => runner.run(cache, 1))).ms);
    const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
    result.decodeAt64 = { meanMs: mean, tokPerS: 1000 / mean, steps: steps.length };
    log(`${label}: decode at ~64 context: ${mean.toFixed(1)} ms/step, ${(1000 / mean).toFixed(1)} tok/s`);
    runner.release(cache);
  }
  return result;
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const info = adapter.info ?? {};
  log(`ort ${ort.env.versions?.web ?? "?"}; adapter ${info.vendor ?? "?"} ${info.architecture ?? ""}`);
  const meta = await (await fetch(new URL("meta.json", modelUrl))).json();
  const results = { ort: ort.env.versions?.web, adapter: info, runs: [] };
  const shards = meta.decoder.externalData.map((p) => p.replace(/^onnx\//, ""));
  const decoder = params.get("decoder") ?? meta.decoder.file.replace(/^onnx\//, "");
  let runner;
  if (params.get("only") !== "opt") {
    runner = await Runner.create(modelUrl, decoder, shards, true);
    results.runs.push(await measure(runner, "zeos"));
    await runner.session.release();
    await runner.embed.release();
  }
  if (optUrl) {
    runner = await Runner.create(
      optUrl,
      "decoder_model_merged_q4f16.onnx",
      ["decoder_model_merged_q4f16.onnx_data", "decoder_model_merged_q4f16.onnx_data_1"],
      false,
    );
    results.runs.push(await measure(runner, "opt"));
    await runner.session.release();
    await runner.embed.release();
  }
  window.benchResult = results;
  log(JSON.stringify(results, null, 2));
} catch (error) {
  window.benchResult = { error: String(error?.stack ?? error) };
  log(`error: ${error?.stack ?? error}`);
}
