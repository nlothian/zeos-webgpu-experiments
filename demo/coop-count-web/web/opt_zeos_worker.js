// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model half of a ZEOS machine backend over the OPT+ZEOS graph
 * (`export/opt_zeos_surgery.py`): Qwen3.5-4B as `onnx-community/Qwen3.5-4B-ONNX-OPT`
 * exports it, with fused DeltaNet kernels, a key mask and a measured attention output.
 * Two sessions: the embedding graph turns ids into `inputs_embeds`, and the decoder runs
 * them against the cache.
 *
 * Like `TransformersWorker` it owns token ids and caches and nothing else, imports
 * nothing but `encodePlain` and `sampleToken` (ONNX Runtime and the tokenizer class are
 * handed in), and returns promises from `append` and `decodeStep`; `SyncModelWorker` in
 * `model_channel.js` makes it synchronous for the Python side.
 *
 * **Pending tokens.** `append` only records ids. The next `decodeStep` runs every id with
 * no cache behind it through the graph, in chunks of at most `maxChunk` (2048) cut at the
 * snapshot positions, under that step's mask, and reads the logits and the attention of
 * the last chunk's last position. A step does not append the id it chooses: `JsMachine`
 * appends it before the next step, which finds it pending, so a step in a run of steps
 * is exactly one forward pass of one token.
 *
 * **The cache** stays where the graph wrote it: on WebGPU every `present*` output is a
 * GPU buffer (`preferredOutputLocation: "gpu-buffer"`) fed back as the next run's past,
 * and only the logits and the attention vector are read back, and only for the last
 * chunk. A run's outputs are new tensors and are never written to, so a set of them is
 * shared, with a reference count, between a context, its snapshots and its forks, and a
 * set is disposed when its count reaches zero. Cutting the softmax layers' cache to `n`
 * positions copies the first `n` of each of its four heads into a new buffer
 * (`copyBufferToBuffer`, no round trip through the CPU).
 *
 * **Snapshots.** The DeltaNet layers' recurrent state and convolution window are not per
 * position, so they cannot be cut. A context keeps the state after every
 * `snapshotEvery` (256) positions, and the state from before its latest run, as
 * references to the graph's own output tensors: no copy, no download, about 25 MB of GPU
 * memory each (24 layers of a 1 MiB recurrent state and a 48 KiB window). At most
 * `maxSnapshots` (16) are kept; past that the one whose removal leaves the smallest gap
 * goes, so old snapshots thin out and recent ones stay dense. A cut (`truncate`, or a
 * step whose mask disagrees over a past position with the mask the state was built under)
 * rewinds to the latest snapshot at or before the position and replays the positions
 * after it, at most `snapshotEvery - 1` of them while no snapshot has been thinned.
 * Keeping the snapshots on the GPU costs memory the CPU would not, but a rewind then costs
 * nothing but the replay, where a downloaded snapshot would cost a 25 MB upload and every
 * snapshot taken a 25 MB download.
 */

import { encodePlain, sampleToken } from "./transformers_worker.js";

/** Positions between two snapshots of a context's recurrent state, and so the most a cut
 * replays. Also the length every prefill chunk is cut at. */
export const SNAPSHOT_EVERY = 256;
/** The most snapshots one context keeps. */
export const MAX_SNAPSHOTS = 16;

/** Whether a `meta.json` describes an OPT+ZEOS export rather than an `export_model.py`
 * one. */
export function isOptZeosMeta(meta) {
  return meta.decoder !== undefined && meta.embedTokens !== undefined;
}

const F16_TO_F32 = (() => {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) table[h] = sign * frac * 2 ** -24;
    else if (exp === 31) table[h] = frac ? NaN : sign * Infinity;
    else table[h] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return table;
})();

/** The first `limit` values of a float16 tensor's data as float32. */
function f16ToF32(data, limit) {
  const out = new Float32Array(limit);
  if (data instanceof Uint16Array) {
    for (let i = 0; i < limit; i++) out[i] = F16_TO_F32[data[i]];
  } else {
    for (let i = 0; i < limit; i++) out[i] = data[i];
  }
  return out;
}

/** A set of tensors with a reference count, disposed when the last holder releases it. */
class Shared {
  constructor(tensors, { permanent = false } = {}) {
    this.tensors = tensors;
    this.refs = 1;
    this.permanent = permanent;
  }

  retain() {
    this.refs += 1;
    return this;
  }

  release() {
    if (this.permanent) return;
    this.refs -= 1;
    if (this.refs < 0) throw new Error("tensor set released twice");
    if (this.refs === 0) {
      for (const t of Object.values(this.tensors)) t.dispose();
      this.tensors = null;
    }
  }
}

class Context {
  constructor(worker, job) {
    /** The context id the interface gave this cache, for `onActivity`. */
    this.job = job;
    this.tokens = [];
    /** Positions with a cache behind them. */
    this.kvLength = 0;
    /** The softmax layers' keys and values for `kvLength` positions, by input name. */
    this.kv = worker.emptyKv;
    /** The DeltaNet layers' state after `kvLength` positions, by input name. */
    this.state = worker.zeroState;
    /** Per cached position, 1 if the cache was built with it visible. */
    this.visibility = new Uint8Array(256);
    /** `{pos, state}` after multiples of `snapshotEvery` positions, ascending. */
    this.snapshots = [];
    /** `{pos, state}` from before the latest run, or null. */
    this.previous = null;
  }

  reserve(positions) {
    if (positions > this.visibility.length) {
      const grown = new Uint8Array(Math.max(positions, this.visibility.length * 2));
      grown.set(this.visibility.subarray(0, this.kvLength));
      this.visibility = grown;
    }
  }

  copy(job) {
    const other = Object.assign(Object.create(Context.prototype), this);
    other.job = job;
    other.tokens = this.tokens.slice();
    other.visibility = this.visibility.slice();
    other.kv = this.kv.retain();
    other.state = this.state.retain();
    other.snapshots = this.snapshots.map(({ pos, state }) => ({ pos, state: state.retain() }));
    other.previous =
      this.previous === null ? null : { pos: this.previous.pos, state: this.previous.state.retain() };
    return other;
  }

  release() {
    this.kv.release();
    this.state.release();
    for (const snap of this.snapshots) snap.state.release();
    this.previous?.state.release();
    this.snapshots = [];
    this.previous = null;
  }
}

export class OptZeosWorker {
  /**
   * @param {object} deps
   * @param {object} deps.ort ONNX Runtime (`onnxruntime-web` or `onnxruntime-node`).
   * @param {object} deps.tokenizer a `Tokenizer` from `@huggingface/tokenizers`.
   * @param {object} deps.meta the export's `meta.json`.
   * @param {object} deps.embed an `InferenceSession` over the embedding graph.
   * @param {object} deps.decoder an `InferenceSession` over the decoder.
   * @param {string} deps.backend the execution provider both run on.
   * @param {object} [deps.device] the `GPUDevice` ONNX Runtime runs on, for WebGPU.
   * @param {(activity: object) => void} [deps.onActivity] as `TransformersWorker`'s.
   * @param {number} [deps.snapshotEvery]
   * @param {number} [deps.maxSnapshots]
   */
  constructor({
    ort,
    tokenizer,
    meta,
    embed,
    decoder,
    backend,
    device = null,
    onActivity = null,
    snapshotEvery = SNAPSHOT_EVERY,
    maxSnapshots = MAX_SNAPSHOTS,
  }) {
    if (meta.blockSize !== 1) throw new Error(`export has blockSize ${meta.blockSize}; expected 1`);
    if (!(maxSnapshots >= 1)) throw new RangeError("maxSnapshots must be at least 1");
    this.ort = ort;
    this.tokenizer = tokenizer;
    // `tokenizerSize` is the name frames.js's `pieces` reads.
    this.meta = { ...meta, tokenizerSize: meta.vocabSize };
    this.embed = embed;
    this.decoder = decoder;
    this.backend = backend;
    this.device = device;
    this.onActivity = onActivity;
    this.snapshotEvery = snapshotEvery;
    this.maxSnapshots = maxSnapshots;
    this.chunk = meta.maxChunk;
    const [, heads, , headDim] = meta.kvShape;
    this.kvHeads = heads;
    this.headDim = headDim;
    /** past input name -> present output name, for both halves of the cache. */
    this.kvNames = [];
    for (const l of meta.fullAttentionLayers) {
      for (const half of ["key", "value"]) {
        this.kvNames.push([`past_key_values.${l}.${half}`, `present.${l}.${half}`]);
      }
    }
    this.stateNames = [];
    for (const l of meta.linearAttentionLayers) {
      this.stateNames.push([`past_conv.${l}`, `present_conv.${l}`]);
      this.stateNames.push([`past_recurrent.${l}`, `present_recurrent.${l}`]);
    }
    const size = (shape) => shape.reduce((a, b) => a * b, 1);
    const zero = {};
    for (const [name] of this.stateNames) {
      const shape = name.startsWith("past_conv") ? meta.convShape : meta.recurrentShape;
      zero[name] = new ort.Tensor("float16", new Uint16Array(size(shape)), shape);
    }
    this.zeroState = new Shared(zero, { permanent: true });
    const empty = {};
    for (const [name] of this.kvNames) {
      empty[name] = new ort.Tensor("float16", new Uint16Array(0), [1, heads, 0, headDim]);
    }
    this.emptyKv = new Shared(empty, { permanent: true });
    this.contexts = new Map();
    this.pieces = new Map();
    /** The logits of the latest run that read them, below `vocabSize`, for diagnostics. */
    this.lastLogits = null;
    /** Runs of the graph, and positions run again because of a cut or a changed mask. */
    this.stats = { runs: 0, positions: 0, reruns: 0, rerunPositions: 0 };
  }

  /** Every output of the decoder but the logits and attention, which are read back. */
  static presentNames(meta) {
    const names = [];
    for (const l of meta.fullAttentionLayers) names.push(`present.${l}.key`, `present.${l}.value`);
    for (const l of meta.linearAttentionLayers) names.push(`present_conv.${l}`, `present_recurrent.${l}`);
    return names;
  }

  /**
   * Build a worker from an export's files.
   *
   * @param {object} options
   * @param {(name: string) => Uint8Array | Promise<Uint8Array>} options.read a file's bytes.
   * @param {(name: string) => Uint8Array | string | Promise<Uint8Array | string>} [options.source]
   *   a graph or weights file as bytes, or as a path or URL ONNX Runtime reads itself.
   *   `read` by default. Files with the same SHA-256 in `meta.files` are read once.
   * @param {string} [options.backend] `webgpu`, `wasm`, or `cpu` (onnxruntime-node).
   * @param {number} [options.numThreads] WebAssembly threads; 1 by default, so the CPU
   *   fallback never needs a cross-origin isolated worker pool.
   */
  static async load({
    ort,
    Tokenizer,
    read,
    source = read,
    backend = "wasm",
    sessionOptions = {},
    numThreads = 1,
    onActivity = null,
    snapshotEvery,
    maxSnapshots,
  }) {
    if (ort.env?.wasm && backend !== "cpu") ort.env.wasm.numThreads = numThreads;
    const decoder = new TextDecoder();
    const json = async (name) => JSON.parse(decoder.decode(await read(name)));
    const meta = await json("meta.json");
    if (!isOptZeosMeta(meta)) throw new Error("meta.json does not describe an OPT+ZEOS export");
    const tokenizer = new Tokenizer(await json("tokenizer.json"), await json("tokenizer_config.json"));
    // The embedding's data and the decoder's second shard are the same tied matrix.
    const bySha = new Map();
    const file = async (name) => {
      const sha = meta.files?.[name]?.sha256;
      if (sha !== undefined && bySha.has(sha)) return bySha.get(sha);
      const value = await source(name);
      if (sha !== undefined) bySha.set(sha, value);
      return value;
    };
    const base = (path) => path.replace(/^.*\//, "");
    const webgpu = backend === "webgpu";
    const create = async (graph, outputs) => {
      const options = {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
        ...sessionOptions,
      };
      const model = await file(graph.file);
      // onnxruntime-node finds the data files beside a model it is given by path.
      if (!(backend === "cpu" && typeof model === "string")) {
        options.externalData = [];
        for (const path of graph.externalData) {
          options.externalData.push({ path: base(path), data: await file(path) });
        }
      }
      if (webgpu) options.preferredOutputLocation = Object.fromEntries(outputs.map((n) => [n, "gpu-buffer"]));
      return ort.InferenceSession.create(model, options);
    };
    const embed = await create(meta.embedTokens, ["inputs_embeds"]);
    const session = await create(meta.decoder, [...OptZeosWorker.presentNames(meta), "logits", "attention"]);
    bySha.clear();
    const device = webgpu ? await ort.env.webgpu.device : null;
    return new OptZeosWorker({
      ort,
      tokenizer,
      meta,
      embed,
      decoder: session,
      backend,
      device,
      onActivity,
      snapshotEvery,
      maxSnapshots,
    });
  }

  // -- the interface -------------------------------------------------------------

  /** `vocabSize` is the tokenizer's vocabulary, added tokens included; the logits are
   * wider (248,320), and no id past the tokenizer's is ever chosen. */
  info() {
    return {
      blockSize: this.meta.blockSize,
      padId: this.meta.padId,
      controlIds: this.meta.controlIds.slice(),
      eosId: this.meta.eosId,
      vocabSize: this.meta.vocabSize,
    };
  }

  tokenize(text) {
    return encodePlain(this.tokenizer, text);
  }

  piece(tokenId) {
    let text = this.pieces.get(tokenId);
    if (text === undefined) {
      if (!(tokenId >= 0 && tokenId < this.meta.vocabSize)) {
        throw new RangeError(`token id ${tokenId} is outside the vocabulary`);
      }
      text = this.tokenizer.decode([tokenId], { skip_special_tokens: false });
      this.pieces.set(tokenId, text);
    }
    return text;
  }

  createContext(jobId) {
    this.contexts.get(jobId)?.release();
    this.contexts.set(jobId, new Context(this, jobId));
  }

  destroyContext(jobId) {
    this.contexts.get(jobId)?.release();
    this.contexts.delete(jobId);
  }

  length(jobId) {
    return this.ctx(jobId).tokens.length;
  }

  /** Record ids; they are run by the next decode step. */
  append(jobId, ids) {
    const ctx = this.ctx(jobId);
    for (const id of ids) {
      if (!(id >= 0 && id < this.meta.vocabSize)) throw new RangeError(`token id ${id} is outside the vocabulary`);
      ctx.tokens.push(id);
    }
  }

  truncate(jobId, n) {
    const ctx = this.ctx(jobId);
    if (!(n >= 0 && n <= ctx.tokens.length)) {
      throw new RangeError(`truncate to ${n} outside [0, ${ctx.tokens.length}]`);
    }
    ctx.tokens.length = n;
    // The last token stays pending, so the step after the cut has a position to run.
    this.rewind(ctx, Math.max(n - 1, 0));
  }

  fork(parentId, childId) {
    const ctx = this.ctx(parentId).copy(childId);
    this.contexts.get(childId)?.release();
    this.contexts.set(childId, ctx);
  }

  async decodeStep(jobId, { allowedBlocks = null, allowedTokens = null, sample = null } = {}) {
    const ctx = this.ctx(jobId);
    const n = ctx.tokens.length;
    if (n === 0) throw new Error(`job ${jobId}: cannot decode an empty context`);
    const allowed = new Uint8Array(n);
    if (allowedBlocks === null) {
      allowed.fill(1);
    } else {
      if (allowedBlocks.length < n) {
        throw new RangeError(`job ${jobId}: allowedBlocks covers ${allowedBlocks.length} blocks of ${n}`);
      }
      for (let b = 0; b < n; b++) allowed[b] = allowedBlocks[b] ? 1 : 0;
      if (!allowed.some((v) => v)) {
        throw new Error(`job ${jobId}: the mask hides every block, so nothing can be attended`);
      }
    }
    if (allowedTokens !== null && allowedTokens.length < this.meta.vocabSize) {
      throw new RangeError(`job ${jobId}: allowedTokens covers ${allowedTokens.length} ids of ${this.meta.vocabSize}`);
    }
    // A step repeated with nothing appended runs its position again.
    if (ctx.kvLength > n - 1) this.rewind(ctx, n - 1);
    const { logits, attention } = await this.fill(ctx, allowed);
    const limit = this.meta.vocabSize;
    const tokenId =
      sample === null ? argmax(logits, allowedTokens, limit) : sampleToken(logits, allowedTokens, limit, sample);
    return { tokenId, attention };
  }

  // -- outside the interface -------------------------------------------------------

  async release() {
    for (const ctx of this.contexts.values()) ctx.release();
    this.contexts.clear();
    await this.decoder.release();
    await this.embed.release();
  }

  /** The logits and attention of the last position under `allowed`, from scratch: one
   * run per chunk with no snapshot or cut involved, every chunk under the same mask.
   * `keep` > 1 also returns `rows`, the logits of the last `keep` positions of the last
   * chunk, oldest first. For tests; leaves every context as it was. */
  async reference(ids, allowed = null, chunk = this.chunk, keep = 1) {
    const job = Symbol("reference");
    const ctx = new Context(this, job);
    ctx.tokens = Array.from(ids);
    const mask = allowed ?? new Uint8Array(ids.length).fill(1);
    try {
      let out = null;
      while (ctx.kvLength < ids.length) {
        const start = ctx.kvLength;
        const count = Math.min(chunk, ids.length - start);
        const last = start + count === ids.length;
        out = await this.run(ctx, start, count, mask, last, last ? keep : 1);
      }
      return out;
    } finally {
      ctx.release();
    }
  }

  // -- internals -------------------------------------------------------------------

  ctx(jobId) {
    const ctx = this.contexts.get(jobId);
    if (ctx === undefined) throw new Error(`no context for job ${jobId}; createContext first`);
    return ctx;
  }

  /** Cut the cache back to at most `target` positions: to the latest of the snapshots
   * and the state from before the latest run at or before it. The positions between
   * are run again by the next `fill`. */
  rewind(ctx, target) {
    if (ctx.kvLength <= target) return;
    let pos = 0;
    let state = this.zeroState;
    while (ctx.snapshots.length > 0 && ctx.snapshots[ctx.snapshots.length - 1].pos > target) {
      ctx.snapshots.pop().state.release();
    }
    const last = ctx.snapshots[ctx.snapshots.length - 1];
    if (last !== undefined) ({ pos, state } = last);
    if (ctx.previous !== null && ctx.previous.pos <= target && ctx.previous.pos > pos) {
      ({ pos, state } = ctx.previous);
    }
    if (pos < target) {
      this.stats.reruns += 1;
      this.stats.rerunPositions += target - pos;
    }
    state.retain();
    if (ctx.previous !== null && ctx.previous.pos > pos) {
      ctx.previous.state.release();
      ctx.previous = null;
    }
    ctx.state.release();
    ctx.state = state;
    const kv = this.sliceKv(ctx.kv, ctx.kvLength, pos);
    ctx.kv.release();
    ctx.kv = kv;
    ctx.kvLength = pos;
  }

  /** The first `n` of `length` positions of a softmax cache, as a new set. */
  sliceKv(kv, length, n) {
    if (n === length) return kv.retain();
    if (n === 0) return this.emptyKv;
    const { ort } = this;
    const dims = [1, this.kvHeads, n, this.headDim];
    const tensors = {};
    if (this.device !== null) {
      const row = this.headDim * 2; // bytes of one position of one head
      const encoder = this.device.createCommandEncoder();
      for (const [name] of this.kvNames) {
        const src = kv.tensors[name].gpuBuffer;
        const dst = this.device.createBuffer({
          size: this.kvHeads * n * row,
          // As ONNX Runtime's own storage buffers.
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        for (let h = 0; h < this.kvHeads; h++) {
          encoder.copyBufferToBuffer(src, h * length * row, dst, h * n * row, n * row);
        }
        tensors[name] = ort.Tensor.fromGpuBuffer(dst, {
          dataType: "float16",
          dims,
          dispose: () => dst.destroy(),
        });
      }
      this.device.queue.submit([encoder.finish()]);
    } else {
      const row = this.headDim;
      for (const [name] of this.kvNames) {
        const src = kv.tensors[name].data;
        const dst = new src.constructor(this.kvHeads * n * row);
        for (let h = 0; h < this.kvHeads; h++) {
          dst.set(src.subarray(h * length * row, h * length * row + n * row), h * n * row);
        }
        tensors[name] = new ort.Tensor("float16", dst, dims);
      }
    }
    return new Shared(tensors);
  }

  /** Make the cache cover every position as `allowed` sees it, running what is
   * missing, and return the last position's logits and attention. */
  async fill(ctx, allowed) {
    const n = ctx.tokens.length;
    let first = 0;
    while (first < ctx.kvLength && ctx.visibility[first] === allowed[first]) first++;
    if (first < ctx.kvLength) this.rewind(ctx, first);
    let out = null;
    while (ctx.kvLength < n) {
      const start = ctx.kvLength;
      const boundary = (Math.floor(start / this.snapshotEvery) + 1) * this.snapshotEvery;
      const count = Math.min(this.chunk, n - start, boundary - start);
      const last = start + count === n;
      if (last) {
        ctx.previous?.state.release();
        ctx.previous = { pos: start, state: ctx.state.retain() };
      }
      out = await this.run(ctx, start, count, allowed, last);
    }
    return out;
  }

  /** One run of the graph over tokens [start, start + count) after a cache of `start`
   * positions, with `visible` giving each position's key mask; commits the outputs to
   * `ctx` and returns the logits and attention when `read` is set. */
  async run(ctx, start, count, visible, read, keep = 1) {
    const { ort } = this;
    const total = start + count;
    const ids = new BigInt64Array(count);
    for (let i = 0; i < count; i++) ids[i] = BigInt(ctx.tokens[start + i]);
    const positions = new BigInt64Array(3 * count);
    for (let r = 0; r < 3; r++) for (let i = 0; i < count; i++) positions[r * count + i] = BigInt(start + i);
    const phase = read && count === 1 ? "decode" : "prefill";
    const activity = { phase, job: typeof ctx.job === "symbol" ? null : ctx.job, start, count, length: ctx.tokens.length };
    this.onActivity?.(activity);
    const began = performance.now();
    const { inputs_embeds } = await this.embed.run({ input_ids: new ort.Tensor("int64", ids, [1, count]) });
    const feed = {
      inputs_embeds,
      key_mask: new ort.Tensor("bool", visible.slice(0, total), [1, total]),
      position_ids: new ort.Tensor("int64", positions, [3, 1, count]),
      num_logits_to_keep: new ort.Tensor("int64", new BigInt64Array([BigInt(keep)]), []),
      ...ctx.kv.tensors,
      ...ctx.state.tensors,
    };
    let out;
    try {
      out = await this.decoder.run(feed);
    } finally {
      inputs_embeds.dispose();
    }
    const kv = {};
    for (const [past, present] of this.kvNames) kv[past] = out[present];
    const state = {};
    for (const [past, present] of this.stateNames) state[past] = out[present];
    ctx.kv.release();
    ctx.kv = new Shared(kv);
    ctx.state.release();
    ctx.state = new Shared(state);
    ctx.reserve(total);
    ctx.visibility.set(visible.subarray(start, total), start);
    ctx.kvLength = total;
    if (total % this.snapshotEvery === 0) this.snapshot(ctx);
    let result = null;
    try {
      if (read) {
        const data = await out.logits.getData();
        const width = this.meta.logitsSize;
        const rows = [];
        for (let k = 0; k < keep; k++) rows.push(f16ToF32(data.subarray(k * width, (k + 1) * width), this.meta.vocabSize));
        const attention = Float32Array.from(await out.attention.getData());
        result = { logits: rows[keep - 1], attention };
        if (keep > 1) result.rows = rows;
        this.lastLogits = result.logits;
      }
    } finally {
      out.logits.dispose();
      out.attention.dispose();
    }
    this.stats.runs += 1;
    this.stats.positions += count;
    this.onActivity?.({ ...activity, ms: performance.now() - began });
    return result;
  }

  snapshot(ctx) {
    ctx.snapshots.push({ pos: ctx.kvLength, state: ctx.state.retain() });
    if (ctx.snapshots.length <= this.maxSnapshots) return;
    // Drop the snapshot whose neighbours are closest, never the latest.
    let drop = 0;
    let gap = Infinity;
    for (let i = 0; i < ctx.snapshots.length - 1; i++) {
      const before = i === 0 ? 0 : ctx.snapshots[i - 1].pos;
      const span = ctx.snapshots[i + 1].pos - before;
      if (span < gap) {
        gap = span;
        drop = i;
      }
    }
    ctx.snapshots.splice(drop, 1)[0].state.release();
  }
}

/** Greedy choice among the allowed ids below `limit`, lowest id on a tie. */
export function argmax(logits, allowedTokens, limit) {
  let best = -1;
  let bestValue = -Infinity;
  for (let id = 0; id < limit; id++) {
    if (allowedTokens !== null && !allowedTokens[id]) continue;
    const value = logits[id];
    if (best === -1 || value > bestValue) {
      best = id;
      bestValue = value;
    }
  }
  if (best === -1) throw new Error("allowedTokens permits no token");
  return best;
}
