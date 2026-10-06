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
 * nothing but `encodePlain`, `pieceBytes` and `sampleToken` (ONNX Runtime and the
 * tokenizer class are handed in), and returns promises from `append` and `decodeStep`;
 * `SyncModelWorker` in `model_channel.js` makes it synchronous for the Python side.
 *
 * **Pending tokens.** `append` only records ids. The next `decodeStep` runs every id with
 * no cache behind it through the graph, in chunks cut at the snapshot positions -- so no
 * chunk is longer than `snapshotEvery` (256), though the graph takes `maxChunk` (2048) --
 * under that step's mask, and reads the logits and the attention of the last chunk's last
 * position. A step does not append the id it chooses: `JsMachine`
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
 * position, so they cannot be cut. A cache keeps the state after every `snapshotEvery`
 * (256) positions, and the state from before its latest run, as references to the
 * graph's own output tensors: no copy, no download, about 25 MB of GPU memory each (24
 * layers of a 1 MiB recurrent state and a 48 KiB window). At most `maxSnapshots` (16) are
 * kept; past that the one whose removal leaves the smallest gap goes, so old snapshots
 * thin out and recent ones stay dense. A cut (`truncate`, or a step whose mask disagrees
 * over a past position with the mask the state was built under) rewinds to the latest
 * snapshot at or before the position and replays the positions after it, at most
 * `snapshotEvery - 1` of them while no snapshot has been thinned. Keeping the snapshots
 * on the GPU costs memory the CPU would not, but a rewind then costs nothing but the
 * replay, where a downloaded snapshot would cost a 25 MB upload and every snapshot taken a
 * 25 MB download.
 *
 * **Two caches per context** (`maxTracks`, 2 by default). A context keeps the cache of
 * its latest step's mask and the cache of the mask before it. A step whose mask agrees
 * with neither over their cached positions starts a new cache from whichever of the two
 * shares more of its history (rewound to the latest snapshot before they part), and the
 * older of the two is dropped. A step whose mask agrees with the kept one switches to it
 * and runs only what that cache has not seen. So a mask that hides some past positions
 * for a few steps and then shows them again -- a tool's name chosen without the tool
 * results in view -- replays once when it is first narrowed and then only catches up,
 * and the cache of the wide mask is never rewound. Each cache costs its own softmax keys
 * and values (32 KB a position) and snapshots. With `maxTracks` 1 a disagreeing mask
 * rewinds the one cache instead.
 *
 * **Hidden runs** (`skipHidden`, on by default). A hidden position leaves the DeltaNet
 * state as it was (beta and the log decay are zero) and enters the convolution as zeros,
 * and no later query can attend its keys. So a run of at least `minSkip` (16, and never
 * fewer than `convShape[2]`, 3) hidden positions, not including the last one, is not run
 * through the graph: the recurrent
 * state is carried as it is, the convolution window becomes zeros, and the softmax cache
 * grows by zeros there in one copy, which the key mask hides. Chunks are also cut where
 * such a run starts. A fresh prefill and a replay under the same mask cut and skip at the
 * same positions, so they still agree bit for bit; and a skipped run aligned with the
 * chunks is bit for bit the run it stands in for, since a hidden position adds exact
 * zeros and multiplies by exact ones. A shorter hidden run -- the note the chat machine
 * hides on every step but the ones that choose a tool's name, say -- runs inside its
 * chunk under the key mask: skipping it would cut the chunk there, and at a few thousand
 * positions a run of the graph costs more than the positions it saves.
 *
 * **A step that can stop.** `decodeStep` also takes `maxChunk`, a smaller run length for
 * this step (chunks are still cut at the snapshot positions and hidden runs), and
 * `shouldStop`, asked before every run of the graph. When it answers true the step
 * returns `{cancelled: true, resident, stats}` at once: the positions already run stay in
 * the cache (`resident` of them), so the next step for the context resumes from there.
 * Chunks are cut as a function of where they start and of `maxChunk`, so a step stopped
 * and then resumed with the same `maxChunk` runs exactly the chunks it would have run
 * uninterrupted, and chooses the same token bit for bit. Resumed with another
 * `maxChunk`, the chunks after the stop are cut differently and agree only to float16
 * rounding. Every answer carries `resident` and `stats` (`positions` run, `chunks`
 * runs, `fillMs` spent in them, not counting a final one-position decode).
 */

import { encodePlain, pieceBytes, sampleToken } from "./transformers_worker.js";

/** Positions between two snapshots of a context's recurrent state, and so the most a cut
 * replays. Also the length every prefill chunk is cut at. */
export const SNAPSHOT_EVERY = 256;
/** The most snapshots one cache keeps. */
export const MAX_SNAPSHOTS = 16;
/** The most caches one context keeps, one per mask history. */
export const MAX_TRACKS = 2;
/** The shortest hidden run carried past rather than run (`skipHidden`). */
export const MIN_SKIP = 16;

/** Throw for a backend other than WebGPU, or for options `load` does not know: a caller
 * passing one expects it to do something. */
export function refuseOptions(where, backend, unknown) {
  if (backend !== "webgpu") throw new Error(`${where}: backend ${backend} is not supported; the model runs on WebGPU only`);
  const names = Object.keys(unknown);
  if (names.length > 0) throw new Error(`${where}: unknown option${names.length > 1 ? "s" : ""} ${names.join(", ")}`);
}

/** What `fill` returns for a step `shouldStop` ended. */
const STOPPED = Symbol("stopped");

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
  /** With `parents`, the tensors are borrowed from those sets: the last release here
   * releases them rather than disposing anything. */
  constructor(tensors, { permanent = false, parents = null } = {}) {
    this.tensors = tensors;
    this.refs = 1;
    this.permanent = permanent;
    this.parents = parents;
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
      if (this.parents === null) for (const t of Object.values(this.tensors)) t.dispose();
      else for (const parent of this.parents) parent.release();
      this.tensors = null;
      this.parents = null;
    }
  }
}

let trackIds = 0;

/** One cache of a context: the softmax layers' keys and values and the DeltaNet state for
 * `kvLength` positions, built under one history of masks (`visibility`). */
class Track {
  constructor(worker) {
    /** Numbers caches for `onActivity`. */
    this.id = ++trackIds;
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

  /** How many leading cached positions were built as `allowed` would show them. */
  agreement(allowed) {
    let p = 0;
    while (p < this.kvLength && this.visibility[p] === allowed[p]) p++;
    return p;
  }

  copy() {
    const other = Object.assign(Object.create(Track.prototype), this);
    other.id = ++trackIds;
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

class Context {
  constructor(worker, job) {
    /** The context id the interface gave this cache, for `onActivity`. */
    this.job = job;
    this.tokens = [];
    /** The cache of the latest step's mask. */
    this.track = new Track(worker);
    /** The cache of the mask before it, or null. */
    this.other = null;
  }

  get snapshots() {
    return this.track.snapshots;
  }

  /** A fork's context: the tokens, and the current cache with its tensors shared. */
  copy(job) {
    const other = Object.create(Context.prototype);
    other.job = job;
    other.tokens = this.tokens.slice();
    other.track = this.track.copy();
    other.other = null;
    return other;
  }

  release() {
    this.track.release();
    this.other?.release();
    this.other = null;
  }
}

export class OptZeosWorker {
  /**
   * @param {object} deps
   * @param {object} deps.ort ONNX Runtime Web's WebGPU build.
   * @param {object} deps.tokenizer a `Tokenizer` from `@huggingface/tokenizers`.
   * @param {object} deps.meta the export's `meta.json`.
   * @param {object} deps.embed an `InferenceSession` over the embedding graph.
   * @param {object} deps.decoder an `InferenceSession` over the decoder.
   * @param {string} deps.backend the execution provider both run on.
   * @param {object} deps.device the `GPUDevice` ONNX Runtime runs on.
   * @param {(activity: object) => void} [deps.onActivity] as `TransformersWorker`'s.
   * @param {number} [deps.snapshotEvery]
   * @param {number} [deps.maxSnapshots]
   * @param {number} [deps.maxTracks] caches per context, 1 or 2.
   * @param {boolean} [deps.skipHidden] carry the state past runs of hidden positions.
   * @param {number} [deps.minSkip] the shortest such run, at least `convShape[2]`.
   */
  constructor({
    ort,
    tokenizer,
    meta,
    embed,
    decoder,
    backend,
    device,
    onActivity = null,
    snapshotEvery = SNAPSHOT_EVERY,
    maxSnapshots = MAX_SNAPSHOTS,
    maxTracks = MAX_TRACKS,
    skipHidden = true,
    minSkip = MIN_SKIP,
  }) {
    if (meta.blockSize !== 1) throw new Error(`export has blockSize ${meta.blockSize}; expected 1`);
    if (!(maxSnapshots >= 1)) throw new RangeError("maxSnapshots must be at least 1");
    if (maxTracks !== 1 && maxTracks !== 2) throw new RangeError("maxTracks is 1 or 2");
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
    this.maxTracks = maxTracks;
    this.skipHidden = skipHidden;
    this.chunk = meta.maxChunk;
    /** The convolution's carried inputs: a hidden run this long leaves none of them. */
    this.convWindow = meta.convShape[2];
    this.minSkip = Math.max(minSkip, this.convWindow);
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
    /** Runs of the graph; positions run again because of a cut or a changed mask; caches
     * started for a new mask and switches back to a kept one; hidden positions carried
     * past without a run. */
    this.stats = { runs: 0, positions: 0, reruns: 0, rerunPositions: 0, tracks: 0, switches: 0, skipped: 0 };
  }

  /** Every output of the decoder but the logits and attention, which are read back. */
  static presentNames(meta) {
    const names = [];
    for (const l of meta.fullAttentionLayers) names.push(`present.${l}.key`, `present.${l}.value`);
    for (const l of meta.linearAttentionLayers) names.push(`present_conv.${l}`, `present_recurrent.${l}`);
    return names;
  }

  /**
   * Build a worker from an export's files, on WebGPU.
   *
   * @param {object} options
   * @param {(name: string) => Uint8Array | Promise<Uint8Array>} options.read a file's bytes.
   * @param {(name: string) => Uint8Array | string | Promise<Uint8Array | string>} [options.source]
   *   a graph or weights file as bytes, or as a URL ONNX Runtime reads itself. `read` by
   *   default. Files with the same SHA-256 in `meta.files` are read once.
   * @param {"webgpu"} [options.backend] the only backend; anything else is refused.
   * @param {number} [options.numThreads] threads for the kernels ONNX Runtime still runs as
   *   WebAssembly beside WebGPU; 1 by default, so no cross-origin isolated pool is needed.
   * Any other option is refused, rather than ignored.
   */
  static async load({
    ort,
    Tokenizer,
    read,
    source = read,
    backend = "webgpu",
    numThreads = 1,
    sessionOptions = {},
    onActivity = null,
    snapshotEvery,
    maxSnapshots,
    maxTracks,
    skipHidden,
    minSkip,
    ...unknown
  }) {
    refuseOptions("OptZeosWorker.load", backend, unknown);
    if (ort.env?.wasm) ort.env.wasm.numThreads = numThreads;
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
    const create = async (graph, outputs) => {
      const options = {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
        ...sessionOptions,
      };
      const model = await file(graph.file);
      options.externalData = [];
      for (const path of graph.externalData) {
        options.externalData.push({ path: base(path), data: await file(path) });
      }
      options.preferredOutputLocation = Object.fromEntries(outputs.map((n) => [n, "gpu-buffer"]));
      return ort.InferenceSession.create(model, options);
    };
    const embed = await create(meta.embedTokens, ["inputs_embeds"]);
    const session = await create(meta.decoder, [...OptZeosWorker.presentNames(meta), "logits", "attention"]);
    bySha.clear();
    const device = await ort.env.webgpu.device;
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
      maxTracks,
      skipHidden,
      minSkip,
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

  pieceBytes(tokenId) {
    return pieceBytes(this.tokenizer, tokenId);
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
    this.rewind(ctx.track, Math.max(n - 1, 0));
    if (ctx.other !== null) this.rewind(ctx.other, Math.max(n - 1, 0));
  }

  fork(parentId, childId) {
    const ctx = this.ctx(parentId).copy(childId);
    this.contexts.get(childId)?.release();
    this.contexts.set(childId, ctx);
  }

  async decodeStep(
    jobId,
    { allowedBlocks = null, allowedTokens = null, sample = null, shouldStop = null, maxChunk = null } = {},
  ) {
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
    if (maxChunk !== null && !(Number.isInteger(maxChunk) && maxChunk >= 1)) {
      throw new RangeError(`job ${jobId}: maxChunk ${maxChunk} is not a positive integer`);
    }
    const stats = { positions: 0, chunks: 0, fillMs: 0 };
    const chunk = maxChunk === null ? this.chunk : Math.min(maxChunk, this.chunk);
    const out = await this.fill(ctx, allowed, { chunk, shouldStop, stats });
    if (out === STOPPED) return { cancelled: true, resident: ctx.track.kvLength, stats };
    const { logits, attention } = out;
    const limit = this.meta.vocabSize;
    const tokenId =
      sample === null ? argmax(logits, allowedTokens, limit) : sampleToken(logits, allowedTokens, limit, sample);
    return { tokenId, attention, resident: ctx.track.kvLength, stats };
  }

  // -- outside the interface -------------------------------------------------------

  async release() {
    for (const ctx of this.contexts.values()) ctx.release();
    this.contexts.clear();
    await this.decoder.release();
    await this.embed.release();
  }

  /** The logits and attention of the last position under `allowed`, from scratch: one
   * run per chunk with no snapshot, cut or skipped run involved, every chunk under the
   * same mask. `keep` > 1 also returns `rows`, the logits of the last `keep` positions of
   * the last chunk, oldest first. For tests; leaves every context as it was. */
  async reference(ids, allowed = null, chunk = this.chunk, keep = 1) {
    const ctx = new Context(this, Symbol("reference"));
    ctx.tokens = Array.from(ids);
    const mask = allowed ?? new Uint8Array(ids.length).fill(1);
    try {
      let out = null;
      while (ctx.track.kvLength < ids.length) {
        const start = ctx.track.kvLength;
        const count = Math.min(chunk, ids.length - start);
        const last = start + count === ids.length;
        out = await this.run(ctx, ctx.track, start, count, mask, last, last ? keep : 1);
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

  /** Cut a cache back to at most `target` positions: to the latest of the snapshots and
   * the state from before the latest run at or before it. The positions between are run
   * again by the next `fill`. */
  rewind(track, target) {
    if (track.kvLength <= target) return;
    let pos = 0;
    let state = this.zeroState;
    while (track.snapshots.length > 0 && track.snapshots[track.snapshots.length - 1].pos > target) {
      track.snapshots.pop().state.release();
    }
    const last = track.snapshots[track.snapshots.length - 1];
    if (last !== undefined) ({ pos, state } = last);
    if (track.previous !== null && track.previous.pos <= target && track.previous.pos > pos) {
      ({ pos, state } = track.previous);
    }
    if (pos < target) {
      this.stats.reruns += 1;
      this.stats.rerunPositions += target - pos;
    }
    state.retain();
    if (track.previous !== null && track.previous.pos > pos) {
      track.previous.state.release();
      track.previous = null;
    }
    track.state.release();
    track.state = state;
    const kv = this.sliceKv(track.kv, track.kvLength, pos);
    track.kv.release();
    track.kv = kv;
    track.kvLength = pos;
  }

  /** The cache a step under `allowed` runs on, made current. */
  select(ctx, allowed) {
    const here = ctx.track.agreement(allowed);
    if (here === ctx.track.kvLength) return ctx.track;
    const there = ctx.other === null ? -1 : ctx.other.agreement(allowed);
    if (ctx.other !== null && there === ctx.other.kvLength) {
      [ctx.track, ctx.other] = [ctx.other, ctx.track];
      this.stats.switches += 1;
      return ctx.track;
    }
    if (this.maxTracks === 1) {
      this.rewind(ctx.track, here);
      return ctx.track;
    }
    // A mask neither cache was built under: start one from whichever shares more of its
    // history, keep the current one, and drop the other.
    const fromOther = there > here;
    const fresh = (fromOther ? ctx.other : ctx.track).copy();
    this.rewind(fresh, fromOther ? there : here);
    ctx.other?.release();
    ctx.other = ctx.track;
    ctx.track = fresh;
    this.stats.tracks += 1;
    return fresh;
  }

  /** The first `n` of `length` positions of a softmax cache, as a new set. */
  sliceKv(kv, length, n) {
    if (n === length) return kv.retain();
    if (n === 0) return this.emptyKv;
    return this.resizeKv(kv, length, n, n);
  }

  /** A softmax cache of `n` positions, as a new set, whose first `keep` are the first
   * `keep` of `kv` (of `length`) and the rest zeros. */
  resizeKv(kv, length, keep, n) {
    const { ort } = this;
    const dims = [1, this.kvHeads, n, this.headDim];
    const tensors = {};
    const row = this.headDim * 2; // bytes of one position of one head
    const encoder = this.device.createCommandEncoder();
    for (const [name] of this.kvNames) {
      const dst = this.device.createBuffer({
        // WebGPU zeroes a new buffer.
        size: this.kvHeads * n * row,
        // As ONNX Runtime's own storage buffers.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      if (keep > 0) {
        const src = kv.tensors[name].gpuBuffer;
        for (let h = 0; h < this.kvHeads; h++) {
          encoder.copyBufferToBuffer(src, h * length * row, dst, h * n * row, keep * row);
        }
      }
      tensors[name] = ort.Tensor.fromGpuBuffer(dst, {
        dataType: "float16",
        dims,
        dispose: () => dst.destroy(),
      });
    }
    this.device.queue.submit([encoder.finish()]);
    return new Shared(tensors);
  }

  /** Make a cache cover every position as `allowed` sees it, running what is missing in
   * runs of at most `chunk`, and return the last position's logits and attention; or
   * `STOPPED`, with the positions run so far cached, if `shouldStop` answers true before
   * a run. `stats` collects the runs. */
  async fill(ctx, allowed, { chunk = this.chunk, shouldStop = null, stats = null } = {}) {
    const n = ctx.tokens.length;
    const track = this.select(ctx, allowed);
    // A step repeated with nothing appended runs its position again.
    if (track.kvLength > n - 1) this.rewind(track, n - 1);
    let out = null;
    while (track.kvLength < n) {
      const start = track.kvLength;
      const boundary = this.nextBoundary(start);
      if (this.skipHidden) {
        const end = this.skippable(allowed, start, n);
        if (end > start) {
          this.skip(ctx, track, start, end);
          continue;
        }
      }
      let count = Math.min(chunk, n - start, boundary - start);
      if (this.skipHidden) {
        for (let q = start + 1; q < start + count; q++) {
          if (!allowed[q] && allowed[q - 1] && this.skippable(allowed, q, n) > q) {
            count = q - start;
            break;
          }
        }
      }
      if (shouldStop?.()) return STOPPED;
      const last = start + count === n;
      if (last) {
        track.previous?.state.release();
        track.previous = { pos: start, state: track.state.retain() };
      }
      const began = performance.now();
      out = await this.run(ctx, track, start, count, allowed, last);
      if (stats !== null) {
        stats.positions += count;
        stats.chunks += 1;
        if (!(last && count === 1)) stats.fillMs += performance.now() - began;
      }
    }
    return out;
  }

  nextBoundary(pos) {
    return (Math.floor(pos / this.snapshotEvery) + 1) * this.snapshotEvery;
  }

  /** Where a hidden run from `start` that can be carried past without a run ends, or
   * `start` if it cannot. It never takes the last position, whose logits are the step's,
   * it is at least `minSkip` long, and it needs `convWindow` hidden positions before the
   * first snapshot position it crosses, so the state there is the one it carries. */
  skippable(allowed, start, n) {
    let end = start;
    while (end < n - 1 && !allowed[end]) end++;
    if (end - start < this.minSkip) return start;
    return Math.min(end, this.nextBoundary(start)) - start >= this.convWindow ? end : start;
  }

  /** Carry a cache past the hidden positions [start, end) without running them: what a
   * run would leave, since a hidden position changes no state (see the module notes). */
  skip(ctx, track, start, end) {
    const kv = this.resizeKv(track.kv, track.kvLength, start, end);
    track.kv.release();
    track.kv = kv;
    const tensors = {};
    for (const [past] of this.stateNames) {
      tensors[past] = past.startsWith("past_conv") ? this.zeroState.tensors[past] : track.state.tensors[past];
    }
    // The new set borrows the recurrent state, so it takes over the track's reference.
    track.state = new Shared(tensors, { parents: [track.state] });
    track.reserve(end);
    track.visibility.fill(0, start, end);
    track.kvLength = end;
    // Every snapshot position crossed holds this same state.
    for (let pos = this.nextBoundary(start); pos <= end; pos += this.snapshotEvery) {
      this.snapshot(track, pos);
    }
    this.stats.skipped += end - start;
    this.onActivity?.({
      phase: "skip",
      job: typeof ctx.job === "symbol" ? null : ctx.job,
      track: track.id,
      start,
      count: end - start,
      length: ctx.tokens.length,
      ms: 0,
    });
  }

  /** One run of the graph over tokens [start, start + count) after a cache of `start`
   * positions, with `visible` giving each position's key mask; commits the outputs to
   * `track` and returns the logits and attention when `read` is set. */
  async run(ctx, track, start, count, visible, read, keep = 1) {
    const { ort } = this;
    const total = start + count;
    const ids = new BigInt64Array(count);
    for (let i = 0; i < count; i++) ids[i] = BigInt(ctx.tokens[start + i]);
    const positions = new BigInt64Array(3 * count);
    for (let r = 0; r < 3; r++) for (let i = 0; i < count; i++) positions[r * count + i] = BigInt(start + i);
    const phase = read && count === 1 ? "decode" : "prefill";
    const activity = {
      phase,
      job: typeof ctx.job === "symbol" ? null : ctx.job,
      track: track.id,
      start,
      count,
      length: ctx.tokens.length,
      hidden: total - visible.subarray(0, total).reduce((a, b) => a + b, 0),
    };
    this.onActivity?.(activity);
    const began = performance.now();
    const { inputs_embeds } = await this.embed.run({ input_ids: new ort.Tensor("int64", ids, [1, count]) });
    const feed = {
      inputs_embeds,
      key_mask: new ort.Tensor("bool", visible.slice(0, total), [1, total]),
      position_ids: new ort.Tensor("int64", positions, [3, 1, count]),
      num_logits_to_keep: new ort.Tensor("int64", new BigInt64Array([BigInt(keep)]), []),
      ...track.kv.tensors,
      ...track.state.tensors,
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
    track.kv.release();
    track.kv = new Shared(kv);
    track.state.release();
    track.state = new Shared(state);
    track.reserve(total);
    track.visibility.set(visible.subarray(start, total), start);
    track.kvLength = total;
    if (total % this.snapshotEvery === 0) this.snapshot(track);
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

  snapshot(track, pos = track.kvLength) {
    track.snapshots.push({ pos, state: track.state.retain() });
    if (track.snapshots.length <= this.maxSnapshots) return;
    // Drop the snapshot whose neighbours are closest, never the latest.
    let drop = 0;
    let gap = Infinity;
    for (let i = 0; i < track.snapshots.length - 1; i++) {
      const before = i === 0 ? 0 : track.snapshots[i - 1].pos;
      const span = track.snapshots[i + 1].pos - before;
      if (span < gap) {
        gap = span;
        drop = i;
      }
    }
    track.snapshots.splice(drop, 1)[0].state.release();
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
