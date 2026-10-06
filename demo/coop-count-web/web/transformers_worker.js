// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model half of a ZEOS machine backend: a Qwen language model exported as one ONNX
 * graph (see `export/export_model.py`), driven by ONNX Runtime, with the tokenizer
 * Transformers.js uses.
 *
 * This module owns token ids and caches and nothing else. Words, segments, rings and the
 * syscall grammar belong to the Python side (`JsMachine`), which calls the methods below.
 * It imports nothing: ONNX Runtime and the tokenizer class are handed in.
 *
 * `append` and `decodeStep` return promises, because `InferenceSession.run` does. The
 * synchronous `ZeosModelWorker` interface the Python side calls is `SyncModelWorker` in
 * `model_channel.js`, which blocks on a thread running this class.
 *
 * Context bookkeeping: a job's last token is *pending*, with no cache behind it, until a
 * decode step. `append` prefills everything before it, and `decodeStep` feeds it through
 * the graph with the step's allowed mask and reads the logits and the attention. A step
 * does not append the id it chooses -- `JsMachine` appends it before the next step, which
 * then finds that id pending -- so every decode step is exactly one forward pass of one
 * token.
 *
 * The cache has three parts (`meta.json` gives their shapes): the softmax layers' keys and
 * values, one entry per position; and, for a hybrid model such as Qwen3.5, each Gated
 * DeltaNet layer's recurrent state and convolution window, a fixed size whatever the
 * length. The state cannot be cut at a position, so a context keeps snapshots of it every
 * `SNAPSHOT_EVERY` positions, and a cut (`truncate`, or a step whose mask disagrees with
 * the one the state was built under) goes back to the latest snapshot at or before the
 * position and runs the tokens after it again. `visibility` records, per cached position,
 * whether the state saw it, so the state always matches the mask of the step reading it.
 *
 * A decode step can also be told to stop (`shouldStop`, asked before every run of the
 * graph, the final decode included) and given a smaller run length (`maxChunk`); a step
 * that stops returns `{cancelled: true, resident, stats}` with what it ran still cached,
 * and the next step resumes from there. `append`'s prefill is not interruptible.
 */

/** Positions between two snapshots of a context's recurrent state: the most a cut ever
 * re-runs. A snapshot of Qwen3.5-2B's state is 20 MB. */
export const SNAPSHOT_EVERY = 256;

/** Tokenise text as plain text: no BOS, and a literal that spells a special token stays
 * text, so nothing foreign can become framing. This is the tokenizer's own pipeline
 * (normaliser, pre-tokeniser, BPE) with the added-token splitter left out. */
export function encodePlain(tokenizer, text) {
  if (text.length === 0) return new Int32Array(0);
  const normalised = tokenizer.normalizer ? tokenizer.normalizer(text) : text;
  if (normalised.length === 0) return new Int32Array(0);
  const words = tokenizer.pre_tokenizer
    ? tokenizer.pre_tokenizer(normalised, { section_index: 0 })
    : [normalised];
  const pieces = tokenizer.model(words);
  const ids = new Int32Array(pieces.length);
  for (let i = 0; i < pieces.length; i++) {
    const id = tokenizer.token_to_id(pieces[i]);
    if (id === undefined) throw new Error(`tokenizer produced unknown piece ${pieces[i]}`);
    ids[i] = id;
  }
  return ids;
}

/** The seeded sampler `opts.sample` asks for, as `chat_machine.sample_index` defines it:
 * rank the allowed ids below `limit` by logit, highest first and the lower id on a tie,
 * keep the first `topK`, weight each by `exp((logit - best) / temperature)`, and take the
 * first whose running total exceeds `u` times the sum (the last if rounding leaves none).
 * `u` is drawn by the caller from a seeded generator, so this function holds no state. */
export function sampleToken(logits, allowedTokens, limit, { temperature, topK, u }) {
  if (!(temperature > 0) || !(topK >= 1) || !(u >= 0 && u < 1)) {
    throw new RangeError(`sample needs temperature > 0, topK >= 1 and 0 <= u < 1`);
  }
  // The topK best so far, best first; small, so insertion is cheaper than a heap.
  const ranked = [];
  for (let id = 0; id < limit; id++) {
    if (allowedTokens !== null && !allowedTokens[id]) continue;
    const value = logits[id];
    if (ranked.length === topK && !(value > logits[ranked[topK - 1]])) continue;
    let at = ranked.length;
    while (at > 0 && value > logits[ranked[at - 1]]) at--;
    ranked.splice(at, 0, id);
    if (ranked.length > topK) ranked.pop();
  }
  if (ranked.length === 0) throw new Error("allowedTokens permits no token");
  const best = logits[ranked[0]];
  const weights = ranked.map((id) => Math.exp((logits[id] - best) / temperature));
  const threshold = u * weights.reduce((a, b) => a + b, 0);
  let running = 0;
  for (let k = 0; k < ranked.length; k++) {
    running += weights[k];
    if (running > threshold) return ranked[k];
  }
  return ranked[ranked.length - 1];
}

class Context {
  constructor(worker) {
    this.stride = worker.stride;
    /** The context id the interface gave this cache, for `onActivity`. */
    this.job = null;
    this.tokens = [];
    this.kv = new Float32Array(0);
    /** Positions with a cache behind them. */
    this.kvLength = 0;
    /** The recurrent state and convolution window after `kvLength` positions. Graph
     * outputs are fresh arrays and never written to, so contexts share them freely. */
    this.state = worker.zeroState;
    this.conv = worker.zeroConv;
    /** Per cached position, 1 if the state took it in, 0 if it was skipped as hidden. */
    this.visibility = new Uint8Array(0);
    /** `{pos, state, conv}` at multiples of SNAPSHOT_EVERY, ascending; position 0 is the
     * zero state and is not stored. */
    this.snapshots = [];
    /** The state before the latest decode step's position, so a step repeated with
     * nothing appended needs no re-run. */
    this.previous = null;
    /** The allowed mask of the most recent decode step, per position, applied to old
     * positions when new tokens are prefilled. Null means every position. */
    this.mask = null;
  }

  reserve(positions) {
    const needed = positions * this.stride;
    if (needed > this.kv.length) {
      const grown = new Float32Array(Math.max(needed, this.kv.length * 2, 256 * this.stride));
      grown.set(this.kv.subarray(0, this.kvLength * this.stride));
      this.kv = grown;
    }
    if (positions > this.visibility.length) {
      const grown = new Uint8Array(Math.max(positions, this.visibility.length * 2, 256));
      grown.set(this.visibility.subarray(0, this.kvLength));
      this.visibility = grown;
    }
  }

  /** Commit a run's outputs for `positions` new positions seen as `visible`. */
  push(result, positions, visible) {
    this.reserve(this.kvLength + positions);
    this.kv.set(result.new_kv.data, this.kvLength * this.stride);
    this.visibility.set(visible, this.kvLength);
    this.kvLength += positions;
    this.state = result.state_out.data;
    this.conv = result.conv_out.data;
    if (this.kvLength % SNAPSHOT_EVERY === 0) {
      this.snapshots.push({ pos: this.kvLength, state: this.state, conv: this.conv });
    }
  }

  copy() {
    const other = Object.assign(Object.create(Context.prototype), this);
    other.tokens = this.tokens.slice();
    other.kv = this.kv.slice(0, Math.max(this.kvLength, 1) * this.stride);
    other.visibility = this.visibility.slice(0, Math.max(this.kvLength, 1));
    other.snapshots = this.snapshots.slice();
    other.mask = this.mask === null ? null : this.mask.slice();
    return other;
  }
}

export class TransformersWorker {
  /**
   * @param {object} deps
   * @param {object} deps.ort ONNX Runtime Web's WebGPU build.
   * @param {object} deps.tokenizer a `Tokenizer` from `@huggingface/tokenizers`.
   * @param {object} deps.meta the export's `meta.json`.
   * @param {object} deps.session an `InferenceSession` over `model.onnx`.
   * @param {string} deps.backend which execution provider the session runs on.
   * @param {(activity: object) => void} [deps.onActivity] told before and after every run
   *   of the graph: `{phase, job, start, count, length}` and then the same with `ms`.
   *   `phase` is `prefill` or `decode`; `start` and `count` are the positions the run
   *   covers and `length` the context's token count. Nothing by default.
   */
  constructor({ ort, tokenizer, meta, session, backend, onActivity = null }) {
    if (meta.blockSize !== 1) throw new Error(`export has blockSize ${meta.blockSize}; expected 1`);
    this.ort = ort;
    this.tokenizer = tokenizer;
    this.meta = meta;
    this.session = session;
    this.backend = backend;
    this.onActivity = onActivity;
    const size = (shape) => shape.reduce((a, b) => a * b, 1);
    this.kvShape = meta.kvShape;
    this.stride = size(meta.kvShape);
    this.zeroState = new Float32Array(size(meta.stateShape));
    this.zeroConv = new Float32Array(size(meta.convShape));
    /** The most new positions one run of the graph takes. */
    this.chunk = meta.maxChunk;
    this.contexts = new Map();
    this.pieces = new Map();
    /** Positions run again because of a cut or a changed mask, for diagnostics. */
    this.stats = { reruns: 0, rerunPositions: 0 };
  }

  /** Build a worker from the files an export wrote, on WebGPU. `read(name)` returns a
   * file's bytes as a Uint8Array (or a promise of them). */
  static async load({ ort, Tokenizer, read, sessionOptions = {}, onActivity = null }) {
    const backend = "webgpu";
    const decoder = new TextDecoder();
    const json = async (name) => JSON.parse(decoder.decode(await read(name)));
    const meta = await json("meta.json");
    const tokenizer = new Tokenizer(await json("tokenizer.json"), await json("tokenizer_config.json"));
    const options = {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
      externalData: [],
      ...sessionOptions,
    };
    // One weights file after another, so only one download is in flight.
    for (const path of meta.weights) options.externalData.push({ path, data: await read(path) });
    const session = await ort.InferenceSession.create(await read("model.onnx"), options);
    return new TransformersWorker({ ort, tokenizer, meta, session, backend, onActivity });
  }

  // -- the interface -------------------------------------------------------------

  /** `vocabSize` is the tokenizer's vocabulary: every id `piece` answers and the only ids
   * a step chooses. The logits are wider (the embedding is padded), but no id past the
   * tokenizer's has a piece, and `argmax` never picks one. */
  info() {
    return {
      blockSize: this.meta.blockSize,
      padId: this.meta.padId,
      controlIds: this.meta.controlIds.slice(),
      eosId: this.meta.eosId,
      vocabSize: this.meta.tokenizerSize,
    };
  }

  tokenize(text) {
    return encodePlain(this.tokenizer, text);
  }

  piece(tokenId) {
    let text = this.pieces.get(tokenId);
    if (text === undefined) {
      if (!(tokenId >= 0 && tokenId < this.meta.tokenizerSize)) {
        throw new RangeError(`token id ${tokenId} is outside the vocabulary`);
      }
      text = this.tokenizer.decode([tokenId], { skip_special_tokens: false });
      this.pieces.set(tokenId, text);
    }
    return text;
  }

  createContext(jobId) {
    const ctx = new Context(this);
    ctx.job = jobId;
    this.contexts.set(jobId, ctx);
  }

  destroyContext(jobId) {
    this.contexts.delete(jobId);
  }

  length(jobId) {
    return this.ctx(jobId).tokens.length;
  }

  async append(jobId, ids) {
    const ctx = this.ctx(jobId);
    for (const id of ids) ctx.tokens.push(id);
    await this.fill(ctx, ctx.tokens.length - 1, this.prefillMask(ctx));
  }

  truncate(jobId, n) {
    const ctx = this.ctx(jobId);
    if (!(n >= 0 && n <= ctx.tokens.length)) {
      throw new RangeError(`truncate to ${n} outside [0, ${ctx.tokens.length}]`);
    }
    ctx.tokens.length = n;
    this.rewind(ctx, Math.max(n - 1, 0));
  }

  fork(parentId, childId) {
    const ctx = this.ctx(parentId).copy();
    ctx.job = childId;
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
    if (maxChunk !== null && !(Number.isInteger(maxChunk) && maxChunk >= 1)) {
      throw new RangeError(`job ${jobId}: maxChunk ${maxChunk} is not a positive integer`);
    }
    const stats = { positions: 0, chunks: 0, fillMs: 0 };
    const chunk = maxChunk === null ? this.chunk : Math.min(maxChunk, this.chunk);
    // A second step with nothing appended in between computes the same position again.
    if (ctx.kvLength > n - 1) this.rewind(ctx, n - 1);
    const filled = await this.fill(ctx, n - 1, allowed, { chunk, shouldStop, stats });
    if (!filled || shouldStop?.()) return { cancelled: true, resident: ctx.kvLength, stats };

    const before = { pos: n - 1, state: ctx.state, conv: ctx.conv };
    const result = await this.run(ctx, n - 1, 1, allowed, "decode");
    stats.positions += 1;
    stats.chunks += 1;
    ctx.push(result, 1, allowed.subarray(n - 1, n));
    ctx.previous = before;
    ctx.mask = allowedBlocks === null ? null : allowed;

    const limit = Math.min(this.meta.tokenizerSize, result.logits.data.length);
    const tokenId =
      sample === null
        ? this.argmax(result.logits.data, allowedTokens)
        : sampleToken(result.logits.data, allowedTokens, limit, sample);
    return { tokenId, attention: Float32Array.from(result.attention.data), resident: ctx.kvLength, stats };
  }

  // -- outside the interface -------------------------------------------------------

  async release() {
    await this.session.release();
  }

  // -- internals -------------------------------------------------------------------

  ctx(jobId) {
    const ctx = this.contexts.get(jobId);
    if (ctx === undefined) throw new Error(`no context for job ${jobId}; createContext first`);
    return ctx;
  }

  /** What a prefill sees: the latest step's mask over the positions it covered, and
   * every position after them. */
  prefillMask(ctx) {
    const total = Math.max(ctx.tokens.length - 1, 0);
    const mask = new Uint8Array(total).fill(1);
    if (ctx.mask !== null) mask.set(ctx.mask.subarray(0, Math.min(ctx.mask.length, total)));
    return mask;
  }

  /** Cut the cache back to at most `target` positions: to the state before the latest
   * decode step if that is the position, otherwise to the latest snapshot at or before
   * it. The positions between are run again by the next `fill`. */
  rewind(ctx, target) {
    if (ctx.kvLength <= target) return;
    if (this.zeroState.length === 0) {
      // Softmax layers only: the cache is per position and is simply cut.
      ctx.kvLength = target;
      ctx.previous = null;
      return;
    }
    let pos = 0;
    let state = this.zeroState;
    let conv = this.zeroConv;
    if (ctx.previous !== null && ctx.previous.pos === target && ctx.kvLength === target + 1) {
      ({ pos, state, conv } = ctx.previous);
    } else {
      while (ctx.snapshots.length > 0 && ctx.snapshots[ctx.snapshots.length - 1].pos > target) {
        ctx.snapshots.pop();
      }
      const last = ctx.snapshots[ctx.snapshots.length - 1];
      if (last !== undefined) ({ pos, state, conv } = last);
      if (pos < target) {
        this.stats.reruns += 1;
        this.stats.rerunPositions += target - pos;
      }
    }
    ctx.snapshots = ctx.snapshots.filter((snap) => snap.pos <= pos);
    ctx.previous = null;
    ctx.kvLength = pos;
    ctx.state = state;
    ctx.conv = conv;
  }

  /** Make the cache cover positions [0, target) as `visible` (one entry per position,
   * at least `target` of them) sees them, running the graph over what is missing in runs
   * of at most `chunk`. False if `shouldStop` answered true before a run, with the runs
   * so far cached; `stats` collects the runs. */
  async fill(ctx, target, visible, { chunk = this.chunk, shouldStop = null, stats = null } = {}) {
    if (this.zeroState.length > 0) {
      let first = 0;
      while (first < ctx.kvLength && ctx.visibility[first] === visible[first]) first++;
      if (first < ctx.kvLength) this.rewind(ctx, first);
    }
    while (ctx.kvLength < target) {
      const start = ctx.kvLength;
      const boundary = (Math.floor(start / SNAPSHOT_EVERY) + 1) * SNAPSHOT_EVERY;
      const count = Math.min(chunk, target - start, boundary - start);
      if (shouldStop?.()) return false;
      const began = performance.now();
      const result = await this.run(ctx, start, count, visible);
      ctx.push(result, count, visible.subarray(start, start + count));
      if (stats !== null) {
        stats.positions += count;
        stats.chunks += 1;
        stats.fillMs += performance.now() - began;
      }
    }
    return true;
  }

  /** One run of the graph over tokens [start, start + count) after a cache of `start`
   * positions, with `visible` giving each position's key mask. */
  async run(ctx, start, count, visible, phase = "prefill") {
    const { ort, meta } = this;
    const ids = new BigInt64Array(count);
    for (let i = 0; i < count; i++) ids[i] = BigInt(ctx.tokens[start + i]);
    const activity = { phase, job: ctx.job ?? null, start, count, length: ctx.tokens.length };
    this.onActivity?.(activity);
    const began = performance.now();
    const result = await this.session.run({
      input_ids: new ort.Tensor("int64", ids, [count]),
      past_kv: new ort.Tensor("float32", ctx.kv.subarray(0, start * this.stride), [start, ...this.kvShape]),
      state: new ort.Tensor("float32", ctx.state, meta.stateShape),
      conv: new ort.Tensor("float32", ctx.conv, meta.convShape),
      key_mask: new ort.Tensor("bool", visible.slice(0, start + count), [start + count]),
    });
    this.onActivity?.({ ...activity, ms: performance.now() - began });
    return result;
  }

  /** Greedy choice among the allowed ids, lowest id on a tie. Ids past the tokenizer's
   * vocabulary are padding rows of the embedding and never chosen. */
  argmax(logits, allowedTokens) {
    const limit = Math.min(this.meta.tokenizerSize, logits.length);
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
}
