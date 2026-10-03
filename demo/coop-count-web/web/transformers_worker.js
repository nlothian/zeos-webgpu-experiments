// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model half of a ZEOS machine backend: a Qwen2 language model exported as two ONNX
 * graphs (see `export/export_model.py`), driven by ONNX Runtime, with the tokenizer
 * Transformers.js uses.
 *
 * This module owns token ids and KV caches and nothing else. Words, segments, rings and
 * the syscall grammar belong to the Python side (`JsMachine`), which calls the methods
 * below. It imports nothing: ONNX Runtime and the tokenizer class are handed in, so the
 * same file runs in a browser worker and under Node.
 *
 * `append` and `decodeStep` return promises, because `InferenceSession.run` does. The
 * synchronous `ZeosModelWorker` interface the Python side calls is `SyncModelWorker` in
 * `sync_client.js`, which blocks on a thread running this class.
 *
 * Context bookkeeping: a job's last token is always *pending*, with no KV behind it.
 * `append` prefills everything before it, and `decodeStep` feeds it through the decode
 * graph, which is the one place the allowed-block mask is applied and attention is
 * measured. So every decode step is exactly one forward pass of one token.
 */

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

class Context {
  constructor(stride) {
    this.stride = stride;
    this.tokens = [];
    this.kv = new Float32Array(0);
    /** Positions with KV behind them. */
    this.kvLength = 0;
    /** The allowed-block mask of the most recent decode step, applied to old keys when
     * new tokens are prefilled. Null means every block. */
    this.mask = null;
  }

  reserve(positions) {
    const needed = positions * this.stride;
    if (needed <= this.kv.length) return;
    const grown = new Float32Array(Math.max(needed, this.kv.length * 2, 256 * this.stride));
    grown.set(this.kv.subarray(0, this.kvLength * this.stride));
    this.kv = grown;
  }

  pushKv(values, positions) {
    this.reserve(this.kvLength + positions);
    this.kv.set(values, this.kvLength * this.stride);
    this.kvLength += positions;
  }

  copy() {
    const other = new Context(this.stride);
    other.tokens = this.tokens.slice();
    other.kv = this.kv.slice(0, Math.max(this.kvLength, 1) * this.stride);
    other.kvLength = this.kvLength;
    other.mask = this.mask === null ? null : this.mask.slice();
    return other;
  }
}

export class TransformersWorker {
  /**
   * @param {object} deps
   * @param {object} deps.ort ONNX Runtime (`onnxruntime-web` or `onnxruntime-node`).
   * @param {object} deps.tokenizer a `Tokenizer` from `@huggingface/tokenizers`.
   * @param {object} deps.meta the export's `meta.json`.
   * @param {object} deps.prefill an `InferenceSession` over `prefill.onnx`.
   * @param {object} deps.decode an `InferenceSession` over `decode.onnx`.
   * @param {string} deps.backend which execution provider the sessions run on.
   */
  constructor({ ort, tokenizer, meta, prefill, decode, backend }) {
    this.ort = ort;
    this.tokenizer = tokenizer;
    this.meta = meta;
    this.prefillSession = prefill;
    this.decodeSession = decode;
    this.backend = backend;
    this.kvShape = [meta.numLayers, 2, meta.numKvHeads, meta.headDim];
    this.stride = this.kvShape.reduce((a, b) => a * b, 1);
    /** Prefill runs in chunks so the score matrix stays small. */
    this.chunk = 256;
    this.contexts = new Map();
    this.pieces = new Map();
  }

  /** Build a worker from the files an export wrote. `read(name)` returns a file's bytes
   * as a Uint8Array (or a promise of them); `backend` is an execution provider name. */
  static async load({ ort, Tokenizer, read, backend = "wasm", sessionOptions = {} }) {
    const decoder = new TextDecoder();
    const json = async (name) => JSON.parse(decoder.decode(await read(name)));
    const meta = await json("meta.json");
    const tokenizer = new Tokenizer(await json("tokenizer.json"), await json("tokenizer_config.json"));
    const weights = await read("weights.bin");
    const options = {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
      externalData: [{ path: "weights.bin", data: weights }],
      ...sessionOptions,
    };
    const prefill = await ort.InferenceSession.create(await read("prefill.onnx"), options);
    const decode = await ort.InferenceSession.create(await read("decode.onnx"), options);
    return new TransformersWorker({ ort, tokenizer, meta, prefill, decode, backend });
  }

  // -- the interface -------------------------------------------------------------

  info() {
    return {
      blockSize: this.meta.blockSize,
      padId: this.meta.padId,
      controlIds: this.meta.controlIds.slice(),
      eosId: this.meta.eosId,
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
    this.contexts.set(jobId, new Context(this.stride));
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
    await this.prefill(ctx);
  }

  truncate(jobId, n) {
    const ctx = this.ctx(jobId);
    if (!(n >= 0 && n <= ctx.tokens.length)) {
      throw new RangeError(`truncate to ${n} outside [0, ${ctx.tokens.length}]`);
    }
    ctx.tokens.length = n;
    ctx.kvLength = Math.min(ctx.kvLength, Math.max(n - 1, 0));
  }

  fork(parentId, childId) {
    this.contexts.set(childId, this.ctx(parentId).copy());
  }

  async decodeStep(jobId, { allowedBlocks = null, allowedTokens = null } = {}) {
    const ctx = this.ctx(jobId);
    const n = ctx.tokens.length;
    if (n === 0) throw new Error(`job ${jobId}: cannot decode an empty context`);
    await this.prefill(ctx);

    const blockSize = this.meta.blockSize;
    const blocks = Math.ceil(n / blockSize);
    const allowed = new Uint8Array(blocks);
    if (allowedBlocks === null) {
      allowed.fill(1);
    } else {
      if (allowedBlocks.length < blocks) {
        throw new RangeError(
          `job ${jobId}: allowedBlocks covers ${allowedBlocks.length} blocks of ${blocks}`,
        );
      }
      for (let b = 0; b < blocks; b++) allowed[b] = allowedBlocks[b] ? 1 : 0;
      if (!allowed.some((v) => v)) {
        throw new Error(`job ${jobId}: the mask hides every block, so nothing can be attended`);
      }
    }

    const { ort } = this;
    const result = await this.decodeSession.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(ctx.tokens[n - 1])]), [1]),
      past_kv: this.pastTensor(ctx, n - 1),
      allowed_blocks: new ort.Tensor("bool", allowed, [blocks]),
    });
    const logits = result.logits.data;
    ctx.pushKv(result.new_kv.data, 1);
    ctx.mask = allowedBlocks === null ? null : allowed;

    const tokenId = this.argmax(logits, allowedTokens);
    ctx.tokens.push(tokenId);
    return { tokenId, attention: Float32Array.from(result.block_attention.data) };
  }

  // -- outside the interface -------------------------------------------------------

  /** The token ids a job holds, for tests and the page. */
  tokens(jobId) {
    return Int32Array.from(this.ctx(jobId).tokens);
  }

  async release() {
    await this.prefillSession.release();
    await this.decodeSession.release();
  }

  // -- internals -------------------------------------------------------------------

  ctx(jobId) {
    const ctx = this.contexts.get(jobId);
    if (ctx === undefined) throw new Error(`no context for job ${jobId}; createContext first`);
    return ctx;
  }

  pastTensor(ctx, positions) {
    return new this.ort.Tensor(
      "float32",
      ctx.kv.subarray(0, positions * this.stride),
      [positions, ...this.kvShape],
    );
  }

  /** Run the prefill graph over everything but the pending last token. */
  async prefill(ctx) {
    const { ort } = this;
    const target = ctx.tokens.length - 1;
    while (ctx.kvLength < target) {
      const start = ctx.kvLength;
      const count = Math.min(this.chunk, target - start);
      const total = start + count;
      const keyMask = new Uint8Array(total).fill(1);
      if (ctx.mask !== null) {
        const blockSize = this.meta.blockSize;
        const covered = Math.min(start, ctx.mask.length * blockSize);
        for (let p = 0; p < covered; p++) keyMask[p] = ctx.mask[Math.floor(p / blockSize)];
      }
      const ids = new BigInt64Array(count);
      for (let i = 0; i < count; i++) ids[i] = BigInt(ctx.tokens[start + i]);
      const result = await this.prefillSession.run({
        input_ids: new ort.Tensor("int64", ids, [count]),
        past_kv: this.pastTensor(ctx, start),
        key_mask: new ort.Tensor("bool", keyMask, [total]),
      });
      ctx.pushKv(result.new_kv.data, count);
    }
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
