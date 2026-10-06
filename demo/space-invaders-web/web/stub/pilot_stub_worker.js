// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The Space Invaders stub model: the model-side twin of
// zeos_space_invaders_web/fake_worker.py (FakePilotWorker), kept to the same behaviour
// by tests/test_stub_parity.py. It is served over coop-count-web's worker channel as the
// real model is, so begin/poll/cancel come from the channel and this file only has to
// honour `shouldStop` and `maxChunk` in `decodeStep`.
//
// What it has, as the Python twin has it:
//   * a fixed vocabulary: the five reserved tokens, then every word of the pilot's
//     commands, of the prompt arm's replies and of the ChatML headers, with and without
//     a leading space, sorted;
//   * a whitespace tokenizer (ASCII whitespace only), unknown words as <unk>;
//   * a script per kind of context, read from the context id "<job>:<descriptor>": a
//     "prompt" context answers the open assistant turn with the next reply (counted over
//     all prompt contexts), a word per
//     token already in the turn, then <|im_end|> (and says <|im_end|> with no assistant
//     turn open); any other context is a pilot and says `write stdout <move>;` for the
//     next move, a word per step (`reads: true` adds `read stdin;` after each), dropping
//     the rest of a command whose next word the step's allowedTokens refuses;
//   * latency: `positionMs` per pending position, filled in runs of at most `maxChunk`
//     with `shouldStop()` checked before each, then `stepMs` for the decode after one
//     more check. A stop returns {cancelled: true, resident, stats} with the positions
//     filled so far resident.
//
// A step chooses a token without changing the context's tokens; the script moves on
// only when the step finishes. A script with no imports or exports, so a module worker
// and Node can both import it for its effect: it defines globalThis.createPilotStubWorker.

(function () {
  "use strict";

  const SPECIAL = ["<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>"];
  const PAD_ID = 0;
  const EOS_ID = 1;
  const UNK_ID = 2;
  const IM_START_ID = 3;
  const IM_END_ID = 4;
  const CONTROL_IDS = [IM_START_ID, IM_END_ID];
  const FRAMING_WORDS = ["system", "user", "assistant"];
  const PROMPT_DESCRIPTOR = "prompt";
  const WORKER_CHUNK = 2048;
  const DEFAULT_MOVES = ["left", "right", "shoot"];
  const WORD = /[ \t\n\r\f\v]*[^ \t\n\r\f\v]+/g;
  const LEADING = /^[ \t\n\r\f\v]+/;
  const SPLIT = /[ \t\n\r\f\v]+/;

  // zeos.machine.seat.words_of: every word but a job's first carries a leading space.
  function wordsOf(command, terminator, lead) {
    const words = command.split(SPLIT).filter((w) => w.length > 0);
    if (words.length === 0) throw new Error("a command needs at least one word");
    if (!words[words.length - 1].endsWith(terminator)) words[words.length - 1] += terminator;
    return words.map((w, i) => (i === 0 && !lead ? w : " " + w));
  }

  function fillChunks(positions, maxChunk) {
    if (!(maxChunk >= 1)) throw new RangeError(`maxChunk is ${maxChunk}; it must be >= 1`);
    const out = [];
    for (let left = positions; left > 0; left -= maxChunk) out.push(Math.min(left, maxChunk));
    return out;
  }

  function byCodeUnit(a, b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  function sleep(ms) {
    return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();
  }

  class PilotStubWorker {
    constructor(options = {}) {
      this.moves = options.moves ?? DEFAULT_MOVES;
      this.replies = options.replies ?? this.moves;
      if (this.moves.length === 0) throw new Error("a pilot script needs at least one move");
      if (this.replies.length === 0) throw new Error("a prompt script needs at least one reply");
      for (const text of [...this.moves, ...this.replies]) {
        if (/[^\x00-\x7f]/.test(text)) throw new Error(`script text ${JSON.stringify(text)} is not ASCII`);
      }
      this.stepMs = options.stepMs ?? 0;
      this.positionMs = options.positionMs ?? 0;
      this.reads = options.reads ?? false;
      this.blockSize = options.blockSize ?? 16;
      this.terminator = options.terminator ?? ";";
      this.chunk = options.chunk ?? WORKER_CHUNK;

      const words = new Set(FRAMING_WORDS);
      const commands = this.moves.length * (this.reads ? 2 : 1);
      for (let n = 0; n < commands; n++) {
        for (const w of wordsOf(this.command(n), this.terminator, false)) words.add(w.trim());
      }
      for (const reply of this.replies) {
        for (const w of reply.split(SPLIT)) if (w.length > 0) words.add(w);
      }
      const all = new Set(words);
      for (const w of words) all.add(" " + w);
      this.vocab = SPECIAL.concat([...all].sort(byCodeUnit));
      this.ids = new Map(this.vocab.map((piece, i) => [piece, i]));
      this.contexts = new Map();
      // Assistant turns answered over every prompt context, so forks of one prefix take turns.
      this.promptTurns = 0;
      // What serveChannel (frames.js) reads off a worker: `pieces` walks
      // meta.tokenizerSize ids, and `backend` names the execution provider.
      this.meta = { tokenizerSize: this.vocab.length, vocabSize: this.vocab.length };
      this.backend = "stub";
    }

    // -- the script ---------------------------------------------------------

    command(n) {
      if (this.reads) {
        return n % 2 ? "read stdin" : `write stdout ${this.moves[Math.floor(n / 2) % this.moves.length]}`;
      }
      return `write stdout ${this.moves[n % this.moves.length]}`;
    }

    reply(n) {
      const words = this.replies[n % this.replies.length].split(SPLIT).filter((w) => w.length > 0);
      return words.map((w) => " " + w).concat([SPECIAL[IM_END_ID]]);
    }

    choose(jobId, ctx, allowed) {
      const ok = (piece) => allowed === null || allowed === undefined || Boolean(allowed[this.ids.get(piece)]);
      if (ctx.descriptor === PROMPT_DESCRIPTOR) return this.chooseReply(jobId, ctx, ok);
      let pending = ctx.pending.slice();
      let issued = ctx.issued;
      if (pending.length > 0 && !ok(pending[0])) pending = [];
      if (pending.length === 0) {
        pending = wordsOf(this.command(issued), this.terminator, ctx.spoken);
        issued += 1;
      }
      const piece = pending[0];
      if (!ok(piece)) throw new Error(`${jobId}: the token mask refuses the script's next word ${JSON.stringify(piece)}`);
      return { tokenId: this.ids.get(piece), pending: pending.slice(1), issued, replyAt: ctx.replyAt, newTurn: false };
    }

    chooseReply(jobId, ctx, ok) {
      const header = this.ids.get("assistant");
      let at = -1;
      for (let i = ctx.ids.length - 2; i >= 0; i--) {
        if (ctx.ids[i] === IM_START_ID && ctx.ids[i + 1] === header) {
          at = i;
          break;
        }
      }
      const newTurn = at >= 0 && at !== ctx.replyAt;
      let piece;
      let issued = ctx.issued;
      if (at < 0) {
        piece = SPECIAL[IM_END_ID];
      } else {
        issued = newTurn ? this.promptTurns : ctx.issued;
        const words = this.reply(issued);
        piece = words[Math.min(ctx.ids.length - at - 2, words.length - 1)];
      }
      if (!ok(piece)) throw new Error(`${jobId}: the token mask refuses the script's next word ${JSON.stringify(piece)}`);
      return { tokenId: this.ids.get(piece), pending: [], issued, replyAt: at >= 0 ? at : ctx.replyAt, newTurn };
    }

    // -- identity -----------------------------------------------------------

    info() {
      return {
        blockSize: this.blockSize,
        padId: PAD_ID,
        controlIds: CONTROL_IDS,
        eosId: EOS_ID,
        vocabSize: this.vocab.length,
      };
    }

    tokenize(text) {
      const out = [];
      for (const match of text.matchAll(WORD)) {
        const run = match[0];
        const word = run.replace(LEADING, "");
        const piece = word.length < run.length ? " " + word : word;
        out.push(this.ids.has(piece) ? this.ids.get(piece) : UNK_ID);
      }
      return Int32Array.from(out);
    }

    piece(tokenId) {
      if (!(tokenId >= 0 && tokenId < this.vocab.length)) {
        throw new RangeError(`token id ${tokenId} is outside a vocabulary of ${this.vocab.length}`);
      }
      return this.vocab[tokenId];
    }

    // -- contexts -----------------------------------------------------------

    ctx(jobId) {
      const ctx = this.contexts.get(jobId);
      if (ctx === undefined) throw new Error(`no context ${jobId}`);
      return ctx;
    }

    fresh(jobId) {
      if (this.contexts.has(jobId)) throw new Error(`context ${jobId} already exists`);
      const at = jobId.indexOf(":");
      const descriptor = at < 0 ? "" : jobId.slice(at + 1);
      return { descriptor, ids: [], kv: 0, issued: 0, replyAt: -1, pending: [], spoken: false };
    }

    createContext(jobId) {
      this.contexts.set(jobId, this.fresh(jobId));
    }

    destroyContext(jobId) {
      this.ctx(jobId);
      this.contexts.delete(jobId);
    }

    length(jobId) {
      return this.ctx(jobId).ids.length;
    }

    append(jobId, ids) {
      const ctx = this.ctx(jobId);
      for (const tokenId of ids) {
        this.piece(tokenId);
        ctx.ids.push(tokenId);
      }
    }

    truncate(jobId, n) {
      const ctx = this.ctx(jobId);
      if (!(n >= 0 && n <= ctx.ids.length)) {
        throw new RangeError(`truncate at ${n} outside [0, ${ctx.ids.length}]`);
      }
      ctx.ids.length = n;
      ctx.kv = Math.min(ctx.kv, n);
      if (ctx.replyAt >= n) ctx.replyAt = -1;
    }

    fork(parentId, childId) {
      const parent = this.ctx(parentId);
      const child = this.fresh(childId);
      child.ids = parent.ids.slice();
      child.kv = parent.kv;
      child.spoken = parent.spoken;
      this.contexts.set(childId, child);
    }

    // -- decoding -----------------------------------------------------------

    async decodeStep(jobId, opts) {
      const ctx = this.ctx(jobId);
      if (ctx.ids.length === 0) throw new Error(`decodeStep on ${jobId}, which holds no tokens`);
      const blocks = opts.allowedBlocks ?? null;
      const expected = Math.ceil(ctx.ids.length / this.blockSize);
      if (blocks !== null && blocks.length !== expected) {
        throw new Error(`allowedBlocks has ${blocks.length} entries for ${expected} blocks`);
      }
      const allowed = opts.allowedTokens ?? null;
      if (allowed !== null && allowed.length !== this.vocab.length) {
        throw new Error(`allowedTokens has ${allowed.length} entries for ${this.vocab.length} ids`);
      }
      const shouldStop = opts.shouldStop ?? (() => false);
      const choice = this.choose(jobId, ctx, allowed);
      const chunks = fillChunks(ctx.ids.length - ctx.kv, opts.maxChunk ?? this.chunk);
      const stats = { positions: 0, chunks: 0, fillMs: 0 };
      const started = Date.now();
      for (const chunk of chunks) {
        if (shouldStop()) return { cancelled: true, resident: ctx.kv, stats };
        await sleep(chunk * this.positionMs);
        ctx.kv += chunk;
        stats.positions += chunk;
        stats.chunks += 1;
        stats.fillMs = Date.now() - started;
      }
      if (shouldStop()) return { cancelled: true, resident: ctx.kv, stats };
      await sleep(this.stepMs);
      ctx.pending = choice.pending;
      ctx.issued = choice.issued;
      ctx.replyAt = choice.replyAt;
      if (choice.newTurn) this.promptTurns += 1;
      ctx.spoken = true;
      return { tokenId: choice.tokenId, attention: null, resident: ctx.kv, stats };
    }
  }

  globalThis.createPilotStubWorker = (options) => new PilotStubWorker(options);
})();
