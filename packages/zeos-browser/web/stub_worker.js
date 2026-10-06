// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// A ZeosModelWorker with no model: the JavaScript twin of
// zeos_coop_count_web/fake_worker.py, and kept to the same behaviour -- the Pyodide
// determinism test runs a case through JsMachine over each and requires the same
// journal bytes.
//
// It exists to exercise every method of the seam from Python under Pyodide. It has a
// fixed vocabulary (five reserved tokens, then every word of the tapes and of the
// ChatML turn headers, with and without a leading space, sorted), a whitespace
// tokenizer, one tape of commands per descriptor, and no attention. Each decode step
// gives the next word of the current command, split as zeos.machine.seat.words_of
// splits it, and refuses a word the token mask forbids.
//
// A script with no imports or exports, so the page's module worker and Node can both
// import it for its effect: it defines globalThis.createStubWorker.
//
// For the decode step that does not block (model_channel.js) it also pretends to have a
// cache: a context has `resident` positions "filled", and a step fills the rest in chunks
// of at most `maxChunk`, asking `shouldStop` before each chunk and before the final
// one-position decode. With `{stepMs, positionMs}` in the options each chunk waits
// `positionMs` per position and the final decode `stepMs` more, with real awaits, so a
// cancel can land between chunks. With both 0 (the default) and no `shouldStop`, a step
// is synchronous and does exactly what it did before.
//
// A real model's step is a function of the context, so a step whose result was dropped
// gives the same token when it is asked again. A tape is not, so a step that was begun
// (it has a `shouldStop`) leaves the tape where it was and records what it would advance
// to; the next step, a truncate or a fork of the context commits that only if the token
// it chose has since been appended.

(function () {
  "use strict";

  const SPECIAL = ["<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>"];
  const PAD_ID = 0;
  const EOS_ID = 1;
  const UNK_ID = 2;
  const CONTROL_IDS = [3, 4];
  const FRAMING_WORDS = ["user", "assistant"];
  // ASCII whitespace only, as in fake_worker.py: \s differs between the two languages.
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

  // Tapes are ASCII (checked in the constructor), where UTF-16 order is code-point order
  // and so matches Python's sorted() on str.
  function byCodeUnit(a, b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  class StubWorker {
    constructor(tapes, options = {}) {
      this.blockSize = options.blockSize ?? 16;
      this.stepMs = options.stepMs ?? 0;
      this.positionMs = options.positionMs ?? 0;
      this.terminator = options.terminator ?? ";";
      this.tapes = new Map(Object.entries(tapes));
      // Python and JavaScript split and sort non-ASCII text differently, so the twin
      // workers would build different vocabularies from it.
      for (const [name, commands] of this.tapes) {
        for (const command of commands) {
          if (/[^\x00-\x7f]/.test(command)) {
            throw new Error(`${name}: tape command ${JSON.stringify(command)} is not ASCII`);
          }
        }
      }
      const words = new Set(FRAMING_WORDS);
      for (const commands of this.tapes.values()) {
        for (const command of commands) {
          for (const w of wordsOf(command, this.terminator, false)) words.add(w.trim());
        }
      }
      const all = new Set(words);
      for (const w of words) all.add(" " + w);
      this.vocab = SPECIAL.concat([...all].sort(byCodeUnit));
      this.ids = new Map(this.vocab.map((piece, i) => [piece, i]));
      this.contexts = new Map();
    }

    // -- identity -----------------------------------------------------------

    /** What `frames.js`'s `pieces` reads, when the stub is served over a channel. */
    get meta() {
      return { tokenizerSize: this.vocab.length };
    }

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
      return {
        tape: this.tapes.get(descriptor) ?? [],
        ids: [],
        issued: 0,
        pending: [],
        spoken: false,
        resident: 0,
        uncommitted: null,
      };
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
      this.commit(ctx);
      ctx.ids.length = n;
      // The last token stays pending, as a model's cache does after a cut.
      ctx.resident = Math.min(ctx.resident, Math.max(n - 1, 0));
    }

    fork(parentId, childId) {
      const parent = this.ctx(parentId);
      this.commit(parent);
      const child = this.fresh(childId);
      child.ids = parent.ids.slice();
      child.spoken = parent.spoken;
      child.resident = parent.resident;
      this.contexts.set(childId, child);
    }

    // -- decoding -----------------------------------------------------------

    decodeStep(jobId, opts) {
      const ctx = this.ctx(jobId);
      this.commit(ctx);
      if (ctx.ids.length === 0) throw new Error(`decodeStep on ${jobId}, which holds no tokens`);
      const blocks = opts.allowedBlocks;
      const expected = Math.ceil(ctx.ids.length / this.blockSize);
      if (blocks !== null && blocks.length !== expected) {
        throw new Error(`allowedBlocks has ${blocks.length} entries for ${expected} blocks`);
      }
      const allowed = opts.allowedTokens;
      if (allowed !== null && allowed.length !== this.vocab.length) {
        throw new Error(`allowedTokens has ${allowed.length} entries for ${this.vocab.length} ids`);
      }
      // A tape has one choice per step, so there is nothing to sample; the options are
      // checked so a caller that sends malformed ones finds out here too.
      const sample = opts.sample ?? null;
      if (sample !== null && !(sample.temperature > 0 && sample.topK >= 1 && sample.u >= 0 && sample.u < 1)) {
        throw new RangeError("sample needs temperature > 0, topK >= 1 and 0 <= u < 1");
      }

      const shouldStop = opts.shouldStop ?? null;
      const maxChunk = opts.maxChunk ?? Infinity;
      if (maxChunk !== Infinity && !(Number.isInteger(maxChunk) && maxChunk >= 1)) {
        throw new RangeError(`maxChunk ${maxChunk} is not a positive integer`);
      }
      const n = ctx.ids.length;
      // A step repeated with nothing appended decodes its position again.
      ctx.resident = Math.min(ctx.resident, n - 1);
      const stats = { positions: 0, chunks: 0, fillMs: 0 };
      const runs = [];
      for (let start = ctx.resident; start < n - 1; start += maxChunk) runs.push(Math.min(maxChunk, n - 1 - start));
      runs.push(1);
      const choose = () => {
        const tokenId = this.choose(ctx, jobId, allowed, shouldStop !== null);
        return { tokenId, attention: null, resident: ctx.resident, stats };
      };
      const stopped = () => ({ cancelled: true, resident: ctx.resident, stats });
      if (this.stepMs === 0 && this.positionMs === 0) {
        if (shouldStop?.()) return stopped();
        stats.positions = n - ctx.resident;
        stats.chunks = runs.length;
        ctx.resident = n;
        return choose();
      }
      return (async () => {
        for (let i = 0; i < runs.length; i++) {
          if (shouldStop?.()) return stopped();
          const last = i === runs.length - 1;
          const began = Date.now();
          await sleep(runs[i] * this.positionMs + (last ? this.stepMs : 0));
          if (!last) stats.fillMs += Date.now() - began;
          stats.positions += runs[i];
          stats.chunks += 1;
          ctx.resident += runs[i];
        }
        return choose();
      })();
    }

    /** The tape's next word, checked against the token mask; advances the tape, or, for
     * a begun step, records the advance for `commit`. */
    choose(ctx, jobId, allowed, deferred) {
      let pending = ctx.pending;
      let issued = ctx.issued;
      if (pending.length === 0) {
        if (issued >= ctx.tape.length) {
          throw new Error(`${jobId} asked for command ${issued + 1} of a tape with ${ctx.tape.length}`);
        }
        pending = wordsOf(ctx.tape[issued], this.terminator, ctx.spoken);
        issued += 1;
      }
      const word = pending[0];
      const tokenId = this.ids.get(word);
      if (allowed !== null && !allowed[tokenId]) {
        // As before begun steps existed: a synchronous step has drawn the command already.
        if (!deferred) Object.assign(ctx, { pending, issued });
        throw new Error(`${jobId}: the token mask refuses the tape's next word ${JSON.stringify(word)}`);
      }
      const advance = { at: ctx.ids.length, tokenId, pending: pending.slice(1), issued };
      if (deferred) ctx.uncommitted = advance;
      else this.advance(ctx, advance);
      return tokenId;
    }

    /** Commit a begun step's advance if its token was appended after it; drop it if not. */
    commit(ctx) {
      const step = ctx.uncommitted;
      if (step === null) return;
      ctx.uncommitted = null;
      if (ctx.ids.length > step.at && ctx.ids[step.at] === step.tokenId) this.advance(ctx, step);
    }

    advance(ctx, { pending, issued }) {
      ctx.pending = pending;
      ctx.issued = issued;
      ctx.spoken = true;
    }
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  globalThis.createStubWorker = (tapes, options) => new StubWorker(tapes, options);
})();
