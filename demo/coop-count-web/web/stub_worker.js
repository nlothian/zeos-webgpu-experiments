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
      return { tape: this.tapes.get(descriptor) ?? [], ids: [], issued: 0, pending: [], spoken: false };
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
    }

    fork(parentId, childId) {
      const parent = this.ctx(parentId);
      const child = this.fresh(childId);
      child.ids = parent.ids.slice();
      child.spoken = parent.spoken;
      this.contexts.set(childId, child);
    }

    // -- decoding -----------------------------------------------------------

    decodeStep(jobId, opts) {
      const ctx = this.ctx(jobId);
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

      if (ctx.pending.length === 0) {
        if (ctx.issued >= ctx.tape.length) {
          throw new Error(
            `${jobId} asked for command ${ctx.issued + 1} of a tape with ${ctx.tape.length}`,
          );
        }
        ctx.pending = wordsOf(ctx.tape[ctx.issued], this.terminator, ctx.spoken);
        ctx.issued += 1;
      }
      const word = ctx.pending[0];
      const tokenId = this.ids.get(word);
      if (allowed !== null && !allowed[tokenId]) {
        throw new Error(`${jobId}: the token mask refuses the tape's next word ${JSON.stringify(word)}`);
      }
      ctx.pending.shift();
      ctx.spoken = true;
      return { tokenId, attention: null };
    }
  }

  globalThis.createStubWorker = (tapes, options) => new StubWorker(tapes, options);
})();
