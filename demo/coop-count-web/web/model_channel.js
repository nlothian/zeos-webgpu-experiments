// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The synchronous `ZeosModelWorker` the Python side calls, over a model that answers
 * asynchronously on another thread.
 *
 * `JsMachine.decode` is synchronous, the kernel loop that calls it is synchronous and
 * stays so, and ONNX Runtime's `session.run` returns a promise. The two meet here: the
 * model runs in its own thread (a Web Worker in the page, a `worker_threads` Worker under
 * Node), and `SyncModelWorker`, on the thread running Pyodide, posts each call to it and
 * then blocks in `Atomics.wait` on a SharedArrayBuffer until the reply has been written
 * into that buffer. The model thread is never blocked, so it receives the message, awaits
 * the session and notifies. `Atomics.wait` is allowed in a Web Worker and on Node's main
 * thread but not on a browser's main thread, which is why Pyodide runs in a worker in the
 * page; SharedArrayBuffer needs a cross-origin isolated page (see `coi.js`).
 *
 * Layout of the buffer: int32 slot 0 is the state (0 idle, 1 waiting, 2 answered), slot 1
 * the reply's length in bytes, slot 2 the abort token and slot 3 the in-flight marker of a
 * begun decode step (both `requestId + 1`, so 0 means none); the reply frame
 * (`frames.js`) starts at byte 16.
 *
 * **A decode step that does not block.** `beginDecodeStep` posts a step and returns its
 * request id at once; `pollDecode(timeoutMs)` waits for its reply for at most `timeoutMs`
 * and returns null while it is still running; `cancelDecode` asks it to stop. The model
 * thread checks the abort slot before every prefill chunk (`shouldStop`, injected by
 * `serveChannel`), so a cancel lands within one chunk, and a step that stopped leaves its
 * context's computed positions cached for the next step to resume from. The result of a
 * cancelled step is always dropped, even if the step had finished when the cancel came:
 * `pollDecode` then reports `{cancelled: true, resident, stats}`, never a token. While a
 * step is in flight every synchronous call refuses (`CHANNEL_BUSY`), so nothing queues
 * behind it; `piece` and a cached `info` never reach the model thread and keep working.
 *
 * **Stale replies.** Every reply names the request it answers, and a reply for any other
 * request is an error: one buffer holds one reply, so it cannot be set aside while the
 * right one is awaited. The only way one arises is a call that timed out, whose reply
 * lands later, so a timed-out call leaves the channel unusable: every later call throws
 * rather than risk taking that reply as its own.
 */

import { decodeFrame, encodeFrame, serveRequest } from "./frames.js";

const IDLE = 0;
const WAITING = 1;
const ANSWERED = 2;
const DATA = 16;
const ABORT = 2;
const IN_FLIGHT = 3;
export const CHANNEL_BYTES = 32 << 20;
/** What a synchronous call throws while a begun decode step is in flight. */
export const CHANNEL_BUSY = "channel busy: decode in flight";

/** Model side: answer every request arriving through `onMessage` into `buffer`. */
export function serveChannel(worker, buffer, onMessage) {
  const state = new Int32Array(buffer, 0, 4);
  const bytes = new Uint8Array(buffer);
  let chain = Promise.resolve();
  onMessage((request) => {
    // Read when the step reaches a chunk boundary, not when the request arrives: the
    // caller stores the token while the step runs.
    const shouldStop = request.begun ? () => Atomics.load(state, ABORT) === request.id + 1 : null;
    chain = chain.then(async () => {
      let frame = encodeFrame(await serveRequest(worker, request, { shouldStop }));
      if (DATA + frame.byteLength > buffer.byteLength) {
        frame = encodeFrame({
          id: request.id,
          ok: false,
          error: `RangeError: a ${frame.byteLength}-byte reply does not fit the channel`,
        });
      }
      bytes.set(frame, DATA);
      Atomics.store(state, 1, frame.byteLength);
      Atomics.store(state, 0, ANSWERED);
      Atomics.notify(state, 0);
    });
  });
}

/** Python side: the `ZeosModelWorker` interface, every method synchronous. */
export class SyncModelWorker {
  /**
   * @param {SharedArrayBuffer} buffer the buffer the model thread answers into.
   * @param {(message: object) => void} post sends a request to the model thread.
   * @param {object} [options]
   * @param {number} [options.timeoutMs] how long one call may take before it throws.
   */
  constructor(buffer, post, { timeoutMs = 600_000 } = {}) {
    this.state = new Int32Array(buffer, 0, 4);
    this.bytes = new Uint8Array(buffer);
    this.post = post;
    this.timeoutMs = timeoutMs;
    this.next = 0;
    this.cachedInfo = null;
    /** Why the channel can no longer be used (a call timed out), or null. */
    this.broken = null;
    // `JsMachine` asks for every piece once at start-up; one call fetches them all.
    this.pieces = this.call("pieces");
    this.backend = this.call("backend");
  }

  call(method, ...args) {
    this.usable();
    if (this.inFlight) throw new Error(CHANNEL_BUSY);
    const id = this.next++;
    Atomics.store(this.state, 0, WAITING);
    this.post({ id, method, args });
    const outcome = Atomics.wait(this.state, 0, WAITING, this.timeoutMs);
    if (outcome === "timed-out") {
      this.broken = `${method} (request ${id}) timed out after ${this.timeoutMs} ms`;
      throw new Error(`model worker did not answer ${method} within ${this.timeoutMs} ms`);
    }
    const reply = this.take(id);
    Atomics.store(this.state, 0, IDLE);
    if (!reply.ok) throw new Error(`model worker ${method}: ${reply.error}`);
    return reply.value;
  }

  usable() {
    if (this.broken !== null) throw new Error(`model channel unusable: ${this.broken}`);
  }

  /** The reply in the buffer, which must answer request `id`. */
  take(id) {
    const length = Atomics.load(this.state, 1);
    const reply = decodeFrame(this.bytes.slice(DATA, DATA + length));
    if (reply.id !== id) {
      this.broken = `a reply to request ${reply.id} arrived for request ${id}`;
      throw new Error(`model channel unusable: ${this.broken}`);
    }
    return reply;
  }

  info() {
    if (this.cachedInfo === null) this.cachedInfo = this.call("info");
    return this.cachedInfo;
  }

  tokenize(text) {
    return this.call("tokenize", text);
  }

  piece(tokenId) {
    if (!(tokenId >= 0 && tokenId < this.pieces.length)) {
      throw new RangeError(`token id ${tokenId} is outside the vocabulary`);
    }
    return this.pieces[tokenId];
  }

  createContext(jobId) {
    this.call("createContext", jobId);
  }

  destroyContext(jobId) {
    this.call("destroyContext", jobId);
  }

  length(jobId) {
    return this.call("length", jobId);
  }

  append(jobId, ids) {
    this.call("append", jobId, Int32Array.from(ids));
  }

  truncate(jobId, n) {
    this.call("truncate", jobId, n);
  }

  fork(parentId, childId) {
    this.call("fork", parentId, childId);
  }

  decodeStep(jobId, opts) {
    return this.call("decodeStep", jobId, stepArgs(opts));
  }

  // -- the decode step that does not block ---------------------------------------

  /** Whether a begun step's result has not yet been taken by `pollDecode`. */
  get inFlight() {
    return Atomics.load(this.state, IN_FLIGHT) !== 0;
  }

  /** Post one decode step and return its request id; never waits. `opts` is
   * `decodeStep`'s, plus `maxChunk`, the most positions one prefill run may take. */
  beginDecodeStep(jobId, opts) {
    this.usable();
    if (this.inFlight) throw new Error(CHANNEL_BUSY);
    const args = stepArgs(opts);
    const id = this.next++;
    // Set before posting: a model thread may answer before `post` returns.
    Atomics.store(this.state, ABORT, 0);
    Atomics.store(this.state, IN_FLIGHT, id + 1);
    Atomics.store(this.state, 0, WAITING);
    try {
      this.post({ id, method: "decodeStep", args: [jobId, args], begun: true });
    } catch (error) {
      Atomics.store(this.state, IN_FLIGHT, 0);
      Atomics.store(this.state, 0, IDLE);
      throw error;
    }
    return id;
  }

  /**
   * Wait at most `timeoutMs` for the begun step's reply; null if it has not landed, or if
   * nothing is in flight. Otherwise the step is no longer in flight and the answer is
   * `{tokenId, attention, cancelled: false, resident, stats}`, or
   * `{cancelled: true, resident, stats}` for a step that was cancelled, whether it
   * stopped early or had already finished. An error reply is thrown.
   */
  pollDecode(timeoutMs = 0) {
    const flight = Atomics.load(this.state, IN_FLIGHT);
    if (flight === 0) return null;
    if (Atomics.wait(this.state, 0, WAITING, Math.max(0, timeoutMs)) === "timed-out") return null;
    const reply = this.take(flight - 1);
    const cancelled = Atomics.load(this.state, ABORT) === flight;
    Atomics.store(this.state, ABORT, 0);
    Atomics.store(this.state, IN_FLIGHT, 0);
    Atomics.store(this.state, 0, IDLE);
    if (!reply.ok) throw new Error(`model worker decodeStep: ${reply.error}`);
    const value = reply.value;
    if (cancelled || value.cancelled === true) {
      return { cancelled: true, resident: value.resident, stats: value.stats };
    }
    return { ...value, cancelled: false };
  }

  /** Ask the begun step to stop before its next prefill chunk; never waits, and a no-op
   * with nothing in flight. The step stays in flight until `pollDecode` returns it. */
  cancelDecode() {
    const flight = Atomics.load(this.state, IN_FLIGHT);
    if (flight !== 0) Atomics.store(this.state, ABORT, flight);
  }
}

/** A step's options as they cross the channel: masks as `Uint8Array`s, the sample's
 * three numbers, and `maxChunk` only when it is set. */
function stepArgs(opts) {
  const { allowedBlocks = null, allowedTokens = null, sample = null, maxChunk = null } = opts ?? {};
  const args = {
    allowedBlocks: allowedBlocks === null ? null : Uint8Array.from(allowedBlocks),
    allowedTokens: allowedTokens === null ? null : Uint8Array.from(allowedTokens),
  };
  if (sample !== null) args.sample = { temperature: sample.temperature, topK: sample.topK, u: sample.u };
  if (maxChunk !== null) args.maxChunk = maxChunk;
  return args;
}
