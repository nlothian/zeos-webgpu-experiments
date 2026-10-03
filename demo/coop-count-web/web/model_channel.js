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
 * the reply's length in bytes; the reply frame (`frames.js`) starts at byte 16.
 */

import { decodeFrame, encodeFrame, serveRequest } from "./frames.js";

const IDLE = 0;
const WAITING = 1;
const ANSWERED = 2;
const DATA = 16;
export const CHANNEL_BYTES = 32 << 20;

/** Model side: answer every request arriving through `onMessage` into `buffer`. */
export function serveChannel(worker, buffer, onMessage) {
  const state = new Int32Array(buffer, 0, 4);
  const bytes = new Uint8Array(buffer);
  let chain = Promise.resolve();
  onMessage((request) => {
    chain = chain.then(async () => {
      let frame = encodeFrame(await serveRequest(worker, request));
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
    // `JsMachine` asks for every piece once at start-up; one call fetches them all.
    this.pieces = this.call("pieces");
    this.backend = this.call("backend");
  }

  call(method, ...args) {
    Atomics.store(this.state, 0, WAITING);
    this.post({ id: this.next++, method, args });
    const outcome = Atomics.wait(this.state, 0, WAITING, this.timeoutMs);
    if (outcome === "timed-out") {
      throw new Error(`model worker did not answer ${method} within ${this.timeoutMs} ms`);
    }
    const length = Atomics.load(this.state, 1);
    const reply = decodeFrame(this.bytes.slice(DATA, DATA + length));
    Atomics.store(this.state, 0, IDLE);
    if (!reply.ok) throw new Error(`model worker ${method}: ${reply.error}`);
    return reply.value;
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
    const { allowedBlocks = null, allowedTokens = null } = opts ?? {};
    return this.call("decodeStep", jobId, {
      allowedBlocks: allowedBlocks === null ? null : Uint8Array.from(allowedBlocks),
      allowedTokens: allowedTokens === null ? null : Uint8Array.from(allowedTokens),
    });
  }
}
