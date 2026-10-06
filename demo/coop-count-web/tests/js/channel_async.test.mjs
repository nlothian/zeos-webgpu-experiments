// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The decode step that does not block: SyncModelWorker's beginDecodeStep, pollDecode and
// cancelDecode over a model thread serving web/stub_worker.js with simulated latency, so
// a step takes long enough to poll, time out on and cancel. No model needed.
//
//   node --test tests/js/channel_async.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";

import { CHANNEL_BUSY } from "../../web/model_channel.js";
import { startNodeModel } from "../../web/node_model_thread.mjs";
import "../../web/stub_worker.js";

const TAPES = { pilot: ["write stdout left;", "write stdout right;", "read stdin;"] };
const ALL = { allowedBlocks: null, allowedTokens: null };
// Every word the tape will say, in order.
const WORDS = ["write", " stdout", " left;", " write", " stdout", " right;", " read", " stdin;"];

async function withStub(options, body) {
  const sync = await startNodeModel({ stub: { tapes: TAPES, options }, timeoutMs: 10_000 });
  try {
    await body(sync);
  } finally {
    await sync.terminate();
  }
}

/** A context for job `jobId` holding `n` copies of one word. */
function context(sync, jobId, n) {
  sync.createContext(jobId);
  const [id] = sync.tokenize("write");
  sync.append(jobId, new Array(n).fill(id));
}

/** Poll until the step lands; fails the test after `ms`. */
function drain(sync, ms = 5000) {
  const reply = sync.pollDecode(ms);
  assert.notEqual(reply, null, `no reply within ${ms} ms`);
  return reply;
}

test("begin never waits; poll times out, then returns the step's result", async () => {
  await withStub({ stepMs: 500 }, async (sync) => {
    context(sync, "a:pilot", 3);
    let began = performance.now();
    const id = sync.beginDecodeStep("a:pilot", ALL);
    assert.ok(performance.now() - began < 50, "beginDecodeStep returned at once");
    assert.equal(typeof id, "number");
    assert.equal(sync.inFlight, true);

    began = performance.now();
    assert.equal(sync.pollDecode(0), null);
    assert.equal(sync.pollDecode(30), null);
    const waited = performance.now() - began;
    assert.ok(waited >= 25, `two polls waited ${waited} ms`);
    assert.equal(sync.inFlight, true, "a timed-out poll leaves the step in flight");

    const reply = drain(sync);
    assert.equal(reply.cancelled, false);
    assert.equal(sync.piece(reply.tokenId), "write");
    assert.equal(reply.attention, null);
    assert.equal(reply.resident, 3);
    assert.deepEqual({ ...reply.stats, fillMs: 0 }, { positions: 3, chunks: 2, fillMs: 0 });
    assert.equal(sync.inFlight, false);
    assert.equal(sync.pollDecode(100), null, "nothing in flight: null at once");
  });
});

test("a poll returns as soon as the reply lands, not at its timeout", async () => {
  await withStub({ stepMs: 40 }, async (sync) => {
    context(sync, "a:pilot", 2);
    const began = performance.now();
    sync.beginDecodeStep("a:pilot", ALL);
    drain(sync, 5000);
    const took = performance.now() - began;
    assert.ok(took < 1000, `took ${took} ms`);
  });
});

test("every call that reaches the model thread refuses while a step is in flight", async () => {
  await withStub({ stepMs: 100 }, async (sync) => {
    context(sync, "a:pilot", 2);
    const info = sync.info();
    sync.beginDecodeStep("a:pilot", ALL);
    for (const call of [
      () => sync.tokenize("write"),
      () => sync.append("a:pilot", [5]),
      () => sync.length("a:pilot"),
      () => sync.decodeStep("a:pilot", ALL),
      () => sync.createContext("b:pilot"),
      () => sync.beginDecodeStep("a:pilot", ALL),
    ]) {
      assert.throws(call, { message: CHANNEL_BUSY });
    }
    assert.equal(sync.piece(5), sync.pieces[5], "piece is answered locally");
    assert.deepEqual(sync.info(), info, "a cached info is answered locally");
    drain(sync);
    assert.equal(sync.length("a:pilot"), 2, "the channel works again once drained");
  });
});

test("a cancel lands between chunks and leaves the filled positions resident", async () => {
  const positionMs = 25;
  await withStub({ positionMs, stepMs: 5 }, async (sync) => {
    const n = 41;
    context(sync, "a:pilot", n);
    sync.beginDecodeStep("a:pilot", { ...ALL, maxChunk: 8 });
    assert.equal(sync.pollDecode(100), null);
    const cancelledAt = performance.now();
    sync.cancelDecode();
    sync.cancelDecode(); // idempotent
    assert.equal(sync.inFlight, true, "a cancelled step stays in flight until drained");
    const reply = drain(sync);
    const latency = performance.now() - cancelledAt;
    assert.equal(reply.cancelled, true);
    assert.equal("tokenId" in reply, false, "no token for a cancelled step");
    assert.ok(latency < 8 * positionMs + 150, `cancel took ${latency} ms, more than one chunk`);
    assert.ok(reply.resident > 0 && reply.resident < n - 1, `resident ${reply.resident}`);
    assert.equal(reply.resident % 8, 0, "stopped at a chunk boundary");
    assert.equal(reply.stats.positions, reply.resident);
    assert.equal(reply.stats.chunks, reply.resident / 8);
    assert.equal(sync.inFlight, false);

    // Resumed: only what the cancelled step did not fill is filled now.
    sync.beginDecodeStep("a:pilot", { ...ALL, maxChunk: 8 });
    const resumed = drain(sync);
    assert.equal(resumed.cancelled, false);
    assert.equal(sync.piece(resumed.tokenId), "write");
    assert.equal(resumed.stats.positions, n - reply.resident);
    assert.equal(resumed.resident, n);
  });
});

test("a step cancelled after it finished is still reported cancelled, and its token is not lost", async () => {
  await withStub({}, async (sync) => {
    context(sync, "a:pilot", 4);
    sync.beginDecodeStep("a:pilot", ALL);
    // Wait for the reply to be written without taking it, then cancel: the race.
    while (Atomics.load(sync.state, 0) !== 2) Atomics.wait(sync.state, 0, 1, 5);
    sync.cancelDecode();
    const reply = drain(sync);
    assert.deepEqual(reply.cancelled, true);
    assert.equal(reply.resident, 4, "the finished step's positions stay resident");
    assert.equal("tokenId" in reply, false);
    // The dropped token was never appended, so the next step chooses it again.
    sync.beginDecodeStep("a:pilot", ALL);
    assert.equal(sync.piece(drain(sync).tokenId), "write");
  });
});

test("an error reply is thrown and clears the step", async () => {
  await withStub({ stepMs: 5 }, async (sync) => {
    sync.beginDecodeStep("missing:pilot", ALL);
    assert.throws(() => sync.pollDecode(5000), /no context missing:pilot/);
    assert.equal(sync.inFlight, false);
    sync.cancelDecode(); // a no-op with nothing in flight
    assert.equal(sync.pollDecode(0), null);
  });
});

test("cancelling every step once and resuming gives the tokens of an uncancelled run", async () => {
  await withStub({ positionMs: 2, stepMs: 4 }, async (sync) => {
    const n = 40;
    context(sync, "plain:pilot", n);
    context(sync, "cut:pilot", n);
    const plain = [];
    const cut = [];
    let cancels = 0;
    for (let i = 0; i < WORDS.length; i++) {
      sync.beginDecodeStep("plain:pilot", { ...ALL, maxChunk: 4 });
      const want = drain(sync);
      plain.push(want.tokenId);
      sync.append("plain:pilot", [want.tokenId]);

      // Cancelled after a while that varies: mid-fill or at the final decode; a step
      // that lands before the cancel is simply taken. The first step fills 40 positions
      // at 2 ms each, so it is always still running when polled with no wait.
      sync.beginDecodeStep("cut:pilot", { ...ALL, maxChunk: 4 });
      let got = sync.pollDecode(i * 3);
      if (got === null) {
        sync.cancelDecode();
        assert.equal(drain(sync).cancelled, true);
        cancels += 1;
        sync.beginDecodeStep("cut:pilot", { ...ALL, maxChunk: 4 });
        got = drain(sync);
      }
      assert.equal(got.cancelled, false);
      cut.push(got.tokenId);
      sync.append("cut:pilot", [got.tokenId]);
    }
    assert.ok(cancels >= 1);
    assert.deepEqual(cut, plain);
    assert.deepEqual(plain.map((id) => sync.piece(id)), WORDS);
  });
});

test("the synchronous decodeStep is unchanged beside the new methods", async () => {
  await withStub({}, async (sync) => {
    context(sync, "a:pilot", 2);
    const step = sync.decodeStep("a:pilot", ALL);
    assert.equal(sync.piece(step.tokenId), "write");
    assert.equal(step.attention, null);
    assert.equal("cancelled" in step, false);
    sync.append("a:pilot", [step.tokenId]);
    sync.beginDecodeStep("a:pilot", ALL);
    assert.equal(sync.piece(drain(sync).tokenId), " stdout");
  });
});

test("the stub without latency or shouldStop answers synchronously, as before", () => {
  const stub = globalThis.createStubWorker(TAPES);
  stub.createContext("a:pilot");
  stub.append("a:pilot", stub.tokenize("write"));
  const step = stub.decodeStep("a:pilot", ALL);
  assert.equal(step instanceof Promise, false);
  assert.equal(stub.piece(step.tokenId), "write");
  // A stop asked before the step is answered at once, still synchronously.
  const stopped = stub.decodeStep("a:pilot", { ...ALL, shouldStop: () => true });
  assert.deepEqual(stopped, { cancelled: true, resident: 0, stats: { positions: 0, chunks: 0, fillMs: 0 } });
});
