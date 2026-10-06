// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// The model cache (web/model_cache.js) over a fake server and an in-memory store with
// the same interface as web/opfs_store.js: what is stored, under which key, and when a
// download counts as complete. And web/sha256.js against node:crypto.
//
//   node --test tests/js/model_cache.test.mjs

import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { test } from "node:test";

import { IntegrityError, QuotaError, hubUrl, modelFiles } from "../../web/model_cache.js";
import { Sha256 } from "../../web/sha256.js";

const REPO = "someone/Model-ZEOS-OPT";
const REV_A = "a".repeat(40);
const REV_B = "b".repeat(40);
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

/** An in-memory store, keyed by the JSON of the key, the way opfs_store.js is by path. */
function memoryStore({ quota = Infinity } = {}) {
  const files = new Map();
  const parts = new Map();
  const id = (key) => JSON.stringify(key);
  const used = () => [...files.values(), ...parts.values()].reduce((a, b) => a + b.byteLength, 0);
  return {
    files,
    parts,
    async complete(key) {
      return files.has(id(key)) ? files.get(id(key)).byteLength : null;
    },
    async partial(key) {
      return parts.get(id(key))?.byteLength ?? 0;
    },
    async *chunks(key, { partial }) {
      const bytes = (partial ? parts : files).get(id(key));
      for (let at = 0; at < bytes.byteLength; at += 1000) yield bytes.slice(at, at + 1000);
    },
    async writer(key, from) {
      let bytes = (parts.get(id(key)) ?? new Uint8Array(0)).slice(0, from);
      parts.set(id(key), bytes);
      return {
        write(chunk) {
          if (used() + chunk.byteLength > quota) {
            throw Object.assign(new Error("quota"), { name: "QuotaExceededError" });
          }
          const next = new Uint8Array(bytes.byteLength + chunk.byteLength);
          next.set(bytes);
          next.set(chunk, bytes.byteLength);
          bytes = next;
          parts.set(id(key), bytes);
        },
        flush() {},
        close() {},
      };
    },
    async commit(key) {
      files.set(id(key), parts.get(id(key)));
      parts.delete(id(key));
    },
    async discard(key) {
      files.delete(id(key));
      parts.delete(id(key));
    },
    async space() {
      return { usage: used(), quota };
    },
  };
}

/**
 * A server holding `files` (name -> bytes) under the Hub's URLs for `revision`. It
 * honours `Range: bytes=N-` unless `ranges` is false, and `cutAfter` (name -> bytes)
 * makes the next response for that name fail after that many bytes.
 */
function hub(files, { revision = REV_A, ranges = true } = {}) {
  const calls = [];
  const cutAfter = new Map();
  const fetch = async (url, init = {}) => {
    const base = hubUrl({ repo: REPO, revision });
    const href = String(url);
    calls.push({ url: href, range: init.headers?.Range ?? null, cache: init.cache });
    if (!href.startsWith(base) || !files.has(href.slice(base.length))) return new Response(null, { status: 404 });
    const name = href.slice(base.length);
    const all = files.get(name);
    const range = ranges ? /^bytes=(\d+)-$/.exec(init.headers?.Range ?? "") : null;
    const from = range ? Number(range[1]) : 0;
    const body = all.slice(from);
    const cut = cutAfter.get(name);
    cutAfter.delete(name);
    let sent = 0;
    const stream = new ReadableStream({
      pull(controller) {
        const chunk = body.slice(sent, sent + 4096);
        if (chunk.byteLength === 0) controller.close();
        else if (cut !== undefined && sent + chunk.byteLength > cut) controller.error(new TypeError("network error"));
        else {
          controller.enqueue(chunk);
          sent += chunk.byteLength;
        }
      },
    });
    const headers = { "content-length": String(body.byteLength) };
    if (range) headers["content-range"] = `bytes ${from}-${all.byteLength - 1}/${all.byteLength}`;
    return new Response(stream, { status: range ? 206 : 200, headers });
  };
  return { fetch, calls, cutAfter };
}

const source = (revision, fetch, store) =>
  modelFiles({ url: hubUrl({ repo: REPO, revision }), cache: { repo: REPO, revision }, fetch, store });
const expect = (bytes) => ({ bytes: bytes.byteLength, sha256: sha(bytes) });

test("sha256 matches node:crypto for every length around a block and chunking", () => {
  for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 1000, 70001]) {
    const bytes = randomBytes(n);
    const hash = new Sha256();
    for (let at = 0; at < n; at += 37) hash.update(bytes.subarray(at, at + 37));
    assert.equal(hash.hex(), sha(bytes), `length ${n}`);
  }
});

test("a cache hit reads the store and fetches nothing", async () => {
  const weights = randomBytes(50_000);
  const server = hub(new Map([["onnx/w.onnx_data", weights]]));
  const store = memoryStore();
  const first = await source(REV_A, server.fetch, store).read("onnx/w.onnx_data", expect(weights));
  assert.deepEqual(Buffer.from(first), weights);
  assert.equal(server.calls.length, 1);
  assert.equal(server.calls[0].cache, "no-store", "the HTTP cache must not keep a second copy");

  const phases = new Set();
  const again = await source(REV_A, server.fetch, store).read("onnx/w.onnx_data", expect(weights), (p) =>
    phases.add(p.phase),
  );
  assert.deepEqual(Buffer.from(again), weights);
  assert.equal(server.calls.length, 1, "no request on a cache hit");
  assert.deepEqual([...phases], ["cache"]);
});

test("the key is repo, revision and path, so another revision never reads this one's file", async () => {
  const oldBytes = randomBytes(3000);
  const newBytes = randomBytes(3000);
  const store = memoryStore();
  await source(REV_A, hub(new Map([["config.json", oldBytes]])).fetch, store).read("config.json", expect(oldBytes));
  assert.deepEqual([...store.files.keys()], [JSON.stringify([REPO, REV_A, "config.json"])]);

  const server = hub(new Map([["config.json", newBytes]]), { revision: REV_B });
  const got = await source(REV_B, server.fetch, store).read("config.json", expect(newBytes));
  assert.deepEqual(Buffer.from(got), newBytes);
  assert.equal(server.calls.length, 1, "revision B downloaded its own file");
  assert.equal(store.files.size, 2);
});

test("meta.json, with nothing expected, is stored and then read from the store", async () => {
  const meta = Buffer.from(JSON.stringify({ files: {} }));
  const server = hub(new Map([["meta.json", meta]]));
  const store = memoryStore();
  assert.deepEqual(Buffer.from(await source(REV_A, server.fetch, store).read("meta.json")), meta);
  assert.deepEqual(Buffer.from(await source(REV_A, server.fetch, store).read("meta.json")), meta);
  assert.equal(server.calls.length, 1);
});

test("an interrupted download is not complete, and the next load resumes it with a Range request", async () => {
  const weights = randomBytes(100_000);
  const server = hub(new Map([["w", weights]]));
  const store = memoryStore();
  server.cutAfter.set("w", 40_000);
  await assert.rejects(source(REV_A, server.fetch, store).read("w", expect(weights)), /download stopped/);
  assert.equal(await store.complete([REPO, REV_A, "w"]), null, "an interrupted file never looks complete");
  const kept = await store.partial([REPO, REV_A, "w"]);
  assert.ok(kept > 0 && kept <= 40_000, `the part keeps what arrived (${kept})`);

  const phases = [];
  const got = await source(REV_A, server.fetch, store).read("w", expect(weights), (p) => phases.push(p.phase));
  assert.deepEqual(Buffer.from(got), weights);
  assert.equal(server.calls.at(-1).range, `bytes=${kept}-`);
  assert.equal(phases[0], "verify", "the stored part is hashed before resuming");
  assert.ok(phases.includes("download"));
  assert.equal(await store.complete([REPO, REV_A, "w"]), weights.byteLength);
  assert.equal(await store.partial([REPO, REV_A, "w"]), 0);
});

test("a server that ignores the Range request restarts the file from its first byte", async () => {
  const weights = randomBytes(30_000);
  const server = hub(new Map([["w", weights]]), { ranges: false });
  const store = memoryStore();
  server.cutAfter.set("w", 10_000);
  await assert.rejects(source(REV_A, server.fetch, store).read("w", expect(weights)));
  const got = await source(REV_A, server.fetch, store).read("w", expect(weights));
  assert.deepEqual(Buffer.from(got), weights);
  assert.equal(await store.complete([REPO, REV_A, "w"]), weights.byteLength);
});

test("a corrupt part is discarded and the file downloaded again from the start", async () => {
  const weights = randomBytes(20_000);
  const server = hub(new Map([["w", weights]]));
  const store = memoryStore();
  const bad = Uint8Array.from(weights.subarray(0, 8000));
  bad[5] ^= 0xff;
  store.parts.set(JSON.stringify([REPO, REV_A, "w"]), bad);
  const got = await source(REV_A, server.fetch, store).read("w", expect(weights));
  assert.deepEqual(Buffer.from(got), weights);
  assert.deepEqual(
    server.calls.map((c) => c.range),
    ["bytes=8000-", null],
  );
});

test("a file whose SHA-256 does not match is rejected and nothing is stored", async () => {
  const weights = randomBytes(10_000);
  const server = hub(new Map([["w", weights]]));
  const store = memoryStore();
  const wrong = { bytes: weights.byteLength, sha256: sha(Buffer.from("something else")) };
  await assert.rejects(source(REV_A, server.fetch, store).read("w", wrong), IntegrityError);
  assert.equal(await store.complete([REPO, REV_A, "w"]), null);
  assert.equal(await store.partial([REPO, REV_A, "w"]), 0);
});

test("a file of the wrong length is rejected and nothing is stored", async () => {
  const weights = randomBytes(10_000);
  const server = hub(new Map([["w", weights]]));
  for (const bytes of [9_000, 11_000]) {
    const store = memoryStore();
    await assert.rejects(
      source(REV_A, server.fetch, store).read("w", { bytes, sha256: sha(weights) }),
      IntegrityError,
    );
    assert.equal(await store.complete([REPO, REV_A, "w"]), null);
    assert.equal(await store.partial([REPO, REV_A, "w"]), 0);
  }
});

test("a stored file of the wrong size is downloaded again", async () => {
  const weights = randomBytes(5000);
  const server = hub(new Map([["w", weights]]));
  const store = memoryStore();
  store.files.set(JSON.stringify([REPO, REV_A, "w"]), new Uint8Array(4000));
  const got = await source(REV_A, server.fetch, store).read("w", expect(weights));
  assert.deepEqual(Buffer.from(got), weights);
  assert.equal(server.calls.length, 1);
});

test("missing bytes count a shared file once and subtract parts; too little quota fails first", async () => {
  const store = memoryStore({ quota: 25_000 });
  const files = source(REV_A, hub(new Map()).fetch, store);
  const shared = { bytes: 10_000, sha256: "1".repeat(64) };
  const entries = [
    ["a", { bytes: 20_000, sha256: "0".repeat(64) }],
    ["b", shared],
    ["c", shared],
  ];
  assert.equal(await files.missing(entries), 30_000);
  store.parts.set(JSON.stringify([REPO, REV_A, "a"]), new Uint8Array(5000));
  assert.equal(await files.missing(entries), 25_000);
  store.files.set(JSON.stringify([REPO, REV_A, "c"]), new Uint8Array(10_000));
  assert.equal(await files.missing(entries), 15_000);
  await assert.rejects(files.checkSpace(15_000), QuotaError, "15 000 more on top of 15 000 stored");
  await files.checkSpace(10_000);
});

test("running out of quota mid-download leaves no part behind", async () => {
  const weights = randomBytes(50_000);
  const store = memoryStore({ quota: 20_000 });
  await assert.rejects(
    source(REV_A, hub(new Map([["w", weights]])).fetch, store).read("w", expect(weights)),
    QuotaError,
  );
  assert.equal(await store.partial([REPO, REV_A, "w"]), 0);
  assert.equal(await store.complete([REPO, REV_A, "w"]), null);
});

test("a local export is fetched each time and never stored", async () => {
  const weights = randomBytes(2000);
  const calls = [];
  const fetch = async (url) => {
    calls.push(String(url));
    return new Response(weights);
  };
  const files = modelFiles({ url: "http://localhost/models/X/", cache: null, fetch, store: null });
  for (let i = 0; i < 2; i++) assert.deepEqual(Buffer.from(await files.read("w", expect(weights))), weights);
  assert.deepEqual(calls, ["http://localhost/models/X/w", "http://localhost/models/X/w"]);
  assert.equal(await files.missing([["w", expect(weights)]]), 0);
});

test("a part that already holds the whole file is checked and stored without a request", async () => {
  const weights = randomBytes(9000);
  const server = hub(new Map([["w", weights]]));
  const store = memoryStore();
  store.parts.set(JSON.stringify([REPO, REV_A, "w"]), Uint8Array.from(weights));
  const got = await source(REV_A, server.fetch, store).read("w", expect(weights));
  assert.deepEqual(Buffer.from(got), weights);
  assert.equal(server.calls.length, 0);
  assert.equal(await store.complete([REPO, REV_A, "w"]), weights.byteLength);
});
