// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * Where the model thread's files come from: a URL, and for a Hugging Face model a cache
 * in the browser's storage, so a reload reads the ~2.4 GB from disk instead of the network.
 *
 * `modelFiles({ url, cache, fetch, store, sha256 })` returns `read(name, expected,
 * onProgress)`, which resolves to the file's bytes as one Uint8Array:
 *
 * - `url` is the export's directory (for the Hub, `https://huggingface.co/<repo>/resolve/
 *   <revision>/`, or a mirror's equivalent; `hubUrl` builds it).
 * - `cache` is `{ repo, revision }`, or null for no cache (a local export, which the HTTP
 *   cache already serves). Every file is stored under the key `[repo, revision, name]`, so
 *   a new revision never reads a file an older one stored.
 * - `expected` is `{ bytes, sha256 }` from the export's `meta.json` (absent for
 *   `meta.json` itself, which comes from the pinned revision's immutable URL).
 * - `onProgress({ phase, loaded, total })` reports `"download"`, `"cache"` (reading a
 *   stored file) and `"verify"` (hashing the stored part of an interrupted download before
 *   resuming it, and finishing the hash of a download).
 *
 * A download is written to the store as a *part* as it streams, hashed as it streams, and
 * copied once into the Uint8Array the worker gets; it becomes a stored file (`commit`) only
 * after its length and SHA-256 match `expected`. So an interrupted download never looks
 * complete: the next load finds the part, hashes what it holds, and asks the server for
 * the rest with a `Range` request (a server that answers 200 instead restarts it). A part
 * that does not hash to `expected` once finished is discarded, and a resumed one is
 * downloaded once more from the start before the load fails.
 *
 * `store` is the storage (`opfs_store.js` in the browser; a Map in the tests):
 *
 *   complete(key) -> number | null        size of the stored file, or null
 *   partial(key)  -> number               bytes in the part (0 when there is none)
 *   chunks(key, { partial }) -> AsyncIterable<Uint8Array>   the stored file or the part
 *   writer(key, from) -> { write(bytes), flush(), close() }  appends to the part, cut at `from`
 *   commit(key)                           the part becomes the stored file
 *   discard(key)                          drop the part and any stored file
 *   space() -> { usage, quota } | null    what the origin's storage holds and allows
 */

import { Sha256 } from "./sha256.js";

export const HUB = "https://huggingface.co";

/** The directory a Hub repo's files resolve under, at one commit; `endpoint` is the Hub
 * or a mirror with its URL layout. */
export function hubUrl({ endpoint = HUB, repo, revision }) {
  return `${endpoint}/${repo}/resolve/${revision}/`;
}

/** A download failed its size or hash check; nothing was stored. */
export class IntegrityError extends Error {}

/** The browser would not store the model; nothing partial is left behind. */
export class QuotaError extends Error {}

const FLUSH_EVERY = 64 * 1024 * 1024;
const gb = (n) => `${(n / 1e9).toFixed(1)} GB`;

function quotaMessage(needed, space) {
  const free = space ? ` (it allows this site ${gb(space.quota)}, of which ${gb(space.usage)} is in use)` : "";
  return (
    `the browser has no room to store the model: it needs ${gb(needed)} more${free}; ` +
    "free disk space, clear this site's data, or use a browser profile with more room"
  );
}

const isQuotaError = (error) => error?.name === "QuotaExceededError";

/**
 * @param {object} options
 * @param {string} options.url the export's directory.
 * @param {{repo: string, revision: string} | null} options.cache
 * @param {typeof fetch} options.fetch
 * @param {object | null} options.store see above; required when `cache` is set.
 */
export function modelFiles({ url, cache, fetch, store }) {
  if (cache !== null && store === null) throw new Error("a cached model source needs a store");
  const keyOf = (name) => [cache.repo, cache.revision, name];

  async function intoBuffer(chunks, out, phase, total, onProgress, hash) {
    let at = 0;
    const parts = out === null ? [] : null;
    for await (const chunk of chunks) {
      if (out !== null) {
        if (at + chunk.byteLength > out.byteLength) throw new IntegrityError(`more bytes than the ${out.byteLength} expected`);
        out.set(chunk, at);
      } else parts.push(chunk);
      hash?.update(chunk);
      at += chunk.byteLength;
      onProgress({ phase, loaded: at, total });
    }
    return { at, parts };
  }

  const join = (parts, length) => {
    const out = new Uint8Array(length);
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.byteLength;
    }
    return out;
  };

  /** The bytes still to download for `files` (`[name, {bytes, sha256}]` pairs): what
   * neither a stored file nor a part already holds. Files with the same SHA-256 are read
   * once (see `OptZeosWorker.load`), so they count once, stored under either name. */
  async function missing(files) {
    if (cache === null) return 0;
    const groups = new Map();
    for (const [name, file] of files) {
      const id = file.sha256 ?? name;
      groups.set(id, [...(groups.get(id) ?? []), [name, file.bytes]]);
    }
    let bytes = 0;
    for (const group of groups.values()) {
      let left = group[0][1];
      for (const [name, size] of group) {
        if ((await store.complete(keyOf(name))) === size) left = 0;
        else left = Math.min(left, size - Math.min(size, await store.partial(keyOf(name))));
      }
      bytes += left;
    }
    return bytes;
  }

  /** Fail before downloading when the origin's storage cannot hold `needed` more bytes. */
  async function checkSpace(needed) {
    if (cache === null || needed === 0) return;
    const space = await store.space();
    if (space !== null && space.quota - space.usage < needed) throw new QuotaError(quotaMessage(needed, space));
  }

  async function download(name, expected, onProgress, { resume }) {
    const key = keyOf(name);
    const total = expected?.bytes ?? 0;
    const out = expected?.bytes !== undefined ? new Uint8Array(expected.bytes) : null;
    const fresh = () => (expected?.sha256 !== undefined ? new Sha256() : null);
    let hash = fresh();
    let from = resume ? await store.partial(key) : 0;
    // A part only resumes into a file of known size that it does not overfill.
    if (from > 0 && !(out !== null && from <= out.byteLength)) from = 0;
    if (from > 0) {
      const read = await intoBuffer(store.chunks(key, { partial: true }), out, "verify", from, onProgress, hash);
      if (read.at !== from) {
        from = 0;
        hash = fresh();
      }
    }
    const resumed = from > 0;
    // A part that already holds every byte (a load stopped between download and commit)
    // needs only its check.
    if (resumed && from === out.byteLength) return finish(from, null);
    const response = await fetch(new URL(name, url), {
      // The bytes go to the store; the HTTP cache keeping a second copy would double the disk.
      cache: "no-store",
      headers: from > 0 ? { Range: `bytes=${from}-` } : {},
    });
    if (!response.ok) throw new Error(`${new URL(name, url)}: HTTP ${response.status}`);
    if (from > 0 && !(response.status === 206 && response.headers.get("content-range")?.startsWith(`bytes ${from}-`))) {
      // The server sent the whole file rather than the rest: take it from the start.
      if (response.status !== 200) {
        await response.body?.cancel();
        throw new Error(`${new URL(name, url)}: HTTP ${response.status} to a range request`);
      }
      from = 0;
      hash = fresh();
    }
    let writer;
    try {
      writer = await store.writer(key, from);
    } catch (error) {
      if (error?.name === "NoModificationAllowedError") {
        throw new Error(`${name} is being downloaded by another tab of this page; close it, or wait and reload`);
      }
      throw error;
    }
    let at = from;
    let flushed = from;
    const parts = out === null ? [] : null;
    try {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (out !== null) {
          if (at + value.byteLength > out.byteLength) throw new IntegrityError(`${name}: longer than its ${out.byteLength} bytes`);
          out.set(value, at);
        } else parts.push(value);
        hash?.update(value);
        await writer.write(value);
        at += value.byteLength;
        if (at - flushed >= FLUSH_EVERY) {
          await writer.flush();
          flushed = at;
        }
        onProgress({ phase: "download", loaded: at, total });
      }
      await writer.flush();
    } catch (error) {
      await writer.close();
      if (isQuotaError(error)) {
        await store.discard(key);
        throw new QuotaError(quotaMessage(total - at, await store.space()));
      }
      if (error instanceof IntegrityError) {
        await store.discard(key);
        throw error;
      }
      // What arrived is stored as the part: the next load resumes from it.
      throw new Error(`${name}: the download stopped after ${at} bytes (${error}); reload to resume it`);
    }
    await writer.close();
    return finish(at, parts);

    async function finish(at, parts) {
      onProgress({ phase: "verify", loaded: at, total });
      const bytes = out ?? join(parts, at);
      const problem =
        out !== null && at !== out.byteLength
          ? `${at} bytes, not ${out.byteLength}`
          : hash !== null && hash.hex() !== expected.sha256
            ? `SHA-256 is not ${expected.sha256}`
            : null;
      if (problem !== null) {
        await store.discard(key);
        if (resumed) return download(name, expected, onProgress, { resume: false });
        throw new IntegrityError(`${new URL(name, url)}: ${problem}; nothing was stored, reload to try again`);
      }
      await store.commit(key);
      return bytes;
    }
  }

  async function read(name, expected, onProgress = () => {}) {
    if (cache === null) {
      const response = await fetch(new URL(name, url));
      if (!response.ok) throw new Error(`${new URL(name, url)}: HTTP ${response.status}`);
      const total = expected?.bytes ?? (Number(response.headers.get("content-length")) || 0);
      const out = expected?.bytes !== undefined ? new Uint8Array(expected.bytes) : null;
      const { at, parts } = await intoBuffer(streamOf(response), out, "download", total, onProgress, null);
      if (out !== null && at !== out.byteLength) throw new IntegrityError(`${name}: ${at} bytes, not ${out.byteLength}`);
      return out ?? join(parts, at);
    }
    const key = keyOf(name);
    const stored = await store.complete(key);
    if (stored !== null && (expected?.bytes === undefined || stored === expected.bytes)) {
      const out = expected?.bytes !== undefined ? new Uint8Array(stored) : null;
      const { at, parts } = await intoBuffer(store.chunks(key, { partial: false }), out, "cache", stored, onProgress, null);
      if (at !== stored) throw new IntegrityError(`the stored ${name} is ${at} bytes, not ${stored}`);
      return out ?? join(parts, at);
    }
    if (stored !== null) await store.discard(key);
    return download(name, expected, onProgress, { resume: true });
  }

  return { read, missing, checkSpace };
}

async function* streamOf(response) {
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    yield value;
  }
}
