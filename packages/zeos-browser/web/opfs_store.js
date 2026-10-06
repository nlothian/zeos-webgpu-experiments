// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The model cache's storage: the Origin Private File System (OPFS), the origin's own
 * sandboxed files, which the browser keeps across reloads and restarts (and, once
 * `navigator.storage.persist()` is granted, does not evict under storage pressure).
 *
 * Layout: `zeos-model-cache/<repo>/<revision>/<path>`, each component URI-encoded (so a
 * path's `/` stays inside one name); a download in progress is `<path>.part` beside it,
 * renamed to `<path>` once it has been checked (`model_cache.js`). A `.part` is written
 * through a synchronous access handle, which only a dedicated worker has (the model thread
 * is one) and which holds the file exclusively, so two tabs never write the same part.
 * Reading takes no lock, and returns the file as a stream: `Blob.arrayBuffer()` of a file
 * over 2 GiB did not finish in Chrome.
 *
 * Why OPFS and not Cache Storage, measured in Chrome 154 on an M1 Max with a 2.07 GB
 * file: OPFS wrote it in 1.4 s and streamed it back in 1.1 s; `Cache.put` took 5.3 s and
 * its read 1.4 s, and after `caches.delete` the origin's usage stayed at 2.07 GB, where
 * removing the OPFS file freed it at once. A `.part` file can also be resumed with a
 * `Range` request, where a cache entry is all or nothing.
 */

const ROOT = "zeos-model-cache";
const PART = ".part";

const notFound = (error) => error?.name === "NotFoundError" || error?.name === "TypeMismatchError";

async function descend(dir, names, create) {
  try {
    for (const name of names) dir = await dir.getDirectoryHandle(name, { create });
  } catch (error) {
    if (!create && notFound(error)) return null;
    throw error;
  }
  return dir;
}

const directory = async (names, create) => descend(await navigator.storage.getDirectory(), names, create);

const dirNames = ([repo, revision]) => [ROOT, encodeURIComponent(repo), encodeURIComponent(revision)];
const fileName = (key) => encodeURIComponent(key[2]);

async function fileHandle(key, suffix) {
  const dir = await directory(dirNames(key), false);
  if (dir === null) return null;
  try {
    return await dir.getFileHandle(fileName(key) + suffix);
  } catch (error) {
    if (notFound(error)) return null;
    throw error;
  }
}

async function remove(dir, name, options) {
  try {
    await dir.removeEntry(name, options);
  } catch (error) {
    if (!notFound(error)) throw error;
  }
}

/** The store `model_cache.js` takes. */
export const opfsStore = {
  async complete(key) {
    const handle = await fileHandle(key, "");
    return handle === null ? null : (await handle.getFile()).size;
  },

  async partial(key) {
    const handle = await fileHandle(key, PART);
    return handle === null ? 0 : (await handle.getFile()).size;
  },

  async *chunks(key, { partial }) {
    const handle = await fileHandle(key, partial ? PART : "");
    if (handle === null) return;
    const reader = (await handle.getFile()).stream().getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      yield value;
    }
  },

  async writer(key, from) {
    const dir = await directory(dirNames(key), true);
    const handle = await dir.getFileHandle(fileName(key) + PART, { create: true });
    const access = await handle.createSyncAccessHandle();
    access.truncate(from);
    let at = from;
    return {
      write(bytes) {
        const written = access.write(bytes, { at });
        if (written !== bytes.byteLength) throw new Error(`wrote ${written} of ${bytes.byteLength} bytes`);
        at += written;
      },
      flush: () => access.flush(),
      close: () => access.close(),
    };
  },

  async commit(key) {
    const dir = await directory(dirNames(key), false);
    await remove(dir, fileName(key));
    const part = await dir.getFileHandle(fileName(key) + PART);
    await part.move(fileName(key));
  },

  async discard(key) {
    const dir = await directory(dirNames(key), false);
    if (dir === null) return;
    await remove(dir, fileName(key) + PART);
    await remove(dir, fileName(key));
  },

  async space() {
    const { usage, quota } = await navigator.storage.estimate();
    return { usage, quota };
  },
};

const USED = ".used-";
/** How long a revision must have gone unused before another revision's load removes it:
 * longer than any load takes, so no tab is still reading it. */
export const UNUSED_FOR_MS = 24 * 60 * 60 * 1000;

const root = () => navigator.storage.getDirectory();

/** Record that a load is using `revision` now: an empty file named `.used-<ms>` in its
 * directory (the previous one removed), which `dropOtherRevisions` reads. */
export async function markUsed({ repo, revision }, { top = root(), now = Date.now() } = {}) {
  const dir = await descend(await top, [ROOT, encodeURIComponent(repo), encodeURIComponent(revision)], true);
  for await (const [name] of dir.entries()) if (name.startsWith(USED)) await remove(dir, name);
  await dir.getFileHandle(`${USED}${now}`, { create: true });
}

/**
 * Remove other revisions of `repo` that no load has used for `UNUSED_FOR_MS` and that
 * were last used before `revision` was: once a load has the newer revision, the older
 * one is never read again. Best-effort, and it skips whatever it is unsure of: a revision
 * with a `.part` (a download in progress, perhaps in another tab), one with no record of
 * its use, and one whose removal fails (a file another tab holds open); each failure is
 * logged and the rest go on. Returns the revisions removed.
 */
export async function dropOtherRevisions(
  { repo, revision },
  { top = root(), now = Date.now(), log = console.warn } = {},
) {
  const repoDir = await descend(await top, [ROOT, encodeURIComponent(repo)], false);
  if (repoDir === null) return [];
  const lastUsed = async (dir) => {
    let used = null;
    for await (const [name] of dir.entries()) {
      if (name.endsWith(PART)) return { busy: true };
      if (name.startsWith(USED)) used = Math.max(used ?? 0, Number(name.slice(USED.length)));
    }
    return { busy: false, used };
  };
  const current = await descend(repoDir, [encodeURIComponent(revision)], false);
  const currentUsed = current === null ? null : (await lastUsed(current)).used;
  const removed = [];
  for await (const [name, dir] of repoDir.entries()) {
    if (name === encodeURIComponent(revision) || dir.kind !== "directory") continue;
    try {
      const { busy, used } = await lastUsed(dir);
      if (busy || used === null || now - used < UNUSED_FOR_MS) continue;
      if (currentUsed !== null && used >= currentUsed) continue;
      await repoDir.removeEntry(name, { recursive: true });
      removed.push(decodeURIComponent(name));
    } catch (error) {
      log(`model cache: left revision ${decodeURIComponent(name)} in place: ${error?.name ?? error}`);
    }
  }
  return removed;
}

/** What the cache holds: one entry per repo and revision, with its files' sizes. */
export async function cachedModels() {
  const root = await directory([ROOT], false);
  if (root === null) return [];
  const models = [];
  for await (const [repo, repoDir] of root.entries()) {
    if (repoDir.kind !== "directory") continue;
    for await (const [revision, revDir] of repoDir.entries()) {
      if (revDir.kind !== "directory") continue;
      const files = [];
      for await (const [name, file] of revDir.entries()) {
        if (file.kind !== "file" || name.startsWith(USED)) continue;
        const partial = name.endsWith(PART);
        files.push({
          path: decodeURIComponent(partial ? name.slice(0, -PART.length) : name),
          bytes: (await file.getFile()).size,
          partial,
        });
      }
      const bytes = files.reduce((a, f) => a + f.bytes, 0);
      models.push({ repo: decodeURIComponent(repo), revision: decodeURIComponent(revision), files, bytes });
    }
  }
  return models;
}

/** Remove the whole cache. Fails while a download holds a part open (in another tab). */
export async function clearModelCache() {
  const root = await navigator.storage.getDirectory();
  await remove(root, ROOT, { recursive: true });
}
