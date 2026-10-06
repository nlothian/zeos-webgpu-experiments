// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * Start the model thread from the page. What comes back is what the thread running
 * Pyodide needs to build a `SyncModelWorker`: the shared buffer replies arrive in, and a
 * MessagePort requests go out on. Both are transferable to another worker, which is how
 * the Pyodide worker gets them.
 */

import { hubUrl } from "./model_cache.js";
import { CHANNEL_BYTES } from "./model_channel.js";
import { cachedModels, clearModelCache } from "./opfs_store.js";

export { clearModelCache };

/**
 * @param {object} options
 * @param {{url: string, cache: {repo: string, revision: string} | null}} options.model
 *   where the export is: its directory (relative to the page, or absolute), and for a
 *   Hugging Face revision the key it is cached under (`modelSource` builds both).
 * @param {string} options.ortWebgpuUrl onnxruntime-web's `ort.webgpu.min.mjs`.
 * @param {string} options.tokenizersUrl `@huggingface/tokenizers`' `tokenizers.min.mjs`.
 * @param {(progress: object) => void} [options.onProgress] download and session progress.
 * @param {(granted: boolean) => void} [options.onPersisted] whether the browser will keep
 *   the model cache under storage pressure, once it answers (cached sources only).
 * @param {(activity: object) => void} [options.onActivity] every run of the graph, before
 *   and after, as `TransformersWorker`'s `onActivity` reports it.
 */
export async function startBrowserModel(options) {
  if (!self.crossOriginIsolated) {
    throw new Error(
      "the page is not cross-origin isolated, so SharedArrayBuffer is unavailable; serve it " +
        "with COOP/COEP headers (serve.py does) or let coi.js install its service worker",
    );
  }
  // Asked as the load starts, and not waited for: a browser may take its time (or ask the
  // user), and the download need not wait on the answer. `onPersisted` gets it.
  const persisted =
    options.model.cache !== null
      ? persistModelCache().then(
          (granted) => (options.onPersisted?.(granted), granted),
          () => false,
        )
      : Promise.resolve(false);
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const channel = new MessageChannel();
  const thread = new Worker(new URL("./model_thread.js", import.meta.url), { type: "module" });
  const absolute = (url) => new URL(url, self.location.href).href;
  const ready = new Promise((resolve, reject) => {
    thread.onmessage = (event) => {
      const data = event.data;
      if (data.progress) options.onProgress?.(data.progress);
      else if (data.activity) options.onActivity?.(data.activity);
      else if (data.ready) resolve(data.backend);
      else reject(new Error(data.error));
    };
    thread.onerror = (event) =>
      reject(new Error(`model thread failed to start: ${event.message ?? "no message"} (${event.filename ?? "?"}:${event.lineno ?? "?"})`));
  });
  thread.postMessage(
    {
      buffer,
      port: channel.port1,
      model: { url: absolute(options.model.url), cache: options.model.cache },
      ortWebgpuUrl: absolute(options.ortWebgpuUrl),
      tokenizersUrl: absolute(options.tokenizersUrl),
    },
    [channel.port1],
  );
  const backend = await ready;
  return { buffer, port: channel.port2, backend, thread, persisted };
}

/**
 * The model a page loads, from its build's `manifest.json`: `model_sources` names what
 * the build offers (`huggingface`: `{name, endpoint, repo, revision}`; `local`: `{name, path}`, an
 * export build.py linked into the page) and `model_source` the default, which a
 * `?model=huggingface` or `?model=local` query parameter overrides. Null when the build
 * offers no model. What comes back is `startBrowserModel`'s `model`, with the `name` to
 * show and a `label` saying where it comes from.
 */
export function modelSource(manifest, search = self.location.search) {
  const sources = manifest.model_sources ?? {};
  const chosen = new URLSearchParams(search).get("model") ?? manifest.model_source ?? null;
  if (chosen === null) return null;
  if (!Object.hasOwn(sources, chosen)) {
    const offered = Object.keys(sources);
    throw new Error(`?model=${chosen}: this build offers ${offered.length ? offered.join(" and ") : "no model"}`);
  }
  const source = sources[chosen];
  if (chosen === "huggingface") {
    const { endpoint, repo, revision } = source;
    return {
      kind: chosen,
      name: source.name,
      url: hubUrl({ endpoint, repo, revision }),
      cache: { repo, revision },
      label: `${new URL(endpoint).host}/${repo} at ${revision.slice(0, 7)}`,
    };
  }
  return { kind: chosen, name: source.name, url: source.path, cache: null, label: `this server, ${source.path}` };
}

const mb = (n) => `${Math.round(n / 1e6)} MB`;
const gb = (n) => `${(n / 1e9).toFixed(1)} GB`;

const VERBS = {
  download: "downloading",
  cache: "loading from the browser's cache",
  verify: "verifying",
};

/** The text and the fraction done (or null) a page shows for a `progress` message. */
export function describeModelProgress(progress) {
  if (progress.phase === "session") {
    return { text: `read ${mb(progress.bytes)}; ONNX Runtime is building the WebGPU session`, fraction: null };
  }
  const { phase, file, loaded, total, files, file_index, bytes, bytes_total } = progress;
  const whole = bytes_total > 0 ? `${mb(bytes)} of ${mb(bytes_total)}` : `${mb(loaded)} of ${mb(total)}`;
  const which = files > 0 ? ` (file ${file_index + 1} of ${files}: ${file})` : ` (${file})`;
  const fraction = bytes_total > 0 ? bytes / bytes_total : total > 0 ? loaded / total : null;
  return { text: `${VERBS[phase] ?? phase} ${whole}${which}`, fraction };
}

/**
 * What the model cache holds and what the browser allows this origin, as a line a page
 * can show: `{models, bytes, usage, quota, persisted, text}`.
 */
export async function modelCacheStatus() {
  const models = await cachedModels();
  const bytes = models.reduce((a, m) => a + m.bytes, 0);
  const { usage, quota } = await navigator.storage.estimate();
  const persisted = await navigator.storage.persisted();
  const held = models.length === 0 ? "no model cached" : `${gb(bytes)} of model cached`;
  const kept = persisted ? "kept until cleared" : "the browser may evict it when short of space";
  return {
    models,
    bytes,
    usage,
    quota,
    persisted,
    text: `${held} (${kept}); this site uses ${gb(usage)} of the ${gb(quota)} the browser allows it`,
  };
}

/** Ask the browser to keep the model cache under storage pressure; the browser decides
 * (Chrome without a prompt, on how much the site is used), and says whether it will. */
export async function persistModelCache() {
  return navigator.storage.persisted().then((done) => done || navigator.storage.persist());
}

/**
 * Wire a page's cache line and "clear cached model" button to the cache, for `source`
 * (`modelSource`'s result). Returns a function that refreshes the line, for after a load.
 * @param {{status: HTMLElement, button: HTMLButtonElement, source: object}} elements
 */
export function modelCacheControls({ status, button, source }) {
  const refresh = async () => {
    if (source.cache === null) {
      status.textContent = `${source.label}; not kept in the browser's storage`;
    } else {
      status.textContent = `${source.label}; ${(await modelCacheStatus()).text}`;
    }
  };
  button.hidden = source.cache === null;
  button.onclick = async () => {
    button.disabled = true;
    try {
      await clearModelCache();
      await refresh();
      status.textContent += "; cleared: the next load downloads it again (a model already loaded stays in use)";
    } catch (error) {
      status.textContent = `could not clear the cache: ${error.message} (is the model still downloading, here or in another tab?)`;
    } finally {
      button.disabled = false;
    }
  };
  refresh().catch((error) => (status.textContent = `${source.label}; storage: ${error.message}`));
  return refresh;
}
