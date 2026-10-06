// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

// Cross-origin isolation for a host that cannot set headers.
//
// The model worker answers Pyodide through a SharedArrayBuffer, which a browser offers
// only to a cross-origin isolated page: one served with
// `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
// require-corp`. serve.py sends both. A plain static host sends neither, so index.html
// registers this service worker, which adds them to every response it relays, and
// reloads once so the page is served through it. Cross-origin requests are left
// untouched: the one the page makes, Pyodide from jsDelivr, already carries
// `Cross-Origin-Resource-Policy: cross-origin`.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }),
  );
});
