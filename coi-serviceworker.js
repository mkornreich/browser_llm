/* coi-serviceworker.js — client-side cross-origin isolation for static hosting.

   GitHub Pages (and any host where you cannot set response headers) can't send the
   Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy pair a page needs before
   the browser grants crossOriginIsolated === true. Without that, SharedArrayBuffer is
   hidden and transformers.js / ONNX Runtime Web run WASM SINGLE-THREADED
   (env.backends.onnx.wasm.numThreads is silently forced to 1). This file is the whole
   workaround: drop it in and the downloaded SmolLM2 fallback can use several threads.

   Loaded as a plain <script src> it runs TWICE, in two different globals:

     • In the PAGE (window): registers itself as a service worker, then reloads ONCE
       so the worker controls the document. It also SELF-HEALS — if the worker ends up
       controlling the page but isolation still didn't take (a browser that can't honor
       SW-injected COEP: old Safari, Firefox private < 140), it gives up PERMANENTLY and
       runs single-threaded instead of reload-looping.

     • In the SERVICE WORKER (no window): on every fetch it re-serves the response with
       COOP + COEP added — which is what actually flips crossOriginIsolated on.

   COEP value: require-corp, NOT credentialless. Every cross-origin asset this app loads
   already satisfies require-corp — jsdelivr sends Cross-Origin-Resource-Policy plus
   Access-Control-Allow-Origin, the Hugging Face weight CDN sends Access-Control-Allow-
   Origin, and ES-module imports are CORS-mode regardless — and require-corp is the ONLY
   COEP value Firefox for Android supports (it has no credentialless). Set
   window.coi.coepCredentialless = true (before this script) only if you add a
   cross-origin asset that lacks CORP.

   Firefox note: service workers only became available in Private Browsing in Firefox
   140 (2025-06-24), so isolation-via-SW works in FF private on 140+ and degrades to
   single-thread below that. The runtime gate below and the self-heal handle both.

   Config (set window.coi = {...} before loading this file):
     coepCredentialless : true → inject credentialless instead of require-corp (default false)
     force              : true → skip the capability gate and always attempt isolation
                          (used by coi-probe.html so it can diagnose any device)

   Adapted in spirit from gzuidhof/coi-serviceworker (MIT); rewritten in this repo's
   style to add the capability gate, the permanent self-heal flag, and a crawler guard.
*/
(function () {
  "use strict";

  var inWindow = typeof window !== "undefined";
  var cfg = (inWindow ? window.coi : self.coi) || {};
  var coepCredentialless = cfg.coepCredentialless === true;   // default false → require-corp

  // ═══ SERVICE-WORKER SIDE ════════════════════════════════════════════════════
  // No window here: this global is the ServiceWorkerGlobalScope.
  if (!inWindow) {
    self.addEventListener("install", function () { self.skipWaiting(); });
    self.addEventListener("activate", function (e) { e.waitUntil(self.clients.claim()); });

    // The page asks us to step aside (self-heal / opt-out): unregister and reload
    // every controlled client back into a plain, un-isolated, un-controlled page.
    self.addEventListener("message", function (e) {
      if (!e.data || e.data.type !== "coi-deregister") { return; }
      self.registration.unregister()
        .then(function () { return self.clients.matchAll(); })
        .then(function (clients) { clients.forEach(function (c) { c.navigate(c.url); }); });
    });

    self.addEventListener("fetch", function (event) {
      var r = event.request;
      // A cross-origin only-if-cached request can't be re-issued; let it pass.
      if (r.cache === "only-if-cached" && r.mode !== "same-origin") { return; }
      // credentialless: strip credentials from cross-origin no-cors requests.
      var request = (coepCredentialless && r.mode === "no-cors")
        ? new Request(r, { credentials: "omit" })
        : r;
      event.respondWith(
        fetch(request).then(function (res) {
          if (res.status === 0) { return res; }   // opaque cross-origin response: leave untouched
          var headers = new Headers(res.headers);
          headers.set("Cross-Origin-Embedder-Policy", coepCredentialless ? "credentialless" : "require-corp");
          if (!coepCredentialless) { headers.set("Cross-Origin-Resource-Policy", "cross-origin"); }
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(res.body, { status: res.status, statusText: res.statusText, headers: headers });
        })
      );
    });
    return;
  }

  // ═══ PAGE SIDE ══════════════════════════════════════════════════════════════
  var nav = navigator;
  // Capture this file's own URL synchronously — document.currentScript is only
  // valid during this initial run, not inside the async callbacks below. We must
  // register from a SAME-ORIGIN url, so this file can't be served from a CDN.
  var SELF_URL = (document.currentScript && document.currentScript.src) || "coi-serviceworker.js";
  var GIVEUP_KEY = "coiUnsupported";   // localStorage: this browser proved it can't isolate
  var RELOADED_KEY = "coiReloaded";    // sessionStorage: the one-time reload already happened

  // Already isolated → the worker did its job on an earlier navigation. Done.
  if (window.crossOriginIsolated) { return; }

  // Don't put crawlers through a reload; running them un-isolated costs nothing.
  if (/bot|crawl|spider|slurp|bingpreview|headless/i.test(nav.userAgent || "")) { return; }

  // This browser already proved it can register the SW but STILL not isolate
  // (old Safari; Firefox private < 140). Never retry — run single-threaded.
  try { if (localStorage.getItem(GIVEUP_KEY)) { return; } } catch (e) { /* storage blocked */ }

  // ── the capability gate: mirror BrowserLLM.shouldCrossOriginIsolate() ──
  // Isolation only speeds up the downloaded SmolLM2 WASM path, so skip the reload
  // cost for anyone who won't touch it: no isolation support, insecure context, no
  // service worker, < 4 cores (the heavy model won't load at a usable speed), or a
  // Prompt API present (Chrome's Nano needs no WASM threads). coi.force bypasses it.
  function hasNanoSurface() {
    try {
      if (typeof LanguageModel !== "undefined" && LanguageModel && LanguageModel.create) { return true; }
      if (window.ai && window.ai.languageModel && window.ai.languageModel.create) { return true; }
    } catch (e) { /* ignore */ }
    return false;
  }
  function shouldIsolate() {
    if (!("crossOriginIsolated" in window)) { return false; }   // old browser: single thread only
    if (!window.isSecureContext) { return false; }
    if (!("serviceWorker" in nav)) { return false; }
    if ((nav.hardwareConcurrency || 0) < 4) { return false; }   // heavy model won't load anyway
    if (hasNanoSurface()) { return false; }                     // Nano path: no WASM threads needed
    return true;
  }

  if (nav.serviceWorker && nav.serviceWorker.controller) {
    // A worker controls this page yet crossOriginIsolated is still false (checked
    // above). Isolation genuinely didn't take on this browser → give up for good and
    // drop the worker, so we never reload-loop.
    try { localStorage.setItem(GIVEUP_KEY, "1"); } catch (e) { /* ignore */ }
    try { nav.serviceWorker.controller.postMessage({ type: "coi-deregister" }); } catch (e) { /* ignore */ }
    return;
  }

  if (!cfg.force && !shouldIsolate()) { return; }
  if (!("serviceWorker" in nav)) { return; }

  nav.serviceWorker.register(SELF_URL).then(function (reg) {
    // Reload once, as soon as a worker is active, so it controls the reloaded page.
    if (reg.active && !nav.serviceWorker.controller) { reloadOnce(); return; }
    nav.serviceWorker.ready.then(function () {
      if (!nav.serviceWorker.controller) { reloadOnce(); }
    });
  }).catch(function (e) {
    // Registration itself failed (storage blocked, Firefox private < 140, etc.).
    // Nothing to heal — just run single-threaded this session.
  });

  function reloadOnce() {
    try {
      if (sessionStorage.getItem(RELOADED_KEY)) { return; }   // guard against a reload loop
      sessionStorage.setItem(RELOADED_KEY, "1");
    } catch (e) { /* storage blocked: skip the reload rather than risk looping */ return; }
    window.location.reload();
  }
})();
