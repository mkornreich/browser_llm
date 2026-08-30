/* coi-probe.js — does THIS browser actually run SmolLM2 multithreaded?

   Cross-origin isolation is the kind of thing that fails SILENTLY: if the document
   isn't crossOriginIsolated, ONNX Runtime Web just warns to the console and pins
   env.backends.onnx.wasm.numThreads = 1, so "why is it single-threaded" looks
   exactly like normal operation. This probe answers the question directly, and it
   checks isolation in all three places that must line up for threads to run:

     1. the DOCUMENT              — crossOriginIsolated + a real SharedArrayBuffer
     2. the MODEL WORKER          — a Blob module worker spawned EXACTLY like
                                    browser-llm.js does (new Worker(blobUrl,
                                    {type:"module"})); isolation must be inherited here
     3. a NESTED worker           — the model worker spawns a child worker, mirroring
                                    ORT's Emscripten pthreads (em-pthread), and the
                                    child must ALSO see isolation + SAB

   Each worker runs ORT's own multithread test — postMessage a SharedArrayBuffer down
   a MessageChannel, which throws unless the context is truly isolated — so a green
   result means threads will genuinely spawn, not just that a flag is set.

   Usage:
     COIProbe.run().then(report => console.log(report));   // structured result
     COIProbe.render("#coi-report");                        // + paint it into an element

   Open coi-probe.html on any browser (including a Firefox private window) to see the
   verdict for that browser with no build step.
*/
var COIProbe = (function () {
  "use strict";

  // ORT's real isMultiThreadSupported() gate: a context that isn't cross-origin
  // isolated throws when you try to postMessage a SharedArrayBuffer.
  function sabTransferable() {
    try {
      if (typeof SharedArrayBuffer === "undefined") { return false; }
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
      return true;
    } catch (e) { return false; }
  }

  // The thread count browser-llm.js would choose for this device (kept in sync with
  // the worker + main-thread paths: numThreads = coi ? min(4, max(2, cores>>1)) : 1).
  function plannedThreads(isolated) {
    var cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 1;
    return isolated ? Math.min(4, Math.max(2, cores >> 1)) : 1;
  }

  // Source for the NESTED worker (the ORT-pthread stand-in). Reports its own view.
  function nestedWorkerSource() {
    return [
      "self.onmessage = function () {",
      "  var sab = false;",
      "  try { if (typeof SharedArrayBuffer !== 'undefined') { new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)); sab = true; } } catch (e) {}",
      "  self.postMessage({",
      "    crossOriginIsolated: (typeof self.crossOriginIsolated !== 'undefined') && self.crossOriginIsolated,",
      "    hasSAB: typeof SharedArrayBuffer !== 'undefined',",
      "    sabTransferable: sab",
      "  });",
      "};"
    ].join("\n");
  }

  // Source for the MODEL worker: reports its own view, then spawns a nested worker
  // (like ORT spawning pthreads) and reports that too. Mirrors browser-llm.js's
  // Blob module worker so the isolation propagation path is identical.
  function modelWorkerSource() {
    return [
      "var NESTED_SRC = " + JSON.stringify(nestedWorkerSource()) + ";",
      "function selfView() {",
      "  var sab = false;",
      "  try { if (typeof SharedArrayBuffer !== 'undefined') { new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)); sab = true; } } catch (e) {}",
      "  return {",
      "    crossOriginIsolated: (typeof self.crossOriginIsolated !== 'undefined') && self.crossOriginIsolated,",
      "    hasSAB: typeof SharedArrayBuffer !== 'undefined',",
      "    sabTransferable: sab",
      "  };",
      "}",
      "self.onmessage = function () {",
      "  var mine = selfView();",
      "  var done = function (nested, err) { self.postMessage({ worker: mine, nested: nested, nestedError: err || null }); };",
      "  var nurl = null;",
      "  try {",
      "    nurl = URL.createObjectURL(new Blob([NESTED_SRC], { type: 'text/javascript' }));",
      "    var child = new Worker(nurl, { type: 'module' });",
      "    var settled = false;",
      "    var t = setTimeout(function () { if (!settled) { settled = true; done(null, 'nested worker timed out'); } }, 4000);",
      "    child.onmessage = function (e) { if (settled) { return; } settled = true; clearTimeout(t); try { child.terminate(); } catch (x) {} try { URL.revokeObjectURL(nurl); } catch (x) {} done(e.data, null); };",
      "    child.onerror = function (e) { if (settled) { return; } settled = true; clearTimeout(t); if (e && e.preventDefault) { e.preventDefault(); } done(null, 'nested worker failed to start (nested/module workers unsupported?)'); };",
      "    child.postMessage('go');",
      "  } catch (e) { done(null, 'could not spawn nested worker: ' + ((e && e.message) || e)); }",
      "};"
    ].join("\n");
  }

  // Spawn the model worker exactly as browser-llm.js does and collect its report.
  function probeWorker() {
    return new Promise(function (resolve) {
      var url, worker, settled = false;
      function finish(val) { if (settled) { return; } settled = true; try { worker && worker.terminate(); } catch (e) {} try { url && URL.revokeObjectURL(url); } catch (e) {} resolve(val); }
      try {
        url = URL.createObjectURL(new Blob([modelWorkerSource()], { type: "text/javascript" }));
        worker = new Worker(url, { type: "module" });
      } catch (e) {
        resolve({ ok: false, error: "module workers unsupported (" + ((e && e.message) || e) + ")", worker: null, nested: null });
        return;
      }
      var timer = setTimeout(function () { finish({ ok: false, error: "model worker timed out", worker: null, nested: null }); }, 6000);
      worker.onerror = function (e) {
        if (e && e.preventDefault) { try { e.preventDefault(); } catch (x) {} }
        clearTimeout(timer);
        finish({ ok: false, error: "model worker failed to start (module worker blocked?)", worker: null, nested: null });
      };
      worker.onmessage = function (e) {
        clearTimeout(timer);
        var d = e.data || {};
        finish({ ok: true, error: null, worker: d.worker || null, nested: d.nested || null, nestedError: d.nestedError || null });
      };
      worker.postMessage("go");
    });
  }

  // Full probe: document view + worker view + nested view + an overall verdict.
  function run() {
    var isolated = (typeof crossOriginIsolated !== "undefined") && crossOriginIsolated;
    var doc = {
      crossOriginIsolated: isolated,
      hasSAB: typeof SharedArrayBuffer !== "undefined",
      sabTransferable: sabTransferable(),
      isSecureContext: typeof isSecureContext !== "undefined" ? isSecureContext : null,
      serviceWorkerControlled: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
      hardwareConcurrency: (navigator && navigator.hardwareConcurrency) || 0
    };
    return probeWorker().then(function (w) {
      var workerOK = !!(w.ok && w.worker && w.worker.crossOriginIsolated && w.worker.sabTransferable);
      var nestedOK = !!(w.ok && w.nested && w.nested.crossOriginIsolated && w.nested.sabTransferable);
      var multithreaded = doc.crossOriginIsolated && doc.sabTransferable && workerOK && nestedOK;
      return {
        multithreaded: multithreaded,
        plannedThreads: plannedThreads(multithreaded),
        document: doc,
        worker: w,
        // A short, human-readable reason the verdict came out the way it did.
        reason: multithreaded
          ? "Isolated end to end — the SmolLM2 WASM model will run multithreaded."
          : !doc.crossOriginIsolated
            ? "Document is not crossOriginIsolated (no COOP/COEP took effect) → single-threaded. If the service worker just registered, this resolves after its one-time reload."
            : !workerOK
              ? "Document is isolated but the model worker isn't (module worker blocked, or isolation didn't inherit) → single-threaded."
              : !nestedOK
                ? "Model worker is isolated but its nested (pthread) worker isn't → ORT threads can't spawn → single-threaded."
                : "Isolated but SharedArrayBuffer isn't transferable → single-threaded."
      };
    });
  }

  // Paint the report into an element (selector or node). Plain, dependency-free.
  function render(target) {
    var el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) { return Promise.resolve(null); }
    el.textContent = "Probing cross-origin isolation…";
    return run().then(function (r) {
      var yn = function (b) { return b ? "yes" : "no"; };
      var w = r.worker || {};
      var wv = w.worker || {};
      var nv = w.nested || {};
      var lines = [
        (r.multithreaded ? "✅" : "⚠️") + " " + r.reason,
        "",
        "planned WASM threads: " + r.plannedThreads,
        "",
        "document:",
        "  crossOriginIsolated: " + yn(r.document.crossOriginIsolated),
        "  SharedArrayBuffer usable: " + yn(r.document.sabTransferable),
        "  secure context: " + (r.document.isSecureContext === null ? "unknown" : yn(r.document.isSecureContext)),
        "  service worker controlling: " + yn(r.document.serviceWorkerControlled),
        "  hardwareConcurrency: " + r.document.hardwareConcurrency,
        "",
        "model worker (Blob module worker):",
        w.ok
          ? "  crossOriginIsolated: " + yn(wv.crossOriginIsolated) + ", SAB usable: " + yn(wv.sabTransferable)
          : "  " + (w.error || "unavailable"),
        "",
        "nested worker (ORT pthread stand-in):",
        w.ok && w.nested
          ? "  crossOriginIsolated: " + yn(nv.crossOriginIsolated) + ", SAB usable: " + yn(nv.sabTransferable)
          : "  " + (w.nestedError || (w.ok ? "not reached" : "unavailable"))
      ];
      el.textContent = lines.join("\n");
      return r;
    });
  }

  return { run: run, render: render, sabTransferable: sabTransferable, plannedThreads: plannedThreads };
})();

if (typeof module !== "undefined" && module.exports) { module.exports = COIProbe; }
