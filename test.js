/* Dependency-free test suite for browser-llm.js — run `node test.js`.
   Covers the parts that are pure logic (no browser): the model catalog and
   context-window constants, the device-capability gates, the Prompt API status
   normalizer, the connection-block decisions, the download-progress math, and
   the generated worker source. The actual model download / inference needs a
   real browser and is not exercised here. */

"use strict";

// ── a tiny test harness ──────────────────────────────────────────────────────
var passed = 0, failed = 0;
function eq(actual, expected, msg) {
  var a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { passed++; }
  else { failed++; console.error("FAIL: " + msg + "\n  expected " + e + "\n  got      " + a); }
}
function ok(cond, msg) { eq(!!cond, true, msg); }

// ── navigator mock helper ─────────────────────────────────────────────────────
// Node ships a read-only `navigator` global, so redefine the property rather
// than assigning to it.
function setNav(n) { Object.defineProperty(global, "navigator", { value: n, configurable: true, writable: true }); }
function clearNav() { Object.defineProperty(global, "navigator", { value: undefined, configurable: true, writable: true }); }

var BrowserLLM = require("./browser-llm.js");

// ── constants ─────────────────────────────────────────────────────────────────
eq(BrowserLLM.CONTEXT_WINDOW.nano, 6144, "nano context window constant");
eq(BrowserLLM.CONTEXT_WINDOW.smol, 8192, "smol context window constant");
eq(BrowserLLM.MODELS.nano.contextWindow, 6144, "MODELS.nano.contextWindow");
eq(BrowserLLM.MODELS.smol.contextWindow, 8192, "MODELS.smol.contextWindow");
eq(BrowserLLM.MODELS.smol.id, "HuggingFaceTB/SmolLM2-360M-Instruct", "smol model id");
eq(BrowserLLM.MODELS.smol.dtype, "int8", "smol dtype");
eq(BrowserLLM.MODELS.nano.backend, "prompt-api", "nano backend");
eq(BrowserLLM.MODELS.smol.backend, "transformers", "smol backend");
ok(BrowserLLM.NANO_LANG.expectedInputs[0].languages[0] === "en", "NANO_LANG english");

// ── tooSlowForLlm ─────────────────────────────────────────────────────────────
setNav({ hardwareConcurrency: 8, deviceMemory: 8 });
eq(BrowserLLM.tooSlowForLlm(), false, "8 cores / 8 GB is fast enough");
setNav({ hardwareConcurrency: 2, deviceMemory: 8 });
eq(BrowserLLM.tooSlowForLlm(), true, "2 cores is too slow");
setNav({ hardwareConcurrency: 8, deviceMemory: 2 });
eq(BrowserLLM.tooSlowForLlm(), true, "2 GB is too slow");
setNav({ hardwareConcurrency: 4 });   // deviceMemory unreported → not penalized
eq(BrowserLLM.tooSlowForLlm(), false, "4 cores, memory unreported → ok");
clearNav();
eq(BrowserLLM.tooSlowForLlm(), true, "no navigator → too slow");

// ── heavyModelUnsupported ─────────────────────────────────────────────────────
setNav({ userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)" });
eq(BrowserLLM.heavyModelUnsupported(), true, "iPhone → heavy model unsupported");
setNav({ userAgent: "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X)" });
eq(BrowserLLM.heavyModelUnsupported(), true, "iPad → unsupported");
setNav({ userAgent: "Mozilla/5.0 (Macintosh)", platform: "MacIntel", maxTouchPoints: 5 });
eq(BrowserLLM.heavyModelUnsupported(), true, "iPadOS-as-Mac (touch) → unsupported");
setNav({ userAgent: "Mozilla/5.0 (Macintosh)", platform: "MacIntel", maxTouchPoints: 0 });
eq(BrowserLLM.heavyModelUnsupported(), false, "real Mac (no touch) → supported");
setNav({ userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" });
eq(BrowserLLM.heavyModelUnsupported(), false, "Windows desktop → supported");

// ── fastEnoughForBackground ───────────────────────────────────────────────────
setNav({ hardwareConcurrency: 8, deviceMemory: 8 });
eq(BrowserLLM.fastEnoughForBackground(), true, "8/8 → fast enough for background");
setNav({ hardwareConcurrency: 6, deviceMemory: 8 });
eq(BrowserLLM.fastEnoughForBackground(), false, "6 cores → not for background");
setNav({ hardwareConcurrency: 8, deviceMemory: 4 });
eq(BrowserLLM.fastEnoughForBackground(), false, "4 GB → not for background");
setNav({ hardwareConcurrency: 16 });  // memory unreported
eq(BrowserLLM.fastEnoughForBackground(), true, "16 cores, memory unreported → ok");

// ── nanoApi / nanoStatus ──────────────────────────────────────────────────────
eq(BrowserLLM.nanoApi(), null, "no Prompt API → nanoApi null");

(async function () {
  eq(await BrowserLLM.nanoStatus(null), "unavailable", "nanoStatus(null) → unavailable");
  eq(await BrowserLLM.nanoStatus({ availability: async function () { return "available"; } }),
    "available", "nanoStatus via availability()");
  eq(await BrowserLLM.nanoStatus({ availability: async function () { return "downloadable"; } }),
    "downloadable", "nanoStatus downloadable");
  eq(await BrowserLLM.nanoStatus({ capabilities: async function () { return { available: "readily" }; } }),
    "available", "nanoStatus legacy readily → available");
  eq(await BrowserLLM.nanoStatus({ capabilities: async function () { return { available: "after-download" }; } }),
    "downloadable", "nanoStatus legacy after-download → downloadable");
  eq(await BrowserLLM.nanoStatus({ capabilities: async function () { return { available: "no" }; } }),
    "unavailable", "nanoStatus legacy no → unavailable");
  eq(await BrowserLLM.nanoStatus({ availability: async function () { throw new Error("x"); } }),
    "unavailable", "nanoStatus that throws → unavailable");

  // ── replyOf ─────────────────────────────────────────────────────────────────
  eq(BrowserLLM.replyOf([{ generated_text: [{ role: "user", content: "hi" }, { role: "assistant", content: " done " }] }]),
    "done", "replyOf chat array → last content trimmed");
  eq(BrowserLLM.replyOf([{ generated_text: "  plain  " }]), "plain", "replyOf plain string");
  eq(BrowserLLM.replyOf([]), "", "replyOf empty → ''");
  eq(BrowserLLM.replyOf(null), "", "replyOf null → ''");

  // ── provider: connectionBlock ─────────────────────────────────────────────────
  var M = { offline: "OFF", saveData: "SAVE", slow: "SLOW" };
  var p = BrowserLLM.create({ messages: M });

  setNav({ onLine: true });
  eq(p.connectionBlock(), null, "online, no connection info → null");
  setNav({ onLine: false });
  eq(p.connectionBlock(), "OFF", "offline → offline message");
  setNav({ onLine: true, connection: { saveData: true } });
  eq(p.connectionBlock(), "SAVE", "save-data → saveData message");
  setNav({ onLine: true, connection: { effectiveType: "2g" } });
  eq(p.connectionBlock(), "SLOW", "2g → slow message");
  setNav({ onLine: true, connection: { effectiveType: "slow-2g" } });
  eq(p.connectionBlock(), "SLOW", "slow-2g → slow message");
  setNav({ onLine: true, connection: { effectiveType: "4g" } });
  eq(p.connectionBlock(), null, "4g → null");

  // weightsCached short-circuits every block
  p.weightsCached = true;
  setNav({ onLine: false, connection: { saveData: true } });
  eq(p.connectionBlock(), null, "weightsCached → never blocks (offline+savedata)");
  p.weightsCached = false;

  // default messages when none supplied
  var pDef = BrowserLLM.create();
  setNav({ onLine: false });
  ok(/one-time download/.test(pDef.connectionBlock()), "default offline copy present");

  // ── provider: llmDownloadBlock ────────────────────────────────────────────────
  setNav({ onLine: false });
  eq(await p.llmDownloadBlock(), "OFF", "no Prompt API + offline → offline block");

  // ── provider: probeWeightsCache via localStorage flag ─────────────────────────
  var store = {};
  global.localStorage = {
    getItem: function (k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
    setItem: function (k, v) { store[k] = String(v); }
  };
  var tag = BrowserLLM.MODELS.smol.id + "|" + BrowserLLM.MODELS.smol.dtype;
  store["cc:tf"] = tag;
  var pc = BrowserLLM.create({ weightsFlagKey: "cc:tf" });
  eq(await pc.probeWeightsCache(), true, "probeWeightsCache: matching flag → true");
  eq(pc.weightsCached, true, "probeWeightsCache set weightsCached");
  var pc2 = BrowserLLM.create({ weightsFlagKey: "cc:tf", dtype: "uint8" });   // different tag
  eq(await pc2.probeWeightsCache(), false, "probeWeightsCache: mismatched tag → false");
  delete global.localStorage;

  // ── provider: download-progress byte-weighting ────────────────────────────────
  var lastPct = -1;
  var pp = BrowserLLM.create({ onProgress: function (pct) { lastPct = pct; } });
  pp.applyDlProgress({ file: "weights", status: "progress", loaded: 50, total: 100 });
  eq(pp.dlPct, 50, "one file 50/100 → 50%");
  pp.applyDlProgress({ file: "config", status: "progress", loaded: 0, total: 100 });
  eq(pp.dlPct, 25, "two files 50/200 → 25% (byte-weighted, not per-file max)");
  pp.applyDlProgress({ file: "config", status: "done" });
  eq(pp.dlPct, 75, "config done → 150/200 → 75%");
  eq(lastPct, 75, "onProgress fired with latest pct");

  // ── worker source ─────────────────────────────────────────────────────────────
  var src = pp.workerSource();
  ok(src.indexOf("HuggingFaceTB/SmolLM2-360M-Instruct") !== -1, "worker source names the model");
  ok(src.indexOf('dtype: "int8"') !== -1, "worker source pins the dtype");
  ok(src.indexOf("@huggingface/transformers@4.2.0") !== -1, "worker source imports transformers");
  ok(src.indexOf("/<\\|[^|]*\\|>/g") !== -1, "worker source strips chat-template tokens with the right regex");
  ok(src.indexOf('type: "boot"') !== -1, "worker source posts a boot message");

  // ── custom model id flows into worker + tag ───────────────────────────────────
  var pCustom = BrowserLLM.create({ modelId: "org/Model", dtype: "q4" });
  ok(pCustom.workerSource().indexOf('"org/Model"') !== -1, "custom modelId flows into worker");
  ok(pCustom.workerSource().indexOf('dtype: "q4"') !== -1, "custom dtype flows into worker");

  // ── download ETA (estimated time remaining) ───────────────────────────────────
  (function () {
    var clock = 0;
    var lastInfo = null;
    var pe = BrowserLLM.create({ now: function () { return clock; }, onProgress: function (pct, info) { lastInfo = info; } });

    eq(pe.downloadEta(), null, "ETA: no samples yet → null");

    // first sample anchors the clock; no rate yet
    clock = 0;
    pe.applyDlProgress({ file: "w", status: "progress", loaded: 0, total: 100 });
    eq(pe.downloadEta(), null, "ETA: only the anchor sample → still null");

    // 1s later, half the bytes are in → rate = 0.5 / 1000ms
    clock = 1000;
    pe.applyDlProgress({ file: "w", status: "progress", loaded: 50, total: 100 });
    var eta = pe.downloadEta();
    ok(eta && !eta.done, "ETA: mid-download → not done");
    ok(eta && Math.abs(eta.etaMs - 1000) < 1, "ETA: 50% in 1s → ~1000ms remaining");
    eq(eta.etaSeconds, 1, "ETA: etaSeconds rounds to 1");
    eq(eta.bytesPerSec, 50, "ETA: 50 bytes in 1s → 50 B/s");
    eq(eta.pct, 50, "ETA: pct carried through");
    ok(lastInfo && Math.abs(lastInfo.etaMs - 1000) < 1, "ETA: onProgress info carries etaMs");
    eq(lastInfo.bytesPerSec, 50, "ETA: onProgress info carries bytesPerSec");

    // download completes → done, zero remaining
    clock = 3000;
    pe.applyDlProgress({ file: "w", status: "done" });
    var etaDone = pe.downloadEta();
    ok(etaDone && etaDone.done, "ETA: 100% → done");
    eq(etaDone.etaMs, 0, "ETA: done → 0ms remaining");
  })();

  // ETA works from Nano's fraction-only progress (no byte totals)
  (function () {
    var clock = 0;
    var pn = BrowserLLM.create({ now: function () { return clock; } });
    clock = 0;  pn.applyDlProgress({ file: "x", status: "progress", loaded: 0, total: 1000 });
    clock = 500; pn.applyDlProgress({ file: "x", status: "progress", loaded: 250, total: 1000 });  // 25% in .5s
    var e = pn.downloadEta();
    ok(e && Math.abs(e.etaMs - 1500) < 2, "ETA: 25% in 0.5s → ~1500ms remaining");
    eq(e.bytesPerSec, 500, "ETA: 250 bytes in 0.5s → 500 B/s");
  })();

  // ── canRun ────────────────────────────────────────────────────────────────────
  function setWin(w) { Object.defineProperty(global, "window", { value: w, configurable: true, writable: true }); }
  function clearWin() { Object.defineProperty(global, "window", { value: undefined, configurable: true, writable: true }); }
  function setNano(on) {
    if (on) { Object.defineProperty(global, "LanguageModel", { value: { create: function () {} }, configurable: true, writable: true }); }
    else { Object.defineProperty(global, "LanguageModel", { value: undefined, configurable: true, writable: true }); }
  }

  // ── hasEnoughCores / threshold constant ───────────────────────────────────────
  eq(BrowserLLM.HEAVY_MODEL_MIN_CORES, 4, "HEAVY_MODEL_MIN_CORES constant = 4");
  setNav({ hardwareConcurrency: 4 });
  eq(BrowserLLM.hasEnoughCores(), true, "hasEnoughCores: 4 cores → true (>= 4)");
  setNav({ hardwareConcurrency: 3 });
  eq(BrowserLLM.hasEnoughCores(), false, "hasEnoughCores: 3 cores → false");
  setNav({});   // hardwareConcurrency undefined → 0
  eq(BrowserLLM.hasEnoughCores(), false, "hasEnoughCores: unreported cores → false");
  clearNav();
  eq(BrowserLLM.hasEnoughCores(), false, "hasEnoughCores: no navigator → false");

  setNav({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Windows NT 10.0)" });
  setNano(false);
  eq(BrowserLLM.canRun(), true, "canRun: fast desktop, heavy model supported → true");
  setNav({ hardwareConcurrency: 2, userAgent: "Mozilla/5.0 (Windows NT 10.0)" });
  eq(BrowserLLM.canRun(), false, "canRun: too slow → false");
  setNav({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)" });
  setNano(false);
  eq(BrowserLLM.canRun(), false, "canRun: iOS (no heavy model) + no Nano → false");
  setNano(true);
  eq(BrowserLLM.canRun(), true, "canRun: iOS but Prompt API present → true (Nano can run)");
  setNano(false);

  // ── shouldCrossOriginIsolate ──────────────────────────────────────────────────
  setNav({ hardwareConcurrency: 8, serviceWorker: {} });
  setWin({ crossOriginIsolated: false, isSecureContext: true });
  setNano(false);
  eq(BrowserLLM.shouldCrossOriginIsolate(), true, "shouldCOI: capable, not isolated, no Nano → true");
  setWin({ crossOriginIsolated: true, isSecureContext: true });
  eq(BrowserLLM.shouldCrossOriginIsolate(), false, "shouldCOI: already isolated → false");
  setWin({ crossOriginIsolated: false, isSecureContext: true });
  setNav({ hardwareConcurrency: 2, serviceWorker: {} });
  eq(BrowserLLM.shouldCrossOriginIsolate(), false, "shouldCOI: < 4 cores → false");
  setNav({ hardwareConcurrency: 8, serviceWorker: {} });
  setNano(true);
  eq(BrowserLLM.shouldCrossOriginIsolate(), false, "shouldCOI: Prompt API present → false (Nano needs no threads)");
  setNano(false);
  setWin({ isSecureContext: true });   // no crossOriginIsolated support
  eq(BrowserLLM.shouldCrossOriginIsolate(), false, "shouldCOI: browser lacks isolation support → false");
  clearWin();
  eq(BrowserLLM.shouldCrossOriginIsolate(), false, "shouldCOI: no window (Node) → false");

  // ── coi-probe: planned thread count stays in sync with browser-llm.js ─────────
  // Must equal the worker's numThreads: coi ? min(4, max(2, cores>>1)) : 1.
  var COIProbe = require("./coi-probe.js");
  setNav({ hardwareConcurrency: 24 });
  eq(COIProbe.plannedThreads(true), 4, "probe threads: 24 cores isolated → 4 (cap)");
  eq(COIProbe.plannedThreads(false), 1, "probe threads: not isolated → 1");
  setNav({ hardwareConcurrency: 6 });
  eq(COIProbe.plannedThreads(true), 3, "probe threads: 6 cores isolated → 3");
  setNav({ hardwareConcurrency: 4 });
  eq(COIProbe.plannedThreads(true), 2, "probe threads: 4 cores isolated → 2 (floor)");
  setNav({});   // hardwareConcurrency undefined → 0
  eq(COIProbe.plannedThreads(true), 2, "probe threads: unknown cores isolated → 2 (floor)");

  // ── capabilities snapshot ─────────────────────────────────────────────────────
  setNav({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Windows NT 10.0)", serviceWorker: {} });
  setWin({ crossOriginIsolated: false, isSecureContext: true });
  setNano(false);
  var cap = BrowserLLM.capabilities();
  eq(cap.canRun, true, "capabilities.canRun");
  eq(cap.hardwareConcurrency, 8, "capabilities.hardwareConcurrency");
  eq(cap.enoughCores, true, "capabilities.enoughCores");
  eq(cap.deviceMemory, 8, "capabilities.deviceMemory");
  eq(cap.nano.supported, false, "capabilities.nano.supported");
  eq(cap.heavyModel.supported, true, "capabilities.heavyModel.supported");
  eq(cap.crossOriginIsolation.recommended, true, "capabilities.crossOriginIsolation.recommended");
  eq(cap.crossOriginIsolation.active, false, "capabilities.crossOriginIsolation.active");
  clearWin();
  setNano(false);

  // ── generateWithRetry ─────────────────────────────────────────────────────────
  function fakeBrain(backend, responder) {
    var calls = [];
    return {
      backend: backend,
      calls: calls,
      generate: function (msgs, o, onToken) {
        var i = calls.length;
        calls.push({ opts: o, hadToken: !!onToken });
        return Promise.resolve().then(function () { return responder(i, onToken); });
      }
    };
  }

  // retries until the filter passes
  var b1 = fakeBrain("nano", function (i) { return i === 0 ? "bad" : "good"; });
  var r1 = await BrowserLLM.generateWithRetry(b1, [], { isBad: function (s) { return s === "bad"; } });
  eq(r1, "good", "generateWithRetry: retries past a bad attempt → good");
  eq(b1.calls.length, 2, "generateWithRetry: took exactly 2 attempts");

  // all-bad exhausts attempts (Nano default = 4) and calls onReject each time
  var rejects = 0;
  var b2 = fakeBrain("nano", function () { return "bad"; });
  var r2 = await BrowserLLM.generateWithRetry(b2, [], {
    isBad: function () { return true; }, onReject: function () { rejects++; }
  });
  eq(r2, null, "generateWithRetry: all bad → null");
  eq(b2.calls.length, 4, "generateWithRetry: Nano default = 4 attempts");
  eq(rejects, 4, "generateWithRetry: onReject fired per rejected attempt");

  // non-Nano default = 2 attempts
  var b3 = fakeBrain("smol-worker", function () { return "bad"; });
  await BrowserLLM.generateWithRetry(b3, [], { isBad: function () { return true; } });
  eq(b3.calls.length, 2, "generateWithRetry: non-Nano default = 2 attempts");

  // postprocess runs before the filter
  var b4 = fakeBrain("nano", function () { return "GOOD"; });
  var r4 = await BrowserLLM.generateWithRetry(b4, [], { postprocess: function (s) { return "[" + s + "]"; } });
  eq(r4, "[GOOD]", "generateWithRetry: postprocess applied");

  // options can vary per attempt (e.g. a temperature schedule)
  var b5 = fakeBrain("smol-worker", function () { return "bad"; });
  await BrowserLLM.generateWithRetry(b5, [], {
    isBad: function () { return true; },
    options: function (a) { return { temperature: 0.6 - a * 0.1 }; }
  });
  eq(b5.calls[0].opts.temperature, 0.6, "generateWithRetry: options(0) → temp 0.6");
  eq(b5.calls[1].opts.temperature, 0.5, "generateWithRetry: options(1) → temp 0.5");

  // onToken forwarded for the current attempt
  var toks = [];
  var b6 = fakeBrain("nano", function (i, onToken) { if (onToken) { onToken("hi"); } return "good"; });
  await BrowserLLM.generateWithRetry(b6, [], { onToken: function (t) { toks.push(t); } });
  eq(toks, ["hi"], "generateWithRetry: onToken forwarded");

  // shouldContinue:false aborts before any attempt
  var b7 = fakeBrain("nano", function () { return "good"; });
  var r7 = await BrowserLLM.generateWithRetry(b7, [], { shouldContinue: function () { return false; } });
  eq(r7, null, "generateWithRetry: shouldContinue false → null");
  eq(b7.calls.length, 0, "generateWithRetry: aborted before generating");

  // a slow non-Nano attempt is bounded by the timeout → null
  var b8 = { backend: "smol-worker", generate: function () { return new Promise(function () {}); } };
  var r8 = await BrowserLLM.generateWithRetry(b8, [], { timeoutMs: 20, isBad: function () { return false; } });
  eq(r8, null, "generateWithRetry: hung non-Nano attempt times out → null");

  // timeout:false overrides the backend default (never time out, even non-Nano)
  var b9 = fakeBrain("smol-worker", function () { return "good"; });
  var r9 = await BrowserLLM.generateWithRetry(b9, [], { timeout: false });
  eq(r9, "good", "generateWithRetry: timeout:false → no timeout applied");
  // timeout:true forces the timeout even on a Nano brain
  var b10 = { backend: "nano", generate: function () { return new Promise(function () {}); } };
  var r10 = await BrowserLLM.generateWithRetry(b10, [], { timeout: true, timeoutMs: 20 });
  eq(r10, null, "generateWithRetry: timeout:true forces timeout on Nano → null");

  clearNav();

  // ── summary ───────────────────────────────────────────────────────────────────
  console.log("\n" + passed + " passed, " + failed + " failed");
  process.exit(failed ? 1 : 0);
})();
