/* ============================================================================
   BROWSER LLM. Get an on-device language model, in the browser, with no server.

   One place that knows how to "get the LLM" for a page: it prefers Chrome's
   built-in Gemini Nano (the Prompt API — on-device, nothing to download from
   us) and, only where Nano is missing, falls back to downloading SmolLM2-360M
   via transformers.js (inside a Web Worker, with a main-thread fallback for the
   browsers that cannot run a module worker). Whatever backend wins, the caller
   gets the SAME tiny shape back:

       var llm   = BrowserLLM.create();
       var brain = await llm.loadGenerator();
       var text  = await brain.generate(
         [{ role: "user", content: 'What is "photosynthesis"?' }],
         { max_new_tokens: 72, temperature: 0.6 },
         function (partial) {  ...live tokens...  });

   It also owns the decisions that go with "should we even try": whether the
   device is too slow / too old for a local model (tooSlowForLlm), whether the
   heavy fallback would crash the tab (heavyModelUnsupported — iOS/iPadOS), and
   whether the network is a bad time to pull ~360 MB (connectionBlock). All UI
   (status text, buttons, prompts, post-processing) stays with the caller; this
   library only produces the brain and reports its state through callbacks.

   Extracted from the Cool Concepts project (mkornreich.me/projects/coolconcepts)
   as the shared source of truth for its word-brain, the sibling of the
   llm_postprocessor repo. Loads in the browser as window.BrowserLLM; the pure
   parts (constants + device/connection logic) also load in Node for testing.

   Compatibility floor matches Cool Concepts: async/await, arrow functions and
   template literals are used, but no `for await`, no bare `import()` (built via
   new Function), and no optional chaining — so a browser that can parse the host
   page's app script can parse this too, and older ones fall through untouched.
   ============================================================================ */
(function (root, factory) {
  var api = factory();
  root.BrowserLLM = api;                                      // browser global
  if (typeof module !== "undefined" && module.exports) {      // Node / bundlers
    module.exports = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var G = typeof globalThis !== "undefined" ? globalThis : this;

  // ── model catalog ──────────────────────────────────────────────────────────
  // The two models this library can be, and the ONE place their identities live
  // (ids, quantization, and context-window sizes). "nano" is Chrome's built-in
  // Gemini Nano via the Prompt API — nothing is downloaded from us. "smol" is
  // the fallback we download ourselves when there is no Prompt API.
  var MODELS = {
    nano: {
      key: "nano",
      label: "Gemini Nano",
      backend: "prompt-api",                 // Chrome's window LanguageModel / window.ai
      // Chrome does not publish a fixed size; a live session reports the real
      // number on session.contextWindow (and the input budget on
      // session.inputQuota). 6144 is the input quota measured on current Chrome
      // builds — a nominal constant, not a promise. Read the live value when a
      // session exists; use this for planning before one does.
      contextWindow: 6144                    // tokens (nominal; see session.contextWindow)
    },
    smol: {
      key: "smol",
      label: "SmolLM2-360M-Instruct",
      backend: "transformers",               // downloaded and run via transformers.js (WASM)
      id: "HuggingFaceTB/SmolLM2-360M-Instruct",
      dtype: "int8",                         // int8 vs uint8: same size/quality, benchmarked flat
      // From the model's config.json: max_position_embeddings = 8192. SmolLM2
      // was trained to 8k (extended from 2k with rope_theta 100000).
      contextWindow: 8192                    // tokens
    }
  };

  // Convenience: just the context-window sizes, keyed by model, as constants.
  var CONTEXT_WINDOW = { nano: MODELS.nano.contextWindow, smol: MODELS.smol.contextWindow };

  // Tell the built-in Prompt API which languages we send and expect back. The
  // spec warns (and may degrade output quality / safety attestation) when a
  // LanguageModel request omits these; shapes that do not know the keys ignore
  // them. Everything here is English.
  var NANO_LANG = {
    expectedInputs: [{ type: "text", languages: ["en"] }],
    expectedOutputs: [{ type: "text", languages: ["en"] }]
  };

  // ── device capability (static; no model / provider needed) ──────────────────

  // A system too old to draw color emoji (Windows XP/Vista/7-era) is far too old
  // to run a local model at a usable speed. Draw one on a canvas and look for a
  // COLORED pixel: tofu boxes and old monochrome glyphs have none. An
  // inconclusive result (no canvas, or a blocked readback on an anti-
  // fingerprinting browser) is treated as capable, so we never penalize privacy.
  // Checked once, remembered.
  var emojiRendersCached = null;
  function emojiRenders() {
    if (emojiRendersCached !== null) { return emojiRendersCached; }
    var ok = true;   // inconclusive (no canvas / blocked readback) → assume capable
    try {
      var cv = document.createElement("canvas");
      cv.width = 32; cv.height = 32;
      var cx = cv.getContext && cv.getContext("2d");
      if (cx && cx.fillText && cx.getImageData) {
        cx.textBaseline = "top";
        cx.font = "28px sans-serif";
        cx.fillText("🧠", 0, 0);            // 🧠
        var d = cx.getImageData(0, 0, 32, 32).data;   // throws if the canvas is blocked
        var colored = false;
        for (var i = 0; i < d.length; i += 4) {
          if (d[i + 3] > 16 &&
              (Math.abs(d[i] - d[i + 1]) > 16 || Math.abs(d[i + 1] - d[i + 2]) > 16)) {
            colored = true; break;                    // a genuinely colored pixel
          }
        }
        ok = colored;   // we could read the pixels → trust it (no color = truly old)
      }
    } catch (e) { ok = true; }   // blocked/unreadable canvas → do not penalize
    emojiRendersCached = ok;
    return ok;
  }

  // The minimum logical cores to bother running the heavy WASM model (or setting
  // up threads for it). Below this, the downloaded model won't load at a usable
  // speed, so we never try. This is the single home for the "hardwareConcurrency
  // >= 4" threshold used by tooSlowForLlm() and shouldCrossOriginIsolate() (and
  // that a page's cross-origin-isolation bootstrap mirrors inline).
  var HEAVY_MODEL_MIN_CORES = 4;
  // Does this machine have enough logical cores (>= HEAVY_MODEL_MIN_CORES) to run
  // the heavy model / use WASM threads? Static; navigator.hardwareConcurrency.
  function hasEnoughCores() {
    if (typeof navigator === "undefined") { return false; }
    return (navigator.hardwareConcurrency || 0) >= HEAVY_MODEL_MIN_CORES;
  }

  // A really slow or old device cannot run a local model at a usable speed: the
  // caller should hide "explain" there and never load or download a model.
  // Chrome's built-in Nano is exempt once it is actually AVAILABLE — Chrome only
  // provisions it on hardware it has vetted, and it costs us no download — so
  // check nanoReady separately and let an available Nano override this.
  function tooSlowForLlm() {
    if (typeof navigator === "undefined") { return true; }
    if (!hasEnoughCores()) { return true; }
    if (typeof navigator.deviceMemory === "number" && navigator.deviceMemory < 4) { return true; }
    if (!emojiRenders()) { return true; }   // no color emoji = an old system
    return false;
  }

  // iOS/iPadOS Safari caps each tab's renderer at ~2 GB (the OS kills it via
  // jetsam) and has no Prompt API, so the ~365 MB SmolLM2 WASM fallback balloons
  // past that during inference and crashes the tab. Detect those devices so the
  // caller can skip the heavy model and show a note instead of crashing. Desktop
  // Safari has a far higher limit and plenty of RAM, so it stays on the normal
  // path.
  function heavyModelUnsupported() {
    if (typeof navigator === "undefined") { return false; }
    var ua = navigator.userAgent || "";
    return (
      /iP(hone|od|ad)/.test(ua) ||
      (navigator.platform === "MacIntel" && (navigator.maxTouchPoints || 0) > 1) // iPadOS-as-Mac
    );
  }

  // A desktop-class machine that can afford background generation (e.g.
  // pre-writing explanations nobody asked for). A luxury for fast computers
  // only: a weak device should not grind its CPU and battery on speculative
  // work. Low-end phones report 4–6 cores, so the bar is 8.
  var BACKGROUND_MIN_CORES = 8;
  function fastEnoughForBackground() {
    if (typeof navigator === "undefined") { return false; }
    if ((navigator.hardwareConcurrency || 0) < BACKGROUND_MIN_CORES) { return false; }
    if (typeof navigator.deviceMemory === "number" && navigator.deviceMemory < 8) { return false; }
    return true;
  }

  // ── "can this device run a local model at all?" ─────────────────────────────
  // A quick, synchronous verdict, so a caller can decide whether to even offer
  // the feature. True when the device is not too slow AND at least one backend is
  // viable: Chrome's Prompt API is present, OR the heavy SmolLM2 fallback can run
  // here (i.e. not iOS/iPadOS, where it OOM-crashes the tab). This is a STATIC
  // capability check — it does not confirm Nano is actually downloaded/available
  // (that is async; see nanoStatus / a provider's prewarm), only that a path
  // exists. An available Nano can still run on a device this calls "too slow",
  // so treat a downloaded Nano as an override where you have that signal.
  function canRun() {
    if (tooSlowForLlm()) { return false; }
    return (nanoApi() !== null) || !heavyModelUnsupported();
  }

  // Whether it is worth setting up cross-origin isolation (SharedArrayBuffer) so
  // the downloaded SmolLM2 WASM model can use several threads. True only when the
  // browser supports isolation and is not already isolated, the context is
  // secure, service workers are available, the machine has >= 4 cores (a slower
  // device never loads the heavy model), and there is NO Prompt API (Nano users
  // never touch the WASM model). This is the gate a page's isolation bootstrap
  // uses to decide whether to register its COOP/COEP service worker and reload
  // into an isolated context — the single home for that "hardwareConcurrency"
  // decision.
  function shouldCrossOriginIsolate() {
    if (typeof window === "undefined" || typeof navigator === "undefined") { return false; }
    if (!("crossOriginIsolated" in window)) { return false; }   // old browser: single thread only
    if (window.crossOriginIsolated) { return false; }           // already isolated: nothing to do
    if (!window.isSecureContext) { return false; }
    if (!("serviceWorker" in navigator)) { return false; }
    if (!hasEnoughCores()) { return false; }                    // heavy model won't load anyway
    if (nanoApi() !== null) { return false; }                   // Nano path: no WASM threads needed
    return true;
  }

  // A full, synchronous capability snapshot — for introspection, debugging, or a
  // one-call summary of everything above.
  function capabilities() {
    var nav = typeof navigator !== "undefined" ? navigator : {};
    var isolated = typeof window !== "undefined" && !!window.crossOriginIsolated;
    return {
      canRun: canRun(),
      tooSlow: tooSlowForLlm(),
      hardwareConcurrency: nav.hardwareConcurrency || 0,
      enoughCores: hasEnoughCores(),                  // >= HEAVY_MODEL_MIN_CORES
      deviceMemory: typeof nav.deviceMemory === "number" ? nav.deviceMemory : null,
      nano: { supported: nanoApi() !== null },        // API surface present (availability is async)
      heavyModel: { supported: !heavyModelUnsupported() },
      fastEnoughForBackground: fastEnoughForBackground(),
      crossOriginIsolation: {
        supported: typeof window !== "undefined" && ("crossOriginIsolated" in window),
        active: isolated,
        recommended: shouldCrossOriginIsolate()
      }
    };
  }

  // ── the Prompt API (Gemini Nano) ────────────────────────────────────────────

  // Chrome's built-in Prompt API, if this browser has it: the modern global
  // LanguageModel, or the older window.ai.languageModel. Null when neither.
  function nanoApi() {
    if (typeof LanguageModel !== "undefined" && LanguageModel && LanguageModel.create) {
      return LanguageModel;
    }
    if (G && G.ai && G.ai.languageModel && G.ai.languageModel.create) {
      return G.ai.languageModel;
    }
    return null;
  }
  // Normalize both API shapes to: "available" | "downloadable" | "downloading" | "unavailable".
  async function nanoStatus(api) {
    try {
      if (!api) { return "unavailable"; }
      if (api.availability) { return await api.availability(); }
      if (api.capabilities) {
        var a = (await api.capabilities()).available;   // "readily" | "after-download" | "no"
        return a === "readily" ? "available"
          : a === "after-download" ? "downloadable" : "unavailable";
      }
    } catch (e) { /* fall through */ }
    return "unavailable";
  }

  // ── shared reply-shape helper ────────────────────────────────────────────────
  // transformers.js returns [{ generated_text: ... }]; pull out the assistant's
  // final message whether it comes back as a chat array or a plain string.
  function replyOf(out) {
    try {
      var g = out[0].generated_text;
      if (Array.isArray(g)) { return (g[g.length - 1].content || "").trim(); }
      if (typeof g === "string") { return g.trim(); }
    } catch (e) { /* fall through */ }
    return "";
  }

  // ── generate with retries + a pluggable output filter ────────────────────────
  // The loop most apps wrap around brain.generate(): keep generating until the
  // output passes a quality filter, or the attempts run out. Each slow (non-Nano)
  // attempt is bounded by a timeout so a wedged WASM pass can't hang forever;
  // Nano's own (legitimate) cold start is never cut off. Tokens are forwarded for
  // the CURRENT attempt only, so late output from a timed-out or superseded
  // attempt never reaches the caller. Returns the accepted reply, or null
  // (nothing passed the filter, the attempt timed out / threw, or the caller
  // aborted via shouldContinue). Everything app-specific is a hook:
  //
  //   brain                a brain from loadGenerator() / buildBrain()
  //   messages             chat messages for brain.generate
  //   opts.options         generation options: an object, or (attempt) => object
  //                        (e.g. to lower temperature on each retry)
  //   opts.attempts        max attempts (default: 4 for a Nano brain, else 2)
  //   opts.timeoutMs       per-attempt timeout for slow backends (default 40000)
  //   opts.timeout         boolean: force the per-attempt timeout on/off for
  //                        every attempt, overriding the backend default (e.g.
  //                        pass your own "is this the slow backend?" signal)
  //   opts.timeoutBackends brain.backend values to bound with the timeout
  //                        (default: every backend except "nano")
  //   opts.postprocess     (rawText) => finalText, run before the filter
  //   opts.isBad           (finalText) => true to reject this attempt and retry
  //                        — the pluggable output filter
  //   opts.onToken         (textSoFar) => void, live tokens for the current
  //                        attempt only
  //   opts.onReject        (finalText, attempt) => void, after a rejected attempt
  //   opts.shouldContinue  () => false to abort early and return null (e.g. a
  //                        superseded request); checked around each attempt
  async function generateWithRetry(brain, messages, opts) {
    opts = opts || {};
    if (!brain || typeof brain.generate !== "function") {
      throw new Error("generateWithRetry: a brain with .generate is required");
    }
    var isNano = brain.backend === "nano";
    var attempts = typeof opts.attempts === "number" ? opts.attempts : (isNano ? 4 : 2);
    var timeoutMs = typeof opts.timeoutMs === "number" ? opts.timeoutMs : 40000;
    var post = typeof opts.postprocess === "function" ? opts.postprocess : function (s) { return s; };
    var isBad = typeof opts.isBad === "function" ? opts.isBad : function () { return false; };
    function cont() { return opts.shouldContinue ? !!opts.shouldContinue() : true; }
    function optionsFor(a) { return typeof opts.options === "function" ? opts.options(a) : (opts.options || {}); }
    function wantsTimeout() {
      if (typeof opts.timeout === "boolean") { return opts.timeout; }
      if (opts.timeoutBackends) { return opts.timeoutBackends.indexOf(brain.backend) !== -1; }
      return !isNano;
    }
    // A fresh token forwarder per attempt (its own `live`), so tokens from an
    // abandoned attempt stop the moment that attempt ends.
    function makeForwarder() {
      var live = true;
      return {
        fn: opts.onToken ? function (t) { if (live && cont()) { try { opts.onToken(t); } catch (e) {} } } : undefined,
        stop: function () { live = false; }
      };
    }
    for (var attempt = 0; attempt < attempts; attempt++) {
      if (!cont()) { return null; }
      var fwd = makeForwarder();
      var raw;
      try {
        var genP = brain.generate(messages, optionsFor(attempt), fwd.fn);
        if (wantsTimeout()) {
          raw = await Promise.race([genP, new Promise(function (_res, rej) {
            setTimeout(function () { rej(new Error("generateWithRetry: attempt timed out")); }, timeoutMs);
          })]);
        } else {
          raw = await genP;
        }
      } catch (e) {
        fwd.stop();       // stop any late tokens from the wedged attempt
        return null;      // timed out or the backend threw → give up gracefully
      }
      fwd.stop();
      if (!cont()) { return null; }
      var reply = post(raw);
      if (!isBad(reply)) { return reply; }              // accepted
      if (opts.onReject) { try { opts.onReject(reply, attempt); } catch (e) {} }
    }
    return null;   // ran out of attempts without a good reply
  }

  // ── default connection copy ──────────────────────────────────────────────────
  // Human-readable reasons to NOT auto-pull the heavy fallback model right now.
  // Neutral defaults; a caller (e.g. Cool Concepts) passes its own voice via
  // config.messages so the exact wording — and behavior — stays identical.
  var DEFAULT_MESSAGES = {
    offline: "You are offline. Setting up the on-device model needs a one-time download the first time it runs.",
    saveData: "Data Saver is on, so the ~360 MB model will not download automatically. Turn it off (or switch to Wi-Fi) and try again.",
    slow: "Your connection looks too slow for the one-time ~360 MB download the model needs. Try again on a faster network."
  };

  var DEFAULT_TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

  function noop() {}

  // ── the worker: transformers.js inference off the main thread ────────────────
  // Built as a string and loaded as a module worker via a Blob, so seconds-long
  // WASM forward passes never freeze the page. Rejects to the main-thread path
  // when the browser cannot run a module worker that imports from a CDN.
  function buildWorkerSource(modelId, dtype, transformersUrl) {
    return `
import { pipeline, TextStreamer, InterruptableStoppingCriteria, env } from "${transformersUrl}";
// SIMD is on by default (needs no cross-origin isolation; the scalar build is
// 2-4x slower). Threads DO need crossOriginIsolated (SharedArrayBuffer). When
// isolated, use ~half the logical cores (>=2, capped at 4) - decode is only
// ~1.5-2.5x faster with threads and each one costs RAM, so more would risk OOM
// on phones for little gain. Not isolated -> pin to 1.
try {
  var _wasm = env.backends.onnx.wasm;
  var _coi = (typeof crossOriginIsolated !== "undefined" && crossOriginIsolated);
  var _cores = (self.navigator && self.navigator.hardwareConcurrency) || 1;
  _wasm.numThreads = _coi ? Math.min(4, Math.max(2, _cores >> 1)) : 1;
} catch (e) {}
var genP = null;
function getGen() {
  if (!genP) {
    genP = pipeline("text-generation", "${modelId}", {
      device: "wasm", dtype: "${dtype}",
      progress_callback: function (p) { self.postMessage({ type: "progress", p: p }); }
    });
  }
  return genP;
}
function replyOf(out) {
  try {
    var g = out[0].generated_text;
    if (Array.isArray(g)) return (g[g.length - 1].content || "").trim();
    if (typeof g === "string") return g.trim();
  } catch (e) {}
  return "";
}
self.onmessage = async function (e) {
  var msg = e.data || {};
  if (msg.type === "load") {
    try { await getGen(); self.postMessage({ type: "ready" }); }
    catch (err) { self.postMessage({ type: "loaderror", message: String((err && err.message) || err) }); }
    return;
  }
  if (msg.type === "generate") {
    var id = msg.id;
    try {
      var gen = await getGen();
      var opts = msg.opts || {};
      var o = { max_new_tokens: opts.max_new_tokens, do_sample: true,
        top_p: opts.top_p || 0.9, temperature: opts.temperature,
        repetition_penalty: opts.repetition_penalty || 1.15 };
      // Optional decoding constraints (transformers.js logits processors), forwarded only when set.
      if (typeof opts.no_repeat_ngram_size === "number") o.no_repeat_ngram_size = opts.no_repeat_ngram_size;
      if (typeof opts.top_k === "number") o.top_k = opts.top_k;
      if (opts.bad_words_ids) o.bad_words_ids = opts.bad_words_ids;
      // Stop sequences: transformers.js 4.2.0 has no stop_strings config, so emulate it by
      // interrupting generation from the stream callback once the output contains a stop string.
      var stops = Array.isArray(opts.stop_strings) ? opts.stop_strings.filter(Boolean) : null;
      var stopper = (stops && stops.length && InterruptableStoppingCriteria) ? new InterruptableStoppingCriteria() : null;
      if (stopper) { o.stopping_criteria = stopper; }
      if (msg.stream && gen.tokenizer && TextStreamer) {
        var streamed = "";
        o.streamer = new TextStreamer(gen.tokenizer, {
          skip_prompt: true,
          callback_function: function (chunk) {
            streamed += chunk;
            var clean = streamed.replace(/<\\|[^|]*\\|>/g, "");
            self.postMessage({ type: "token", id: id, text: clean });
            if (stopper) { for (var si = 0; si < stops.length; si++) { if (clean.indexOf(stops[si]) !== -1) { stopper.interrupt(); break; } } }
          }
        });
      }
      var out = await gen(msg.messages, o);
      self.postMessage({ type: "result", id: id, text: replyOf(out) });
    } catch (err) {
      self.postMessage({ type: "error", id: id, message: String((err && err.message) || err) });
    }
  }
};
self.postMessage({ type: "boot" });
`;
  }

  // ── the provider: owns model state and produces the brain ────────────────────
  function create(config) {
    config = config || {};
    var messages = {};
    var srcMsg = config.messages || {};
    messages.offline  = srcMsg.offline  != null ? srcMsg.offline  : DEFAULT_MESSAGES.offline;
    messages.saveData = srcMsg.saveData != null ? srcMsg.saveData : DEFAULT_MESSAGES.saveData;
    messages.slow     = srcMsg.slow     != null ? srcMsg.slow     : DEFAULT_MESSAGES.slow;

    var transformersUrl = config.transformersUrl || DEFAULT_TRANSFORMERS_URL;
    var bootTimeoutMs = typeof config.workerBootTimeoutMs === "number" ? config.workerBootTimeoutMs : 20000;
    // Clock for the download-ETA estimate (injectable so it can be tested
    // deterministically). Monotonic performance.now() where available.
    var nowFn = typeof config.now === "function" ? config.now
      : (typeof performance !== "undefined" && performance.now)
        ? function () { return performance.now(); }
        : function () { return Date.now(); };
    // Smoothing for the download-rate estimate (0..1; higher trusts the latest
    // sample more). The rate is an EMA of progress-per-ms across samples, so the
    // ETA tracks the CURRENT speed rather than the average since the start.
    var etaSmoothing = typeof config.etaSmoothing === "number" ? config.etaSmoothing : 0.35;
    // Where the "this browser already downloaded the fallback model" flag lives.
    // Keyed to the model+dtype, so a model/dtype change invalidates a stale
    // "ready". Cool Concepts passes "cc:tf" to keep its returning-visitor signal.
    var weightsFlagKey = config.weightsFlagKey || "browser_llm:tf-ready";

    var onProgress   = config.onProgress   || noop;   // (pct, {loaded,total,files,etaMs,etaSeconds,bytesPerSec})
    var onModelReady = config.onModelReady || noop;   // ()
    var onModelError = config.onModelError || noop;   // (err)
    var onNanoStatus = config.onNanoStatus || noop;   // (nanoReady:boolean)
    var onStateChange = config.onStateChange || noop; // () — nanoReady/modelReady/weightsCached changed

    var smol = MODELS.smol;
    var TF_MODEL_ID = config.modelId || smol.id;
    var TF_DTYPE = config.dtype || smol.dtype;
    // Opt-in for a caller whose configured model is small enough to run within a
    // constrained device's memory. heavyModelUnsupported() hard-stops the
    // on-device model on iOS/iPadOS because the ~365 MB default OOM-crashes the
    // tab there; a caller running a much smaller model (e.g. a ~137 MB int8 one)
    // can set this to run it on those devices anyway.
    var allowConstrained = !!config.allowConstrainedDevice;
    // Nano-only mode: use Chrome's built-in Gemini Nano (Prompt API) exclusively and NEVER download
    // or fall back to the SmolLM2 model. buildBrain() throws instead of falling back when Nano is
    // absent or fails to start, and prewarm never prefetches the fallback weights. For callers that
    // only want the zero-download built-in model.
    var nanoOnly = !!config.nanoOnly;
    var TF_TAG = TF_MODEL_ID + "|" + TF_DTYPE;
    var TF_CONFIG_URL = "https://huggingface.co/" + TF_MODEL_ID + "/resolve/main/config.json";

    var self_ = {
      // public, read-only-ish state (mirrors Cool Concepts' flags)
      modelReady: false,
      loading: false,         // a brain build is in flight (started, not yet ready/failed)
      nanoReady: false,       // built-in AI usable now (runs offline)
      weightsCached: false,   // the fallback model already downloaded on this device
      dlPct: 0,               // overall model-download %, byte-weighted across files
      MODELS: MODELS,
      CONTEXT_WINDOW: CONTEXT_WINDOW
    };

    // internal state
    var dlFiles = {};                 // file -> {loaded,total}
    var generatorPromise = null;      // memoized brain
    var tfPipelinePromise = null;
    var tfWorkerRef = null;
    var tfSpareDropped = false;
    var tfPrimary = false;            // true once transformers is THE brain (its progress may paint)
    var tf = null;                    // the transformers.js module (for main-thread TextStreamer)

    // download-ETA estimation state
    var etaStart = null;              // ms timestamp of the first progress sample
    var etaLastT = null;              // ms timestamp of the last sample
    var etaLastFrac = 0;              // last aggregate progress fraction (0..1)
    var etaRate = null;               // smoothed progress fraction per ms (EMA)
    var etaLastBytes = 0;             // last aggregate loaded bytes (0 when unknown)
    var etaByteRate = null;           // smoothed bytes per ms (EMA), when byte totals are known

    function fireState() { try { onStateChange(); } catch (e) {} }

    function resetEta() {
      etaStart = null; etaLastT = null; etaLastFrac = 0;
      etaRate = null; etaLastBytes = 0; etaByteRate = null;
    }
    // Feed one progress reading (frac in 0..1; byte counts optional, 0 when
    // unknown) into the rate estimate.
    function sampleProgress(frac, loadedBytes, totalBytes) {
      if (typeof frac !== "number" || !isFinite(frac)) { return; }
      if (frac < 0) { frac = 0; } else if (frac > 1) { frac = 1; }
      var bytes = loadedBytes || 0;
      var t = nowFn();
      if (etaStart === null) {          // first sample: anchor, no rate yet
        etaStart = t; etaLastT = t; etaLastFrac = frac; etaLastBytes = bytes;
        return;
      }
      var dt = t - etaLastT;
      if (dt <= 0) {                    // same tick / clock stall: only raise the level
        if (frac > etaLastFrac) { etaLastFrac = frac; }
        if (bytes > etaLastBytes) { etaLastBytes = bytes; }
        return;
      }
      var dFrac = frac - etaLastFrac;   // progress since last sample (ignore regressions)
      if (dFrac > 0) {
        var instRate = dFrac / dt;      // fraction per ms
        etaRate = (etaRate === null) ? instRate : (etaSmoothing * instRate + (1 - etaSmoothing) * etaRate);
      }
      var dBytes = bytes - etaLastBytes;
      if (dBytes > 0) {
        var instByte = dBytes / dt;     // bytes per ms
        etaByteRate = (etaByteRate === null) ? instByte : (etaSmoothing * instByte + (1 - etaSmoothing) * etaByteRate);
      }
      etaLastT = t;
      if (frac > etaLastFrac) { etaLastFrac = frac; }
      if (bytes > etaLastBytes) { etaLastBytes = bytes; }
    }
    function bytesPerSecOrNull() { return (etaByteRate && etaByteRate > 0) ? Math.round(etaByteRate * 1000) : null; }
    // Estimated time remaining for the in-flight model download, or null when
    // there is nothing to estimate yet (no samples, or the rate has stalled).
    // Shape: { etaMs, etaSeconds, bytesPerSec, loaded, total, pct, done }.
    function downloadEta() {
      var loaded = 0, total = 0, f;
      for (f in dlFiles) { loaded += dlFiles[f].loaded; total += dlFiles[f].total; }
      if (self_.modelReady || etaLastFrac >= 1) {
        return { etaMs: 0, etaSeconds: 0, bytesPerSec: bytesPerSecOrNull(),
          loaded: loaded, total: total, pct: self_.dlPct, done: true };
      }
      if (etaRate === null || etaRate <= 0) { return null; }   // not enough info yet / stalled
      var remainingFrac = 1 - etaLastFrac;
      var etaMs = remainingFrac / etaRate;
      return { etaMs: etaMs, etaSeconds: Math.round(etaMs / 1000), bytesPerSec: bytesPerSecOrNull(),
        loaded: loaded, total: total, pct: self_.dlPct, done: false };
    }

    // ── returning-visitor cache probe ──────────────────────────────────────────
    function markTfReady() {          // the downloaded model finished loading this session
      self_.weightsCached = true;
      try { localStorage.setItem(weightsFlagKey, TF_TAG); } catch (e) { /* private mode */ }
      fireState();
    }
    // transformers.js stores small files (config, tokenizer) in the Cache API but
    // the big weights go to the HTTP disk cache. So "downloaded before" is: (1) a
    // flag WE set on a completed load, or (2) the model's config.json sitting in
    // the Cache API. A hint, not a guarantee (eviction/incognito can lie).
    function probeWeightsCache() {
      try {
        if (localStorage.getItem(weightsFlagKey) === TF_TAG) {
          self_.weightsCached = true; fireState(); return Promise.resolve(true);
        }
      } catch (e) { /* private mode → fall through to the cache probe */ }
      try {
        if (typeof caches === "undefined" || !caches.match) { return Promise.resolve(false); }
        return caches.match(TF_CONFIG_URL).then(function (hit) {
          self_.weightsCached = !!hit;
          fireState();
          return self_.weightsCached;
        }).catch(function () { return false; });
      } catch (e) { return Promise.resolve(false); }
    }

    // ── download progress accounting ───────────────────────────────────────────
    function emitProgress() {
      var loaded = 0, total = 0, f;
      for (f in dlFiles) { loaded += dlFiles[f].loaded; total += dlFiles[f].total; }
      var eta = downloadEta();
      try {
        onProgress(self_.dlPct, {
          loaded: loaded, total: total, files: dlFiles,
          etaMs: eta ? eta.etaMs : null,
          etaSeconds: eta ? eta.etaSeconds : null,
          bytesPerSec: eta ? eta.bytesPerSec : null
        });
      } catch (e) {}
    }
    // Fold one transformers.js download-progress event into the overall percent,
    // byte-weighted across files (a per-file max would jump to 100% the instant
    // the tiny config finishes, long before the big weights file is done).
    function applyDlProgress(p) {
      if (!p || !p.file) { return; }
      if (p.status === "progress" && p.total) {
        dlFiles[p.file] = { loaded: p.loaded || 0, total: p.total };
      } else if (p.status === "done" && dlFiles[p.file]) {
        dlFiles[p.file].loaded = dlFiles[p.file].total;   // finished = fully loaded
      } else {
        return;
      }
      var loaded = 0, total = 0, f;
      for (f in dlFiles) { loaded += dlFiles[f].loaded; total += dlFiles[f].total; }
      self_.dlPct = total ? Math.round(loaded / total * 100) : 0;
      if (total) { sampleProgress(loaded / total, loaded, total); }
      emitProgress();
    }
    // Nano's one-time built-in-model download (Chrome-managed, shared across
    // sites) reports its own progress. The modern spec gives loaded as a 0..1
    // fraction (no total); some impls give loaded/total bytes.
    function nanoMonitor(m) {
      if (!m || !m.addEventListener) { return; }
      m.addEventListener("downloadprogress", function (e) {
        var hasBytes = !!(e && e.total && e.loaded > 1);
        var p = hasBytes ? e.loaded / e.total : (e ? e.loaded : 0);
        if (typeof p === "number" && isFinite(p)) {
          self_.dlPct = Math.max(self_.dlPct, Math.round(p * 100));
          if (hasBytes) { sampleProgress(e.loaded / e.total, e.loaded, e.total); }
          else { sampleProgress(p, 0, 0); }
          emitProgress();
        }
      });
    }

    // ── connection gating ──────────────────────────────────────────────────────
    // Returns a human message when the heavy fallback should NOT auto-download
    // right now, or null to go ahead. A returning visitor (weights cached) always
    // gets null — there is nothing to fetch, and it even works offline.
    function connectionBlock() {
      if (self_.weightsCached) { return null; }
      if (typeof navigator === "undefined") { return null; }
      if (navigator.onLine === false) { return messages.offline; }
      var conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      if (conn) {
        if (conn.saveData) { return messages.saveData; }
        if (conn.effectiveType === "slow-2g" || conn.effectiveType === "2g") { return messages.slow; }
      }
      return null;
    }
    // Like connectionBlock, but first give Nano a chance: if the Prompt API is
    // available/downloadable/downloading, it manages its own small download and
    // there is nothing for us to fetch — so never block.
    async function llmDownloadBlock() {
      if (self_.modelReady) { return null; }
      var api = nanoApi();
      if (api) {
        try {
          var st = await nanoStatus(api);
          if (st === "available" || st === "downloadable" || st === "downloading") { return null; }
        } catch (e) { /* fall through to the transformers path */ }
      }
      return connectionBlock();
    }

    // ── Nano backend ────────────────────────────────────────────────────────────
    // messages are [ ...prefix, finalUserTurn ]: a session is created with the
    // prefix (system + few-shot + any prior Q&A) and prompted with the final user
    // turn. Streams token-by-token when a callback is given.
    function makeNanoBrain(api) {
      return {
        backend: "nano",
        generate: async function (msgs, opts, onToken) {
          opts = opts || {};
          var initial = msgs.slice(0, -1).map(function (m) {
            return { role: m.role, content: m.content };
          });
          var last = msgs[msgs.length - 1];
          var session;
          var createOpts = { initialPrompts: initial,
            expectedInputs: NANO_LANG.expectedInputs, expectedOutputs: NANO_LANG.expectedOutputs };
          // Nano wants temperature+topK together, or neither.
          if (typeof opts.temperature === "number") {
            createOpts.temperature = opts.temperature;
            createOpts.topK = 3;
          }
          try { session = await api.create(createOpts); }
          catch (e) { session = await api.create({ initialPrompts: initial,
            expectedInputs: NANO_LANG.expectedInputs, expectedOutputs: NANO_LANG.expectedOutputs }); }
          try {
            if (onToken && session.promptStreaming) {
              var stream = session.promptStreaming(last.content);
              var full = "";
              // Hand-rolled async iteration instead of "for await": that syntax
              // is a parse error on older browsers, and this file must PARSE
              // everywhere even though this path only runs on modern Chrome.
              var it = stream[Symbol.asyncIterator] ? stream[Symbol.asyncIterator]() : null;
              while (it) {
                var step = await it.next();
                if (step.done) { break; }
                var chunk = step.value;
                // Chrome <=130 streamed the full text each time; newer streams
                // deltas. Detect which, and accumulate correctly either way.
                if (chunk.length >= full.length && chunk.indexOf(full) === 0) { full = chunk; }
                else { full += chunk; }
                onToken(full);
              }
              return full;
            }
            return await session.prompt(last.content);
          } finally {
            if (session && session.destroy) { try { session.destroy(); } catch (e) { /* ignore */ } }
          }
        }
      };
    }

    // ── transformers.js (SmolLM2) backend ────────────────────────────────────────
    // Wrap the worker in the same {generate} shape as the Nano/main-thread brains.
    // Each call is a round-trip: tokens stream back as "token" messages, the final
    // reply arrives as "result". An id keeps two concurrent jobs from crossing.
    function makeTfWorkerBrain(worker) {
      var nextId = 0, pending = {};
      worker.onmessage = function (e) {
        var msg = e.data || {};
        var job = pending[msg.id];
        if (!job) { return; }
        if (msg.type === "token") { if (job.onToken) { try { job.onToken(msg.text); } catch (e2) {} } return; }
        if (msg.type === "result") { delete pending[msg.id]; job.resolve(msg.text || ""); return; }
        if (msg.type === "error") { delete pending[msg.id]; job.reject(new Error(msg.message || "worker error")); }
      };
      // A hard crash (e.g. an OOM kill on a borderline phone) fires onerror, not a
      // "message"; without this the in-flight generate() would never settle and
      // would dead-lock every future call. Reject the pending job(s) so the caller
      // can release its lock and show a "got stuck" note.
      worker.onerror = function (ev) {
        if (ev && ev.preventDefault) { try { ev.preventDefault(); } catch (e2) {} }
        var ids = Object.keys(pending);
        for (var i = 0; i < ids.length; i++) {
          var job = pending[ids[i]]; delete pending[ids[i]];
          try { job.reject(new Error("word-brain worker crashed")); } catch (e3) {}
        }
      };
      return {
        backend: "smol-worker",
        generate: function (msgs, opts, onToken) {
          return new Promise(function (resolve, reject) {
            var id = ++nextId;
            pending[id] = { resolve: resolve, reject: reject, onToken: onToken };
            worker.postMessage({ type: "generate", id: id,
              messages: msgs, opts: opts || {}, stream: !!onToken });
          });
        }
      };
    }

    // Spin up the inference worker and download SmolLM2 inside it. Rejects
    // {fallback:true} when the browser cannot run a module worker that imports
    // from a CDN (older Safari); the caller then uses the main thread. A real
    // download/init failure rejects {fallback:false} so we do NOT retry the whole
    // ~360 MB on the main thread.
    function loadTfWorker() {
      return new Promise(function (resolve, reject) {
        var worker, url, booted = false, settled = false;
        var src = buildWorkerSource(TF_MODEL_ID, TF_DTYPE, transformersUrl);
        try {
          url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
          worker = new Worker(url, { type: "module" });
          tfWorkerRef = worker;                       // so dropTfSpare can free it
        } catch (e) { reject({ fallback: true }); return; }
        // No "boot" (the module import was blocked) → treat the worker as unusable.
        var bootTimer = setTimeout(function () {
          if (booted || settled) { return; }
          settled = true; try { worker.terminate(); } catch (e) {}
          reject({ fallback: true });
        }, bootTimeoutMs);
        worker.onerror = function (e) {
          // Mark the error handled so a blocked CDN does not splash a console
          // error on page load. We fall back (or fail politely) on our own.
          if (e && e.preventDefault) { try { e.preventDefault(); } catch (e2) { /* ignore */ } }
          if (settled) { return; }
          settled = true; clearTimeout(bootTimer);
          try { worker.terminate(); } catch (e3) {}
          reject({ fallback: true });     // setup failed; a post-boot error comes as a message
        };
        worker.onmessage = function (e) {
          var msg = e.data || {};
          if (msg.type === "boot") {
            booted = true;
            if (url) { try { URL.revokeObjectURL(url); } catch (e2) {} url = null; }
            worker.postMessage({ type: "load" });   // start the one-time model download
          } else if (msg.type === "progress") {
            if (tfPrimary) { applyDlProgress(msg.p); }   // a silent prefetch paints nothing
          } else if (msg.type === "ready") {
            if (settled) { return; }
            settled = true; clearTimeout(bootTimer);
            worker.onerror = null;                   // makeTfWorkerBrain installs the per-request/crash handler
            markTfReady();                           // remember this browser has the model
            resolve(makeTfWorkerBrain(worker));
          } else if (msg.type === "loaderror") {
            if (settled) { return; }
            settled = true; clearTimeout(bootTimer);
            try { worker.terminate(); } catch (e2) {}
            reject({ fallback: false, message: msg.message });
          }
        };
      });
    }

    // Main-thread pipeline (fallback), only where a module worker cannot be used.
    // Always the WASM backend (transformers.js 4.2.0's WebGPU path throws on init
    // and has no safe in-place fallback once its weights have downloaded).
    function loadTfPipelineMain() {
      // import() is ES2020 SYNTAX: written literally it would kill this whole file
      // on older parsers. Built inside Function, it only gets parsed on the modern
      // browsers that actually reach this fallback.
      var dynImport = new Function("u", "return import(u)");
      return dynImport(transformersUrl).then(function (t) {
        tf = t;                          // keep the module around for TextStreamer
        try {                            // SIMD on, threads capped (see the worker note)
          var w = t.env.backends.onnx.wasm;
          var coi = (typeof crossOriginIsolated !== "undefined" && crossOriginIsolated);
          var cores = (navigator && navigator.hardwareConcurrency) || 1;
          w.numThreads = coi ? Math.min(4, Math.max(2, cores >> 1)) : 1;
        } catch (e) { /* ignore */ }
        return t.pipeline(
          "text-generation", TF_MODEL_ID,
          { device: "wasm", dtype: TF_DTYPE,
            progress_callback: function (p) { if (tfPrimary) { applyDlProgress(p); } } }
        );
      }).then(function (gen) { markTfReady(); return makeTfBrain(gen); });
    }

    // Wrap a main-thread transformers.js pipeline in the same {generate} shape.
    function makeTfBrain(gen) {
      return {
        backend: "smol-main",
        generate: async function (msgs, opts, onToken) {
          opts = opts || {};
          var o = { max_new_tokens: opts.max_new_tokens, do_sample: true,
            top_p: opts.top_p || 0.9, temperature: opts.temperature,
            repetition_penalty: opts.repetition_penalty || 1.15 };
          // Optional decoding constraints (transformers.js logits processors), forwarded only when set.
          if (typeof opts.no_repeat_ngram_size === "number") o.no_repeat_ngram_size = opts.no_repeat_ngram_size;
          if (typeof opts.top_k === "number") o.top_k = opts.top_k;
          if (opts.bad_words_ids) o.bad_words_ids = opts.bad_words_ids;
          // Stop sequences: emulate via an interruptable stopping criteria driven by the stream (4.2.0
          // has no stop_strings config).
          var stops = Array.isArray(opts.stop_strings) ? opts.stop_strings.filter(Boolean) : null;
          var stopper = (stops && stops.length && tf && tf.InterruptableStoppingCriteria) ? new tf.InterruptableStoppingCriteria() : null;
          if (stopper) { o.stopping_criteria = stopper; }
          if (onToken && tf && tf.TextStreamer && gen.tokenizer) {
            var streamed = "";
            o.streamer = new tf.TextStreamer(gen.tokenizer, {
              skip_prompt: true,
              callback_function: function (chunk) {
                streamed += chunk;
                var clean = streamed.replace(/<\|[^|]*\|>/g, "");
                onToken(clean);
                if (stopper) { for (var si = 0; si < stops.length; si++) { if (clean.indexOf(stops[si]) !== -1) { stopper.interrupt(); break; } } }
              }
            });
          }
          var out = await gen(msgs, o);
          return replyOf(out);
        }
      };
    }

    // The heavy fallback, memoized so a page-load prefetch and the click path
    // share one download. Worker first; main thread only where a module worker
    // cannot be used. Hard-stops on devices where the model OOM-crashes the tab,
    // unless the caller opted in via allowConstrainedDevice (a small enough model).
    function loadTfPipeline() {
      if (tfPipelinePromise) { return tfPipelinePromise; }
      if (heavyModelUnsupported() && !allowConstrained) {
        return Promise.reject(new Error("on-device model unsupported on this device"));
      }
      tfSpareDropped = false;
      tfPipelinePromise = loadTfWorker().catch(function (e) {
        if (e && e.fallback === false) { throw (e.message ? new Error(e.message) : e); }
        if (tfSpareDropped) { throw (e && e.message ? e : new Error("spare dropped")); }
        return loadTfPipelineMain();
      });
      tfPipelinePromise.then(null, function () { tfPipelinePromise = null; });  // failed: let a later call retry
      return tfPipelinePromise;
    }
    // Nano won at click time: the fallback model's files stay cached, but the
    // spare worker's RAM is freed and its fallbacks are stopped.
    function dropTfSpare() {
      tfSpareDropped = true;
      if (tfWorkerRef) { try { tfWorkerRef.terminate(); } catch (e) { /* ignore */ } tfWorkerRef = null; }
      tfPipelinePromise = null;
    }

    // Nano first (no download from us); only if it is missing or fails to start
    // do we download SmolLM2.
    async function buildBrain() {
      var api = nanoApi();
      if (api) {
        var status = await nanoStatus(api);
        if (status === "available" || status === "downloadable" || status === "downloading") {
          try {
            // Prepare Nano (downloads the built-in model once if needed). A fresh
            // session per call, so this just warms/downloads, then frees it.
            var warm = await api.create({ monitor: nanoMonitor,
              expectedInputs: NANO_LANG.expectedInputs, expectedOutputs: NANO_LANG.expectedOutputs });
            if (warm && warm.destroy) { try { warm.destroy(); } catch (e) { /* ignore */ } }
            dropTfSpare();   // Nano is the brain: free any prefetched spare (files stay cached)
            return makeNanoBrain(api);
          } catch (e) {
            if (nanoOnly) { throw e; }   // nanoOnly: surface the Nano error, never fall back to SmolLM2
            /* Nano failed to start → fall through to transformers */
          }
        } else if (nanoOnly) {
          throw new Error("browser_llm: nanoOnly is set but Gemini Nano is unavailable (status: " + status + ").");
        }
      } else if (nanoOnly) {
        throw new Error("browser_llm: nanoOnly is set but the Gemini Nano Prompt API is not present.");
      }
      tfPrimary = true;                // transformers is the brain: its progress may paint
      return await loadTfPipeline();   // no usable built-in AI → download our own model
    }

    // The one call most callers use: get the brain, building it once. Sets
    // modelReady and fires onModelReady on success; on failure it resets so a
    // later call can retry.
    function loadGenerator() {
      if (generatorPromise) { return generatorPromise; }
      self_.loading = true;           // a build is now in flight
      fireState();
      generatorPromise = buildBrain();
      generatorPromise.then(
        function () {
          self_.modelReady = true; self_.loading = false;
          fireState();
          try { onModelReady(); } catch (e) {}
        },
        function (err) {
          generatorPromise = null; self_.loading = false;
          self_.dlPct = 0; dlFiles = {}; resetEta();
          try { onModelError(err); } catch (e) {}   // caller re-syncs; no UI hook, matching a silent reset
        }
      );
      return generatorPromise;
    }

    // Warm the brain in the background as soon as the page settles, so the first
    // real request does not sit through the whole download. Same connection rules
    // as the click path (offline / Data Saver / very slow link block the auto-
    // pull). Call this on load and again on "online".
    function prewarm() {
      var api = nanoApi();
      if (!api) {
        self_.nanoReady = false; fireState();
        try { onNanoStatus(false); } catch (e) {}
        prewarmModel();
        return;
      }
      nanoStatus(api).then(function (status) {
        self_.nanoReady = (status === "available");   // downloaded & usable (works offline)
        fireState();
        try { onNanoStatus(self_.nanoReady); } catch (e) {}
        prewarmModel();
      }).catch(function () {
        self_.nanoReady = false; fireState();
        try { onNanoStatus(false); } catch (e) {}
        prewarmModel();
      });
    }
    function prewarmModel() {
      if (generatorPromise || self_.modelReady) { return; }
      if (tooSlowForLlm() && !self_.nanoReady) { return; }   // weak device: no model, ever
      var api = nanoApi();
      if (nanoOnly && !api) { return; }   // nanoOnly: no Prompt API → nothing to prewarm (never SmolLM2)
      if (api) {
        nanoStatus(api).then(function (st) {
          if (st === "downloadable" || st === "downloading") {
            // nanoOnly: do NOT prefetch the SmolLM2 fallback; Nano installs on the first user gesture.
            if (nanoOnly) { return; }
            // Installing Nano needs a user gesture (Chrome refuses silent model
            // downloads), so its create() waits for the first click. Meanwhile,
            // prefetch the fallback brain right away: both models arrive as soon
            // as possible, and if Nano cannot start after all, nobody sits
            // through a cold download.
            if (!connectionBlock()) { loadTfPipeline().then(null, function () { /* click path retries */ }); }
            return;
          }
          // "available": warm it now (installed, no gesture needed). Anything
          // else: the transformers path is the brain.
          llmDownloadBlock().then(function (block) {
            if (!block) { loadGenerator(); }
          }).catch(function () { /* stay lazy */ });
        }).catch(function () { /* stay lazy; the click path handles it */ });
        return;
      }
      llmDownloadBlock().then(function (block) {
        if (!block) { loadGenerator(); }
      }).catch(function () { /* any doubt → stay lazy; the click path handles it */ });
    }

    // Public methods on the provider.
    self_.tooSlowForLlm = tooSlowForLlm;
    self_.heavyModelUnsupported = heavyModelUnsupported;
    self_.fastEnoughForBackground = fastEnoughForBackground;
    self_.emojiRenders = emojiRenders;
    self_.canRun = canRun;
    self_.hasEnoughCores = hasEnoughCores;
    self_.shouldCrossOriginIsolate = shouldCrossOriginIsolate;
    self_.capabilities = capabilities;
    self_.nanoApi = nanoApi;
    self_.nanoStatus = function () { return nanoStatus(nanoApi()); };
    self_.connectionBlock = connectionBlock;
    self_.llmDownloadBlock = llmDownloadBlock;
    self_.probeWeightsCache = probeWeightsCache;
    self_.loadGenerator = loadGenerator;
    self_.buildBrain = buildBrain;
    self_.generateWithRetry = generateWithRetry;
    self_.loadTfPipeline = loadTfPipeline;
    self_.dropTfSpare = dropTfSpare;
    self_.prewarm = prewarm;
    // exposed for tests / advanced callers
    self_.workerSource = function () { return buildWorkerSource(TF_MODEL_ID, TF_DTYPE, transformersUrl); };
    self_.applyDlProgress = applyDlProgress;      // exposed for testing the byte-weighting
    self_.downloadFiles = function () { return dlFiles; };
    self_.downloadEta = downloadEta;              // estimated time remaining for the model download

    return self_;
  }

  return {
    VERSION: "1.0.0",
    // model catalog + context windows (constants)
    MODELS: MODELS,
    CONTEXT_WINDOW: CONTEXT_WINDOW,
    NANO_LANG: NANO_LANG,
    // static device / capability helpers (no provider needed)
    tooSlowForLlm: tooSlowForLlm,
    heavyModelUnsupported: heavyModelUnsupported,
    fastEnoughForBackground: fastEnoughForBackground,
    emojiRenders: emojiRenders,
    canRun: canRun,
    hasEnoughCores: hasEnoughCores,
    HEAVY_MODEL_MIN_CORES: HEAVY_MODEL_MIN_CORES,
    shouldCrossOriginIsolate: shouldCrossOriginIsolate,
    capabilities: capabilities,
    // static Prompt API helpers
    nanoApi: nanoApi,
    nanoStatus: nanoStatus,
    replyOf: replyOf,
    // generation loop (retries + pluggable output filter)
    generateWithRetry: generateWithRetry,
    // factory
    create: create
  };
});
