/* Demo glue for browser-llm.js. Not part of the library — just a page to try it.
   Shows the model catalog + context-window constants, reports which backend the
   library picked, paints download progress, and streams tokens live. */
(function () {
  "use strict";

  var promptEl = document.getElementById("prompt");
  var goBtn = document.getElementById("go");
  var outEl = document.getElementById("out");
  var statusEl = document.getElementById("status");
  var backendEl = document.getElementById("backend-note");
  var progressEl = document.getElementById("progress");
  var barEl = document.getElementById("bar");
  var statusNano = document.getElementById("status-nano");
  var statusSmol = document.getElementById("status-smol");
  var deviceNote = document.getElementById("device-note");

  // Show the context-window constants straight from the catalog.
  document.getElementById("ctx-nano").textContent =
    BrowserLLM.CONTEXT_WINDOW.nano.toLocaleString() + " tokens (nominal)";
  document.getElementById("ctx-smol").textContent =
    BrowserLLM.CONTEXT_WINDOW.smol.toLocaleString() + " tokens";

  // The device gates the same way the library will decide whether to run.
  var tooSlow = BrowserLLM.tooSlowForLlm();
  var noHeavy = BrowserLLM.heavyModelUnsupported();
  var bits = [];
  if (tooSlow) { bits.push("this device is flagged too slow/old for a local model"); }
  if (noHeavy) { bits.push("the ~360 MB fallback would crash this device (iOS/iPadOS), so only Nano can run here"); }
  deviceNote.textContent = bits.length
    ? "Heads up: " + bits.join("; ") + "."
    : "This device can run a local model.";

  var llm = BrowserLLM.create({
    onProgress: function (pct, info) {
      progressEl.hidden = false;
      barEl.style.width = Math.max(2, pct) + "%";
      var eta = (info && info.etaSeconds != null) ? " (~" + fmtEta(info.etaSeconds) + " left)" : "";
      statusEl.textContent = "Downloading the fallback model… " + pct + "%" + eta;
    },
    onModelReady: function () {
      progressEl.hidden = true;
      statusEl.textContent = "";
    },
    onNanoStatus: function (ready) {
      statusNano.textContent = ready ? "available (runs on-device, offline)" : "not available here";
    },
    onStateChange: function () {
      if (llm.nanoReady) { statusNano.textContent = "available (runs on-device, offline)"; }
      if (llm.weightsCached) { statusSmol.textContent = "already downloaded on this device"; }
    }
  });

  // Probe the returning-visitor cache, then warm in the background.
  llm.probeWeightsCache().then(function () {
    if (llm.weightsCached) { statusSmol.textContent = "already downloaded on this device"; }
    if (typeof requestIdleCallback === "function") { requestIdleCallback(llm.prewarm); }
    else { setTimeout(llm.prewarm, 600); }
  });

  // "1m 20s" / "45s" from a whole number of seconds.
  function fmtEta(s) {
    if (s < 60) { return s + "s"; }
    var m = Math.floor(s / 60), r = s % 60;
    return r ? (m + "m " + r + "s") : (m + "m");
  }

  var busy = false;
  goBtn.addEventListener("click", function () {
    if (busy) { return; }
    var prompt = (promptEl.value || "").trim();
    if (!prompt) { return; }

    // If there is genuinely no way to run a model here, say so instead of hanging.
    if (tooSlow && !llm.nanoReady && !llm.modelReady) {
      statusEl.textContent = "This device is too slow/old to run a local model.";
      return;
    }

    busy = true;
    goBtn.disabled = true;
    outEl.textContent = "";
    backendEl.textContent = "";
    statusEl.textContent = llm.modelReady ? "Thinking…" : "Setting up the model…";

    llm.llmDownloadBlock().then(function (block) {
      if (block) {                       // offline / Data Saver / very slow link
        statusEl.textContent = block;
        busy = false; goBtn.disabled = false;
        return;
      }
      llm.loadGenerator().then(function (brain) {
        backendEl.textContent = "backend: " + brain.backend;
        statusEl.textContent = "Thinking…";
        return brain.generate(
          [{ role: "user", content: prompt }],
          { max_new_tokens: 96, temperature: 0.6, top_p: 0.9, repetition_penalty: 1.15 },
          function (partial) {
            statusEl.textContent = "";
            outEl.textContent = partial;
          }
        ).then(function (text) {
          outEl.textContent = text || "(no answer)";
        });
      }).catch(function (e) {
        statusEl.textContent = "The model got stuck: " + ((e && e.message) || e);
      }).then(function () {
        busy = false; goBtn.disabled = false;
      });
    });
  });
})();
