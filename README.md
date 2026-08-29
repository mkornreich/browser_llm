# Browser LLM

Get an on-device language model in the browser, with **no server and no API keys**. One small, dependency-free file that knows how to "get the LLM" for a page:

- **Main model — Gemini Nano.** If the browser has Chrome's built-in [Prompt API](https://developer.chrome.com/docs/ai/prompt-api), it is used directly. The model runs on-device and is downloaded/managed by Chrome, so **there is nothing to download from you**, and it works offline once installed.
- **Fallback model — SmolLM2-360M-Instruct.** Where there is no Prompt API, the library downloads [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (~360 MB, one time) and runs it with [transformers.js](https://github.com/huggingface/transformers.js) — inside a Web Worker, with a main-thread fallback for browsers that cannot run a module worker.

Whichever backend wins, the caller gets the **same tiny shape** back: a `brain` with one `generate()` method. It streams tokens, and everything runs locally — no text ever leaves the page.

This is the shared source of truth for the word-brain in [Cool Concepts](https://mkornreich.me/projects/coolconcepts/), extracted so the model logic lives in one place — the sibling of the [llm_postprocessor](https://github.com/mkornreich/llm_postprocessor) repo (which does the text clean-up *after* the model speaks).

## Try it

Open `index.html` in a browser. No build step, no server, no dependencies (the fallback model is pulled from a CDN only if your browser has no Prompt API).

```
# or serve it, if your browser blocks file:// scripts
python3 -m http.server 8000    # then open http://localhost:8000
```

## Use it

```html
<script src="browser-llm.js"></script>
<script>
  const llm = BrowserLLM.create();
  const brain = await llm.loadGenerator();          // Nano, or SmolLM2 if no Nano
  const text = await brain.generate(
    [{ role: "user", content: 'What is "photosynthesis"?' }],
    { max_new_tokens: 72, temperature: 0.6 },
    partial => { /* live tokens, called as they stream in */ });
</script>
```

`generate(messages, opts, onToken?)` takes OpenAI-style chat `messages` (`{ role, content }`), generation `opts` (`max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`), and an optional `onToken(textSoFar)` callback for streaming. It returns the finished reply as a string.

## The models, and their context windows

The one place model identities live is `BrowserLLM.MODELS`, and the **context-window sizes are stored as constants**:

| Model | Key | Backend | Context window | Notes |
| --- | --- | --- | --- | --- |
| Gemini Nano | `nano` | Chrome Prompt API | **6144 tokens** (`BrowserLLM.CONTEXT_WINDOW.nano`) | Nominal. Chrome does not publish a fixed size; a live session reports the real number on `session.contextWindow` (input budget on `session.inputQuota`). |
| SmolLM2-360M-Instruct | `smol` | transformers.js (WASM) | **8192 tokens** (`BrowserLLM.CONTEXT_WINDOW.smol`) | From the model's `config.json` (`max_position_embeddings`). |

```js
BrowserLLM.CONTEXT_WINDOW;            // { nano: 6144, smol: 8192 }
BrowserLLM.MODELS.smol.id;            // "HuggingFaceTB/SmolLM2-360M-Instruct"
BrowserLLM.MODELS.smol.dtype;         // "int8"
```

## Device & connection gates

Before it tries anything, the library answers "should we even run a model here?" — the same checks Cool Concepts uses:

| Call | Answers |
| --- | --- |
| `BrowserLLM.canRun()` | `true` when this device can run **some** local model: not too slow, and either the Prompt API is present or the heavy fallback can run here. The one-call "should I even offer this?" check. (Static capability — it doesn't confirm Nano is downloaded; that's async.) |
| `BrowserLLM.tooSlowForLlm()` | `true` on a device too slow/old for a usable local model (< 4 cores, < 4 GB where reported, or no color-emoji support = a very old system). An **available Nano overrides this** — Chrome only ships Nano to vetted hardware. |
| `BrowserLLM.heavyModelUnsupported()` | `true` on iOS/iPadOS, where the ~360 MB WASM fallback OOM-crashes the tab. Those devices can still use Nano; they just never download SmolLM2. |
| `BrowserLLM.fastEnoughForBackground()` | `true` on desktop-class hardware (≥ 8 cores, ≥ 8 GB) that can afford speculative background generation. |
| `BrowserLLM.hasEnoughCores()` | `true` when `navigator.hardwareConcurrency >= BrowserLLM.HEAVY_MODEL_MIN_CORES` (4) — the single home for the "enough cores to bother with the heavy WASM model / threads" threshold, used by `tooSlowForLlm()` and `shouldCrossOriginIsolate()`. |
| `BrowserLLM.shouldCrossOriginIsolate()` | `true` when it's worth setting up cross-origin isolation (SharedArrayBuffer) so the WASM fallback can use several threads: isolation is supported and not already active, secure context, service worker available, **≥ 4 cores** (`hasEnoughCores()`), and no Prompt API. The single home for a page's isolation-bootstrap gate. |
| `BrowserLLM.capabilities()` | A full synchronous snapshot: `{ canRun, tooSlow, hardwareConcurrency, deviceMemory, nano:{supported}, heavyModel:{supported}, fastEnoughForBackground, crossOriginIsolation:{supported, active, recommended} }`. |
| `provider.connectionBlock()` | A human message when the ~360 MB fallback should **not** auto-download right now (offline / Data Saver / 2G), or `null` to go ahead. A returning visitor whose weights are cached always gets `null`. |
| `provider.llmDownloadBlock()` | Like `connectionBlock()`, but returns `null` first if Nano can serve the request (Nano manages its own small download). |

`canRun`, `tooSlowForLlm`, `heavyModelUnsupported`, `fastEnoughForBackground`, `hasEnoughCores`, `shouldCrossOriginIsolate`, `capabilities`, `emojiRenders`, `nanoApi`, `nanoStatus`, and `replyOf` are available both **statically** (`BrowserLLM.x`) and on a provider instance.

## API

`BrowserLLM.create(config)` returns a provider. `config` is all optional:

| Field | Default | What |
| --- | --- | --- |
| `onProgress(pct, info)` | — | Fallback-model download progress. `pct` is 0–100 (byte-weighted across files); `info` is `{ loaded, total, files, etaMs, etaSeconds, bytesPerSec }` — the last three are the **estimated time remaining** (see `downloadEta()`), or `null` before there's enough data. |
| `onModelReady()` | — | The brain finished building. |
| `onModelError(err)` | — | Building the brain failed (state is reset so a later call retries). |
| `onNanoStatus(ready)` | — | Nano availability was (re)checked during `prewarm()`. |
| `onStateChange()` | — | Any of `nanoReady` / `modelReady` / `weightsCached` changed. |
| `messages` | neutral copy | `{ offline, saveData, slow }` strings returned by `connectionBlock()`. |
| `transformersUrl` | jsDelivr `@huggingface/transformers@4.2.0` | Where transformers.js is imported from. |
| `weightsFlagKey` | `"browser_llm:tf-ready"` | `localStorage` key for the "already downloaded here" flag. |
| `modelId` / `dtype` | SmolLM2-360M / `int8` | Override the fallback model. |
| `workerBootTimeoutMs` | `20000` | How long to wait for the worker to boot before falling back to the main thread. |
| `etaSmoothing` | `0.35` | EMA weight (0–1) for the download-rate estimate behind `downloadEta()`; higher tracks the latest speed more closely. |
| `now` | `performance.now` | Clock for the ETA estimate. Injectable for deterministic testing. |

Provider methods and state:

| Member | Does |
| --- | --- |
| `loadGenerator()` → `Promise<brain>` | Build the brain once (memoized); sets `modelReady`. **The main entry.** |
| `generateWithRetry(brain, messages, opts)` → `Promise<string \| null>` | The generation loop most apps want: retry until the output passes a **pluggable filter** (`opts.isBad`), or attempts run out. Bounds each slow (non-Nano) attempt with a timeout, forwards streamed tokens for the current attempt only, and takes hooks for `options(attempt)`, `postprocess`, `onToken`, `onReject`, and `shouldContinue`. Defaults: 4 attempts for a Nano brain / 2 otherwise, 40 s timeout on non-Nano. Also on `BrowserLLM.*`. |
| `prewarm()` | Warm the brain in the background as soon as the page settles. Call on load and on `"online"`. |
| `probeWeightsCache()` → `Promise<bool>` | Detect a returning visitor whose fallback weights are already on disk. |
| `downloadEta()` → `{ etaMs, etaSeconds, bytesPerSec, loaded, total, pct, done }` \| `null` | **Estimated time remaining** for the in-flight model download. `null` until there's enough data (or if the rate stalls); `done: true` once finished. `etaMs`/`etaSeconds` come from an EMA of progress-per-ms, so they track the current speed; `bytesPerSec` is present when byte totals are known (the SmolLM2 download) and `null` for Nano's fraction-only progress. |
| `connectionBlock()` / `llmDownloadBlock()` | The connection gates above. |
| `dropTfSpare()` | Free a prefetched fallback worker (Nano won; its files stay cached). |
| `modelReady` / `nanoReady` / `weightsCached` / `dlPct` | Live state flags. |

## Plugging into Cool Concepts (no visible changes)

Cool Concepts today has all of this inline in its `index.html`. Swapping in this library is a **behavior-preserving** refactor — the page looks and acts exactly the same:

1. Load it next to the postprocessor scripts, in CORS mode (the page is cross-origin isolated):
   ```html
   <script src="https://mkornreich.github.io/browser_llm/browser-llm.js" crossorigin></script>
   ```
2. Create the provider with Cool Concepts' exact wording and its returning-visitor key, and wire the callbacks to the existing UI functions:
   ```js
   var llm = BrowserLLM.create({
     weightsFlagKey: "cc:tf",                     // keep the same returning-visitor flag
     messages: {
       offline:  "You are offline. “What is it?” needs a one-time download the first time it runs!",
       saveData: "Data Saver is on, so I will not auto-pull the ~360 MB word-brain. Turn it off (or switch to Wi-Fi) and tap “What is it?” again.",
       slow:     "Your connection looks too slow for the one-time ~360 MB download the word-brain needs. Try again on a faster network."
     },
     onProgress:   function () { if (!explainBox.hidden && !llm.modelReady) { showDownloadStatus(); } },
     onModelReady: function () { if (!explainBox.hidden) { explainStatus.innerHTML = '<span class="cc-dot"></span>Thinking…'; } pumpDesc(); queueBackfill(); },
     onNanoStatus: function () { updateExplainState(); },
     onStateChange: updateExplainState
   });
   ```
3. Replace the inline `tooSlowForLlm` / `heavyModelUnsupported` / `nanoApi` / `buildBrain` / `loadTfPipeline` / `connectionBlock` / `prewarm` / … with the `llm.*` equivalents, and read `llm.modelReady` / `llm.nanoReady` / `llm.weightsCached` in place of the local flags.
4. The download percentage now lives in the library, so **`showDownloadStatus()` reads `llm.dlPct`** in place of the removed local `dlPct` (its only writers — `nanoMonitor` and `applyDlProgress` — moved into the library, and the library also resets `dlPct` on a failed load). Without this the counter would sit at 0 and the status would never count up. Equivalently, read the percent straight off the callback: `onProgress: function (pct) { … showDownloadStatus(pct); }`.

The generation retry loop, the badness filter, the prompts, and the `PostProcessor` text clean-up stay in Cool Concepts — this library only produces the brain and reports its state. The Cool Concepts repo is **not** modified here.

## Compatibility

Same floor as the Cool Concepts app script: `async`/`await`, arrow functions, and template literals are used, but no `for await`, no bare `import()` (built with `new Function`), and no optional chaining — so any browser that can parse the host page's app script can parse this too, and older ones fall through untouched.

## Files

| File | What |
| --- | --- |
| `browser-llm.js` | The library. Model catalog + context windows, device/connection gates, and the Nano→SmolLM2 brain. |
| `index.html` / `app.js` / `style.css` | The demo page. |
| `test.js` | Dependency-free test suite for the pure logic — run `node test.js`. |

## Credits

- Fallback model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (HuggingFaceTB), run via [transformers.js](https://github.com/huggingface/transformers.js).
- Main model: Google's Gemini Nano, via Chrome's built-in [Prompt API](https://developer.chrome.com/docs/ai/prompt-api).
