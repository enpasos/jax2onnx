# Roadmap

## Planned

* Expand deployment-readiness reporting beyond the basic validation guide:
  operator inventory, dtype/shape summaries, stricter shape-inference checks,
  and reusable report artifacts for exported models.
* Extend target-oriented validation guidance beyond the documented ONNX Runtime
  CPU and Web/WASM flows, especially for mobile deployments where practical.
* Broaden capability-matrix coverage across dtype and shape variants, including
  BF16, dynamic dimensions, and non-square inputs.
* Add focused end-to-end deployment examples for small vision, RL-style, and
  numerical models.
* Continue targeted coverage work for JAX, Flax NNX/Linen, Equinox, SotA
  examples, and physics/simulation use cases.

## Upcoming Version

### **jax2onnx 0.14.1**

* **Add targeted Opset 27 coverage:** Preserve the Opset 23 default while using
  native FP16/BF16 `Range` for `opset=27` exports and keeping older exports on
  the cast fallback; cover new JAX 0.10 resize methods where the ONNX mapping is
  exact.
* **Refresh the validation stack:** Update the upcoming documented runtime
  stack to JAX `0.10.2`, Equinox `0.13.8`, ONNX `1.22.0`, ONNX Runtime
  `1.27.0`, and keep Web validation on the latest stable
  `onnxruntime-web` release until a matching `1.27.x` Web package is published.

## Current Version

### **jax2onnx 0.14.0**

* **Add a browser/WASM export profile:** Introduce `export_mode="web"` for
  `to_onnx(...)` so browser deployments can produce a single self-contained
  `.onnx` artifact without stale `.onnx.data` sidecars, ready to serve directly
  to `onnxruntime-web/wasm`; invalid export modes fail early.
* **Expose ONNX Runtime Web parity checks:** Add
  `allclose_onnxruntime_web(...)` to compare Python ONNX Runtime CPU output
  with `onnxruntime-web/wasm` in Node.js or a Playwright-driven
  Chrome/Chromium browser, with browser-safe tensor serialization for common
  numeric, boolean, and string feeds.
* **Document and automate browser deployment validation:** Add a Browser/WASM
  guide, API/getting-started updates, a Quickstart Web MLP helper, Node.js and
  Chrome smoke/full validation scripts, opt-in full-suite Web validation gates,
  and scheduled/manual CI coverage without making the heavy Web run part of
  every PR.
* **Harden Web runtime parity edge cases:** Fix shape metadata and runtime
  behavior surfaced by Web validation across shape-sensitive lowerings,
  attention metadata, and JAX sort NaN ordering through ONNX `TopK`.
* **Improve BF16 export confidence:** Fix BF16 Linen pool-add exports and add
  capability-matrix coverage across representative JAX NumPy operations and Flax
  Linen layers, including checker/shape-inference validation, dtype preservation,
  and ReferenceEvaluator parity.
* **Fix SiLU/Swish exports:** Patch Flax NNX `silu`/`swish` alongside JAX
  `silu`, and emit ONNX `Swish` for opset 24+ while preserving older-opset and
  shared-`Sigmoid` behavior.
* **Refresh the validation stack:** Update the documented runtime stack to JAX
  `0.10.1`, Equinox `0.13.8`, ONNX Runtime `1.26.0`, and add documented
  `onnxruntime-web` versions for Web validation.



## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
