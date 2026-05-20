# Diagnostics Report Target State

The current diagnostics report work should be treated as experimental. It is
useful as an internal model inventory and ONNX Runtime CPU smoke-test helper,
but it is not yet a stable external API.

## Current Maturity

- Static model facts are on a solid path: ONNX checker status, strict shape
  inference, opset imports, graph inventory, public I/O metadata, dtype counts,
  and serialized model size are directly observable.
- ONNX Runtime CPU validation is sample-based. It proves behavior only for the
  provided runtime inputs and provider configuration.
- `ort-web` and `ort-mobile` target findings are currently static hints. They
  are not target compatibility guarantees because no web or mobile runtime is
  executed.
- The JSON payload is derived directly from dataclasses and does not yet have a
  schema version or compatibility policy.
- The exported Python API is broader than a stable surface should be.

## Stable External Target

A stable diagnostics API should separate observed facts from advisory linting.
The core report may only present compatibility claims that are backed by an
actual validation run.

Stable report data should include:

- `schema_version` for JSON and persisted reports.
- Model facts derived from ONNX protobuf inspection.
- Validation facts from ONNX checker and strict shape inference.
- Runtime execution facts tied to a concrete runtime package, version, provider,
  platform, and input profile.
- Explicit `validated`, `failed`, and `not_evaluated` states.
- Golden-file coverage for JSON and Markdown output.

Advisory checks should be isolated from the core report, for example under a
`static_lints` or `advisory_findings` section. They must not be phrased as
runtime compatibility.

## Required Work

1. Narrow the public API to a small stable entrypoint set, such as
   `analyze_model`, `analyze_jax_export`, `write_model_report`, and
   `evaluate_model_report_gate`.
2. Add JSON schema versioning and compatibility tests for persisted report
   payloads.
3. Replace target-profile heuristics with runtime adapters where compatibility
   claims are needed.
4. Keep `ort-cpu` validation backed by real ONNX Runtime execution and record
   runtime version/provider metadata.
5. Add an `ort-web` adapter only when CI can run `onnxruntime-web` through a
   concrete Node, WASM, WebGPU, or browser harness.
6. Add an `ort-mobile` adapter only when CI can run a defined ONNX Runtime
   mobile build through an emulator, device farm, or equivalent reproducible
   harness.
7. Add integration coverage with larger representative models, ONNX Functions,
   dynamic shapes, less common dtypes, invalid models, missing inputs, shape
   mismatches, and numeric parity failures.
8. Document the boundary between facts, validated runtime behavior, and
   advisory findings in user-facing docs.

Until these items are done, the diagnostics module should remain experimental.
