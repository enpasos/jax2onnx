# Deployment Readiness Roadmap Notes

This internal note expands the short public roadmap bullets into possible
implementation directions. It is intentionally exploratory: the items below are
candidate tracks, not release commitments.

## Product Thesis

Position `jax2onnx` less as a raw converter and more as a deployment bridge from
JAX, Flax, and Equinox into the ONNX ecosystem. The strongest user value is not
just "an ONNX file was produced", but "the exported model is valid, typed,
shape-inferred, numerically checked, and understandable for the intended
runtime."

## Candidate User Segments

| Segment | Likely need | Useful proof point |
| ------- | ----------- | ------------------ |
| Product and MLOps teams | Ship JAX-originated models into non-Python stacks. | Validation report, runtime smoke test, stable inputs/outputs. |
| Web, mobile, and edge developers | Small models that run without a Python runtime. | `ort-web` / mobile diagnostics and end-to-end examples. |
| JAX, Flax, and Equinox teams | Avoid the `jax2tf` / TensorFlow / `tf2onnx` detour. | Migration guidance and direct parity checks. |
| RL teams | Export small policies into simulators, games, robotics, or browsers. | Policy examples with clear action-head conventions. |
| Scientific and simulation users | Export numerical kernels with strict dtype and shape expectations. | Float64/BF16 validation profiles and control-flow coverage. |
| Runtime and hardware partners | Understand whether exported graphs fit a target runtime. | Operator inventory, target warnings, benchmarkable examples. |

## Track 1: Export Check And Report

Possible goal: provide a first-class way to inspect deployment readiness after
or during export.

Candidate report fields:

- `jax2onnx`, JAX, ONNX, and ONNX Runtime versions.
- Model name, opset imports, ONNX IR version, and model size.
- Input and output names, dtypes, shapes, symbolic dimensions, and layout hints.
- Operator inventory grouped by domain and op type.
- ONNX checker result.
- Strict shape inference result.
- Dtype preservation summary, especially BF16/FP16/FP64.
- Numeric parity result against JAX for supplied sample inputs.
- Optional runtime smoke-test result for selected targets.
- Warnings for custom functions, unsupported target operators, or unvalidated
  dynamic dimensions.

Initial implementation could be a library helper before becoming a CLI. A CLI
shape might eventually look like:

```bash
jax2onnx check model.onnx --target ort-cpu --report report.md
```

## Track 2: Target-Oriented Diagnostics

Possible goal: introduce target profiles that start as validation guidance and
warnings, without promising full compatibility.

Candidate targets:

- `ort-cpu`: conservative baseline using ONNX checker, strict shape inference,
  and ONNX Runtime execution where available.
- `ort-web`: focus on model size, operator inventory, custom functions, and
  web-sensitive operators.
- `ort-mobile`: focus on compact graphs, reduced operator sets, dynamic-shape
  risks, and portable dtypes.
- Later candidates: `ort-cuda`, `tensorrt`, `openvino`, `qnn`, `coreml`.

The first version should avoid changing lowering behavior. It can simply report
what the graph contains and where target-specific validation is incomplete.

## Track 3: Dtype And Shape Capability Matrix

Possible goal: turn the BF16 capability matrix into a reusable framework for
multiple deployment concerns.

Candidate dimensions:

- Dtypes: `float32`, `float64`, `bfloat16`, and later `float16` where runtime
  support is practical.
- Shapes: static shapes, dynamic batch, dynamic spatial dimensions,
  non-square image-like inputs, small and larger resolutions.
- Component families: JAX NumPy arithmetic/reductions, Dense/Linear, Conv,
  Pool, Norm, MatMul, Gather/Take, reshape/layout ops, and attention-adjacent
  building blocks.

Useful checks:

- Export succeeds without special-case test code.
- ONNX checker passes.
- Strict shape inference passes.
- Public floating inputs and outputs keep the requested dtype.
- Floating initializers and intermediate value info keep the requested dtype
  unless a case explicitly allows promotion.
- ONNX execution matches JAX within documented tolerances.

## Track 4: End-To-End Deployment Examples

Possible goal: show practical workflows that appeal to users outside the core
JAX contributor audience.

Candidate examples:

- Small vision model: variable/non-square input resolution, BF16/FP32 variants,
  ONNX Runtime validation, and simple preprocessing guidance.
- RL policy: deterministic and stochastic policy heads, stable input/output
  naming, and execution from Python plus one non-Python runtime.
- Numerical model: compact simulation or filtering step with Float64 parity and
  explicit tolerance profile.
- Web example: small exported model loaded with ONNX Runtime Web or Node.js.

Each example should prefer a narrow, maintainable scope over broad model-family
claims.

## Track 5: Migration And Failure Guidance

Possible goal: reduce adoption friction for users replacing multi-step export
pipelines.

Candidate docs:

- "From `jax2tf` / `tf2onnx` to `jax2onnx`" migration page.
- Missing-primitive troubleshooting guide.
- Plugin-author starter path for library maintainers.
- Guidance on what belongs inside the ONNX graph versus application-side
  preprocessing or postprocessing.

## Track 6: Larger Model Families

Possible goal: keep SOTA examples useful while avoiding premature broad
promises.

Candidate focus areas:

- Reproducible weight-loading and export recipes.
- External-data handling and model-size reporting.
- Stable input/output conventions.
- Runtime validation and parity summaries.
- Later investigation of KV-cache patterns for autoregressive models.

This track is strategically valuable but should build on the report and target
diagnostics work first.

## Practical Sequencing

1. Build a small internal report data model and markdown renderer.
2. Reuse existing checker, strict shape inference, and parity helpers from the
   capability tests.
3. Add operator and dtype inventory helpers for exported ONNX models.
4. Add `ort-cpu` as the first target profile because it is the least ambiguous.
5. Add `ort-web` and `ort-mobile` as warning-only profiles.
6. Add one small end-to-end example that consumes the report.
7. Expand the capability matrix across one additional shape axis before adding
   more model families.

## Non-Goals For The First Pass

- Do not claim universal target-runtime compatibility.
- Do not make target profiles silently rewrite graphs until diagnostics are
  trustworthy.
- Do not start with large LLM exports as the primary adoption story.
- Do not let examples depend on generated `.onnx` artifacts committed to git.
- Do not expand public promises faster than CI-backed validation.

