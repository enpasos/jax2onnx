# Roadmap

## Planned

* Explore deployment-readiness reports for exported models: ONNX checker,
  strict shape inference, operator inventory, dtype/shape summaries, and
  numeric parity checks.
* Add target-oriented validation guidance for ONNX Runtime CPU, web, and mobile
  deployments where practical.
* Broaden capability-matrix coverage across dtype and shape variants, including
  BF16, dynamic dimensions, and non-square inputs.
* Add focused end-to-end deployment examples for small vision, RL-style, and
  numerical models.
* Continue targeted coverage work for JAX, Flax NNX/Linen, Equinox, SotA
  examples, and physics/simulation use cases.


## Upcoming Version

### **jax2onnx 0.13.2**

* **Fix BF16 Linen pool exports:** Treat `jnp.bfloat16` as a floating dtype in
  `flax.linen.pool(..., reduce_fn=jax.lax.add)` by using JAX dtype
  classification, and add an end-to-end issue #221 regression test covering
  ONNX export, strict shape inference, BF16 type preservation, and Reference
  Evaluator execution.
* **Broaden BF16 capability coverage:** Add a reusable capability-matrix test
  helper that reuses existing plugin/export metadata, runs ONNX checker plus
  strict shape inference, verifies public IO and all floating graph tensors keep
  the requested dtype, and compares ONNX ReferenceEvaluator output against JAX.
* **Validate representative BF16 surfaces:** Cover BF16 export/runtime behavior
  across JAX NumPy arithmetic, reductions, transpose/concat/take/matmul, and
  Flax Linen Dense, Conv, AvgPool, MaxPool, pool-add, LayerNorm, and GroupNorm,
  including non-square Conv and AvgPool shape variants.
* **Tighten Linen Conv dtype metadata:** Propagate the requested dtype to
  `nn.Conv` parameters via `param_dtype=with_requested_dtype()` in Linen Conv
  test metadata so BF16 capability checks exercise BF16 weights as well as BF16
  inputs; remove leftover Conv debug prints.
* **Fix NNX SiLU/Swish lowering:** Patch `flax.nnx.silu` and `flax.nnx.swish`
  alongside `jax.nn.silu`, add Flax NNX metadata coverage, and rewrite captured
  `Mul(x, Sigmoid(x))` patterns to ONNX `Swish` for opset 24+ while preserving
  opset <24 behavior and multi-consumer `Sigmoid` cases.
* **Refresh roadmap state:** Move the completed `0.13.1` notes into the current
  version section and reserve the upcoming section for `0.13.2`.


## Current Version

### **jax2onnx 0.13.1**

* **Refactor the IR-only converter pipeline:** Split tracing, context setup,
  equation lowering, output binding, model assembly, layout adaptation, late
  finalization, and postprocessing into focused helpers so each phase can be
  tested directly.
* **Centralize lowering dispatch:** Add shared converter helpers for plugin
  lookup, plugin metadata identification, primitive/function dispatch, returned
  value binding, and nested JAXPR lowering across top-level conversion, ONNX
  Function bodies, control-flow subgraphs, and direct nested-JAXPR wrappers.
* **Strengthen lowering guardrails:** Verify every non-drop equation output is
  bound immediately after lowering, preserve the current equation scope while
  plugin metadata is staged, report nested JAXPR missing-plugin errors with
  source context, accept name-matched graph-connected outputs across `onnx_ir`
  value wrappers, scope constant-folder producer maps during nested lowering,
  require equation inputs to be bound before plugin dispatch, and keep
  stacktrace metadata plus primitive-call recording on the shared lowering path.
* **Tighten converter utility surfaces:** Consolidate legacy context typing onto
  `LoweringContextProtocol`, centralize IR dtype/shape coercion in shared
  utilities, normalize `onnx_ir` function-container iteration, route manually
  constructed nodes through the IR builder metadata path, and remove the unused
  optional shape-inference placeholder.
* **Raise first-party typing coverage:** Enable `onnx_ir` type checking for the
  converter, type the lowering-builder/plugin protocols and shared helper
  modules, extend strict mypy coverage across JAX/LAX/NumPy, Flax Linen/NNX,
  Equinox, example plugins, public API, quickstart, debug logging, and
  `expect_graph` helpers, and collapse the mypy target to package-wide
  `jax2onnx` coverage with only sandbox and optional MaxText/MaxDiffusion
  integrations excluded.
* **Harden the IR optimizer:** Add an opt-in strict optimizer failure mode,
  replace the hard-coded optimizer sequence with a named pass registry that
  declares top-level/function-body applicability, fix constant folding for
  multi-output primitives by tracking each produced output independently, and
  split graph reference helpers into a focused optimizer utility module.
* **Improve layout and function-body robustness:** Route NCHW input/output
  adaptation through a dedicated layout adapter, share output-binding guardrails
  with control-flow and nested lowering helpers, and preserve function signatures
  while optimizing function body graphs; keep function-body attribute overrides
  isolated from same-named top-level overrides.
* **Harden `@onnx_function` registration:** Extract registration and marker
  helpers, support stable display-name overrides, prefer `type` over `name` when
  both are provided, reject conflicting namespace/name re-decoration, keep the
  ONNX-function registries synchronized, reject non-function registry
  collisions, and expose more precise public API overloads.
* **Expand regression coverage:** Add focused tests for converter phase helpers,
  output binding, lowering dispatch, primitive-call recording, IR constants,
  strict optimizer failures, optimizer pass ordering, function-container
  iteration, nested JAXPR error reporting, function override isolation,
  builder-routed node insertion, constant-folder scoping, and `@onnx_function`
  conflict cases.
* **Separate public docs from maintainer workflows:** Keep user-facing guides
  focused on stable export, validation, and reference material; move coverage
  generation, sample-model publishing, pinned refs, and SotA maintenance notes
  into maintainer documentation.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
