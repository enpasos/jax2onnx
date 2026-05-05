# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.


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
