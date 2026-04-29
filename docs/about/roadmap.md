# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.


## Upcoming Version

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
  source context, and keep primitive-call recording on the shared lowering path.
* **Tighten converter utility surfaces:** Consolidate legacy context typing onto
  `LoweringContextProtocol`, centralize IR dtype/shape coercion in shared
  utilities, normalize `onnx_ir` function-container iteration, and remove the
  unused optional shape-inference placeholder.
* **Harden the IR optimizer:** Add an opt-in strict optimizer failure mode,
  replace the hard-coded optimizer sequence with a named pass registry that
  declares top-level/function-body applicability, fix constant folding for
  multi-output primitives by tracking each produced output independently, and
  simplify consumer lookup fallbacks for wrapped or renamed IR values.
* **Improve layout and function-body robustness:** Route NCHW input/output
  adaptation through a dedicated layout adapter, share output-binding guardrails
  with control-flow and nested lowering helpers, and preserve function signatures
  while optimizing function body graphs.
* **Harden `@onnx_function` registration:** Extract registration and marker
  helpers, support stable display-name overrides, prefer `type` over `name` when
  both are provided, reject conflicting namespace/name re-decoration, keep the
  ONNX-function registries synchronized, reject non-function registry
  collisions, and expose more precise public API overloads.
* **Expand regression coverage:** Add focused tests for converter phase helpers,
  output binding, lowering dispatch, primitive-call recording, IR constants,
  strict optimizer failures, optimizer pass ordering, function-container
  iteration, nested JAXPR error reporting, and `@onnx_function` conflict cases.
* **Document the converter design changes:** Add the converter design review,
  update the IR optimizer guide with the declarative pass list and strict failure
  mode, link the review from the architecture guide and docs navigation,
  document shared lowering-dispatch responsibilities, and clarify
  `@onnx_function` override and conflict rules.


## Current Version

### **jax2onnx 0.13.0**

* **Refresh compatibility floors:** Move the documented stack to JAX `0.10.0`, Equinox `0.13.7`, and `onnx-ir` `0.2.1`; update GPT-OSS examples to the new `jnp.clip(min=..., max=...)` API and tolerate legacy `scan` payload params removed upstream.
* **Broaden attention export coverage:** Extend `jax.nn.dot_product_attention` support for grouped-query attention and unbatched `TNH` inputs by reusing the shared IR KV-head expansion path already exercised by Flax NNX.
* **Lean harder on `onnx-ir` helpers:** Replace local graph/value plumbing with `onnx-ir` helpers for bulk `rename_values`, `replace_all_uses_with`, `create_value_mapping`, constant tensor reads, tensor/NumPy dtype mapping, scalar/tensor attribute conversion, and IRBuilder mapping attributes.
* **Tighten dtype fidelity:** Preserve explicit dtype requests across casts, array creation, `full`, `eye`, `linspace`, `arange`, index outputs, bitwise casts, shifts, and `iota`, backed by exact-dtype converter tests.
* **Complete generated component coverage:** Bring generated JAX LAX, JAX NumPy, Flax Linen/NNX, and Equinox NN coverage matrices to zero missing entries, with guard tests that fail on future drift.
* **Expand JAX NumPy lowering coverage:** Add or harden coverage for `searchsorted`, `digitize`, `histogram`, `histogram2d`, `histogramdd`, `interp`, `nancumprod`, `frexp`, `ldexp`, `spacing`, 1D convolution reuse, `packbits`, `argpartition`, `lexsort`, `polyval`, `intersect1d`, triangular index helpers, `roots`, and static linear `polyfit`.
* **Add small static linear algebra lowerings:** Cover static small-matrix/system paths for `jnp.linalg.inv`, `solve`, `tensorsolve`, and `tensorinv`, including generated tests and coverage-doc refreshes.
* **Improve LAX and control-flow coverage:** Add `lax.linalg.ormqr`, strengthen gather/scatter/sort/top-k/split/reduce helper coverage, fix single-output split-style lowerings, and keep custom IO-name rewrites on the `onnx-ir` bulk rename path.
* **Improve coverage documentation quality:** Add stale-doc checks for coverage generators, snapshot/zero-missing guards, ONNX metadata-vs-lowering asymmetry guards, and an ONNX operator coverage matrix with next-action buckets for the remaining open operators.



 
 
## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
