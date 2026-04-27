# Converter Design Review

This page documents the current implementation of `jax2onnx/converter` and
records improvement opportunities found during a source-level review. It
complements the higher-level [Architecture Overview](architecture.md): that page
explains the intended shape of the system, while this page describes how the
converter package actually implements it.

## Scope

The converter package is the IR-only core of the export pipeline. It receives a
Python callable plus abstract input specs, traces the callable to a JAXPR, asks
registered plugins to lower each JAX primitive, assembles an `onnx_ir.Model`,
and performs conservative cleanup before the public API serializes or returns
the model.

The package is intentionally not responsible for operator semantics. Those live
in `jax2onnx/plugins`. It is also not responsible for final protobuf
serialization; that remains in `jax2onnx/user_interface.py`.

## Module Map

| Module | Primary role |
| ------ | ------------ |
| `conversion_api.py` | End-to-end conversion orchestration: plugin activation, JAXPR tracing, context setup, equation lowering, model assembly, optimization, late shape normalization. |
| `ir_context.py` | Plugin-facing lowering context. Owns the JAX var to IR value map, constant binding, symbolic dimension origins, function-mode behavior, and small graph input helpers. |
| `ir_builder.py` | Thin wrapper around `onnx_ir._tape.Builder` plus graph-owned live lists, initializer deduplication, deterministic names, and optional stacktrace metadata. |
| `function_scope.py` | Builds ONNX Function bodies in child `IRContext` instances and snapshots them into `onnx_ir.Function` definitions. |
| `lower_dimexpr.py` | Lowers JAX symbolic dimension expressions to runtime ONNX shape arithmetic using registered dimension origins. |
| `ir_constants.py` | Minimal recursive constant evaluator used by plugins that can fold primitive outputs from known inputs. |
| `ir_optimizations.py` | Destructive, IR-only graph cleanup passes and their top-level `optimize_graph` orchestrator. |
| `ir_postprocess.py` | Export-time model postprocessing: loosen intermediate shapes and promote constant payloads for double-precision exports. |
| `typing_support.py` | Runtime-checkable protocols and small data classes shared between converter and plugins. |

## End-to-End Flow

`conversion_api.to_onnx` is the central orchestrator.

1. It temporarily sets `jax_enable_x64` to match `enable_double_precision`.
2. It normalizes user input specs into `jax.ShapeDtypeStruct` instances. String
   dimensions are converted through a shared `jax.export.SymbolicScope` so equal
   strings become shared symbolic dimensions during tracing.
3. It activates the plugin world around `jax.make_jaxpr`. Activation imports all
   plugins, applies declared monkey patches, enters leaf-plugin primitive
   bindings, and backfills selected autodiff transpose rules.
4. It traces the wrapped callable, validates optional layout indices and custom
   I/O-name arity, and creates an `IRContext`.
5. It binds closed-JAXPR constants as graph initializers or function constants,
   then binds positional JAXPR inputs as graph inputs. `inputs_as_nchw` is
   handled by declaring an external NCHW input and inserting an NCHW-to-NHWC
   `Transpose` before binding the JAX variable.
6. It iterates over `jaxpr.eqns`. For each equation it resolves
   `PLUGIN_REGISTRY[eqn.primitive.name]` and dispatches either to
   `PrimitiveLowering.lower(...)` or to a `FunctionLowering` handler. Optional
   stacktrace metadata is staged on the builder while the plugin emits nodes.
7. It binds graph outputs from JAXPR outvars. `outputs_as_nchw` inserts a final
   NHWC-to-NCHW `Transpose` and appends that value as the graph output.
8. It builds an `onnx_ir.Model`, attaches collected `onnx_ir.Function` objects,
   and ensures function domains appear in model opset imports.
9. It runs `optimize_graph` in a non-fatal wrapper, applies late attribute
   overrides, fixes missing `Concat(axis)` attributes, consumes function-hit
   bookkeeping, and normalizes symbolic shape labels to `ir.SymbolicDim`.
10. The public API then calls `postprocess_ir_model`, materializes dynamic
    `input_params` as graph inputs when needed, applies final custom I/O names,
    and either returns IR, returns protobuf, or writes a file.

The important sequencing property is that `conversion_api` returns a precise
IR model. Shape loosening is intentionally deferred to `user_interface.py` so
the converter and structural tests can inspect precise metadata.

## Core Data Model

### Values and ownership

The builder creates one live `ir.Graph` at construction time. `IRBuilder.inputs`,
`outputs`, `nodes`, and `initializers` are views over that graph rather than
detached buffers. Reassigning these properties clears and repopulates graph-side
containers; mutating them in place is safe.

Initializers are special because `onnx_ir.Graph.initializers` is keyed by name.
`IRBuilder` wraps it in `_InitializerList` and rejects duplicate names unless the
payload is identical. This preserves object identity for nodes that already
reference an initializer value.

### Variable binding

`IRContext` and `IRBuilder` maintain a JAX variable to `ir.Value` map. Plugins
obtain existing values through `ctx.get_value_for_var(...)` and bind equation
outputs with `ctx.bind_value_for_var(...)`. Literals and closed-over constants
are converted to `ir.Value` constants by the context.

The converter has a few legacy compatibility surfaces:

- `IRContext._inputs`, `_nodes`, `_initializers`, and `_var2val` mirror builder
  state because older plugins still touch these attributes.
- `conversion_api._IRBuildContext` and `plugin_system._IRBuildContext` remain as
  compatibility aliases for plugins that import them under `TYPE_CHECKING`.

These shims keep older plugin code working, but the architectural direction is
builder-first and context-method-first.

### Dtypes and precision

The converter has a global precision mode per conversion. When double precision
is enabled, floating constants and float32 JAX values are promoted to ONNX
`DOUBLE` unless a function body has requested float32 preservation. When double
precision is disabled, floating values are normalized toward float32 unless the
source dtype explicitly requires another supported ONNX type.

The dtype conversion logic appears in both `conversion_api.py` and
`ir_context.py`; see the improvement list below.

### Symbolic dimensions

Symbolic dimension support has two layers:

- Static metadata: symbolic JAX dimensions are converted into strings during
  shape stamping and finally normalized to `ir.SymbolicDim`.
- Runtime dimension materialization: `IRContext` records, for each symbolic
  dimension, which input value and axis supplied it. `LowerDimExpr` turns those
  origins into `Shape(start=axis, end=axis+1)` values and composes arithmetic
  with ONNX `Add`, `Mul`, `Div`, `Mod`, `Min`, `Max`, `Pow`, and `Concat`.

This split lets plugins preserve symbolic metadata while still building runtime
shape tensors for `Reshape` and related ops.

## Function Bodies

`FunctionScope` builds ONNX Function bodies in a child `IRContext`.

- The child context inherits the parent opset and precision mode.
- Function mode is enabled so constants are emitted as `Constant` nodes. ONNX
  Functions cannot own graph initializers.
- Parent input values are mirrored as fresh function inputs. Symbolic dimension
  origins are remapped from parent values to those function inputs.
- `end(...)` snapshots child inputs, outputs, nodes, and attribute overrides into
  a `FunctionDef`.
- `to_ir_function()` clones the child graph, updates body opset imports, and
  creates an `onnx_ir.Function` whose domain and name match the call-site node.

Function definitions are attached to the model after the main graph is built.
The model-level opset imports are then extended with each non-empty function
domain.

## Optimization And Postprocessing

`ir_optimizations.optimize_graph` runs after lowering and before late attribute
overrides. The current pass sequence includes:

- name fixing via `onnx_ir` common passes,
- redundant cast removal,
- transpose folding around reductions, add forests, and elementwise chains,
- reshape pair and identity reshape removal,
- common subexpression elimination,
- lifting constants to initializers,
- `Mul(Rsqrt(x))` to `Div(1, Sqrt(x))` rewriting,
- dropout training-mode constant inlining,
- unary and elementwise shape propagation,
- dead node removal,
- orphan transpose removal,
- unused graph input pruning.

Function bodies receive a similar cleanup sequence, except function inputs are
not pruned because they are part of the function signature.

`ir_postprocess.postprocess_ir_model` is separate from optimization. It loosens
intermediate shapes for runtime flexibility and can promote float32 constant
payloads to float64 for double-precision export.

## Key Invariants

- Converter and plugin code stay IR-only; ONNX protobuf types are not imported
  under `converter/` or `plugins/`.
- Plugins emit through `ctx.builder` whenever possible, pass `_outputs` as a
  sequence, and keep dtype/shape metadata on `ir.Value` objects.
- Constants in top-level graphs are initializers; constants in function bodies
  are `Constant` nodes.
- Symbolic dimensions must never be coerced to integers unless they are already
  concrete.
- Optimizer passes are destructive but conservative. They must be IR-only and
  preserve graph outputs, function signatures, and live value references.
- The public API performs export-only shape loosening after the converter has
  produced a precise IR model.

## Strengths

- The converter has a clear inversion-of-control boundary: primitive semantics
  live in plugins and the core dispatches only by primitive name.
- The builder wrapper shields most code from `onnx_ir` ownership and initializer
  edge cases.
- Symbolic-dimension origins are represented explicitly enough for runtime shape
  lowering without global graph introspection.
- Function mode correctly avoids initializers in ONNX Function bodies.
- Stacktrace metadata is opt-in and local to builder calls, so normal exports do
  not pay the metadata cost.

## Potential Improvements

### 1. Add the missing equation output-binding guardrail

`architecture.md` says the core asserts every `eqn.outvar` is bound after a
plugin lowers an equation. The current implementation relies on plugins to do
this and only fails later if a downstream equation or graph output requests an
unbound value. Add a small helper after plugin dispatch:

- ignore JAX drop variables,
- verify every non-drop outvar is present in the context map,
- include primitive name, equation index, and outvar index in the error.

This would turn late graph-shape failures into immediate plugin-contract
failures.

Status: implemented for the top-level `conversion_api.to_onnx` equation loop.
The guard also accepts returned `ir.Value` objects and binds them generically
when a plugin returns values instead of calling `ctx.bind_value_for_var(...)`.
A useful follow-up is reusing the same guard in plugins that lower nested JAXPRs
manually, such as JIT/custom-call style wrappers.

### 2. Consolidate legacy context shims

`conversion_api._IRBuildContext` duplicates parts of `IRContext` and is mostly
used as a type-checking import by older plugins. Prefer a single public
`LoweringContextProtocol` or a lightweight alias exported from one module. Keep
the old import path temporarily with a deprecation comment, then remove the
duplicate implementation once plugins no longer depend on it.

Status: implemented. The duplicate `_IRBuildContext` class was removed from
`conversion_api`; both legacy import paths now alias
`LoweringContextProtocol`, and the shared protocol documents the legacy context
methods and mirror attributes that still exist during migration.

### 3. Split `conversion_api.to_onnx`

`to_onnx` currently owns tracing, layout adaptation, plugin dispatch, function
attachment, optimization error handling, attribute overrides, Concat fixes, and
shape finalization. Extract the late phases into small modules or helpers:

- `trace_to_jaxpr(...)`,
- `lower_jaxpr_to_ir(...)`,
- `attach_ir_functions(...)`,
- `apply_late_attr_overrides(...)`,
- `finalize_value_shapes(...)`.

This would make failures easier to isolate and would let tests cover each phase
without constructing a full export.

Status: in progress. JAXPR tracing, function attachment, late attribute/Concat
postprocessing, and final shape normalization now live in module-level helpers
with direct tests. The JAXPR-to-IR lowering phase is still inline in `to_onnx`.

### 4. Make layout adaptation a first-class helper

`inputs_as_nchw` and `outputs_as_nchw` are currently hard-coded inside
`to_onnx` and mutate private context fields directly. Move this into a
`LayoutAdapter` helper that uses public context methods, stamps symbolic origins
consistently, and owns rank/permutation validation. This keeps the core loop
focused on JAXPR lowering.

### 5. Remove or implement no-op configuration surfaces

Two public-looking surfaces are currently misleading:

- `run_optional_shape_inference(...)` always returns the model unchanged.
- `record_primitive_calls_file` is stored on `IRContext` but not consumed in the
  converter package.

Either wire these features end-to-end with tests, or mark/remove them so callers
do not assume they are active diagnostics.

### 6. Centralize dtype and shape coercion helpers

`conversion_api.py`, `ir_context.py`, and `ir_builder.py` each contain local
helpers for dtype and shape conversion. Move the shared rules into one IR utility
module and keep only policy-specific wrappers near their call sites. This would
reduce drift around float promotion, symbolic label handling, and integer
normalization.

### 7. Make optimizer failures optionally strict

`conversion_api.to_onnx` logs optimizer failures and continues. That is friendly
for exports but risky for CI and development because structural regressions can
hide behind a warning. Add a strict mode, for example an environment flag or
internal parameter, that re-raises optimizer failures in tests and debugging
sessions.

### 8. Make the optimizer pipeline declarative

`optimize_graph` is a hard-coded sequence. A small pass registry with names,
graph/function applicability, and debug flags would improve documentation,
testing, and traceability. It would also help keep
`advanced_topics/ir_optimizer.md` synchronized with the implemented pass list.

### 9. Extend constant folding for multi-output primitives

`ConstantFolder.install_producers` maps each outvar to its producing equation,
but `try_evaluate` does not track which output index is being requested. If a
handler returns multiple outputs, each outvar could see the whole handler result
instead of its corresponding element. Store `(eqn, output_index)` and cache each
output independently.

### 10. Normalize function container handling

Some converter paths handle model functions as either dictionaries or sequences,
while `optimize_graph` assumes `ir_model.functions.values()`. If `onnx_ir`
surface variants differ, this becomes a portability hazard. Add one helper that
iterates functions safely and use it across conversion, optimization, and
postprocessing.

## Suggested Next Steps

1. Reuse the output-binding guardrail in nested-JAXPR lowering paths that
   currently run their own equation loops.
2. Extract late finalization helpers from `conversion_api.to_onnx` before making
   broader behavior changes.
3. Consolidate context typing and dtype/shape helpers once the guardrail has
   tests; those refactors touch wider plugin-facing surfaces.
