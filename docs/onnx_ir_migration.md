# jax2onnx → onnx\_ir migration plan (Draft RFC)

*Last updated: 2025-08-26 (pm)*

## 1) Context & goals

**Goal:** Introduce an alternative converter pipeline based on an ONNX IR builder (“onnx\_ir”), while keeping the existing pipeline fully functional. We migrate incrementally, testcase-by-testcase (or component-by-component), without breaking users and without regressing CI.

**Non-goals (for now):**

* Changing the public user-facing API beyond an optional feature flag.
* Rewriting all plugins at once.
* Forcing all tests to run through the new pipeline immediately.

## 2) High-level approach

* Run two pipelines side-by-side:

  * **Old:** `converter/` + `plugins/` (status quo)
  * **New:** `converter2/` + `plugins2/` using `onnx_ir` builder
* Introduce a **feature flag** (``) that routes calls to either pipeline.
* Migrate the smallest, low-risk testcases first; keep CI green by isolating failures to the new path (xfail/skip where appropriate).
* When parity is reached, flip the default to the new pipeline, then remove the old code.

## 3) Terminology & naming

* **onnx\_ir:** New internal IR builder layer to construct ONNX graphs/models.
* **plugins2:** New plugin API for the `onnx_ir` pipeline.
* **converter2:** New converter stack that emits `onnx_ir` and serializes to ONNX.

> ⚠️ Keep `plugins2` / `converter2` names stable until the legacy stack is removed.

## 4) Repository layout

(Target structure unchanged; IR-specific tests live under `*2` subfolders.)

```
jax2onnx/
  user_interface.py
  converter/
  plugins/
  converter2/
  plugins2/
  plugin/__init__.py         # shared registry (see §5)
  ir/                        # optional: shared onnx_ir helpers
  sandbox/
    onnx_ir_*.py
  tests/
    t_generator.py
    conftest.py
    primitives/
    examples/
    extra_tests/
    primitives2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
    examples2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
    extra_tests2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
  scripts/
    generate_tests.py
```

### 4a) Module map (converter2 & plugins2)

A concise, practical map of the new layers—what each file owns and how they interact.

#### converter2

* **`conversion_api.py` — Orchestrator (public entry)**

  * Normalizes input specs (via `frontend._normalize_inputs_for_tracing`).
  * Traces `fn` with `jax.make_jaxpr` (threads `input_params`).
  * Creates an `IRContext` (holds builder, dtype policy, symbolic origin tables).
  * Binds constants and graph inputs from the ClosedJaxpr.
  * Iterates `jaxpr.eqns`; dispatches to `plugins2` by primitive name.
  * Finalizes outputs and materializes `onnx.ModelProto` via builder.
  * Wraps plugin activation: `import_all_plugins`, `apply_monkey_patches`, per-plugin `plugin_binding()`.

* **`frontend.py` — Input spec normalization & tracing helpers**

  * `_normalize_inputs_for_tracing(inputs, default_float=…)` → list of `jax.ShapeDtypeStruct` honoring f32/f64 policy.
  * Optional `trace_to_jaxpr(...)` convenience wrapper.

* **`ir_context.py` — Stateful lowering context**

  * Owns the **IR builder** and JAX-var → IR-value mapping.
  * `add_input_for_invar(var, i)` registers graph inputs; records symbolic dim origins `(Value, axis)` in `_sym_origin` / `_sym_origin_str`.
  * `bind_const_for_var(literal_or_array, np_array)` emits initializers for constants.
  * `get_value_for_var(var, name_hint=…, prefer_np_dtype=…)` materializes values (handles JAX `Literal`).
  * `add_outputs_from_vars(outvars)` closes graph outputs.
  * Utilities: `fresh_name(...)`, `cast_like(tensor, exemplar)`.
  * `to_model_proto(name=…)` delegates to builder for serialization.

* **`ir_builder.py` — Minimal ONNX IR assembler (using `onnx_ir`)**

  * Holds `inputs`, `outputs`, `nodes`, `initializers`, counters, and var→value map.
  * Creates `ir.Value` with proper `TensorType`/`Shape` respecting float policy.
  * `add_node(op_type, inputs, outputs, **attrs)` appends `ir.Node`.
  * Tracks symbolic dim origins (filled by `IRContext`).
  * `to_model_proto(name, ir_version=10)` → `ir.Graph`/`ir.Model` → save with `onnx_ir.save`, reload with `onnx.load_model` to return `onnx.ModelProto`.

#### plugins2

* **`plugin_system.py` — Registry & glue**

  * Global `PLUGIN_REGISTRY2` (primitive name → plugin instance).
  * Base types (`PrimitiveLeafPlugin`, optional Function-level plugin type).
  * Discovery (`import_all_plugins()`), legacy interop (`apply_monkey_patches()`), per-plugin activation (`plugin_binding()`).

* **Typical leaf plugins** (examples):

  * `jax/lax/add.py`, `jax/lax/sub.py`, `jax/lax/mul.py`: elementwise arithmetic → ONNX `Add`/`Sub`/`Mul`; use `ctx.cast_like` for dtype alignment.
  * `jax/lax/broadcast_in_dim.py`: emits `Reshape`/`Expand`, builds **dynamic target shapes** with `Shape → Gather → Concat` to avoid int-coercion of symbolic dims.
  * `jax/nnx/linear.py`: lowers `nnx.Linear` via `MatMul`/`Gemm` (+ `Add` for bias), preserving batch symbols.

#### Interaction flow (end-to-end)

1. **API call** → `converter2.conversion_api.to_onnx(...)`.
2. **Input prep** → `frontend._normalize_inputs_for_tracing(...)` (f32/f64 policy; keep symbolic names like `"B"`).
3. **Trace** → `jax.make_jaxpr` → `ClosedJaxpr`.
4. **Context** → `IRContext` created; constants bound; inputs registered; symbolic origins recorded.
5. **Lowering** → for each `eqn`: lookup plugin in `PLUGIN_REGISTRY2` → `plugin.lower(ctx, eqn)` emits nodes onto `ctx.builder`.
6. **Finalize** → `ctx.add_outputs_from_vars(...)` → `ctx.to_model_proto(...)` → `onnx.ModelProto`.

#### Cross-cutting policies

* **Float policy:** `enable_double_precision` flows from `to_onnx` → context/builder and governs default float dtype, literal casting, and IR tensor types.
* **Symbolic shapes:** Never coerce symbolic dims to Python ints. Plugins must construct target shapes at runtime (`Shape`/`Gather`/`Concat`) and can query dimension origins from `IRContext`.
* **IR version:** Pinned to an ORT-safe value (IR v10) for broad runtime compatibility.

## 5) Moving `plugin_system.py`

* **Action:** Move `plugin_system.py` → `jax2onnx/plugin/__init__.py` (or `plugin/registry.py`).
* **Shim:** Keep `jax2onnx/plugin_system.py` that re-exports from `jax2onnx/plugin` with a `DeprecationWarning`.

## 6) Feature flag & routing

**Default remains ``** during migration. We flip later.

* **In tests (legacy subtrees):** each testcase may set `: bool` (default `False`).
* **In `tests/*2` subtrees:** the flag is **forced to `True`** via that subtree’s `conftest.py`.
* **In API:** `to_onnx(*, : bool | None = None, ...)`.
* **Env toggles:** `JAX2ONNX_USE_ONNX_IR` overrides both. Optional `JAX2ONNX_SHADOW_COMPARE` for dev/CI.

Optional “shadow mode” remains the same.

## 7) Test strategy

* **Single** `tests/` tree with dual subtrees (`*2` forces IR).
* Markers: `@pytest.mark.ir_only`, `@pytest.mark.legacy_only`, `@pytest.mark.ir_xfail`.
* The generator threads `` from testcase metadata into `user_interface.to_onnx`.

### Status update (new)

* The **`broadcast_in_dim` family** is green on the IR path (static + symbolic-B variants) with the new symbolic-shape handling and dynamic target-shape construction.

## 8) Sandbox

No change.

## 9) `plugins2` design (v2 API)

* Emit `onnx_ir` nodes (prefer the builder over raw protobuf).
* **Dynamic shape rule (new):** **never** coerce potentially symbolic dims to ints (`np.asarray(..., dtype=np.int64)` will crash). Use runtime shape extraction (`Shape` + `Gather`) and **build target shapes via concat** of 1-D pieces.
* **Identity cases:** prefer an explicit `Identity` node when ONNX graph needs a distinct SSA name.

## 10) `converter2` architecture

Stages unchanged. Two **implementation details added**:

1. **Symbolic inputs for tracing (new):**
   `_as_sds_list` recognizes string dims (e.g. `"B"`) in `input_shapes` and calls **`jax.export.symbolic_shape`** to create JAX symbolic `DimSize`s. Equal names map to the same symbol. This fixes `jax.make_jaxpr` errors like “Shapes must be 1D sequences of integer scalars”.

2. **Symbol origin tracking (new):**
   `IRContext.add_input_for_invar` records **where each symbolic dim came from**:
   `ctx._sym_origin[dim_obj] = (input_value, axis)` (and a string-key fallback).
   Plugins can then call `ctx.get_symbolic_dim_origin(dim)` to recover `(Value, axis)` and build runtime shapes with `Shape → Gather`.

## 11) Public API & docs

* `to_onnx(..., : bool | None = None)` is **experimental**.
* Add a doc page: “Adopting the ONNX IR pipeline (experimental)” with **symbolic shape guidance** (see §12.2) and examples.

## 12) Migration phases & milestones

### Phase 0 – Framing

* [x] Routing + env var.
* [x] `converter2` MVP (ClosedJaxpr walk + `onnx_ir` emission).
* [x] `plugins2` entry path + registry integration.
* [x] `tests/primitives2` subtree with IR forced.
* [ ] CI: smoke job for `*2`.
* [x] Single generator/script.

### Phase 1 – Core math & tensor ops

* [ ] add/mul/sub/div/neg/cast
* [x] reshape  *(via `Reshape` in `broadcast_in_dim`)*
* [x] concat   *(for assembling dynamic target shapes)*
* [x] gather   *(for symbolic dim extraction from `Shape`)*
* [x] expand / broadcast\_in\_dim  **(landed)**
* [ ] transpose / slice
* [ ] matmul

### Phase 2 – Shape/Index ops + NNX basics

* [ ] where/select/cumsum/reduce\*
* [ ] minimal NNX: linear/conv/batch\_norm (static shapes)

### Phase 3 – Control flow & dynamic shapes

* [ ] while\_loop / scan / cond; broaden NNX; start Equinox track

### Phase 4 – Parity & flip

* [ ] parity (see §14), flip default to IR, deprecate and remove legacy

## 13) Risks & mitigations

* **Dynamic shape misuse** → plugins2 must **not** cast symbolic dims to ints. Use the runtime `Shape/Gather/Concat` pattern (see §12.2).
* **Attribute typing drift** → prefer the **mapping form** (`attributes={"axis": 0}`) with `onnx_ir.Node`. It’s accepted and keeps tests green. (Using `Attr` objects is fine too, but not required.)
* **Import churn / JAX versions** → we use `jax.extend.core` for `Literal/Var` on JAX ≥0.6; avoid touching `jax.core` directly.

## 14) Parity definition (flip DoD)

No change.

## 15) Developer ergonomics

* Logging namespaces unchanged.
* `JAX2ONNX_DEBUG_IR_DUMP=1` unchanged.
* Errors: include primitive, shapes/dtypes, and repro hint.

## 16) Example snippets

### 16.1 Symbolic inputs in `_as_sds_list` (new)

```python
# Given inputs like [("B", 49, 256)]
names = ["B"]
name2sym = {n: jax_export.symbolic_shape(n)[0] for n in names}
dims = tuple(name2sym[d] if isinstance(d, str) else int(d) for d in spec)
sds = jax.ShapeDtypeStruct(dims, jnp.float32)  # or float64 if requested
```

### 16.2 Recording origins (new)

```python
# in IRContext.add_input_for_invar(...)
for ax, d in enumerate(aval.shape):
    if not isinstance(d, (int, np.integer)):
        self._sym_origin[d] = (val, ax)
        self._sym_origin_str[str(d)] = (val, ax)
```

### 16.3 Building dynamic target shapes (new)

```python
# For shape=(B, 1, D), create a 1-D INT64 tensor of length 3:
pieces = []
# B → Shape(input) → Gather([axis_of_B])  (produces shape (1,))
# 1 → const [1] (shape (1,))
# D → Shape(input) → Gather([axis_of_D])
tgt_shape = Concat(pieces, axis=0)  # shape (3,)
Expand(x_reshaped, tgt_shape)
```

## 17) Documentation & comms

* README section: “Experimental ONNX IR pipeline”.
* Release notes: add **symbolic shape rules** and “don’t cast dims” warnings.
* Invite contributors to port specific primitives with the new helpers.

## 18) Decisions

1. **IR implementation:** `onnx_ir` (ir-py).
2. **Opset strategy:** same as legacy for now.
3. **Scope:** migrate existing plugins progressively.
4. **Performance:** measure later unless regressions appear.
5. **Trace determinism:** no expected problems.

---

### Current MVP status (updated)

* `converter2` lowers ClosedJaxpr and emits IR for `tanh` **and** `broadcast_in_dim` (including symbolic batch).
* JAX 0.6 compatibility: using `jax.extend.core` for `Literal`/`Var`.
* IR version pinned for ORT compatibility (IR\_VERSION=10).
* New **symbolic shape & origin tracking** in place; dynamic target shapes are assembled at runtime (`Shape/Gather/Concat`), avoiding int-coercion of symbolic dims.

### TL;DR

* Legacy stays default; IR is opt-in (`` or `JAX2ONNX_USE_ONNX_IR=1`).
* One `tests/` tree; `*2` subtrees force IR.
* New rules for symbolic shapes are in; `broadcast_in_dim` is green on IR.
* Keep iterating primitive-by-primitive; flip after parity.

