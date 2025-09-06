# Goals (what “done” looks like)

1. You can decorate a class or function with `@onnx_function` and get:

   * A **single ONNX FunctionProto** emitted and referenced by a call-site node, not an inlined body.
   * Stable function naming & deduplication (same body → 1 function def, multiple call-sites).
   * Correct numerics and shapes (incl. symbolic dims).
2. The example

   ```
   tests/examples/onnx_functions_000.py::000_one_function_on_outer_layer
   ```

   produces **exactly 1** function instance and passes numeric checks.
3. No core special-cases: implementation stays behind the plugin2 façades (Pattern A).

---

# Constraints & guardrails

* **Core remains op-agnostic.** The core only asks the registry for a handler and hosts the lowering context; it never “knows” about onnx\_function.
* **Pattern A** (“plugins own the outputs”) stays: function plugins must bind outvars or return Values that the core binds.
* **IR-first**: we target the `onnx_ir` builder in converter2; FunctionProto creation must fit the same IRContext/IRBuilder we use for graphs.
* **Immutability & backend API**: when editing node inputs during any rewrite, use `Node.replace_input_with(idx, value)`. Don’t assign to `node.inputs` item-wise.

---

# What exists today (old world behaviors we will re-use)

> Summary based on the old `plugin_system` you already ported into plugins2:

* **Decorator** `@onnx_function(target)`:

  * Creates a `FunctionPlugin` with a private `Primitive(name)`.
  * Registers in:

    * `ONNX_FUNCTION_REGISTRY2[name] = target`
    * `ONNX_FUNCTION_PRIMITIVE_REGISTRY2[name] = (primitive, target)`
    * `ONNX_FUNCTION_PLUGIN_REGISTRY2[name] = plugin`
* **Monkey patch**:

  * If `target` is a class, patch its `__call__`; for functions, patch the function symbol.
  * Bind `primitive.bind(...)` at call time.
  * For classes, pass an `instance_key` so we can recover the bound `self` (via `INSTANCE_MAP2`) during lowering & abstract eval.
* **Abstract eval**:

  * `FunctionPlugin.abstract_eval_with_kwargs(...)` forwards to `jax.eval_shape` over the original function/instance, preserving symbolic dims.
* **Lowering (old)**:

  * Uses a `function_handler` helper to build an ONNX Function body via a builder, then emits a call-site node.

This is most of what we need; we’ll just align the **lowering** to the **new IRContext/IRBuilder** and add a few “new-world” niceties (dedup, parameter lifting rules, shape/attr handling).

---

# Gaps to close in the new world

1. **FunctionProto builder in converter2**
   We need a small API on top of `IRBuilder` to:

   * Start a function scope (collect local nodes/initializers/value\_infos),
   * Lower the function body (re-enter converter2 on the function target),
   * End the function (commit FunctionProto to the IR model) and get a **callable op** metadata (domain/name).
2. **Deduplication**
   Cache function bodies by **shape-signature + target identity** (and optionally `instance_key` for class-based targets) to avoid multiple defs.
3. **Input/Output arity**
   Map the function’s **call-site JAXPR** args/outs to FunctionProto inputs/outputs deterministically, preserving symbolic names where possible.
4. **Parameter & scalar kwargs policy**

   * Scalars (bool/int/float) → **0-D initializers** internal to the function body (do not balloon function signature).
   * Arrays (weights) resolved by leaf plugins inside the body → stay as initializers inside the function body.
5. **Nested functions**
   Allow `@onnx_function` inside another `@onnx_function`. For v1: dedup across the **whole model**; inner functions get their own defs and call-sites.
6. **Tests**
   Port the simple example; add 3 small unit tests for function lowering boundaries.

---

# Proposed design (new-world)

## A. Converter2 integration

* Add a **FunctionRegistry** to `IRContext` (or alongside `IRBuilder`) that tracks:

  * `key → (function_name, already_emitted)`
    where `key` = `(qualified_target_name, input_aval_shape_sig, dtype_sig, capture_hash)`

* Add a **FunctionScope** helper:

  ```python
  with ctx.builder.function_scope(func_name, inputs=[...], outputs=[...]) as fctx:
      # lower function body using converter2 recursively, but into fctx
  # returns FunctionProto handle; register to model
  ```

  Minimal version can be a simple push/pop of node/initializer lists into a separate buffer while lowering the body.

* **Call-site emission**:

  * After (re)building/locating the FunctionProto, insert a node:

    * `op_type = func_name`, `domain = ""` or a private domain (e.g. `"ai.enpasos.experimental"`),
    * inputs/outputs: pass through from the call-site eqn invars/outvars.

## B. FunctionPlugin in plugins2

* **get\_handler(converter)** already exists — keep using it.

* Implement handler to:

  1. Build a **dedup key**:

     * `qualified_name(target)` or the class’ dotted name,
     * `input_aval_signature`: tuple of shapes/dtypes (keep symbolic tokens),
     * `capture_key`: for classes, the `instance_key` id plus a **shallow config** (static fields we’ll lift as constants).
  2. Check the registry; if not present:

     * Create a function scope (`function_scope`).
     * Re-enter **converter2** on the original callable (bound to instance if needed) with the same `inputs` traced as **function inputs**.
     * End scope and **register** the FunctionProto; store `(func_name, emitted=True)` under the key.
  3. Emit a call-site node bound to `eqn.invars → eqn.outvars`.

* **Parameter lifting** (in the handler before function-scope):

  * Mirror your working logic for scalar kwargs: lift to **initializers** inside the function body, keep JAX kwargs intact for tracing correctness.
  * Avoid passing “static” kwargs as function inputs (keeps signature stable).

## C. Minimal IR utilities (for stability)

* **Name policy**: stable, deterministic `function_name`, e.g.:

  ```
  {qualified_target}__sig{hash(shape+dtypes)}__v1
  ```
* **ValueInfo stamping**: reuse your input/output stamping helper on function inputs/outputs so symbols like `"B"` survive.
* **Replace-input rule**: when rewriting, always call `node.replace_input_with(idx, val)` (the backend forbids direct `inputs[i]=...`).

---

# Work plan (iterative)

## Phase 0 — Trace “as-is” sanity

* [ ] Confirm current plugins2 `FunctionPlugin` gets engaged (add temporary log in `plugin_system.FunctionPlugin.get_patch_fn`).
* [ ] Run the demo without any lowering (inline body) to confirm tracing & abstract eval **do** work in converter2.

**Exit**: demo runs, but function is inlined (0 function defs).
**Risk**: none; we’re just confirming plumbing.

## Phase 1 — Function scope + call-site

* [ ] Add `builder.function_scope(name, inputs, outputs)` context manager buffering nodes/initializers.
* [ ] In handler, for first encounter:

  * [ ] Compute key, enter scope, recursively call converter2 on the original callable; exit scope → commit FunctionProto.
* [ ] Emit call-site node; for repeats use cached definition.

**Tests**:

* [ ] `expected_number_of_function_instances == 1` passes on `000_one_function_on_outer_layer`.
* [ ] Numeric parity vs inlined model (small tolerances).

**Exit**: 1 function def, N call-sites. Numerics pass.

## Phase 2 — Signatures & stamping polish

* [ ] Input/output naming: preserve symbolic labels from `ShapedArray`s.
* [ ] Ensure shapes/dtypes carried on FunctionProto inputs/outputs.

**Tests**:

* [ ] Add a symbolic batch test for SuperBlock (e.g., `("B", 10, 3)`).
* [ ] Verify exported ONNX contains those symbols in Function input/output `value_info`.

## Phase 3 — Dedup & nested functions

* [ ] Implement keying & dedup (target + aval signature + capture key).
* [ ] Add nested function demo (e.g., function inside function; 1 outer, 1 inner def).
* [ ] Ensure re-use across multiple call-sites (same shapes) produces 1 def per unique body.

**Tests**:

* [ ] 2 call-sites → 1 function def for identical shapes.
* [ ] Nested case produces exactly 2 defs (outer + inner).

## Phase 4 — Realistic kwargs & captures

* [ ] Scalar kwargs (bool/int/float) → 0-D initializers inside function body.
* [ ] Arrays/weights are handled by leaf plugins inside the function; no function input growth.
* [ ] Class-based targets: bind `instance_key`, recover `self`, forward to the original bound method.
* [ ] RNG/deterministic kwargs (e.g., `dropout(deterministic=True)`) flow unchanged; numerics must match.

**Tests**:

* [ ] The provided SuperBlock with `deterministic=True` works; dropout contributes no randomness.
* [ ] One additional test that passes a scalar kwarg into a function and confirms it is **not** an input but an initializer.

## Phase 5 — IR optimizer compatibility pass

* [ ] Ensure the IR optimizer runs before serialization and **does not** fold into/through Function boundaries.
* [ ] If the optimizer sees a `Transpose…` pair inside a Function body, it’s fine to fold **inside the function** scope.

**Tests**:

* [ ] CNN tests still pass when functions are present elsewhere.
* [ ] A micro test: function body containing `Transpose → Relu → Transpose` gets folded internally (same call-site).

---

# Acceptance criteria

* ✅ `tests/examples/onnx_functions_000.py::000_one_function_on_outer_layer`

  * Exactly **1** function def in the model.
  * Numeric parity with the eager/original.
  * Pass on both static and symbolic batch shapes.

* ✅ Function dedup across repeated shapes (no duplicate defs).

* ✅ No core entanglement: converter2 stays plugin-agnostic; functions live in plugins2.

---

# Risks & mitigations

* **Permutations of attr/shape storage across `onnx_ir` builds**
  We already hardened attribute/shape readers in the optimizer — re-use those patterns if needed for FunctionProto stamping.

* **State capture for class-based targets** (`self` members):

  * Mitigation: rely on `instance_key` + `INSTANCE_MAP2`. Only pass **data** through tracing; avoid relying on Python objects at lower time.

* **Nested function recursion & caching**:

  * Mitigation: carry the FunctionRegistry in `IRContext` and use a “during-build” sentinel to avoid re-entrancy loops.

---

# Parking lot (nice-to-haves, defer until v2)

* Function **inlining** toggle (e.g., for backends that don’t support FunctionProto).
* Function **versioning** and stable **domain** (`ai.enpasos.functions`).
* A helper to export a **library of functions** separately from models (re-usable defs).

 