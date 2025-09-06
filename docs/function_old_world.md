# How ONNX **functions** work in the “old-world” converter (updated)

> This is the ground-truth description of how **functions** are exported to ONNX in the legacy (aka “old-world”) `jax2onnx` pipeline. It covers both `nnx.Module.__call__` **and native functions** decorated with `@onnx_function`, shows where the hooks live, how bodies are built and cached, how nesting works, and why tests find exactly the expected number of functions.

---

## 0) Scope and supported targets

`@onnx_function` can be applied to:

* **Class callables** — typically `nnx.Module.__call__`.
  Example: `@onnx_function class SuperBlock(nnx.Module): … def __call__(self, x): …`
* **Native (free-standing) functions** — plain Python callables in a module.
  Example:

  ```python
  @onnx_function
  def my_block(x, scale: float = 1.0): ...
  ```

Both paths produce the **same effect** during tracing: a dedicated JAX **Primitive** marks each call as an ONNX function boundary, instead of letting JAX inline the ops into the surrounding graph.

---

## 1) Big picture lifecycle

```
Decorate target with @onnx_function
      │
      ▼
(1) Decoration-time wrapping
    • Replace original callable with a wrapper that binds a unique Primitive
    • Remember the original callable for abstract eval / eager impl
    • Register a FunctionPlugin under that Primitive's name
      │
      ▼
(2) JAX tracing
    • Calls to the target emit the custom Primitive into the jaxpr
      │
      ▼
(3) JAXPR walk (converter)
    • Dispatcher sees the Primitive → delegates to function_handler(...)
      │
      ▼
(4) Build-or-reuse the Function body
    • Compute a dedup key (target + aval signature + capture state)
    • If new: create a FunctionScope (child builder) and recursively lower
      the original callable into that child scope; close → FunctionProto
    • Cache & attach FunctionProto to the ONNX model (model.functions)
      │
      ▼
(5) Emit the call site
    • Insert a parent-graph Node with op_type=function_name, domain=...
    • Hook up call-site inputs/outputs to that Node
```

If the same target is called again with the same signature, step (4) is skipped thanks to the dedup cache; only a new call-site node is emitted.

---

## 2) Where the pieces live (old-world code map)

* **Decoration and registry**

  * `jax2onnx.plugins.plugin_system`

    * `onnx_function` decorator (wraps classes and functions)
    * `register_example` (used by test generator)
    * global plugin/primitive registries

* **Conversion and function building**

  * `jax2onnx.converter.function_handling`

    * `function_handler(...)` — main entry called from equation lowering
    * `setup_sub_converter(...)` — builds a **child** builder/converter for function scope
    * `handle_function_parameters(...)`, `process_scalar_parameters(...)` — kwargs lifting
  * (older helper) `jax2onnx.converter.onnx_function_builder` — explicit builder utilities for FunctionProto

* **JAXPR walk**

  * `jax2onnx.converter.jaxpr_converter` — dispatch table and per-equation lowering

---

## 3) Decoration-time wrapping (the essential hook)

At import time, the `@onnx_function` decorator:

* **Creates a unique Primitive name** (e.g., `onnx_fn::pack.mod.SuperBlock` for a class or `onnx_fn::pack.mod.my_block` for a function).
* **Registers** a `FunctionPlugin` instance under that name.
* **Wraps the target** so that when you call it:

  * JAX **binds that Primitive** into the jaxpr **instead of inlining** operations,
  * The wrapper **remembers** the original callable:

    * for class targets, stores a bound method per instance via `instance_key` (see §6),
    * for function targets, stores the plain function object.

Because the Primitive is distinct, the converter can intercept calls reliably.

---

## 4) Tracing & jaxpr content

* When you trace the user function, every call to a decorated target becomes an equation with our custom **Primitive** and a vector of **inputs** plus **static kwargs** (e.g., `deterministic=True`).
* For class targets, the wrapper also includes an **`instance_key`** parameter, which lets the converter recover the `self` object/state during lowering (see §6).

---

## 5) Build-or-reuse decision (dedup key)

To avoid duplicate function definitions, the handler computes a **deduplication key**:

```
FunctionKey = (
  qualified_target_name,      # "module.path.Class" or "module.path.func"
  input_signature,            # tuple of (shape tuple, dtype) per input aval (symbols preserved)
  capture_signature,          # e.g., (id(self),) or a shallow digest of class config
)
```

* The **input signature** is built from JAX **avals**, not concrete arrays, so it preserves **symbolic dims** like `"B"`.
* The **capture signature** separates distinct class instances if their state/config matters.

If the key is present: **reuse** the previously built `FunctionProto`.
If absent: **build** a new one in a function scope (next section).

---

## 6) Building a Function body (FunctionScope)

When a body needs to be built:

1. **Create a child scope**
   `setup_sub_converter` creates a new `OnnxBuilder` + new converter instance **pointing at a fresh graph buffer**. This is the effective “FunctionScope”.

2. **Map parent inputs → function inputs**
   The scope allocates child input `Value`s mirroring the types/shapes (including symbols) of the parent call-site inputs.

3. **Restore the **original callable** and forward kwargs**

   * **Class target**: use `instance_key` to get the original instance; call its bound `__call__`.
   * **Function target**: call the remembered Python function object.
   * **Static kwargs** (e.g., `deterministic=True`) are forwarded *into* the body lowering; they do **not** become function inputs. Leaf plugins may lift scalars into **initializers** inside the function body (see §7).

4. **Recursively lower into the child scope**
   Temporarily swap the converter’s active builder so any leaf plugin lowering lands in the **child** graph. This is **exactly the same code path** as parent lowering; only the destination builder differs.

5. **Close the scope & register**
   Finalize function outputs (child `Value`s), materialize a **FunctionProto**, attach it to the model’s `functions`, and **cache** it under the dedup key.

> This isolation is what makes **nested functions** work: inner calls bind their own primitives and are handled recursively within the child scope (see §9).

---

## 7) Static arguments and parameter lifting

* **Static kwargs** (bools/ints/floats, small tuples) are **not** added as ONNX inputs.
  They influence which nodes are emitted inside the function body. If a scalar is needed in the graph, leaf plugins **lift** it as a 0-D initializer.
* The old-world code that decides this is in:

  * `handle_function_parameters(...)`
  * `process_scalar_parameters(...)`
    It handles edge cases (bools, tracers, etc.) and aims to keep signatures stable while producing deterministic graphs.

---

## 8) Emitting the call site

Once the `FunctionProto` exists (built or reused), the handler inserts a **single** `NodeProto` in the **parent** graph:

* `op_type = function_name` (the FunctionProto’s name),
* `domain = function_domain` (often `""` in the old world),
* `inputs = parent call-site inputs`,
* `outputs = parent call-site outputs`.

In ONNX, a node whose `(domain, op_type)` matches a registered `FunctionProto` is a **function invocation**.

---

## 9) Nested functions

Because decorated inner targets also bind **their** primitives, calls inside the body:

* are intercepted during **child** lowering,
* build inner `FunctionProto`s within the same **model**, using the child scope’s own function registry/cache,
* and emit **call nodes** **inside** the outer function body.

This composes naturally:

* Outer body may contain calls to inner functions.
* Each unique `(target, aval signature, capture)` produces **one** definition; multiple call sites reuse it.

---

## 10) Shapes, dtypes, symbols

* The child scope stamps **input/output ValueInfo** for the `FunctionProto` using the **avals** of the mapped inputs/outputs — preserving symbolic labels like `"B"`.
* This way, symbol names are consistent across parent graph and function bodies.

---

## 11) Optimizations and boundaries

* Standard graph optimizations (dead-code, transpose-folding, etc.) run **inside** a function scope and **outside** it.
* The function call boundary is treated as a **black box**; no cross-boundary folding is attempted.
* Some old-world flows support **optional inlining** of trivial wrappers (prior to final serialization) to allow more aggressive optimizations — but that is opt-in and separate from the function mechanism.

---

## 12) Test discovery (why “examples2” are found)

`scripts/generate_tests.py` discovers example testcases by walking the **legacy registry** via `jax2onnx.plugins.plugin_system.register_example`.

During migration, mirror any new-world example registration to the **legacy registrar** so examples in both `tests/examples/...` and `tests/examples2/...` continue to be generated **without** changing the generator.

---

## 13) Failure modes & debugging checklist

**Symptom:** *“Expected 1 ONNX function, found 0”*

1. **Decorator not applied** (or applied to the wrong symbol):

   * For classes, ensure you decorated the **class** that is **actually constructed and called**.
   * For free functions, ensure the module symbol you call is the **wrapped** one (import order matters).

2. **Primitive not in the jaxpr:**

   * Log the jaxpr or print equations; you should see your custom Primitive.
   * If not, the wrapper didn’t bind; verify the decorator ran at import time.

3. **Body built but not attached:**

   * Ensure `function_handler` uses the **parent model builder** to attach the new `FunctionProto` (the old-world does this in `setup_sub_converter` return path).

4. **Wrong context when emitting:**

   * The call-site node must be added to the **parent** builder; bodies go to the **child** builder.

5. **Dedup key mismatch:**

   * If the key is too broad (missing capture) or too narrow (includes volatile ids), you might end up skipping build or over-building. Use a stable capture signature.

6. **Numeric mismatches:**

   * Forward static kwargs like `deterministic=True`; missing flags often change the emitted body subtly.

Logs to add while debugging:

* “wrapping target … → primitive …”
* “function key … hit/miss”
* “built FunctionProto name=… inputs=… outputs=…”
* “emitted call site (domain, op\_type) …”

---

## 14) Design review (strengths / weaknesses)

**Strengths**

1. **Precise hook via Primitives** — guarantees a recognizable boundary in the jaxpr.
2. **Robust dedup** — key = target + avals + capture; avoids duplicates, supports reuse.
3. **Clean isolation** — child builder/scope reuses the **same** leaf-lowering code; no special casing.
4. **Static args done right** — stay kwargs, not runtime inputs; scalars lifted to initializers when needed.

**Weaknesses / modernization opportunities**

1. **Global instance map** — `instance_key` + global registry works but is implicit; a scoped/contextual instance carrier would be cleaner.
2. **Implicit “FunctionScope”** — implemented by “new builder + sub-converter” on the fly; a first-class `FunctionScope` API (context manager) would make the intent clearer and less error-prone.
3. **Cross-boundary optimizations** — ONNX functions are black boxes; provide an *optional inlining pass* pre-export when users want whole-graph optimizations.
4. **Parameter lifting rules** — the current helpers work but have grown organically; consider a declarative annotation (e.g., “this kwarg is compile-time only”, “this kwarg is a constant tensor input”, etc.).

---

## 15) Minimal reference pseudocode

```python
# DECORATION (classes and free functions)
def onnx_function(target):
    prim_name = f"onnx_fn::{qualified_name(target)}"
    plugin = FunctionPlugin(prim_name, target)
    registry[prim_name] = plugin

    if inspect.isclass(target):
        orig = target.__call__
        target.__call__ = plugin.wrap(orig, is_class=True)
    else:
        mod = inspect.getmodule(target)
        setattr(mod, target.__name__, plugin.wrap(target, is_class=False))
    return target

# JAXPR DISPATCH
def lower_eqn(converter, eqn):
    if eqn.primitive in function_plugins:
        return function_handler(converter, eqn, eqn.params)
    return lower_leaf_equation(...)

# FUNCTION HANDLER (class or function targets)
def function_handler(conv, eqn, params):
    ctx = conv.builder.ctx
    key = make_key(qualified_target, aval_sig(eqn.invars), capture_sig(params))

    fdef = ctx.functions.get(key)
    if fdef is None:
        sub = setup_sub_converter(parent=conv)     # child builder & converter
        in_parent = [ctx.value_for_var(v) for v in eqn.invars]
        in_child  = sub.map_inputs(in_parent)      # create child function inputs

        saved = conv.builder
        try:
            conv.builder = sub.builder              # redirect lowering to child
            out_child = call_original_callable(in_child, static_kwargs(params))
        finally:
            conv.builder = saved

        fdef = sub.finalize(out_child)             # -> FunctionProto
        ctx.functions.put(key, fdef)               # attach + cache

    emit_call_node(ctx, fdef, eqn.invars, eqn.outvars)
```

---

## 16) Bottom line

* The old-world pipeline **works** by making function boundaries explicit via a **dedicated Primitive**, then **recursively lowering** the original callable into a **child scope** to get a `FunctionProto`, and **reusing** definitions via a strong dedup key.
* It supports **both** class targets and **native functions** decorated with `@onnx_function`.
* Nested functions fall out naturally from the same mechanism.
* Test discovery depends on registering examples into the **legacy** registry.

This is the implementation we want to faithfully mirror in the new IR-based pipeline — with modernization around scope management, instance handling, and (optional) pre-export inlining for users who want cross-boundary optimization.
