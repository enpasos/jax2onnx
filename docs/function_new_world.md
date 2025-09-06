# ONNX **functions** in the “new-world” (converter2/IR)

*A design you can implement, test, and scale to arbitrary nesting*

---

## 0) Goals

* Support **both** `nnx.Module.__call__` **and native Python functions** decorated with `@onnx_function`.
* Export each decorated call as a **true ONNX FunctionProto** (in `model.functions`), not inline nodes.
* Work for **nested functions** (functions calling functions), with a **single definition** per unique body (dedup).
* Preserve **symbolic dimensions** and **dtypes** across boundaries.
* Keep the **core** converter2 plugin-agnostic; implement function logic with well-defined extension points.
* Avoid global mutable state; make function building **scoped to the active conversion**.

---

## 1) High-level lifecycle (new world)

```
@onnx_function applied to a class __call__ or a free function
      │
      ▼
(1) Decoration-time: wrap the target so calls bind a unique JAX Primitive.
    • remember the original callable (class-bound or free fn)
    • register a FunctionPlugin keyed by the Primitive name
      │
      ▼
(2) JAX tracing: each call emits the Primitive into the jaxpr (not inlined).
      │
      ▼
(3) Converter2 jaxpr walker encounters the Primitive.
    It calls FunctionPlugin.get_handler(...)(converter, eqn, params).
      │
      ▼
(4) Build-or-reuse the function body in a FunctionScope (child context).
    • compute a dedup key: target + input aval signature + capture signature
    • if new: lower original callable recursively into a child IR context
      (weights captured as Constant nodes; scalars as attributes or constants)
    • register FunctionProto into model.functions; cache it
      │
      ▼
(5) Emit the parent call-site node: op_type=function_name, domain=...
    • wire parent inputs/outputs → the call node
```

This mirrors the “old-world” semantics but uses first-class converter2 IR concepts.

---

## 2) IR extensions (builder/context)

Add **three** minimal, focused facilities:

### 2.1 `FunctionRegistry` (per IRContext)

A scoped cache living on the **IRContext**:

```python
class FunctionKey(NamedTuple):
    qualified_name: str              # "pkg.mod.Class" or "pkg.mod.func"
    input_sig: tuple[tuple[tuple[Any, ...], str], ...]  # ((shape, dtype), ...)
    capture_sig: tuple[Any, ...]     # stable capture token (e.g., instance digest)

class FunctionDef:  # internal handle
    name: str       # fully resolved function name (unique)
    domain: str     # "" or "ai.enpasos.fn"
    inputs: list[ir.ValueInfo]
    outputs: list[ir.ValueInfo]
    body_nodes: list[ir.Node]
    # (no initializers; FunctionProto uses Constant nodes instead)

class FunctionRegistry:
    def get(self, key: FunctionKey) -> FunctionDef | None: ...
    def put(self, key: FunctionKey, fdef: FunctionDef) -> None: ...
    def all(self) -> Iterable[FunctionDef]: ...
```

Attach an instance at `IRContext._function_registry`.

### 2.2 `FunctionScope` (first-class child scope)

A context manager that creates an **isolated subgraph buffer** where the body will be lowered:

```python
class FunctionScope:
    def __init__(self, parent_ctx: IRContext, name: str, domain: str = ""): ...
    def begin(self, parent_inputs: list[ir.Value]) -> list[ir.Value]:
        """Create function inputs in child ctx, mapped from parent Values."""
    def end(self, child_outputs: list[ir.Value]) -> FunctionDef:
        """Seal inputs/outputs, return FunctionDef registered on the parent."""
    @property
    def ctx(self) -> IRContext: ...
```

Implementation: internally allocate a **child IRContext** with its own node buffer; map the parent `Value` metadata (shape/dtype/symbols) onto newly created input `ValueInfo` for the function.

> Important: **FunctionProto cannot contain initializers**. Inside the function scope, emit tensors via **Constant** nodes (attributes hold the tensor), not initializers.

### 2.3 `IRBuilder.function_scope(...)` sugar

```python
@contextmanager
def function_scope(self, name: str, domain: str = "") -> Iterator[FunctionScope]:
    fscope = FunctionScope(self._ctx, name, domain)
    try:
        yield fscope
    finally:
        pass
```

This is syntactic sugar for convenience in plugins/handlers.

---

## 3) Plugin-side API (no core coupling)

### 3.1 `@onnx_function` decorator (new world)

Keep the decorator able to target **both** kinds:

* **Classes**: wrap `__call__`
* **Free functions**: patch module symbol

At decoration time:

* **Create** a unique JAX Primitive name, e.g. `onnx_fn::{qualified_name}`.
* **Instantiate** a `FunctionPlugin(prim_name, target)`; store it in the plugins2 registry.
* **Wrap** the callable so that calling it **binds the Primitive** and passes a **small, static** `instance_token` param for class targets.

> Do **not** use global weakmaps. Instead, use a **conversion-scoped** instance table (see §6) that the wrapper writes into via a ContextVar pointing at the current conversion context.

### 3.2 `FunctionPlugin.get_handler(...)`

The handler gets called from the jaxpr walker:

```python
def handler(converter, eqn, params):
    ctx = converter.builder._ctx  # parent IRContext
    # 1) Resolve original callable (class instance or free function)
    orig, capture_sig = resolve_callable(params, ctx)   # see §6
    # 2) Build dedup key
    in_sig = aval_signature(eqn.invars)                 # shapes (with symbols) + dtypes
    key = FunctionKey(qualified_name=self.name, input_sig=in_sig, capture_sig=capture_sig)
    # 3) Build or reuse
    fdef = ctx._function_registry.get(key)
    if fdef is None:
        fname = stable_function_name(self.name, in_sig, capture_sig)
        with converter.builder.function_scope(fname, domain="") as fscope:
            # Map parent inputs -> child inputs
            in_vals_parent = [ctx.get_value_for_var(v) for v in eqn.invars]
            in_vals_child = fscope.begin(in_vals_parent)

            # Temporarily redirect lowering into the child scope
            with temporarily_use_child_builder(converter, fscope.ctx):
                out_vals_child = lower_original_callable(orig, in_vals_child, static_kwargs=params)

            fdef = fscope.end(as_list(out_vals_child))
        ctx._function_registry.put(key, fdef)
    # 4) Emit call-site node in parent graph
    emit_call_node(ctx, fdef, eqn.invars, eqn.outvars)
```

All heavy lifting (name gen, value lookup, stamping) comes from IRContext/IRBuilder.

---

## 4) Stable naming, signatures & captures

### 4.1 Function name

Deterministic and debuggable:

```
{qualified_target_name}__sig{short_hex(hash(input_sig, capture_sig))}
```

Examples:

* `pkg.mod.SuperBlock__sig7b31`
* `pkg.mod.my_block__sig94f2`

### 4.2 Input signature (avals)

From `eqn.invars`:

* **Shapes**: preserve symbolic tokens (e.g., `"B"`).
* **Dtypes**: `str(dtype)` for stability.
* Signature element: `((dim0, dim1, ...), "float32")`

### 4.3 Capture signature (no globals)

* **Class targets (`nnx.Module`)**: compute a **shallow digest** of relevant config/state:

  * Example: sorted `(field_name, shape, dtype)` for parameters or `setup` attrs, plus module class name.
  * If too heavy, combine **(module class name, id(instance))** **and** persist a **per-conversion** instance table mapping `instance_id -> instance`. This avoids globals and keeps keys stable within a single run.
* **Free functions**: `()`.

> Provide an overridable `capture_signature_of(target)` hook for advanced plugins (e.g., include `approximate` flag for GELU).

---

## 5) Static kwargs and constants (inside the function)

* **Static kwargs** (bool/int/float) are not function inputs. They affect **which nodes** are emitted.
* If a scalar is required as a tensor, emit a **Constant** node in the **function body** via a builder helper:

```python
val = builder.constant_tensor(name="const_alpha", np_value=np.array(0.1, dtype=np.float32))
```

> Remember: FunctionProto has **no initializers**, only nodes & attributes.

---

## 6) Class instances — no global weakmaps

Introduce a **conversion-scoped** context:

```python
# At converter2 entry (once per to_onnx call):
CURRENT_CONVERSION = ContextVar("J2O_CURRENT_CONVERSION")
class ConversionScope:
    def __init__(self):
        self.instance_table: dict[int, object] = {}
```

* Enter `ConversionScope()` at the top of the conversion; set the ContextVar.
* In the **decorator wrapper** for class targets:

  * Lookup the current scope: `scope = CURRENT_CONVERSION.get(None)`
  * Allocate `token = id(self)` and store `scope.instance_table[token] = self`
  * Call `primitive.bind(..., instance_token=token)` (static param)
* In the **handler**:

  * Get `token = params.pop("instance_token", None)`
  * Resolve instance: `scope.instance_table[token]`
    This is **scoped**, thread-safe, and not global.

> If the converter is invoked without the scope (e.g., direct call), raise a clear error.

---

## 7) Recursively lowering the body

Provide a tiny utility to **temporarily** redirect lowering to the child context:

```python
@contextmanager
def temporarily_use_child_builder(converter, child_ctx):
    saved = converter.builder
    try:
        converter.builder = BuilderFacade(child_ctx)  # exposes ._ctx, .fresh_name, .constant_tensor, etc.
        yield
    finally:
        converter.builder = saved
```

`BuilderFacade` narrows the surface the function body needs (fresh names, constants, maybe a few tiny helpers). All leaf plugins keep working because they read from `converter.builder`.

---

## 8) Call-site emission in the parent graph

After `fdef` is known:

```python
in_vals  = [ctx.get_value_for_var(v) for v in eqn.invars]
out_vals = [ctx.get_value_for_var(v) for v in eqn.outvars]
node = ir.Node(
  op_type=fdef.name,
  domain=fdef.domain,   # "" or "ai.enpasos.fn"
  inputs=in_vals,
  outputs=out_vals,
  name=ctx.builder.fresh_name(fdef.name),
)
ctx.add_node(node)
```

The serializer emits:

* `model.graph` = parent graph with the call node,
* `model.functions` = all `FunctionProto`s from `ctx._function_registry`.

---

## 9) Nesting

Function bodies often call **other** `@onnx_function` targets:

* Because the decorator bound a Primitive, inner calls manifest as Primitive equations when the body is lowered in the child scope.
* The handler logic (dedup + `FunctionScope`) runs **again** in that child context; inner `FunctionProto`s are built and cached in the **same** `IRContext._function_registry` (shared by parent & child via the parent pointer).
* The outer function’s body will contain **call nodes** (op\_type = inner function name). At serialization, both defs appear under `model.functions`.

No extra wiring needed: the mechanism composes naturally.

---

## 10) Optimizations

* Run IR optimizations **inside** function scopes and **outside** them; treat call nodes as **opaque boundaries** (no cross-boundary folding).
* Optionally expose a pre-export **inlining pass** for trivial wrappers (opt-in) if users want whole-graph optimizations.

---

## 11) Test & tooling compatibility

* Keep `plugins2.register_example(...)` **mirroring** to the legacy `plugins.register_example` so `scripts/generate_tests.py` continues to discover examples (both `examples` and `examples2`) **without any generator changes**.
* Unit tests:

  * static & dynamic batch: expect **exactly 1** function def for `SuperBlock`.
  * nested demo: outer+inner count as 2 definitions; dedup across sites.
  * numeric parity with the inlined model (same tolerances).
  * symbols preserved across the boundary (e.g., `"B"` on inputs/outputs).

---

## 12) Edge cases & policies

* **Multiple outputs**: return a list/tuple; `FunctionScope.end` stamps all outputs.
* **Zero-input functions**: supported; call node has 0 inputs.
* **RNG / dropout**: respect `deterministic=True` (static kwarg), ensuring body is deterministic.
* **Captured weights**: leaf plugins inside the function body must create **Constant** nodes (NOT initializers).
* **Dtype promotion**: follow the same dtype rules inside the body (e.g., f64 variant).

---

## 13) Implementation plan (phased)

**Phase 1 — Scaffolding**

* Add `FunctionRegistry`, `FunctionScope`, `IRBuilder.function_scope`.
* Add `CURRENT_CONVERSION` scope and pass it at converter2 top.
* Harden builder facade (fresh names, constants, add node).

**Phase 2 — Plugin glue**

* Make `@onnx_function` work for classes **and** free functions.
* Implement `FunctionPlugin` handler per §3 with dedup key.

**Phase 3 — Serializer**

* Serialize `FunctionDef` as **ONNX FunctionProto** (domain `""` or `ai.enpasos.fn`).
* Emit parent call nodes.

**Phase 4 — Tests**

* Port `examples2/onnx_functions_000` and friends; assert `len(model.functions)`.

**Phase 5 — Quality**

* Symbols, numeric parity, nesting demos.
* Optional: inliner pass (off by default).

---

## 14) Minimal code sketch (just interfaces)

```python
# converter2/ir_context.py
class IRContext:
    def __init__(self):
        self._nodes = []
        self._function_registry = FunctionRegistry()
        # ...

# converter2/function_scope.py
class FunctionScope:
    def __init__(self, parent_ctx, name, domain=""):
        self.parent_ctx = parent_ctx
        self.child_ctx = IRContext()   # or a child view using same symbol table
        self.name = name; self.domain = domain
        self._in = []; self._out = []

    def begin(self, parent_inputs):
        self._in = [self.child_ctx.add_input_like(v) for v in parent_inputs]
        return self._in

    def end(self, child_outputs):
        self._out = list(child_outputs)
        fdef = FunctionDef(self.name, self.domain,
                           inputs=stamp(self._in), outputs=stamp(self._out),
                           body_nodes=list(self.child_ctx._nodes))
        self.parent_ctx._function_registry.put(
           FunctionKey(...built earlier...), fdef
        )
        return fdef

# plugins2/plugin_system.py (core parts)
def onnx_function(target): ...  # wraps classes & free functions, binds Primitive

class FunctionPlugin:
    def get_handler(self, converter):
        def handler(conv, eqn, params):
            ctx = conv.builder._ctx
            orig, cap = resolve_callable(params)  # scope.instance_table[token]
            key = FunctionKey(self.name, aval_sig(eqn.invars), cap)
            fdef = ctx._function_registry.get(key)
            if fdef is None:
                fname = stable_name(self.name, ...)
                with conv.builder.function_scope(fname) as fscope:
                    in_parent = [ctx.get_value_for_var(v) for v in eqn.invars]
                    in_child = fscope.begin(in_parent)
                    with temporarily_use_child_builder(conv, fscope.ctx):
                        out_child = self._orig_fn(*in_child, **static_kwargs(params))
                fdef = fscope.end(as_list(out_child))
            emit_call_node(ctx, fdef, eqn.invars, eqn.outvars)
        return handler
```

---

## 15) Why this works (and scales)

* It preserves the **old-world semantics** (Primitive hook + recursive lowering in an isolated scope) while using **first-class** converter2 IR constructs.
* It eliminates **global state** via a **conversion-scoped** instance map.
* It supports **arbitrary nesting** by construction.
* It separates concerns cleanly:

  * decorator: *mark & route calls*
  * handler: *build or reuse function defs, emit call nodes*
  * IR builder: *own the graph, functions, and serialization*

---

## 16) Acceptance criteria

* `tests/examples2/test_onnx_functions_000::test_000_one_function_on_outer_layer`:

  * **Exactly one** FunctionProto in `model.functions`.
  * Numeric parity with eager call, static **and** `"B"`-symbolic batch.
* Nested demo produces correct count (outer+inner) and reuse across sites.
* No failures in unrelated primitive tests (function boundaries are opaque).

---

## 17) Future niceties (defer)

* Public API for **user-selected inlining**.
* Pluggable **capture signature** providers for custom modules.
* A small **library** export (shared functions across models).

---

**Bottom line:**
Implement `FunctionRegistry` + `FunctionScope`, wire `@onnx_function` to a Primitive + `FunctionPlugin` handler that **builds or reuses** function bodies in a child IR context, **emits call sites** in the parent, and **serializes** FunctionProtos. This gives you the same power and reliability as the old world, with cleaner boundaries and no global state — and it scales naturally to nested functions.
