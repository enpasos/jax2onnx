# Big idea

The converter is a tiny, generic **JAXPR → IR** engine. It knows nothing about NNX, Conv, Pool, or any specific op. Its only job is:

1. **Discover** plugins (inversion of control),
2. **Activate** whatever they declare (monkey-patching to produce crisp primitives),
3. **Trace** your function to a **ClosedJaxpr**,
4. **Lower** each equation by handing it to a plugin that claimed that primitive,
5. **Assemble** an IR graph,
6. **Optimize** the IR graph with a small, safe, plugin-agnostic pass,
7. **Finalize** a valid ONNX model (stamp shapes/dtypes, prune, serialize).

Everything op-specific — layouts, padding math, attribute shapes, NHWC↔NCHW, etc. — stays in plugins.

---

# Roles & responsibilities

## Core (plugin-agnostic)

* **Plugin discovery.** Recursively import `plugins2/*`. Plugins self-register into a registry keyed by **primitive name** (string). The core never sees concrete classes like `nnx.Conv`.
* **Activation window.** Core enters a context that applies *whatever patches plugins declare*. This context **wraps tracing** so patched high-level calls (e.g., `nnx.Conv.__call__`) emit the right primitive names. No allowlists; no special-cases.
* **Tracing.** `make_jaxpr(fn)(*shape_specs)` yields a **ClosedJaxpr**: `(constvars, invars, eqns, outvars)`.
* **IR assembly.** Walk equations in order; for each equation:

  * Look up `PLUGIN_REGISTRY2[eqn.primitive.name]`.
  * Give it the equation and a **lowering context**; it emits IR nodes/values.
  * Assert that **every** `eqn.outvars[i]` is bound to an IR value before moving on (generic guardrail).
* **IR optimization (safe, structure-only).** Run small, local rewrites that don’t encode op semantics, e.g. folding redundant layout ping-pongs (see below).
* **Finalize.** Add model inputs/outputs, stamp symbolic dim labels (e.g. `"B"`), prune dead nodes/initializers, serialize `ModelProto`.

## Plugin (op-specific)

Each plugin describes one primitive (or one high-level function). It has three standard pieces:

* **Binding specs (monkey-patching).** “When a user calls X, bind primitive named P.”
  Example: patch `flax.nnx.Conv.__call__` so the *traced* program contains `primitive.name == "nnx.conv"`. If NNX exposes multiple symbols, *the plugin* lists them all. The core just applies what’s declared.

* **Abstract eval (shape/dtype).** Given JAX abstract values (`ShapedArray`), return the result’s abstract value (or tuple). No real compute; just shape math (use `lax.*` if helpful). This is used by JAX during tracing.

* **Lowering (IR emission).** Given a `LoweringContext` and the equation:

  * Pull IR inputs via `ctx.get_value_for_var(eqn.invars[i])`.
  * Create IR nodes (Conv, Transpose, Reshape, …).
  * Produce IR outputs and **bind** them to `eqn.outvars[i]` via `ctx.bind_value_for_var(...)`.
  * Return nothing (binding suffices) or return the produced values (the core will bind any unbound outvars generically).

That’s it. The contract is tiny and uniform across all primitives.

---

# Data flow end-to-end

```
User fn + shape specs
       │
       ▼
[Activation Context]  ←— plugins declare patches; core applies them (no names)
       │
       ▼
ClosedJaxpr = make_jaxpr(fn)(*specs)
       │
       ├── constvars  → ctx.bind_const_for_var(...)
       ├── invars     → ctx.add_input_for_invar(...)
       │
       └── eqns:  e₀, e₁, …, eₙ
                  │
                  ├─ core reads eᵢ.primitive.name (string)
                  ├─ plugin = REGISTRY[name]
                  ├─ plugin.lower(ctx, eᵢ)   ← emits IR nodes
                  └─ core asserts eᵢ.outvars all bound
       │
       └── outvars    → ctx.add_outputs_from_vars(...)
       │
       ▼
IR graph → **IR optimizer** → stamp shapes/symbols → prune → ONNX ModelProto
```

No step above references “Conv”, “Tanh”, or any specific op in the **core**. All knowledge sits behind the primitive name string chosen by the plugin.

---

# The lowering context (what plugins see)

A small, stable API:

* `get_value_for_var(var, *, name_hint=None) -> IRValue`
  Materialize (or retrieve) the IR value corresponding to a JAX var (const/invar/intermediate). Handles literals by creating constant initializers.

* `bind_value_for_var(var, value: IRValue)`
  Declare that this IR value is the output of `var` (an equation outvar). This is the *only* binding contract the core depends on.

* Minimal utilities the plugin can rely on (implemented once in the core):

  * Name generator (`fresh_name`),
  * Helpers for constants and attributes,
  * Optionally a couple of generic IR helpers (`emit_node`, tiny wrappers for Shape/Gather/Unsqueeze where dynamic dims are needed).

### Output binding pattern (must follow)

When lowering, always **reuse the IRValue pre-allocated for each equation outvar**. The canonical flow is:

1. Fetch inputs via `ctx.get_value_for_var(eqn.invars[i])`.
2. Pre-allocate outputs with `ctx.get_value_for_var(eqn.outvars[i])`.
3. Emit nodes whose `outputs=[...]` point to those pre-allocated values (write final results directly into them).
4. Optionally stamp dtype/shape metadata on the same values.

Avoid producing temporary outputs and “attaching” them afterwards; that pattern bypasses the var→value map and leads to orphaned tensors. Keeping the contract tight here means downstream equations always receive the correct tensor without extra bookkeeping.

The plugin uses *IR*, not a high-level “builder” that might vary. That keeps plugins robust: they create nodes with `(op_type, inputs, outputs, attributes)` and let the core do the rest.

### Functions (converter-owned call boundaries)

Decorators such as `@onnx_function` register a plugin that lowers the call into a
**FunctionScope**. At runtime the handler:

1. Builds a `FunctionKey` from the qualified target name, input aval signature,
   and capture signature (class instance id/config).
2. Reuses an existing definition if the key is cached; otherwise it opens a
   `FunctionScope`, maps parent inputs to fresh function inputs, and recursively
   lowers the original callable inside the child IRContext (constants are emitted
   as `Constant` nodes because FunctionProto bodies cannot own initializers).
3. Seals the scope into an `onnx_ir.Function`, records any attribute overrides,
   and caches the result.
4. Emits a call-site node in the parent graph with `domain/op_type` matching the
   function and wires the original in/out values to that node.

Because the definition is keyed on avals and capture signature, identical calls
share a single function body, while shape/config changes trigger new entries.

---

# Shapes & symbolic dims

* **Inputs.** If the user gives symbolic strings (e.g., `"B"`), the core creates JAX symbolic dims so the jaxpr records symbols instead of numbers.
* **Abstract eval.** Preserve `_DimExpr` symbols—call `jax.eval_shape` on the
  original callable (with `ShapeDtypeStruct` inputs) instead of doing manual
  shape math. Never cast symbolic dims to ints.
* **Dynamic shapes in IR.** When an IR op needs runtime sizes (e.g., a flatten), plugins use:

  * `Shape(x)` → shape vector,
  * `Gather(shape, axis=i)` → ith dimension,
  * `Unsqueeze/Concat` → assemble a runtime shape tensor,
  * `Reshape(x, shape_tensor)`.
* **Output stamping.** After lowering, the core restamps inputs/outputs so symbolic labels survive through ONNX `ValueInfo` (only where no concrete size is present). `IRContext` tracks the origin tensor/axis for each symbol so helpers like `dim_as_value` can materialize runtime shapes via `Shape → Gather → Squeeze`.

---

# Determinism & graph hygiene

* **Deterministic names.** The core’s `fresh_name` yields deterministic per-graph names; initializers keep stable names based on plugin hints (when feasible).
* **Single-node policy.** If a plugin needs `Reshape`, it emits one `Reshape` and a *single* constant shape initializer if static; it avoids const-only `Concat`.
* **Pruning.** A simple backwards mark from graph outputs removes dead nodes and unused initializers. This keeps **`match="exact"`** tests strict.
* **No dangling inputs.** The core asserts every outvar is bound; graph is built in jaxpr order so edges are naturally well-formed.

---

# IR optimizer (plugin-agnostic)

We run a small, safe, structure-only pass on the IR **before serialization**. Current rule:

## Fold redundant Transpose pairs

**Pattern:** `Transpose → [pure elementwise]* → Transpose`
**Condition:** the two permutations compose to identity.
**Middle ops allowed:** pure elementwise that don’t reorder elements (`Relu`, `Gelu`, `Elu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Dropout`, `Cast`, `CastLike`, `Identity`, etc.).
**Not folded:** Anything across non-elementwise (e.g., `AveragePool`, `Conv`, …).

### How we match (robust to IR backend differences)

* **Follow the real consumer chain** by **name *or* object identity** (some backends wrap/rename `Value`s).
* **Skip unrelated nodes** (Const/Shape/etc.) that don’t consume the current tensor.
* **Require single-consumer** at each hop (no branching rewires).
* **Check permutations** via `perm` attributes if available.
* **Fallback** when `perm` isn’t readable: if shapes before `T1` and after `T2` are equal and the middle is strictly elementwise, treat composition as identity and fold.

### Rewiring / deletion (backend rules)

* `onnx_ir.Node.inputs` is **immutable**; use the backend API
  **`Node.replace_input_with(index:int, value:Value)`** to edit inputs.
* Rewire **all** consumers of `T2`’s output (by name or object) to the kept tensor.
* Rewire **graph/model outputs** (output lists) and the **var→value** map so no edge points to removed nodes.
* Delete `T2` then `T1` (higher index first).

This pass is **op-agnostic** (no Conv/Pool knowledge), conservative (no branching), and stable across IR backends (works with unnamed values and different attr representations).

---

# Testing expectations (how “exact” works)

* **Anchored path checks.** Tests can say:
  `Transpose → Conv → Relu → AveragePool → …`
  With `match="exact"`, the test fails if required ops are missing or **extra** ops are present between anchors.
* **CNN sentinel.** The CNN static test is a canary: if Conv doesn’t lower, flatten will see `{B,14,14,1}` and fail a later `Reshape(B,3136)`. With the optimizer, extra `Transpose…Transpose` pairs around **Relu** are eliminated; pairs around **AveragePool** remain (by design).

See `docs/expect_graph_reference.md` for a focused reference on writing
`expect_graph` checks (shapes, counts, symbols, and helper flags).

---

# Typical plugin lifecycle (concrete but generic)

1. **Register**
   `@register_primitive(jaxpr_primitive="…")` puts an instance in the registry under that **string** key. The core will later match that key with `eqn.primitive.name`.

2. **Patch**
   `binding_specs()` returns `MonkeyPatchSpec`s: “replace `module.symbol` with `prim.bind(…)` shim”. If there are multiple aliases, the plugin lists them. The core just applies them all.

3. **Abstract eval**
   `def abstract_eval(*avals, **params):` returns a `ShapedArray` (or tuple) describing the outputs. Use `jax.eval_shape` on `lax.*` helpers if that’s easier, but never call the patched function (it would recurse).

4. **Lower**
   `def lower(ctx, eqn):`

   * `x = ctx.get_value_for_var(eqn.invars[0])`
   * Create IR nodes (e.g., Transpose, Conv, CastLike, …),
   * `ctx.bind_value_for_var(eqn.outvars[0], y)`

That’s the whole contract.

---

# Failure modes & how the architecture contains them

* **Patch activation window too late.** If activation doesn’t wrap **tracing**, the jaxpr will never contain the plugin’s primitive names. The core still doesn’t special-case anything; you just see “no plugin for primitive ‘foo’”. Fix = activate around `make_jaxpr`.

* **Plugin forgets to bind the output.** Then the core’s *generic* guardrail catches it and fails the build at the exact primitive, without central knowledge of op names.

* **Multiple symbols for the same high-level op.** Plugins add multiple patch specs. The core applies them all — still no names or allow-lists in the core.

* **An unfinished plugin gets imported.** If it also patches a runtime path it can trip tracing/lowering. Fix in the plugin: either complete it or don’t patch until ready. The core does not and should not maintain an allow/deny list.

---

# Architectural Guarantees

* **Inversion of control.** The only dynamic choice the core makes is:
  “Given `eqn.primitive.name` (a string), ask the registry for a handler.”
  There is zero knowledge of concrete ops or frameworks.

* **Uniform contracts.** Every plugin implements the same three hooks. The core only provides generic services (var→value map, name generator, constant creation, and a place to put nodes).

* **No central policy on which plugins are ‘on’.** Activation applies *whatever* plugins declare. If a plugin shouldn’t change tracing yet, it shouldn’t publish a monkey-patch — that decision is local to the plugin, not the core.

---

# TL;DR blueprint (for maintainers)

1. **Core** = small, generic: discover, activate, trace, loop eqns, call `plugin.lower`, assert outputs bound, **run IR optimizer**, finalize ONNX.
   No plugin names. Ever.

2. **Plugins** = specific: declare patches (all aliases), implement `abstract_eval`, implement `lower` (bind outvars), own all op semantics.

3. **Context** = minimal API for plugins: `get_value_for_var`, `bind_value_for_var`, `fresh_name`, plus a couple IR conveniences; no framework knowledge.

4. **Optimizer** = tiny, safe, and IR-only: fold layout ping-pongs across pure elementwise ops; match by name **or** object; never mutate `Node.inputs` directly — use `replace_input_with`.
