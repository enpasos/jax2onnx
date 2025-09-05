Here’s a drop-in section for your design doc that nails **Pattern A** as the canonical output wiring contract and aligns the rest of the doc to it.

---

# Canonical output wiring 

**Goal:** every equation’s ONNX/IR node(s) must produce the **same** `IRValue` object that the converter associates with the equation’s **outvar(s)**. This guarantees downstream equations “see” the correct tensor and prevents orphan nodes.

## Contract (required for all plugins)

In `lower(ctx, eqn)`:

1. **Fetch inputs from the context**

   ```python
   xs = [ctx.get_value_for_var(v) for v in eqn.invars]
   ```
2. **Pre-allocate outputs from the context**

   ```python
   ys = [ctx.get_value_for_var(v) for v in eqn.outvars]
   ```
3. **Emit nodes whose `outputs=[...]` reference those exact `ys`**

   ```python
   node = ir.Node(
       op_type="SomeOp",
       domain="",
       inputs=[...xs...],
       outputs=[ys[0], ys[1], ...],   # ← write directly into pre-allocated out slots
       name=ctx.fresh_name("SomeOp"),
       attributes=[...],
   )
   ctx.add_node(node)
   ```
4. **(Optional) Stamp dtype/shape on `ys[i]` if the plugin can infer it**
   (helps strict tests and better ValueInfo)

   ```python
   # _stamp_type_and_shape(ys[0], inferred_shape)
   ```
5. **Return nothing** (or return `ys`—the core will ignore/only use for checks).

> Never “attach” or “rename” an output post-hoc. The **node must output the `IRValue` obtained from `ctx.get_value_for_var(eqn.outvars[i])`.**

### Why this Pattern 

* The converter’s var→value map already associates each **outvar** with the `IRValue` you pre-allocated.
* When your node writes into that same object, consumers later calling `ctx.get_value_for_var(outvar)` get the **identical** tensor — no placeholders, no islands, no pruned initializers.

### Anti-patterns (don’t do this)

* ❌ Build a node that returns a fresh `IRValue` (e.g., `y_tmp`) and then try to “attach/rename” it later.
  This can silently skip the central var→value map; the next equation won’t find your result.
* ❌ Emit nodes with outputs you never bind to `eqn.outvars[i]`.
* ❌ Depend on private context fields to force bindings (e.g., `_var_values`, ad-hoc setters).  

---

# Example: Conv (NHWC → NCHW → Conv → NCHW → NHWC)

```python
def lower(self, ctx, eqn):
    x_var, w_var, b_var = eqn.invars[:3]
    y_var = eqn.outvars[0]

    # 1) Inputs & pre-allocated output
    x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
    w = ctx.get_value_for_var(w_var, name_hint=ctx.fresh_name("kernel"))
    b = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("bias")) if use_bias else None
    y = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))   # ← Pattern A

    # 2) NHWC -> NCHW (write into a temp)
    x_nchw = ir.Value(name=ctx.fresh_name("to_nchw"), type=x.type, shape=None)
    ctx.add_node(ir.Node("Transpose","", [x],[x_nchw],
                         name=ctx.fresh_name("Transpose"),
                         attributes=[ir.Attr("perm", ir.AttributeType.INTS, (0,3,1,2))]))

    # 3) Conv (temp out)
    y_nchw = ir.Value(name=ctx.fresh_name("y_nchw"), type=x.type, shape=None)
    inputs = [x_nchw, w] + ([b] if use_bias else [])
    ctx.add_node(ir.Node("Conv","", inputs,[y_nchw],
                         name=ctx.fresh_name("Conv"),
                         attributes=[...strides/dilations/pads/group...]))

    # 4) NCHW -> NHWC, **into `y` (the pre-allocated outvar value)**
    ctx.add_node(ir.Node("Transpose","", [y_nchw],[y],
                         name=ctx.fresh_name("Transpose"),
                         attributes=[ir.Attr("perm", ir.AttributeType.INTS, (0,2,3,1))]))

    # 5) (optional) shape/dtype stamping on `y`
    # _stamp_type_and_shape(y, inferred_nhwc)

    # Done: no return needed; `y` already carries the outvar.
```

> Note how the **last node outputs directly to `y`** (the pre-allocated outvar). There is no attach/rename step.

---

# Example: Linear with flatten + restore (multi-node)

```python
def lower(self, ctx, eqn):
    x_var, k_var, b_var = eqn.invars
    y_var = eqn.outvars[0]

    x = ctx.get_value_for_var(x_var)
    k = ctx.get_value_for_var(k_var)
    b = ctx.get_value_for_var(b_var) if use_bias else None

    # Optional flatten → Gemm → final Reshape
    x2d = ir.Value(name=ctx.fresh_name("x2d"), type=x.type, shape=None)
    shape2d = const_i64(ctx, [-1, in_features])
    ctx.add_node(ir.Node("Reshape","", [x, shape2d],[x2d], name=ctx.fresh_name("Reshape")))

    gemm_out = ir.Value(name=ctx.fresh_name("gemm_out"), type=x.type, shape=None)
    inputs = [x2d, k] + ([b] if use_bias else [])
    ctx.add_node(ir.Node("Gemm","", inputs,[gemm_out], name=ctx.fresh_name("Gemm"),
                         attributes=[...alpha/beta/trans...]))

    # Pre-alloc final output and write into it
    y = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))    # ← Pattern A
    final_shape = ...  # static or built via Shape/Slice/Concat
    ctx.add_node(ir.Node("Reshape","", [gemm_out, final_shape],[y], name=ctx.fresh_name("Reshape")))

    # Optional: stamp shape/dtype on y
```

---

# Plugin responsibilities (updated)

* **Binding specs:** declare monkey-patch shims; the core applies them during the activation window.
* **Abstract eval:** return precise `ShapedArray`(s); preserve symbolic dims.
* **Lower (Pattern A):**

  * Inputs via `ctx.get_value_for_var(eqn.invars[i])`
  * Outputs via **pre-allocated** `ctx.get_value_for_var(eqn.outvars[i])`
  * All emitted nodes must **write into those outputs**
  * (Optional) `_stamp_type_and_shape` on outputs when known
  * No post-hoc attach/rename helpers; no private context pokes

> The converter enforces this with a generic guardrail: after each `eqn`, **all `outvars` must be bound**.
 
---

# Quick checklist (per `lower`)

* [ ] All inputs fetched via `get_value_for_var`.
* [ ] All outputs **pre-allocated** via `get_value_for_var(outvar)`.
* [ ] Final node(s) **output directly to those pre-allocated `IRValue`s**.
* [ ] No attach/rename helpers used.
* [ ] Weight/bias initializers appear in node `inputs`.
* [ ] (Optional) Output shape/dtype stamped when available.

 
