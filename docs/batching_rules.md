### What *is* a “batching rule” in JAX?

`vmap` is JAX’s automatic vectoriser.
When you wrap a function with `jax.vmap`, JAX:

1. **Lifts** every primitive operation executed by that function into a *batched* context (each operand now has an extra leading “batch” axis).
2. **Consults** a **batching rule** registered for that primitive to figure out
   *how the primitive should treat* that new axis and
   *where that axis ends up* in the result (or whether it disappears).

A *batching rule* is therefore **the contract that lets a primitive work inside
`vmap`**.

```python
def add_batch_rule(xs, dims, *, **params):
    # `xs`   = operands with possible batch axes
    # `dims` = tuple giving the axis of each operand that is batched,
    #          or None if that operand wasn’t batched
    # must return (result_value, result_batch_axis)
```

JAX keeps the lookup table you saw printed in your log:

```
batching.primitive_batchers[primitive] -> batching_rule
```

If a primitive *has no rule*, `vmap` will raise
`NotImplementedError: batching rule for '<primitive>' not implemented`.

---

### Why you sometimes need to write your own rule

* All primitives in **`lax.*`** and most high‑level NumPy wrappers already
  ship with rules (see the long list in your output).
* If you **define a *new* primitive** – e.g. `j2o_einsum_p` in your
  plugin – you must also supply:

  * **`def_impl` / lowering** – how to actually run it,
  * **`def_abstract_eval`** – how to compute its output shape & dtype *symbolically*,
  * **`primitive_batchers[my_primitive] = my_batch_rule`** – how to act
    when operands carry an extra batch dimension inserted by `vmap`.

Without the last piece, any user code that calls
`jax.vmap(my_einsum, in_axes=0)(…)` (including indirect uses inside Flax
attention) will break.

---

### Anatomy of a simple batching rule (example – `add`)

```python
def _add_batch(xs, dims, **kwargs):
    x, y = xs
    dx, dy = dims
    # 1. Move both batch axes to the same position (front) if necessary
    if dx != 0:
        x = lax.moveaxis(x, dx, 0)
    if dy != 0:
        y = lax.moveaxis(y, dy, 0)
    # 2. Broadcast operands so shapes match
    y = jnp.broadcast_to(y, x.shape)
    # 3. Call the primitive’s *manual* implementation
    out = lax.add(x, y, **kwargs)
    # 4. Tell vmap that the result’s batch axis is at position 0
    return out, 0
batching.primitive_batchers[lax.add_p] = _add_batch
```

Most rules fall into just a few templates:

| helper in JAX        | When used                                          | What it does                                                        |
| -------------------- | -------------------------------------------------- | ------------------------------------------------------------------- |
| `vectorized_batcher` | unary ops (`sin`, `exp`, `floor`, …)               | Just move the batch axis if needed; element‑wise op can ignore it.  |
| `broadcast_batcher`  | binary ops that broadcast (`add`, `mul`, `pow`, …) | Align & broadcast both operands, then run scalar op.                |
| custom functions     | e.g. `conv_general_dilated`, `dot_general`         | Need non‑trivial reshaping/transpose or a call to a batched XLA op. |

Your log shows exactly those three classes of rules being registered.

---

### Batching rules **and** jax2onnx

jax2onnx traces user code under a **symbolic “batch size”** to build an
ONNX graph – this tracing uses the same machinery as `vmap`.
Therefore, if your plugin introduces a primitive but forgets to define a
batching rule, conversion will fail as soon as that primitive appears
under a `vmap` (which is precisely what happens inside
`jax.nn.dot_product_attention`).

**Key point for your `Einsum` plugin**

* Because you *replace* the NumPy wrapper with a new primitive,
  you must also give it a batching rule that mimics NumPy’s behaviour:
  prepend an ellipsis (`...`) to every operand spec that is being batched
  and shift the batch axis to the front.

The minimal rule from the previous answer:

```python
def _einsum_batch_rule(vals, dims, *, equation, **params):
    # Promote batch axes into leading '...' in the equation
    if any(d is not None for d in dims) and "..." not in equation:
        in_specs, out_spec = equation.split("->")
        new_in_specs = []
        for spec, d in zip(in_specs.split(","), dims):
            new_in_specs.append("..." + spec if d is not None else spec)
        equation = ",".join(new_in_specs) + "->..." + out_spec
    result = j2o_einsum_p.bind(*vals, equation=equation, **params)
    return result, 0
batching.primitive_batchers[j2o_einsum_p] = _einsum_batch_rule
```

ensures:

* Any singleton axis inserted by `vmap` is treated as part of `...`.
* The output keeps the batch dimension in the leading position,
  exactly what Flax’s attention core expects.

---

### Debugging tips

* **Print your rule** after registration to confirm it made it into
  `batching.primitive_batchers` (like the log you showed).
* To see how JAX rewrites code under `vmap`, use
  `jax.make_jaxpr(lambda x: vmap(fun)(x))(dummy_input)` –
  missing rules will raise immediately.
* Keep the rule **stateless** and cheap: it runs during tracing, not at
  runtime.

---

#### TL;DR

*Batching rules* are the glue that lets JAX’s `vmap` (and therefore
jax2onnx’s symbolic tracer) understand what a primitive does with a new
batch axis.
If you define a fresh primitive in a plugin, you **must** ship:

1. an implementation,
2. an abstract‑eval,
3. a batching rule.

Get those three right, and your primitive will behave just like the
built‑ins under every transformation JAX applies.
