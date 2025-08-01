# Converting Flax Linen → ONNX via the NNX Bridge in `jax2onnx` (Flax ≥ 0.11)

`jax2onnx` now includes **first-class support** for models wrapped with `flax.nnx.bridge.ToNNX`.

The new handler automatically transforms a wrapped, _stateful_ object into a **pure, stateless function** that can be symbolically traced and exported to ONNX without shape-mismatch errors.

---

## 1. Why a Dedicated Handler?

- **`nnx.bridge.ToNNX` is stateful** – it stores parameters, RNG streams, and other mutable data.

- **Symbolic tracing** (e.g. `jax.make_jaxpr`) must be stateless – only pure functions whose outputs depend solely on their inputs can be safely traced.

Exporting a `ToNNX` object as-is mixes these two worlds and previously led to `flax.errors.ScopeParamShapeError`.

The dedicated handler resolves this by _isolating_ the stored variables and tracing a fresh, stateless `apply` function instead.

---

## 2. What the Handler Does

| Step | Action                                                                                                   | Effect                                              |
|------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 1    | **Detect** `isinstance(model, flax.nnx.bridge.ToNNX)`                                                    | Opt-in only when the wrapper is present             |
| 2    | **Select variable leaves** (`params`, `batch_stats`, …) with a small utility                             | Ignores anything that is _not_ a `nnx.Variable`     |
| 3    | **Convert** NNX variables → Linen variables via `flax.nnx.bridge.variables.nnx_attrs_to_linen_vars`      | Produces a canonical Flax “variables” dict          |
| 4    | **Suspend Flax’s shape check** (when symbolic dims appear) for the duration of tracing                   | Prevents `ScopeParamShapeError` in dynamic-batch    |
| 5    | **Create** a pure function                                                                               | `def pure_apply(*args):`<br>`    fresh = linen_cls(**ctor_kwargs)`<br>`    return fresh.apply({"params": params}, *args)`<br>No stateful closure; ready for `jax.make_jaxpr` |
| 6    | **Trace & export** with the existing `jax2onnx` pipeline                                                 | Works for static **and** dynamic shapes             |

---

## 3. User Workflow

1. **Keep your model code unchanged** – define normal Linen modules and wrap them once:

   ```python
   linen_model = MyModule(**cfg)
   bridged     = nnx.bridge.ToNNX(linen_model, rngs=nnx.Rngs(0))
   bridged     = nnx.bridge.lazy_init(bridged, dummy_input)
   ```

2. **Call `jax2onnx.to_onnx(..)`** directly on `bridged`:

   ```python
   onnx_model = jax2onnx.to_onnx(
       bridged,
       input_specs=[jax.ShapeDtypeStruct(("B", 10), jnp.float32)],
       model_name="my_model",
       opset=21,
   )
   ```

3. **Done!** Dynamic batch dimensions (`"B"`) and other symbolic shapes are now supported.

---

## 4. Supported Patterns

| Layer / Pattern                  | Status |
|----------------------------------|--------|
| `nn.Dense`, `nn.Conv`, `nn.LayerNorm`, … | ✅     |
| Nested Linen sub-modules         | ✅     |
| Dynamic batch dim `"B"`          | ✅     |
| Multiple inputs / outputs        | ✅     |

> *Note*: Additional variable collections (e.g. `batch_stats`) will be forwarded exactly as they appear in the NNX wrapper. If you add new collections, no change inside `jax2onnx` is required.

---

## 5. Examples

See the **`jax2onnx/plugins/examples/linen_bridge/`** directory:

- `dense.py` – single Dense → ReLU
- `conv.py` – 2-D Conv with dynamic height/width
- `mlp.py` – two-layer MLP

(All examples are registered automatically in the test-suite.)

---

## 6. Extending the Handler

The implementation is intentionally **minimal**:

- ~100 LOC (`jax2onnx/converter/linen_handler.py`)
- No monkey-patching of Flax internals (except a short, scoped suspension of the parameter-shape check)
- Relies only on **public** Flax ≥ 0.11 APIs

If future Flax/NNX releases add new bridge types or variable kinds, you can extend the handler with a few extra `isinstance` checks—**no changes** to the core conversion pipeline are required.

---

## 7. Takeaways

- You can now export **any** Flax Linen model wrapped with `nnx.bridge.ToNNX` to ONNX using `jax2onnx` – with *dynamic shapes* and *without* brittle hacks.
- The solution leverages Flax’s official conversion utilities, making it both **robust** and **forward-compatible**.

Happy converting! 🎉