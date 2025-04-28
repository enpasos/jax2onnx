# Symbolic‑Dimension Support in **jax2onnx**

This guide is for plugin authors who want to add or migrate primitive‑handlers so that they **round‑trip JAX symbolic shapes to ONNX symbolic dimensions**.

---
## 1. Why we need it
*   JAX ≥ 0.6.0 can treat dimensions as symbols (e.g. `"B"`) that are only resolved at run‑time.
*   ONNX supports the same idea through `dim_param`.
*   Until now `jax2onnx` converted those strings *literally* and many custom primitives lost the information during their `abstract_eval` step.

> **Key idea**: keep the symbol alive during tracing by delegating shape‑inference to JAX itself (`jax.eval_shape`).

---
## 2. High‑level flow
1. **User API** – in `to_onnx` the user still writes
   ```python
   to_onnx(fn, input_shapes=[("B", 1, 8)])
   ```
2. **conversion_api** – converts each string (e.g. `"B"`) to a real JAX `_DimExpr` using `export.symbolic_shape`.  These objects live in the `ShapeDtypeStruct`s that seed `jax.make_jaxpr`.
3. **Plugins** – every primitive handler gets those symbolic objects inside `aval.shape`.
4. **abstract_eval** – the handler runs `jax.eval_shape` on the **original JAX op** to obtain an output `ShapeDtypeStruct`, converts that to a `ShapedArray` and returns it.
5. **ONNX builder** – keeps a `var_to_symbol_map` so that when the final graph is written the symbol name (`"B"`) is restored into `dim_param`.

---
## 3. Boiler‑plate for a plugin

```python
class <MyPrimitive>Plugin(PrimitiveLeafPlugin):
    _ORIGINAL_OP: Callable | None = None  # filled by patch

    # --- abstract_eval --------------------------------------------------
    @staticmethod
    def abstract_eval(*avals: core.ShapedArray, **params):
        axis: int = params["axis"]  # example extra param

        # 1. Sanity checks
        if not all(isinstance(a, core.ShapedArray) for a in avals):
            raise TypeError("expected ShapedArray inputs")

        # 2. Specs for eval_shape
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]

        # 3. helper using the *un‑patched* op
        orig = <MyPrimitive>Plugin._ORIGINAL_OP
        def _helper(*xs):
            return orig(xs, axis=axis)  # call original op

        result = jax.eval_shape(_helper, *specs)
        out = jax.tree_util.tree_leaves(result)[0]
        return core.ShapedArray(out.shape, out.dtype)

    # --- patch_info -----------------------------------------------------
    @staticmethod
    def patch_info():
        def _creator(orig_fn):
            <MyPrimitive>Plugin._ORIGINAL_OP = orig_fn
            return PatchedCallableWrapper(orig_fn, jnp.<op>_p)
        return {
            "patch_targets": [jnp],
            "target_attribute": "<op>",
            "patch_function": _creator,
        }
```

That is *all* that is needed—no manual symbolic math, no shape strings.

---
## 4. Migration checklist
| ✓ | Step |
|---|------|
| ☐ | Capture the original JAX function in `patch_info` and store it on the plugin class. |
| ☐ | Rewrite `abstract_eval` to use **only** `jax.eval_shape` (or `jax.export` if lowering is actually needed – rare). |
| ☐ | Ensure extra params (e.g. `axis`) are **plain `int` / `bool` / enum**, _never_ tracers.  Use `int(axis)` as safeguard. |
| ☐ | Do **not** call `jax.numpy` inside `abstract_eval` – always the stored original op to avoid recursion. |
| ☐ | Add/extend test‑cases with symbolic batches: `("B", …)` and verify `expected_output_shapes`. |

---
## 5. Known pitfalls & remedies
| Symptom | Root cause | Fix |
|---------|-----------|------|
| `UnexpectedTracerError` in abstract_eval | Tried to do arithmetic directly on tracers | Don’t.  Hand control to `jax.eval_shape`. |
| `AssertionError ctx.axis_size_env is None` inside MLIR | You used `jax.export` inside abstract_eval **with** lowering; not supported while outer trace is running | Switch to `jax.eval_shape` or use `jax.export` _without lowering_ (`lower=False` once available). |
| Infinite recursion | helper function calls the patched op which re‑enters primitive | Always call the **original** un‑patched op. |

---
## 6. Example: finished `concatenate` plugin
See `jax2onnx/plugins/jax/numpy/concatenate.py` in the repo – the tests:
```
pytest tests/primitives/test_jnp.py::Test_concatenate -v
```
all pass including the dynamic‑symbolic‑batch case.

---
## 7. Extending to new primitives
1. Copy the skeleton above.
2. Replace `<op>` / `<MyPrimitive>` and parameter handling.
3. Add ONNX emission code in `to_onnx` if missing.
4. Add pytest case with symbolic dim(s).

You now have a primitive that **just works** for static, dynamic and symbolic shapes 👏.

