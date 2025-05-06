# ✅ Symbolic Dimension Support in `jax2onnx`

This guide explains how to **round-trip symbolic shapes** (like `"B"`) from JAX to ONNX in a robust and maintainable way—both for **plugin authors** and the **core tracing logic**.

---

## 🔍 Why Symbolic Dimensions Matter

* JAX ≥ 0.6 uses `_DimExpr` symbols for dimensions like `"B"`, only resolved at runtime.
* ONNX supports `dim_param="B"` for the same idea.
* Without care, symbolic info is lost in `abstract_eval`, and `dim_as_value` can’t infer origins during ONNX export.

---

## 🔧 How It Works

### 🔁 Round-tripping symbolic dimensions

1. `to_onnx(fn, input_shapes=[("B", 64, 14, 14)])` is called.
2. `conversion_api` converts `"B"` → `_DimExpr`, stored in all `aval.shape`s.
3. `abstract_eval` for a primitive calls `jax.eval_shape(orig_fn)` to retain symbolic shape algebra.
4. `var_to_symbol_map` records string names like `"B"` → ONNX `dim_param="B"`.

### 🧠 Metadata for Runtime Extraction

To enable `dim_as_value.to_onnx` to extract symbolic dims like `B`:

```python
# in Jaxpr2OnnxConverter
self.symbolic_dim_to_origin: dict[_DimExpr, tuple[str, int]] = {}

# in trace_jaxpr()
for input_var, aval in zip(jaxpr.invars, symbolic_avals):
    for axis, dim in enumerate(aval.shape):
        if isinstance(dim, _DimExpr):
            self.symbolic_dim_to_origin[dim] = (tensor_name, axis)
```

Then, during ONNX export, symbolic dimensions are extracted as:

```python
Shape → Gather(axis) → Squeeze → Cast
```

---

## 🧩 Plugin Template for Symbolic Support

```python
class MyPrimitivePlugin(PrimitiveLeafPlugin):
    _ORIG_CALL = None

    @staticmethod
    def abstract_eval(*avals, **params):
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
        def _impl(*xs): return MyPrimitivePlugin._ORIG_CALL(SimpleNamespace(...), *xs)
        out = jax.eval_shape(_impl, *specs)
        return core.ShapedArray(out.shape, out.dtype)

    @staticmethod
    def get_monkey_patch(orig_fn):
        MyPrimitivePlugin._ORIG_CALL = orig_fn
        def patched(self, x): return MyPrimitivePlugin._bind_primitive(...)
        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.MyPrimitiveClass],
            "target_attribute": "__call__",
            "patch_function": MyPrimitivePlugin.get_monkey_patch,
        }
```

See [`conv.py`](https://github.com/enpasos/jax2onnx/blob/main/jax2onnx/plugins/flax/nnx/conv.py) or `concatenate.py` for full examples.

---

## ✅ Migration & Pitfall Checklist

| ✅ | Item                                                              |
| - | ----------------------------------------------------------------- |
| ☐ | `abstract_eval` uses only `jax.eval_shape` on the original op.    |
| ☐ | No manual shape math or jnp calls inside `abstract_eval`.         |
| ☐ | All symbolic dims traced via `symbolic_dim_to_origin`.            |
| ☐ | Unit test includes at least one symbolic input like `("B", ...)`. |
| ☐ | `dim_as_value.to_onnx` constructs correct `Shape→Gather→Squeeze`. |

---

## 🔥 Benefits

* **Correct**: Handles dynamic & symbolic shapes, no fallbacks.
* **Maintainable**: Shape logic centralized in real ops via `eval_shape`.
* **Portable**: ONNX `dim_param` + runtime shape extraction supported.
* **Debuggable**: Errors clearly raised if symbolic dims are untracked.

---

Use this pattern for all future plugins and core primitives involving symbolic shapes.

