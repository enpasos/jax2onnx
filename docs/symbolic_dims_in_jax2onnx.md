# Symbolic-Dimension Support in **jax2onnx**

This guide is for plugin authors who want to add or migrate primitive-handlers so that they **round-trip JAX symbolic shapes to ONNX symbolic dimensions**.

---
## 1 Why we need it
*   JAX ≥ 0.6 treats dimensions as symbols (e.g. `"B"`) that are only resolved at run-time.
*   ONNX supports the same idea through `dim_param`.
*   Without explicit care, a primitive’s `abstract_eval` can destroy the symbol and the exported model becomes fully-static.

> **Key idea** Let **JAX** do the algebra by calling `jax.eval_shape` on the *original* implementation; never re-implement shape maths by hand.

---
## 2 High-level flow
1. **User API**  
   ```python
   to_onnx(fn, input_shapes=[("B", 1, 8)])
   ```
2. **conversion_api** – strings like `"B"` are replaced by true `_DimExpr` objects and stored inside every `ShapeDtypeStruct`.
3. **Plugin → abstract_eval** – receives those symbolic objects in `aval.shape` and calls `jax.eval_shape` on the **un-patched** JAX op.
4. **var_to_symbol_map** – the builder tracks which `_DimExpr` maps to which original string and writes `dim_param="B"` into the ONNX graph.

---
## 3 Boiler-plate for a plugin (new pattern)

```python
from types import SimpleNamespace
import jax, jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

class <MyPrimitive>Plugin(PrimitiveLeafPlugin):
    _ORIG_CALL: callable | None = None      # filled once in patch

    # ------------------------------------------------------------
    # abstract_eval – delegate to original __call__ via eval_shape
    # ------------------------------------------------------------
    @staticmethod
    def abstract_eval(*in_avals: core.ShapedArray, **params):
        if <MyPrimitive>Plugin._ORIG_CALL is None:
            raise RuntimeError("original op not captured yet")

        # build ShapeDtypeStruct specs
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in in_avals]

        def _helper(*xs):
            """Call the pristine method on a dummy object."""
            dummy = SimpleNamespace(
                # every field the real method expects:
                kernel  = SimpleNamespace(value=xs[1]),
                bias    = SimpleNamespace(value=xs[2]) if len(xs) > 2 else None,
                strides = params["strides"],
                padding = params["padding"],
                # plus helpers sometimes referenced by nnx modules
                promote_dtype        = lambda args, dtype=None: args,
                conv_general_dilated = jax.lax.conv_general_dilated,
            )
            return <MyPrimitive>Plugin._ORIG_CALL(dummy, xs[0])

        out_spec = jax.eval_shape(_helper, *specs)
        out_spec = jax.tree_util.tree_leaves(out_spec)[0]       # scalar output
        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    # ------------------------------------------------------------
    # monkey-patch – capture original & inject primitive binding
    # ------------------------------------------------------------
    @staticmethod
    def _bind_primitive(x, kernel, bias, **attrs):
        return nnx.<my_primitive>_p.bind(x, kernel, bias, **attrs)

    @staticmethod
    def get_monkey_patch(orig_fn):          # <-- orig_fn provided by framework
        <MyPrimitive>Plugin._ORIG_CALL = orig_fn     # capture once

        def patched(self, x):
            # optional zero-bias convenience
            bias = self.bias.value if self.bias is not None else jnp.zeros(
                (self.kernel.value.shape[-1],), self.kernel.value.dtype
            )
            return <MyPrimitive>Plugin._bind_primitive(
                x, self.kernel.value, bias,
                strides=self.strides, padding=self.padding,
                dimension_numbers=getattr(self, "dimension_numbers", None),
            )

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.<MyPrimitiveClass>],
            "target_attribute": "__call__",
            "patch_function": <MyPrimitive>Plugin.get_monkey_patch,
        }
```

A real-world example is **`conv.py`** in the repo.

---
## 4 Migration checklist
| ✓ | Step |
|---|------|
| ☐ | `patch_info` returns a **factory** that receives `orig_fn`; store it on the class. |
| ☐ | `abstract_eval` uses only `jax.eval_shape` on that original fn. |
| ☐ | Build a `SimpleNamespace` carrying *all* attributes that the original method reads. |
| ☐ | Never import or call jax.numpy inside `abstract_eval` – let the real op do the work. |
| ☐ | Unit tests include at least one symbolic-batch shape such as `("B", 32, 32, 3)`. |

---
## 5 Common pitfalls
| Symptom | Cause | Remedy |
|---------|-------|--------|
| `UnexpectedTracerError` | Manual math on tracers | Delegate to `jax.eval_shape`. |
| `…got an unexpected keyword argument …` inside eval_shape | Your dummy instance misses an attribute the original method accesses | Add that field to the `SimpleNamespace`. |
| Extra `Transpose` nodes for constant kernels | You left the transpose in the graph | Pre-transpose the numpy constant *before* registering it as an initializer (see `conv.py`). |

---
## 6 Reference implementation
* **`jax2onnx/plugins/flax/nnx/conv.py`** – full pattern with constants vs runtime tensors, dummy instance, etc.
* **`jax2onnx/plugins/jax/numpy/concatenate.py`** – light-weight unary example.

---
## 7 Adding a new primitive

1. Copy the boiler-plate above.
2. Fill in `<MyPrimitive>` placeholders and ONNX emission logic in `to_onnx`.
3. Provide tests with both static and symbolic shapes.
4. Enjoy automatic support for dynamic & symbolic dimensions 👏.
