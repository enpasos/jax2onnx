# jax2onnx 🌟

`jax2onnx` converts your JAX/Flax functions directly into the ONNX format.

![img.png](https://enpasos.github.io/jax2onnx/images/jax2onnx.png)

## ✨ Key Features

- **Simple API**  
  Convert any JAX/Flax model to ONNX using `to_onnx(...)`

- **Model structure preserved**  
  With `@onnx_function`, submodules appear as named functions in the ONNX graph (e.g. in Netron). Useful for readability and reuse.

- **Dynamic input support**  
  Use abstract dimensions like `'B'` or pass scalars as runtime inputs. Models stay flexible without retracing.

- **Plugin-based extensibility**  
  Add support for new primitives by writing small, local plugins.

- **Netron-friendly outputs**  
  All generated ONNX graphs include shape/type annotations and are structured for clear visualization.
---

## 🚀 Quickstart

Convert your JAX callable to ONNX in just a few lines:

```python
import onnx
from flax import nnx
from jax2onnx import to_onnx

# Define a simple MLP (from Flax docs)
class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs): 
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs) 
    def __call__(self, x): 
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)

# Instantiate model
my_callable = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))

# Convert to ONNX
onnx_model = to_onnx(my_callable, [("B", 30)])

# Save the model
onnx.save_model(onnx_model, "my_callable.onnx")
```
 
🔎 See it visualized:  [`my_callable.onnx`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/my_callable.onnx)

---

## 🧠 ONNX Functions — Minimal Example

ONNX functions help encapsulate reusable subgraphs. Simply use the `@onnx_function` decorator to make your callable an ONNX function.
Just an @onnx_function decorator to make your callable an ONNX function

```python
from onnx import save_model
from flax import nnx
from jax2onnx import onnx_function, to_onnx

# just an @onnx_function decorator to make your callable an ONNX function
@onnx_function
class MLPBlock(nnx.Module):
  def __init__(self, dim, *, rngs):
    self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
    self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
    self.batchnorm = nnx.BatchNorm(dim, rngs=rngs)
  def __call__(self, x):
    return nnx.gelu(self.linear2(self.batchnorm(nnx.gelu(self.linear1(x)))))

# Use it inside another module
class MyModel(nnx.Module):
  def __init__(self, dim, *, rngs):
    self.block1 = MLPBlock(dim, rngs=rngs)
    self.block2 = MLPBlock(dim, rngs=rngs)
  def __call__(self, x):
    return self.block2(self.block1(x))

callable = MyModel(256, rngs=nnx.Rngs(0))
model = to_onnx(callable, [(100, 256)])
save_model(model, "docs/onnx/model_with_function.onnx")
```

🔎 See it visualized: [`model_with_function.onnx`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/model_with_function.onnx)

---

## 📅 Roadmap and Releases


### **Planned Versions**
- **Ongoing**: Expanding JAX component coverage.
- **0.5.1**: Some more ONNX function support ... function reuse, make graph optimizer work within functions, allow user friendly var names  
- **0.5.0**: Improved dynamic batch dimension handling by leveraging [shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html) for more robust and flexible model export.

### **Current Productive Version**
- **0.4.4** *(PyPI)*: Added support for `lax.cos`, `lax.cosh`, `lax.sin`, `lax.sinh` and `lax.scatter` primitives. 

### **Past Versions**
- **0.4.3**: Fixed a bug in the validation of JAX callable outputs against their ONNX counterparts. This fix exposed previously hidden failing tests, which are now fixed.
- **0.4.2**: Cleanup and fixes to the basic ONNX function release.
- **0.4.1** *(ONNX functions)*: Introducing simple ONNX function support. Making use of ONNX functions is easy for the user: just a `@onnx_function` decorator making a callable an ONNX function.
  Each `@onnx_function` decorator creates a new ONNX function instance on the call graph.
- **0.3.2**: relaxed the minimum Python version to 3.10.
- **0.3.0**: Streamlined the plugin system with automatic registration and simplified integration of custom primitives.
- **0.2.0** *(First PyPI Release)*: Rebased the implementation on `jaxpr`, improving usability and adding low-level `lax` components.
- **0.1.0** *(Initial Approach, Not Released to PyPI)*: Produced ONNX exports for some `nnx` components and `nnx`-based examples, including a VisualTransformer.

---

## ❓ Troubleshooting

If conversion doesn't work out of the box, it could be due to:

- **Non-dynamic function references:**  
  JAXPR-based conversion requires function references to be resolved dynamically at call-time.  
  **Solution:** Wrap your function call inside a lambda to enforce dynamic resolution:
  ```python
  my_dynamic_callable_function = lambda x: original_function(x)
  ```

- **Unsupported primitives:**  
  The callable may use a primitive not yet or not fully supported by `jax2onnx`.  
  **Solution:** Write a [plugin](#how-to-contribute) to handle the unsupported function (this is straightforward!).

---

## 🧩 Supported JAX/ONNX Components


<!-- AUTOGENERATED TABLE START -->

| JAX Component | ONNX Components | Testcases | Since |
|:-------------|:---------------|:---------|:------|
| [jnp.add](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html) | [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | [`add`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/add.onnx) ✅ | v0.1.0 |
| [jnp.concat](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html) | [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html) | [`concat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/concat.onnx) ✅<br>[`concat_abstract_middle_dim_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/concat_abstract_middle_dim_dynamic.onnx) ✅<br>[`concat_abstract_middle_dim`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/concat_abstract_middle_dim.onnx) ✅ | v0.1.0 |
| [jnp.einsum](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html) | [Einsum](https://onnx.ai/onnx/operators/onnx__Einsum.html) | [`einsum`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum.onnx) ✅<br>[`einsum_preferred_element_type`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_preferred_element_type.onnx) ✅<br>[`einsum_matmul`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_matmul.onnx) ✅<br>[`einsum_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_dynamic.onnx) ✅<br>[`einsum_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic.onnx) ✅<br>[`einsum_dynamic_matmul_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul_dynamic.onnx) ✅<br>[`einsum_dynamic_matmul`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul.onnx) ✅<br>[`einsum_transpose`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_transpose.onnx) ✅<br>[`einsum_dynamic_transpose_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_transpose_dynamic.onnx) ✅<br>[`einsum_dynamic_transpose`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_transpose.onnx) ✅<br>[`einsum_dynamic_matmul2_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul2_dynamic.onnx) ✅<br>[`einsum_dynamic_matmul2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul2.onnx) ✅<br>[`einsum_dynamic_matmul3_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul3_dynamic.onnx) ✅<br>[`einsum_dynamic_matmul3`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_matmul3.onnx) ✅<br>[`einsum_outer_product`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_outer_product.onnx) ✅<br>[`einsum_trace`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_trace.onnx) ✅<br>[`einsum_sum`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_sum.onnx) ✅<br>[`einsum_broadcast`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_broadcast.onnx) ✅<br>[`einsum_reduce`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_reduce.onnx) ✅<br>[`einsum_permute`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_permute.onnx) ✅<br>[`einsum_dynamic_outer_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_outer_dynamic.onnx) ✅<br>[`einsum_dynamic_outer`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_outer.onnx) ✅<br>[`einsum_dynamic_reduce_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_reduce_dynamic.onnx) ✅<br>[`einsum_dynamic_reduce`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/einsum_dynamic_reduce.onnx) ✅ | v0.1.0 |
| [jnp.matmul](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html) | [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) | [`matmul_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_2d.onnx) ✅<br>[`matmul_1d_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_1d_2d.onnx) ✅<br>[`matmul_2d_1d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_2d_1d.onnx) ✅<br>[`matmul_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic_dynamic.onnx) ✅<br>[`matmul_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic.onnx) ✅<br>[`matmul_dynamic_a_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic_a_dynamic.onnx) ✅<br>[`matmul_dynamic_a`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic_a.onnx) ✅<br>[`matmul_dynamic_b_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic_b_dynamic.onnx) ✅<br>[`matmul_dynamic_b`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_dynamic_b.onnx) ✅<br>[`matmul_1d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_1d.onnx) ✅<br>[`matmul_3d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/matmul_3d.onnx) ✅ | v0.1.0 |
| [jnp.reshape](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html) | [Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | [`reshape_1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_1.onnx) ✅<br>[`reshape_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_2.onnx) ✅<br>[`reshape_3`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_3.onnx) ✅<br>[`reshape_4_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_4_dynamic.onnx) ✅<br>[`reshape_4`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_4.onnx) ✅<br>[`reshape_to_scalar`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_to_scalar.onnx) ✅<br>[`reshape_from_scalar`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/reshape_from_scalar.onnx) ✅ | v0.1.0 |
| [jnp.shape](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.shape.html) | [Shape](https://onnx.ai/onnx/operators/onnx__Shape.html) | [`shape_basic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/shape_basic.onnx) ✅<br>[`shape_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/shape_dynamic_dynamic.onnx) ✅<br>[`shape_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/shape_dynamic.onnx) ✅ | 0.4.0 |
| [jnp.squeeze](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html) | [Squeeze](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | [`squeeze_single_dim`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_single_dim.onnx) ✅<br>[`squeeze_multiple_dims`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_multiple_dims.onnx) ✅<br>[`squeeze_vit_output`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_vit_output.onnx) ✅<br>[`squeeze_dynamic_batch_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_dynamic_batch_dynamic.onnx) ✅<br>[`squeeze_dynamic_batch`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_dynamic_batch.onnx) ✅<br>[`squeeze_all_dims`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_all_dims.onnx) ✅<br>[`squeeze_negative_axis`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_negative_axis.onnx) ✅<br>[`squeeze_negative_axis_tuple`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_negative_axis_tuple.onnx) ✅<br>[`squeeze_dynamic_and_negative_axis_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_dynamic_and_negative_axis_dynamic.onnx) ✅<br>[`squeeze_dynamic_and_negative_axis`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/squeeze_dynamic_and_negative_axis.onnx) ✅ | v0.1.0 |
| [jnp.tile](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html) | [Tile](https://onnx.ai/onnx/operators/onnx__Tile.html) | [`tile_repeats_tensor`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_repeats_tensor.onnx) ✅<br>[`tile_a`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_a.onnx) ✅<br>[`tile_b`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_b.onnx) ✅<br>[`tile_c`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_c.onnx) ✅<br>[`tile_d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_d.onnx) ✅<br>[`tile_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_dynamic_dynamic.onnx) ✅<br>[`tile_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_dynamic.onnx) ✅<br>[`tile_pad`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/tile_pad.onnx) ✅ | v0.1.0 |
| [jnp.transpose](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html) | [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) | [`transpose_basic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_basic.onnx) ✅<br>[`transpose_reverse`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_reverse.onnx) ✅<br>[`transpose_4d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_4d.onnx) ✅<br>[`transpose_square_matrix`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_square_matrix.onnx) ✅<br>[`transpose_high_dim`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_high_dim.onnx) ✅<br>[`transpose_no_axes`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_no_axes.onnx) ✅<br>[`transpose_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_dynamic_dynamic.onnx) ✅<br>[`transpose_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/jnp/transpose_dynamic.onnx) ✅ | v0.1.0 |
| [lax.add](https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html) | [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | [`add`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/add.onnx) ✅ | v0.2.0 |
| [lax.argmax](https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html) | [ArgMax](https://onnx.ai/onnx/operators/onnx__ArgMax.html) | [`argmax_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/argmax_test1.onnx) ✅<br>[`argmax_test2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/argmax_test2.onnx) ✅ | v0.2.0 |
| [lax.argmin](https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html) | [ArgMin](https://onnx.ai/onnx/operators/onnx__ArgMin.html) | [`argmin_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/argmin_test1.onnx) ✅<br>[`argmin_test2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/argmin_test2.onnx) ✅ | v0.2.0 |
| [lax.broadcast_in_dim](https://jax.readthedocs.io/en/latest/jax-primitives.html) | [Expand](https://onnx.ai/onnx/operators/onnx__Expand.html)<br>[Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | [`broadcast_in_dim`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/broadcast_in_dim.onnx) ✅<br>[`broadcast_in_dim_2d_to_3d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/broadcast_in_dim_2d_to_3d.onnx) ✅<br>[`broadcast_in_dim_scalar`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/broadcast_in_dim_scalar.onnx) ✅ | v0.2.0 |
| [lax.concatenate](https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html) | [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html) | [`concatenate`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/concatenate.onnx) ✅<br>[`concatenate_axis1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/concatenate_axis1.onnx) ✅<br>[`concatenate_dynamic_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/concatenate_dynamic_dynamic.onnx) ✅<br>[`concatenate_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/concatenate_dynamic.onnx) ✅<br>[`concatenate_3d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/concatenate_3d.onnx) ✅ | v0.2.0 |
| [lax.conv](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv.html) | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | [`conv`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/conv.onnx) ✅<br>[`conv2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/conv2.onnx) ✅ | v0.2.0 |
| [lax.convert_element_type](https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html) | [Cast](https://onnx.ai/onnx/operators/onnx__Cast.html) | [`convert_element_type`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/convert_element_type.onnx) ✅ | v0.2.0 |
| [lax.cos](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cos.html) | [Cos](https://onnx.ai/onnx/operators/onnx__Cos.html) | [`cos`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/cos.onnx) ✅ | v0.4.4 |
| [lax.cosh](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cosh.html) | [Cosh](https://onnx.ai/onnx/operators/onnx__Cosh.html) | [`cosh`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/cosh.onnx) ✅ | v0.4.4 |
| [lax.device_put](https://jax.readthedocs.io/en/latest/jax.device_put.html) | [Identity](https://onnx.ai/onnx/operators/onnx__Identity.html) | [`device_put_array`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/device_put_array.onnx) ✅<br>[`device_put_scalar`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/device_put_scalar.onnx) ✅ | v0.4.0 |
| [lax.div](https://docs.jax.dev/en/latest/_autosummary/jax.lax.div.html) | [Div](https://onnx.ai/onnx/operators/onnx__Div.html) | [`div`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/div.onnx) ✅ | v0.2.0 |
| [lax.dot_general](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html) | [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) | [`dot_general`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/dot_general.onnx) ✅ | v0.2.0 |
| [lax.dynamic_slice](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html) | [Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) | [`dynamic_slice_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/dynamic_slice_test1.onnx) ✅<br>[`dynamic_slice_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/dynamic_slice_2d.onnx) ✅<br>[`dynamic_slice_3d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/dynamic_slice_3d.onnx) ✅ | v0.1.0 |
| [lax.eq](https://docs.jax.dev/en/latest/_autosummary/jax.lax.eq.html) | [Equal](https://onnx.ai/onnx/operators/onnx__Equal.html) | [`eq`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/eq.onnx) ✅ | v0.2.0 |
| [lax.exp](https://docs.jax.dev/en/latest/_autosummary/jax.lax.exp.html) | [Exp](https://onnx.ai/onnx/operators/onnx__Exp.html) | [`exp`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/exp.onnx) ✅ | v0.2.0 |
| [lax.gather](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html) | [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html) | [`gather`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/gather.onnx) ✅ | v0.2.0 |
| [lax.gt](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gt.html) | [Greater](https://onnx.ai/onnx/operators/onnx__Greater.html) | [`gt`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/gt.onnx) ✅ | v0.2.0 |
| [lax.integer_pow](https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html) | [Pow](https://onnx.ai/onnx/operators/onnx__Pow.html) | [`integer_pow`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/integer_pow.onnx) ✅ | v0.2.0 |
| [lax.log](https://docs.jax.dev/en/latest/_autosummary/jax.lax.log.html) | [Log](https://onnx.ai/onnx/operators/onnx__Log.html) | [`log`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/log.onnx) ✅ | v0.2.0 |
| [lax.lt](https://docs.jax.dev/en/latest/_autosummary/jax.lax.lt.html) | [Less](https://onnx.ai/onnx/operators/onnx__Less.html) | [`lt`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/lt.onnx) ✅ | v0.2.0 |
| [lax.max](https://docs.jax.dev/en/latest/_autosummary/jax.lax.max.html) | [Max](https://onnx.ai/onnx/operators/onnx__Max.html) | [`max`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/max.onnx) ✅ | v0.2.0 |
| [lax.min](https://docs.jax.dev/en/latest/_autosummary/jax.lax.min.html) | [Min](https://onnx.ai/onnx/operators/onnx__Min.html) | [`min_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/min_test1.onnx) ✅ | v0.1.0 |
| [lax.mul](https://docs.jax.dev/en/latest/_autosummary/jax.lax.mul.html) | [Mul](https://onnx.ai/onnx/operators/onnx__Mul.html) | [`mul_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/mul_test1.onnx) ✅<br>[`mul_test2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/mul_test2.onnx) ✅ | v0.1.0 |
| [lax.ne](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ne.html) | [Equal](https://onnx.ai/onnx/operators/onnx__Equal.html)<br>[Not](https://onnx.ai/onnx/operators/onnx__Not.html) | [`ne`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/ne.onnx) ✅ | v0.2.0 |
| [lax.neg](https://docs.jax.dev/en/latest/_autosummary/jax.lax.neg.html) | [Neg](https://onnx.ai/onnx/operators/onnx__Neg.html) | [`neg`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/neg.onnx) ✅ | v0.2.0 |
| [lax.reduce_max](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html) | [ReduceMax](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) | [`reduce_max`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/reduce_max.onnx) ✅ | v0.2.0 |
| [lax.reduce_min](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html) | [ReduceMin](https://onnx.ai/onnx/operators/onnx__ReduceMin.html) | [`reduce_min`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/reduce_min.onnx) ✅ | v0.2.0 |
| [lax.reduce_sum](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html) | [ReduceSum](https://onnx.ai/onnx/operators/onnx__ReduceSum.html) | [`reduce_sum`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/reduce_sum.onnx) ✅ | v0.2.0 |
| [lax.reshape](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html) | [Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | [`reshape`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/reshape.onnx) ✅ | v0.2.0 |
| [lax.scatter](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html) | [ScatterElements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) | [`scatter_set_axis0`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/scatter_set_axis0.onnx) ✅<br>[`scatter_set_middle`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/scatter_set_middle.onnx) ✅ | v0.4.4 |
| [lax.sin](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sin.html) | [Sin](https://onnx.ai/onnx/operators/onnx__Sin.html) | [`sin`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sin.onnx) ✅ | v0.4.4 |
| [lax.sinh](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sinh.html) | [Sinh](https://onnx.ai/onnx/operators/onnx__Sinh.html) | [`sinh`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sinh.onnx) ✅ | v0.4.4 |
| [lax.slice](https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html) | [Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) | [`slice_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/slice_test1.onnx) ✅ | v0.1.0 |
| [lax.sort](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html) | [TopK](https://onnx.ai/onnx/operators/onnx__TopK.html) | [`sort_1d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sort_1d.onnx) ✅<br>[`sort_1d_empty`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sort_1d_empty.onnx) ✅<br>[`sort_1d_single`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sort_1d_single.onnx) ✅<br>[`sort_1d_larger`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sort_1d_larger.onnx) ✅<br>[`sort_1d_specific_values`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sort_1d_specific_values.onnx) ✅ | v0.2.0 |
| [lax.sqrt](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sqrt.html) | [Sqrt](https://onnx.ai/onnx/operators/onnx__Sqrt.html) | [`sqrt`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sqrt.onnx) ✅ | v0.2.0 |
| [lax.square](https://docs.jax.dev/en/latest/_autosummary/jax.lax.square.html) | [Mul](https://onnx.ai/onnx/operators/onnx__Mul.html) | [`square`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/square.onnx) ✅ | v0.2.0 |
| [lax.squeeze](https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html) | [Squeeze](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | [`squeeze`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/squeeze.onnx) ✅ | v0.2.0 |
| [lax.stop_gradient](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html) | [Identity](https://onnx.ai/onnx/operators/onnx__Identity.html) | [`stop_gradient`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/stop_gradient.onnx) ✅ | v0.2.0 |
| [lax.sub](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sub.html) | [Sub](https://onnx.ai/onnx/operators/onnx__Sub.html) | [`sub_test1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sub_test1.onnx) ✅<br>[`sub_test2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/sub_test2.onnx) ✅ | v0.1.0 |
| [lax.tanh](https://docs.jax.dev/en/latest/_autosummary/jax.lax.tanh.html) | [Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html) | [`tanh`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/tanh.onnx) ✅ | v0.2.0 |
| [lax.transpose](https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html) | [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) | [`transpose_basic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/lax/transpose_basic.onnx) ✅ | v0.2.0 |
| [nn.softmax](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html) | [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html) | [`softmax`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nn/softmax.onnx) ✅<br>[`softmax_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nn/softmax_2d.onnx) ✅<br>[`softmax_3d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nn/softmax_3d.onnx) ✅ | v0.1.0 |
| [nnx.avg_pool](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool) | [AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html)<br>[Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) | [`avg_pool`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool.onnx) ✅<br>[`avg_pool_same_padding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_same_padding.onnx) ✅<br>[`avg_pool_default_padding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_default_padding.onnx) ✅<br>[`avg_pool_stride1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_stride1.onnx) ✅<br>[`avg_pool_single_batch`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_single_batch.onnx) ✅<br>[`avg_pool_dynamic_batch_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_dynamic_batch_dynamic.onnx) ✅<br>[`avg_pool_dynamic_batch`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_dynamic_batch.onnx) ✅<br>[`avg_pool_stride_none`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_stride_none.onnx) ✅<br>[`avg_pool_count_include_pad_false`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/avg_pool_count_include_pad_false.onnx) ✅ | v0.1.0 |
| [nnx.batch_norm](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) | [BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | [`batch_norm_simple`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_simple.onnx) ✅<br>[`batch_norm_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_2d.onnx) ✅<br>[`batch_norm_2d_use_bias_false`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_2d_use_bias_false.onnx) ✅<br>[`batch_norm_2d_use_scale_false`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_2d_use_scale_false.onnx) ✅<br>[`batch_norm_4d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_4d.onnx) ✅<br>[`batch_norm_4d_use_bias_false`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_4d_use_bias_false.onnx) ✅<br>[`batch_norm_4d_use_scale_false`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_4d_use_scale_false.onnx) ✅<br>[`batch_norm_minimal`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/batch_norm_minimal.onnx) ✅ | v0.1.0 |
| [nnx.conv](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv) | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)<br>[Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) | [`conv_basic_bias_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_basic_bias_dynamic.onnx) ✅<br>[`conv_basic_bias`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_basic_bias.onnx) ✅<br>[`conv_basic_bias_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_basic_bias_2.onnx) ✅<br>[`conv_basic_bias_3`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_basic_bias_3.onnx) ✅<br>[`conv_stride2_bias`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_stride2_bias.onnx) ✅<br>[`conv_no_bias_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_no_bias_dynamic.onnx) ✅<br>[`conv_no_bias`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_no_bias.onnx) ✅<br>[`conv_valid_padding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_valid_padding.onnx) ✅<br>[`conv_stride1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_stride1.onnx) ✅<br>[`conv_stride2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_stride2.onnx) ✅<br>[`conv_different_kernel`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_different_kernel.onnx) ✅<br>[`conv_float64`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_float64.onnx) ✅<br>[`conv_single_batch`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_single_batch.onnx) ✅<br>[`conv_large_batch`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/conv_large_batch.onnx) ✅ | v0.1.0 |
| [nnx.dot_product_attention](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.dot_product_attention) | [Cast](https://onnx.ai/onnx/operators/onnx__Cast.html)<br>[Div](https://onnx.ai/onnx/operators/onnx__Div.html)<br>[Einsum](https://onnx.ai/onnx/operators/onnx__Einsum.html)<br>[Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)<br>[Shape](https://onnx.ai/onnx/operators/onnx__Shape.html)<br>[Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html)<br>[Sqrt](https://onnx.ai/onnx/operators/onnx__Sqrt.html) | [`dpa_basic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_basic.onnx) ✅<br>[`dpa_diff_heads_embed`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_diff_heads_embed.onnx) ✅<br>[`dpa_batch4_seq16`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_batch4_seq16.onnx) ✅<br>[`dpa_float64`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_float64.onnx) ✅<br>[`dpa_heads1_embed4`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_heads1_embed4.onnx) ✅<br>[`dpa_heads8_embed8`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_heads8_embed8.onnx) ✅<br>[`dpa_batch1_seq2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_batch1_seq2.onnx) ✅<br>[`dpa_batch8_seq4`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_batch8_seq4.onnx) ✅<br>[`dpa_axis1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dpa_axis1.onnx) ✅ | v0.1.0 |
| [nnx.dropout](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout) | [Dropout](https://onnx.ai/onnx/operators/onnx__Dropout.html) | [`dropout_init_params`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dropout_init_params.onnx) ✅<br>[`dropout_call_params`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/dropout_call_params.onnx) ✅ | v0.1.0 |
| [nnx.einsum](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/array.html#flax.nnx.Einsum) | [Add](https://onnx.ai/onnx/operators/onnx__Add.html)<br>[Einsum](https://onnx.ai/onnx/operators/onnx__Einsum.html) | [`einsum_module_with_bias`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/einsum_module_with_bias.onnx) ✅<br>[`einsum_module_no_bias`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/einsum_module_no_bias.onnx) ✅ | v0.4.2 |
| [nnx.elu](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html) | [Elu](https://onnx.ai/onnx/operators/onnx__Elu.html) | [`elu`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/elu.onnx) ✅ | v0.1.0 |
| [nnx.gelu](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html) | [Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html) | [`gelu`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/gelu.onnx) ✅<br>[`gelu_1`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/gelu_1.onnx) ✅<br>[`gelu_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/gelu_2.onnx) ✅<br>[`gelu_3`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/gelu_3.onnx) ✅ | v0.1.0 |
| [nnx.group_norm](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm) | [GroupNormalization](https://example.com/onnx_GroupNormalization_doc) | [`group_norm`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/group_norm.onnx) ✅<br>[`group_norm_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/group_norm_2.onnx) ✅ | v0.3.0 |
| [nnx.layer_norm](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm) | [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) | [`layer_norm`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/layer_norm.onnx) ✅<br>[`layer_norm_multiaxis`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/layer_norm_multiaxis.onnx) ✅ | v0.1.0 |
| [nnx.leaky_relu](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.leaky_relu.html) | [LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html) | [`leaky_relu`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/leaky_relu.onnx) ✅ | v0.1.0 |
| [nnx.linear](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html) | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html)<br>[Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | [`linear_2d`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_2d.onnx) ✅<br>[`linear_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_dynamic.onnx) ✅<br>[`linear`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear.onnx) ✅ | v0.1.0 |
| [nnx.linear_general](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral) | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html)<br>[Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | [`linear_general_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_general_dynamic.onnx) ✅<br>[`linear_general`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_general.onnx) ✅<br>[`linear_general_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_general_2.onnx) ✅<br>[`linear_general_3`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_general_3.onnx) ✅<br>[`linear_general_4`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/linear_general_4.onnx) ✅ | v0.1.0 |
| [nnx.log_softmax](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_softmax.html) | [LogSoftmax](https://onnx.ai/onnx/operators/onnx__LogSoftmax.html) | [`log_softmax`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/log_softmax.onnx) ✅ | v0.1.0 |
| [nnx.max_pool](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool) | [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html)<br>[Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) | [`max_pool`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/max_pool.onnx) ✅<br>[`max_pool_same_padding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/max_pool_same_padding.onnx) ✅ | v0.1.0 |
| [nnx.relu](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html) | [Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) | [`relu`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/relu.onnx) ✅<br>[`relu_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/relu_2.onnx) ✅ | v0.1.0 |
| [nnx.rms_norm](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.RMSNorm) | [RMSNormalization](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html) | [`rms_norm`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/rms_norm.onnx) ✅<br>[`rms_norm_2`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/rms_norm_2.onnx) ✅ | v0.3.0 |
| [nnx.sigmoid](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid) | [Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html) | [`sigmoid`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/sigmoid.onnx) ✅ | v0.1.0 |
| [nnx.softmax](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html) | [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html) | [`softmax`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/softmax.onnx) ✅ | v0.1.0 |
| [nnx.softplus](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html) | [Softplus](https://onnx.ai/onnx/operators/onnx__Softplus.html) | [`softplus`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/softplus.onnx) ✅ | v0.1.0 |
| [nnx.tanh](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh) | [Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html) | [`tanh`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/primitives/nnx/tanh.onnx) ✅ | v0.1.0 |

<!-- AUTOGENERATED TABLE END -->

**Legend:**  
✅ = Passed  
❌ = Failed  
➖ = No testcase yet

---

## 🎯 Examples

<!-- AUTOGENERATED EXAMPLES TABLE START -->

| Component | Description | Testcases | Since |
|:----------|:------------|:----------|:------|
| AutoEncoder | A simple autoencoder example. | [`simple_autoencoder`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_autoencoder.onnx) ✅ | v0.2.0 |
| CNN | A simple convolutional neural network (CNN). | [`simple_cnn`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_cnn.onnx) ✅ | v0.1.0 |
| ClassificationHead | Classification head for Vision Transformer | [`classification_head`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/classification_head.onnx) ✅ | v0.4.0 |
| ClassificationHeadFlatten | Classification head for Vision Transformer | [`classification_head_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/classification_head_flat.onnx) ✅ | v0.4.0 |
| ConcatClsToken | Concatenate CLS token to the input embedding | [`concat_cls_token`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/concat_cls_token.onnx) ✅ | v0.4.0 |
| ConcatClsTokenFlatten | Concatenate CLS token to the input embedding | [`concat_cls_token_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/concat_cls_token_flat.onnx) ✅ | v0.4.0 |
| ConvEmbedding | Convolutional Token Embedding for MNIST with hierarchical downsampling. | [`mnist_conv_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/mnist_conv_embedding.onnx) ✅ | v0.1.0 |
| ConvEmbeddingFlatten | Convolutional Token Embedding for MNIST with hierarchical downsampling. | [`mnist_conv_embedding_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/mnist_conv_embedding_flat.onnx) ✅ | v0.1.0 |
| FeedForward | MLP in Transformer | [`feed_forward`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/feed_forward.onnx) ✅ | v0.1.0 |
| FeedForwardFlatten | MLP in Transformer | [`feed_forward_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/feed_forward_flat.onnx) ✅ | v0.1.0 |
| MLP | A simple Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation. | [`simple_mlp_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_mlp_dynamic.onnx) ✅<br>[`simple_mlp`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_mlp.onnx) ✅<br>[`simple_mlp_with_call_params_dynamic`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_mlp_with_call_params_dynamic.onnx) ✅<br>[`simple_mlp_with_call_params`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/simple_mlp_with_call_params.onnx) ✅ | v0.1.0 |
| MultiHeadAttention | This is a multi-head attention module implemented by Flax/nnx that has no ONNX correspondent on the same granularity. | [`multihead_attention`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/multihead_attention.onnx) ✅ | v0.2.0 |
| PatchEmbedding | Cutting the image into patches and linearly embedding them. | [`patch_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/patch_embedding.onnx) ✅ | v0.1.0 |
| PatchEmbeddingFlatten | Cutting the image into patches and linearly embedding them. | [`patch_embedding_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/patch_embedding_flat.onnx) ✅ | v0.1.0 |
| PositionalEmbedding | Add positional embedding to the input embedding | [`positional_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/positional_embedding.onnx) ✅ | v0.4.0 |
| PositionalEmbeddingFlatten | Add positional embedding to the input embedding | [`positional_embedding_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/positional_embedding_flat.onnx) ✅ | v0.4.0 |
| TransformerBlock | Transformer from 'Attention Is All You Need.' | [`transformer_block`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/transformer_block.onnx) ✅ | v0.1.0 |
| TransformerBlockFlatten | Transformer from 'Attention Is All You Need.' | [`transformer_block_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/transformer_block_flat.onnx) ✅ | v0.1.0 |
| TransformerStack | Stack of Transformer blocks | [`transformer_stack`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/transformer_stack.onnx) ✅ | v0.1.0 |
| TransformerStackFlatten | Stack of Transformer blocks | [`transformer_stack_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/transformer_stack_flat.onnx) ✅ | v0.1.0 |
| VisionTransformer | A Vision Transformer (ViT) model for MNIST with configurable embedding type. | [`vit_conv_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/vit_conv_embedding.onnx) ✅<br>[`vit_patch_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/vit_patch_embedding.onnx) ✅ | v0.2.0 |
| VisionTransformerFlatten | A Vision Transformer (ViT) model for MNIST with configurable embedding type. | [`vit_conv_embedding_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/vit_conv_embedding_flat.onnx) ✅<br>[`vit_patch_embedding_flat`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/nnx/vit_patch_embedding_flat.onnx) ✅ | v0.2.0 |
| onnx_functions_000 | one function on an outer layer. | [`000_one_function_on_outer_layer`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/000_one_function_on_outer_layer.onnx) ✅ | v0.4.0 |
| onnx_functions_001 | one function on an inner layer. | [`001_one_function_inner`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/001_one_function_inner.onnx) ✅ | v0.4.0 |
| onnx_functions_002 | two nested functions. | [`002_two_nested_functions`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/002_two_nested_functions.onnx) ✅ | v0.4.0 |
| onnx_functions_003 | two nested functions. | [`003_two_simple_nested_functions`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/003_two_simple_nested_functions.onnx) ✅ | v0.4.0 |
| onnx_functions_004 | nested function plus component | [`004_nested_function_plus_component`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/004_nested_function_plus_component.onnx) ✅ | v0.4.0 |
| onnx_functions_005 | nested function plus more components | [`005_nested_function_plus_component`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/005_nested_function_plus_component.onnx) ✅ | v0.4.0 |
| onnx_functions_006 | one function on an outer layer. | [`006_one_function_outer`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/006_one_function_outer.onnx) ✅ | v0.4.0 |
| onnx_functions_008 | transformer block with nested mlp block no call parameter | [`008_transformer_block`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/008_transformer_block.onnx) ✅ | v0.4.0 |
| onnx_functions_009 | transformer block using decorator on class and function | [`009_transformer_block`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/009_transformer_block.onnx) ✅ | v0.4.0 |
| onnx_functions_010 | transformer stack | [`010_transformer_stack`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/010_transformer_stack.onnx) ✅ | v0.4.0 |
| onnx_functions_012 | Vision Transformer (ViT) | [`012_vit_conv_embedding`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/012_vit_conv_embedding.onnx) ✅ | v0.4.0 |
| onnx_functions_013 | Vision Transformer (ViT) | [`013_vit_conv_embedding_with_call_params`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/013_vit_conv_embedding_with_call_params.onnx) ✅<br>[`013_vit_conv_embedding_with_internal_call_params`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/013_vit_conv_embedding_with_internal_call_params.onnx) ✅ | v0.4.0 |
| onnx_functions_014 | one function on an outer layer. | [`014_one_function_with_input_param_with_default_value`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/014_one_function_with_input_param_with_default_value.onnx) ✅<br>[`014_one_function_without_input_param_with_default_value`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/014_one_function_without_input_param_with_default_value.onnx) ✅ | v0.4.0 |
| onnx_functions_015 | one function on an outer layer. | [`015_one_function_with_input_param_without_default_value`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/015_one_function_with_input_param_without_default_value.onnx) ✅ | v0.4.0 |
| onnx_functions_016 | nested function plus more components | [`016_internal_function_with_input_param_with_default_value`](https://netron.app/?url=https://enpasos.github.io/jax2onnx/onnx/examples/onnx_functions/016_internal_function_with_input_param_with_default_value.onnx) ✅ | v0.4.0 |

<!-- AUTOGENERATED EXAMPLES TABLE END -->

---

## 📌 Dependencies

**Versions of Major Dependencies:**

| Library       | Versions |  
|:--------------|:---------| 
| `JAX`         | 0.6.0    | 
| `Flax`        | 0.10.6   | 
| `onnx`        | 1.17.0   |  
| `onnxruntime` | 1.21.1   |  

*Note: For more details, check `pyproject.toml`.*

---

## ⚠️ Limitations

- Currently not all JAX/Flax components are supported (you can easily help expand this coverage!).
- Function references need dynamic resolution at call-time.
- ONNX graph composition is done in-memory before saving to disk, potentially causing memory issues with very large models.

---

## 🤝 How to Contribute

We warmly welcome contributions!

**How you can help:**

- **Add a plugin:** Extend `jax2onnx` by writing a simple Python file in [`jax2onnx/plugins`](./jax2onnx/plugins). 
a custom primitive or an example.
- **Bug fixes & improvements:** PRs and issues are always welcome.

---

## 💾 Installation

Install from PyPI:

```bash
pip install jax2onnx  
```

Or get the latest development version from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ jax2onnx
```

---

## 📜 License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE) for details.

---

## 🌟 Special Thanks

Special thanks for plugin contributions to: @burakksen

Special thanks to @lutzroeder for making shapes internal to ONNX function visible in his great Netron viewer.

- [ONNX: Function value_info support #1447](https://github.com/lutzroeder/netron/issues/1447)


Special thanks to the community members involved in:

- [Flax Feature Request #4430](https://github.com/google/flax/issues/4430)
- [JAX Feature Request #26430](https://github.com/jax-ml/jax/issues/26430)

A huge thanks especially to [@limarta](https://github.com/limarta), whose elegant [jaxpr-to-ONNX demonstration](https://gist.github.com/limarta/855a88cc1c0163487a9dc369891147ab) significantly inspired this project.

---

**Happy converting! 🎉**


