# Enabling Flax Linen Support in jax2onnx

This document details the work done to enable support for [Flax Linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html) models in `jax2onnx`.

## 1. The Challenge

Initial attempts to convert Flax Linen models using `jax2onnx` failed with a `TypeError` related to JAX tracing:
```
TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(int32[]), 16)
```
This error occurred because `jax2onnx` activates a set of "plugins" that monkey-patch JAX primitives to capture the ONNX graph. One specific plugin, `JnpShapePlugin`, was monkey-patching `jnp.shape`. This interfered with Flax Linen's internal initialization tracing (specifically within `flax.linen.Dense`), where it relies on `jnp.shape` returning concrete values to determine kernel shapes. The monkey patch caused it to return a `JitTracer` instead, leading to the failure.

## 2. The Fix

The solution was to disable the monkey-patching of `jnp.shape` within the `JnpShapePlugin`.

**File:** `jax2onnx/plugins/jax/numpy/shape.py`

```python
class JnpShapePlugin(PrimitiveLeafPlugin):
    # ...
    def get_patch_params(self):
        # We generally DO NOT want to patch jnp.shape logic because it
        # breaks internal helper logic in libraries like Flax that rely
        # on shape inspection during init.
        # Returning an empty list disables the monkey patch.
        return [] 
```

By ensuring `jnp.shape` behaves natively during tracing, Flax Linen can correctly resolve shapes during its initialization phase, allowing the rest of the `jax2onnx` conversion process to proceed normally using JAXPR tracing.

## 4. Testing Infrastructure

To ensure ongoing support and prevent regressions, we integrated Linen examples into the `jax2onnx` test suite.

**Location:** `jax2onnx/plugins/examples/linen/`

We introduced a pattern for testing stateful Linen modules within the stateless `jax2onnx` test harness:

### The `ToNNX` Bridge Pattern
Since `jax2onnx` tests typically expect a simple stateless callable, we bridge Linen modules into NNX and run a one-time lazy init with a dummy input.

```python
import jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge

def linen_to_nnx(module_cls, input_shape=(1, 32), dtype=jnp.float32, rngs=None, **kwargs):
    module = module_cls(**kwargs)
    model = bridge.ToNNX(module, rngs=rngs)
    dummy_x = jnp.zeros(input_shape, dtype=dtype)
    if isinstance(rngs, nnx.Rngs):
        if "params" in rngs:
            rngs = rngs["params"].key.value
        elif "default" in rngs:
            rngs = rngs["default"].key.value
        else:
            raise ValueError("NNX RNGs must define a 'params' or 'default' stream.")
    if rngs is None:
        model.lazy_init(dummy_x)
        return model
    model.lazy_init(dummy_x, rngs=rngs)
    return lambda *args, **kwargs: model(*args, rngs=rngs, **kwargs)
```

### Registration
We use `register_example` to expose the test case to the test generator (`tests/t_generator.py`).

```python
register_example(
    component="LinenDense",
    context="examples.linen",
    testcases=[{
        "testcase": "simple_linen_dense",
        "callable": construct_and_call(
            linen_to_nnx,
            module_cls=SimpleDense,
            input_shape=(1, 32),
            dtype=with_requested_dtype(),
            rngs=with_rng_seed(0),
            features=16,
        ),
        "input_shapes": [(1, 32)],
    }]
)
```

## 5. Usage Example

Here is a minimal script to convert a Linen model to ONNX:

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax2onnx import to_onnx

class MyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        return nn.relu(x)

# 1. Initialize Model
model = MyModel()
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 32))
variables = model.init(key, dummy_input)

# 2. Define functional wrapper
def apply_fn(x):
    return model.apply(variables, x)

# 3. Convert to ONNX
onnx_model = to_onnx(
    apply_fn,
    inputs=[dummy_input], # or [jax.ShapeDtypeStruct((1,32), jnp.float32)]
    model_name="my_linen_model"
)

# 4. Save
with open("my_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```
