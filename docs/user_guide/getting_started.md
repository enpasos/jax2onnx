# Getting Started

## Quickstart

Install and export your first model in minutes:

```bash
pip install jax2onnx
```

Convert your JAX callable to ONNX in just a few lines:

```python
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

# Export straight to disk without keeping the proto in memory
to_onnx(
    my_callable,
    [("B", 30)],
    return_mode="file",
    output_path="my_callable.onnx",
)
```
 
🔎 See it visualized:  [`my_callable.onnx`](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/my_callable.onnx)

## ONNX Functions — Minimal Example

ONNX functions help encapsulate reusable subgraphs. Simply use the `@onnx_function` decorator to make your callable an ONNX function.

```python
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
to_onnx(
    callable,
    [(100, 256)],
    return_mode="file",
    output_path="model_with_function.onnx",
)
```

🔎 See it visualized: [`model_with_function.onnx`](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/model_with_function.onnx)

## Troubleshooting

If conversion doesn't work out of the box, it could be due to:

- **Non-dynamic function references:**  
  JAXPR-based conversion requires function references to be resolved dynamically at call-time.  
  **Solution:** Wrap your function call inside a lambda to enforce dynamic resolution:
  ```python
  my_dynamic_callable_function = lambda x: original_function(x)
  ```

- **Unsupported primitives:**  
  The callable may use a primitive not yet or not fully supported by `jax2onnx`.  
  **Solution:** Write a [plugin](https://enpasos.github.io/jax2onnx/design#plugin-op-specific) to handle the unsupported function (this is straightforward!).

Looking for provenance details while debugging? Check out the new [Stacktrace Metadata guide](../readme/stacktrace_metadata.md).
