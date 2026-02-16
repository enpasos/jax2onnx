# Layout Optimization (NCHW Support)

## Overview

By default, JAX models often use **NHWC** layout for image processing (e.g. `(Batch, Height, Width, Channels)`), whereas ONNX runtimes and tools typically prefer **NCHW** `(Batch, Channels, Height, Width)`.

`jax2onnx` automatically inserts transpose operations to handle layout mismatches when converting layers like Convolution. However, this can lead to redundant transposes at the model boundaries if your input data is already available in NCHW format.

To optimize this, you can use the `inputs_as_nchw` and `outputs_as_nchw` arguments in `to_onnx`.

## Usage

- **`inputs_as_nchw`**: A sequence of input indices (e.g. `[0]`) that you promise to provide in NCHW layout. `jax2onnx` will insert a transpose at the beginning of the graph to convert NCHW to NHWC for the internal JAX logic. This often cancels out with the initial transpose expected by Conv layers, removing it entirely.
- **`outputs_as_nchw`**: A sequence of output indices that you want to be returned in NCHW layout.
- **`input_names` / `output_names`**: Optional sequences for explicit boundary names (for example `["image"]` and `["prediction"]`) when exporting.

## Example

```python
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

def my_conv_model(x):
    # x is NHWC in JAX logic
    return jax.lax.conv_general_dilated(
        x, kernel, (1, 1), 'SAME', ('NHWC', 'HWIO', 'NHWC')
    )

# Convert with layout optimization
# We tell jax2onnx that input 0 will be provided as NCHW, 
# and we want output 0 to be NCHW.
onnx_model = to_onnx(
    my_conv_model,
    inputs=[jax.ShapeDtypeStruct((1, 32, 32, 3), jnp.float32)], # Shape is still specified as NHWC logic
    inputs_as_nchw=[0],
    outputs_as_nchw=[0],
    input_names=["image_nchw"],
    output_names=["features_nchw"],
)
```
