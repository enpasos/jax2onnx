import jax
from jax2onnx import to_onnx
import numpy as np, onnxruntime as ort, jax.numpy as jnp


@jax.jit
def f(x):
    return jnp.squeeze(x, axis=-2)


m = to_onnx(f, [("B", 2, 1)])
print([d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])
