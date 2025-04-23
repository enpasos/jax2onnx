import jax.numpy as jnp, onnx
from jax2onnx import to_onnx


def fn(x):
    def inner(y):  # lowered as ONNX Function
        return jnp.squeeze(y, (-1, -3))

    return inner(x)


model = to_onnx(fn, inputs=[(1, "B", 1)])

# --- graph level ---
assert model.graph.input[0].type.tensor_type.shape.dim[1].dim_param == "B"

# --- inside the function ---
f = model.functions[0]
assert f.input[0].type.tensor_type.shape.dim[1].dim_param == "B"
