# jax2onnx/sandbox/issue139.py

import jax
from jax2onnx import to_onnx
import onnxruntime as ort


def function(x):
    a = jax.nn.silu(x)
    b = jax.nn.silu(a)
    return b


to_onnx(
    function,
    [(1,)],
    return_mode="file",
    output_path="test.onnx",
)
session = ort.InferenceSession("test.onnx")
