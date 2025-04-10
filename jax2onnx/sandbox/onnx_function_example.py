# file: jax2onnx/sandbox/onnx_function_example.py


import os

import onnx
from flax import nnx

from jax2onnx import to_onnx
from jax2onnx.plugin_system import onnx_function


@onnx_function
class MLPBlock(nnx.Module):
    """
    Represents a simple MLP block with a GELU activation function.
    This block will be converted to an ONNX function.
    """

    def __call__(self, x):
        # Apply the GELU activation function to the input.
        return nnx.gelu(x)


@onnx_function
class TransformerBlock(nnx.Module):
    """
    Represents a transformer block. Additional details can be added here.
    This block will also be converted to an ONNX function.
    """

    def __init__(self):
        # Initialization logic for the transformer block.
        pass

    def __call__(self, x):
        return self.mlp(x) + x


top_model = TransformerBlock()
onnx_model = to_onnx(top_model, [("B", 10, 256)])
output_path = "./docs/onnx/sandbox/one_function.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)
