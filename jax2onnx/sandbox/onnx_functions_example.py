# file: jax2onnx/sandbox/onnx_functions_example.py

from jax2onnx.onnx_functions import enhanced_to_onnx, onnx_function
from flax import nnx
import os
import onnx


@onnx_function
class MLPBlock(nnx.Module):
    def __call__(self, x):
        return nnx.gelu(x)


@onnx_function
class TransformerBlock(nnx.Module):
    def __init__(self):
        self.mlp = MLPBlock()

    def __call__(self, x):
        return self.mlp(x) + x


top_model = TransformerBlock()
onnx_model = enhanced_to_onnx(top_model, [("B", 10, 256)])
output_path = "./docs/onnx/sandbox/transformer_block.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)
