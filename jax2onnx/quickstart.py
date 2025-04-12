# file: quickstart.py

import onnx
from flax import nnx

from jax2onnx import to_onnx


# Example: A minimal MLP (from Flax documentation)
class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs):
        # Initialize the layers of the MLP: two linear layers, dropout, and batch normalization.
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        # Apply the layers in sequence with activation and dropout.
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


# Instantiate model
my_callable = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))


onnx_model = to_onnx(my_callable, [("B", 30)])

onnx.save_model(onnx_model, "docs/onnx/my_callable.onnx")
