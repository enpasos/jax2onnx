# file: jax2onnx/examples/mlp.py

import jax
from flax import nnx
from flax.nnx import BatchNorm, Dropout, Linear

from jax2onnx.plugin_system import register_example


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.dropout = Dropout(rate=0.1, deterministic=True, rngs=rngs)
        self.bn = BatchNorm(dmid, use_running_average=True, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True):
        x = nnx.gelu(
            self.dropout(self.bn(self.linear1(x)), deterministic=deterministic)
        )
        return self.linear2(x)


register_example(
    component="MLP",
    description="A simple Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.nnx",
    children=["nnx.Linear", "nnx.Dropout", "nnx.BatchNorm", "nnx.gelu"],
    testcases=[
        {
            "testcase": "simple_mlp",
            "callable": MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(17)),
            "input_shapes": [("B", 30)],
        },
        {
            "testcase": "simple_mlp_with_call_params",
            "callable": MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(17)),
            "input_shapes": [("B", 30)],
            "input_params": {
                "deterministic": True,
            },
        },
    ],
)
