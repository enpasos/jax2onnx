# jax2onnx/plugins/examples/nnx/sequential.py

from __future__ import annotations

from flax import nnx
from typing import Final

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


_double_relu: Final = nnx.Sequential(
    nnx.relu,
    nnx.relu,
)

register_example(
    component="SequentialReLU",
    description="Two stateless nnx.relu activations chained via nnx.Sequential.",
    source="https://flax.readthedocs.io/en/latest/nnx/index.html",
    since="v0.7.1",
    context="examples.nnx",
    children=["nnx.Sequential", "nnx.relu"],
    testcases=[
        {
            "testcase": "sequential_double_relu",
            "callable": _double_relu,
            "input_shapes": [(5,)],
            "expected_output_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Max:5 -> Max:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)


class ComplexParentWithResidual(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.initial_op = nnx.Linear(in_features=16, out_features=16, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(in_features=16, out_features=32, rngs=rngs),
            lambda x: nnx.relu(x),
            nnx.Linear(in_features=32, out_features=16, rngs=rngs),
        )
        self.layernorm = nnx.LayerNorm(num_features=16, rngs=rngs)

    def __call__(self, x):
        x_residual = self.initial_op(x)
        ffn_output = self.ffn(x_residual)
        return self.layernorm(x_residual + ffn_output)


register_example(
    component="SequentialWithResidual",
    description="nnx.Sequential nested within a residual block to regress earlier bugs.",
    source="Internal bug report",
    since="v0.7.1",
    context="examples.nnx",
    children=["nnx.Sequential", "nnx.Linear", "nnx.relu", "nnx.LayerNorm"],
    testcases=[
        {
            "testcase": "sequential_nested_with_residual",
            "callable": construct_and_call(
                ComplexParentWithResidual,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 16)],
            "expected_output_shapes": [(1, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gemm:Bx16 -> Add:Bx16 -> LayerNormalization:Bx16"],
                symbols={"B": 1},
                no_unused_inputs=True,
            ),
        },
    ],
)
