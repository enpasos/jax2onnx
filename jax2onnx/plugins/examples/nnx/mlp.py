# jax2onnx/plugins/examples/nnx/mlp.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from flax import nnx

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.nnx.dropout import post_check_onnx_graph
from jax2onnx.plugins.flax.test_utils import expect_mlp_precision


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, deterministic=True, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, use_running_average=True, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        x = nnx.gelu(
            self.dropout(self.bn(self.linear1(x)), deterministic=deterministic)
        )
        return self.linear2(x)


def _mlp_graph_check(batch: str) -> Callable[[Any], bool]:
    common = {"Gemm": 2, "Dropout": 1, "Gelu": 1}
    return expect_mlp_precision(
        expect_graph(
            [
                (
                    f"Gemm:{batch}x20 -> BatchNormalization:{batch}x20 -> "
                    f"Dropout:{batch}x20 -> Gelu:{batch}x20 -> Gemm:{batch}x10",
                    {"counts": {**common, "BatchNormalization": 1}},
                )
            ],
            symbols={"B": None},
            must_absent=["Not"],
            no_unused_inputs=True,
        ),
        expect_graph(
            [
                (
                    f"Gemm:{batch}x20 -> Sub:{batch}x20 -> Mul:{batch}x20 -> "
                    f"Add:{batch}x20 -> Dropout:{batch}x20 -> Gelu:{batch}x20 -> "
                    f"Gemm:{batch}x10",
                    {
                        "counts": {
                            **common,
                            "BatchNormalization": 0,
                            "Sub": 1,
                            "Add": 2,
                            "Sqrt": 1,
                            "Div": 1,
                            "Mul": 2,
                        }
                    },
                ),
                # NNX promotes input, statistics and affine parameters together,
                # then forms this coefficient before scaling the centered input.
                f"Add:1x20 -> Sqrt:1x20 -> Div:1x20 -> Mul:1x20 -> Mul:{batch}x20",
            ],
            symbols={"B": None},
            must_absent=["Not"],
            no_unused_inputs=True,
        ),
    )


register_example(
    component="MLP",
    description="A simple Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="0.1.0",
    context="examples.nnx",
    children=["nnx.Linear", "nnx.Dropout", "nnx.BatchNorm", "nnx.gelu"],
    testcases=[
        {
            "testcase": "simple_mlp_static",
            "callable": construct_and_call(
                MLP,
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": _mlp_graph_check("7"),
        },
        {
            "testcase": "simple_mlp",
            "callable": construct_and_call(
                MLP,
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": _mlp_graph_check("B"),
        },
        {
            "testcase": "simple_mlp_with_call_params",
            "callable": construct_and_call(
                MLP,
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "input_params": {"deterministic": True},
            "post_check_onnx_graph": post_check_onnx_graph,
        },
    ],
)
