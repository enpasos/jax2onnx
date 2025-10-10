# jax2onnx/plugins/examples/nnx/mlp.py

from __future__ import annotations
import jax
from flax import nnx

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.nnx.dropout import post_check_onnx_graph


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, deterministic=True, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, use_running_average=True, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

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
            "testcase": "simple_mlp_static",
            "callable": construct_and_call(
                MLP,
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": expect_graph(
                [
                    # edge-shape after each node (leaving that node)
                    "Gemm:7x20 -> BatchNormalization:7x20 -> Dropout:7x20 -> Gelu:7x20 -> Gemm:7x10",
                ],
                must_absent=["Not"],  # ensure no Not left anywhere
                no_unused_inputs=True,  # fail if 'deterministic' or other inputs dangle
                mode="all",
            ),
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
            "post_check_onnx_graph": expect_graph(
                [
                    # edge-shape after each node (leaving that node)
                    "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10",
                ],
                symbols={"B": None},  # allow dynamic batch sizes
                must_absent=["Not"],  # ensure no Not left anywhere
                no_unused_inputs=True,  # fail if 'deterministic' or other inputs dangle
                mode="all",
            ),
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
