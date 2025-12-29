# jax2onnx/plugins/examples/linen/mlp.py

from __future__ import annotations

from flax import linen as nn

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.nnx.dropout import post_check_onnx_graph
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class LinenMLP(nn.Module):
    din: int
    dmid: int
    dout: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        x = nn.Dense(self.dmid)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.gelu(x)
        return nn.Dense(self.dout)(x)


register_example(
    component="LinenMLP",
    description="A simple Linen MLP with BatchNorm, Dropout, and GELU activation.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.11.0",
    context="examples.linen",
    children=[
        "flax.linen.Dense",
        "flax.linen.BatchNorm",
        "flax.linen.Dropout",
        "flax.linen.gelu",
    ],
    testcases=[
        {
            "testcase": "simple_linen_mlp_static",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": expect_graph(
                [
                    "Gemm:7x20 -> BatchNormalization:7x20 -> Dropout:7x20 -> Gelu:7x20 -> Gemm:7x10",
                ],
                must_absent=["Not"],
                no_unused_inputs=True,
                mode="all",
            ),
        },
        {
            "testcase": "simple_linen_mlp",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": expect_graph(
                [
                    "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10",
                ],
                symbols={"B": None},
                must_absent=["Not"],
                no_unused_inputs=True,
                mode="all",
            ),
        },
        {
            "testcase": "simple_linen_mlp_with_call_params",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
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
