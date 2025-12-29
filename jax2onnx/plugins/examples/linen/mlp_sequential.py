# jax2onnx/plugins/examples/linen/mlp_sequential.py

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


class LinenMLPSequential(nn.Module):
    din: int
    dmid: int
    dout: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Dense(self.dmid, dtype=self.dtype, param_dtype=self.dtype),
                nn.BatchNorm(
                    use_running_average=True,
                    dtype=self.dtype,
                    param_dtype=self.dtype,
                ),
                nn.Dropout(rate=self.dropout_rate, deterministic=True),
                nn.gelu,
                nn.Dense(self.dout, dtype=self.dtype, param_dtype=self.dtype),
            ]
        )(x)


register_example(
    component="LinenMLPSequential",
    description="A Linen MLP built from flax.linen.Sequential.",
    source=(
        "https://flax-linen.readthedocs.io/en/latest/api_reference/"
        "flax.linen/layers.html#flax.linen.Sequential"
    ),
    since="v0.11.0",
    context="examples.linen",
    children=[
        "flax.linen.Sequential",
        "flax.linen.Dense",
        "flax.linen.BatchNorm",
        "flax.linen.Dropout",
        "flax.linen.gelu",
    ],
    testcases=[
        {
            "testcase": "simple_linen_mlp_sequential_static",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLPSequential,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": expect_graph(
                [
                    "Gemm:7x20 -> BatchNormalization:7x20 -> Dropout:7x20 -> "
                    "Gelu:7x20 -> Gemm:7x10",
                ],
                must_absent=["Not"],
                no_unused_inputs=True,
                mode="all",
            ),
        },
        {
            "testcase": "simple_linen_mlp_sequential",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLPSequential,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": expect_graph(
                [
                    "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> "
                    "Gelu:Bx20 -> Gemm:Bx10",
                ],
                symbols={"B": None},
                must_absent=["Not"],
                no_unused_inputs=True,
                mode="all",
            ),
        },
    ],
)
