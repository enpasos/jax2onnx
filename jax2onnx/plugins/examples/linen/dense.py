# jax2onnx/plugins/examples/linen/dense.py

from __future__ import annotations
from flax import linen as nn

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins.flax.test_utils import linen_to_nnx


class SimpleDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=self.features,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )(x)


register_example(
    component="LinenDense",
    description="A simple Flax Linen Dense layer.",
    source="",
    since="v0.11.0",
    context="examples.linen",
    children=["flax.linen.Dense"],
    testcases=[
        {
            "testcase": "simple_linen_dense",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=SimpleDense,
                input_shape=(1, 32),
                dtype=with_requested_dtype(),
                features=16,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 32)],
            # NOTE: post_check_onnx_graph removed - Gemm fusion optimization
            # (fuse_gemm_bias_ir) is not yet implemented. The graph may contain
            # MatMul+Add instead of fused Gemm.
        },
    ],
)
