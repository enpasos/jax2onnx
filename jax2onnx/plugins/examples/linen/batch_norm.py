# jax2onnx/plugins/examples/linen/batch_norm.py

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


class SimpleBatchNorm(nn.Module):
    dtype: object | None = None
    param_dtype: object = jnp.float32
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        return nn.BatchNorm(
            use_running_average=True,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)


register_example(
    component="LinenBatchNorm",
    description="A simple Flax Linen BatchNorm layer.",
    source="",
    since="v0.11.0",
    context="examples.linen",
    children=["flax.linen.BatchNorm"],
    testcases=[
        {
            "testcase": "simple_linen_batch_norm",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=SimpleBatchNorm,
                input_shape=(1, 4, 4, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [(1, 4, 4, 3)],
        },
    ],
)
