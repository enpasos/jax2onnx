# jax2onnx/plugins/examples/nnx/channel_attention_resblock.py

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)

filters: int = 32
kernel_size: tuple[int, int] = (3, 3)


class ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.conv2 = nnx.Conv(filters, filters // 8, kernel_size=(1, 1), rngs=rngs)
        self.conv3 = nnx.Conv(filters // 8, filters, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, input: jax.Array) -> jax.Array:
        x = nnx.silu(self.conv0(input))
        f = self.conv1(x)
        x = jnp.mean(f, axis=(1, 2), keepdims=True)
        x = nnx.silu(self.conv2(x))
        x = nnx.sigmoid(self.conv3(x))
        x = (x * f) + input
        return x


register_example(
    component="ResBlock",
    description=(
        "Residual block with squeeze-and-excite channel attention " "(from issue #168)."
    ),
    source="https://github.com/enpasos/jax2onnx/issues/168",
    since="0.12.0",
    context="examples.nnx",
    children=["nnx.Conv", "nnx.silu", "nnx.sigmoid", "jax.numpy.mean"],
    testcases=[
        {
            "testcase": "resblock_channel_attention_static",
            "callable": construct_and_call(
                ResBlock,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, filters)],
            "expected_output_shapes": [(1, 8, 8, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    ("ReduceMean", {"counts": {"ReduceMean": 1}}),
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "resblock_channel_attention_static_nchw",
            "callable": construct_and_call(
                ResBlock,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, filters)],
            "inputs_as_nchw": [0],
            "outputs_as_nchw": [0],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Conv:1x32x8x8 -> ReduceMean:1x32x1x1 -> Conv:1x4x1x1",
                        {"counts": {"ReduceMean": 1}},
                    ),
                    "Sigmoid:1x32x1x1 -> Mul:1x32x8x8 -> Add:1x32x8x8",
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "resblock_channel_attention_dynamic_hw",
            "callable": construct_and_call(
                ResBlock,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, "H", "W", filters)],
            "expected_output_shapes": [(1, "H", "W", filters)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "resblock_channel_attention_dynamic_hw_nchw",
            "callable": construct_and_call(
                ResBlock,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, "H", "W", filters)],
            "inputs_as_nchw": [0],
            "outputs_as_nchw": [0],
            "run_only_f32_variant": True,
        },
    ],
)
