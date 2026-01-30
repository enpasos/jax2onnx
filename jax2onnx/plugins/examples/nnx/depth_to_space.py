# jax2onnx/plugins/examples/nnx/depth_to_space.py

from __future__ import annotations

import jax
import dm_pix as pix
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class DepthToSpace(nnx.Module):
    def __init__(self, block_size: int = 2, *, rngs: nnx.Rngs):
        self.block_size = block_size

    def __call__(self, x: jax.Array) -> jax.Array:
        return pix.depth_to_space(x, self.block_size)


class ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv1 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = nnx.silu(self.conv0(x))
        y = self.conv1(y)
        return y + x


class DepthToSpaceResNet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.res_blocks = nnx.List(ResBlock(rngs=rngs) for _ in range(4))
        self.conv1 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 4, kernel_size=(3, 3), rngs=rngs)
        self.depth_to_space = DepthToSpace(block_size=2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        conv0 = self.conv0(x)
        y = conv0
        for block in self.res_blocks:
            y = block(y)
        y = self.conv1(y)
        y = y + conv0
        y = self.conv2(y)
        y = self.depth_to_space(y)
        return y


register_example(
    component="DepthToSpaceResNet",
    description=("Residual conv stack followed by dm_pix.depth_to_space upsampling."),
    source="Internal issue #145",
    since="0.11.2",
    context="examples.nnx",
    children=[
        "dm_pix.depth_to_space",
        "nnx.Conv",
        "nnx.List",
        "nnx.silu",
        "jax.numpy.reshape",
        "jax.numpy.transpose",
    ],
    testcases=[
        {
            "testcase": "depth_to_space_resnet_static",
            "callable": construct_and_call(
                DepthToSpaceResNet,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(1, 16, 16, 1)],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x1x8x8 -> Conv:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Mul:1x32x8x8 -> Conv:1x32x8x8 -> Add:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Mul:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Add:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Mul:1x32x8x8 -> Conv:1x32x8x8 -> Add:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Add:1x32x8x8 -> Conv:1x4x8x8 -> "
                        "Transpose:1x8x8x4 -> Reshape:1x8x8x2x2x1 -> "
                        "Transpose:1x8x2x8x2x1 -> Reshape:1x16x16x1",
                        {
                            "counts": {
                                "Conv": 11,
                                "Add": 5,
                                "Mul": 4,
                                "Sigmoid": 4,
                                "Transpose": 3,
                                "Reshape": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "depth_to_space_resnet_inputs_outputs_as_nchw",
            "callable": construct_and_call(
                DepthToSpaceResNet,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(1, 1, 16, 16)],
            "inputs_as_nchw": [0],
            "outputs_as_nchw": [0],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Conv:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Add:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Mul:1x32x8x8 -> Conv:1x32x8x8 -> Add:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Mul:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Add:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8 -> Conv:1x4x8x8 -> Transpose:1x8x8x4 -> "
                        "Reshape:1x8x8x2x2x1 -> Transpose:1x8x2x8x2x1 -> "
                        "Reshape:1x16x16x1 -> Transpose:1x1x16x16",
                        {
                            "counts": {
                                "Conv": 11,
                                "Add": 5,
                                "Mul": 4,
                                "Sigmoid": 4,
                                "Transpose": 3,
                                "Reshape": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
