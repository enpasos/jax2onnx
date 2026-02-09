# jax2onnx/plugins/examples/nnx/nested_residual_group.py

from __future__ import annotations

import jax
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)

filters: int = 32
kernel_size: tuple[int, int] = (3, 3)
blocks: int = 2
groups: int = 3


class _ResGroup(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.res_blocks = nnx.List(_ResBlock(rngs=rngs) for _ in range(blocks))
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = input
        for block in self.res_blocks:
            x = block(x)
        x = self.conv0(x)
        x = x + input
        return x


class _ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = nnx.silu(self.conv0(input))
        x = self.conv1(x)
        x = x + input
        return x


class _ResGroupWithLeadConv(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.res_blocks = nnx.List(_ResBlock(rngs=rngs) for _ in range(blocks))
        self.conv0 = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, input: jax.Array) -> jax.Array:
        x = self.conv0(input)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv1(x)
        x = x + input
        return x


class _ResNetWithNestedGroups(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.stem = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.groups = nnx.List(_ResGroup(rngs=rngs) for _ in range(groups))
        self.tail = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, input: jax.Array) -> jax.Array:
        x = self.stem(input)
        for group in self.groups:
            x = group(x)
        x = self.tail(x)
        return x


class _ResNetWithNestedGroupsLeadConv(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.stem = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.groups = nnx.List(_ResGroupWithLeadConv(rngs=rngs) for _ in range(groups))
        self.tail = nnx.Conv(
            filters,
            filters,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, input: jax.Array) -> jax.Array:
        x = self.stem(input)
        for group in self.groups:
            x = group(x)
        x = self.tail(x)
        return x


register_example(
    component="NestedResidualGroup",
    description=(
        "Nested residual blocks inside a residual group; regression harness for issue #173."
    ),
    source="https://github.com/enpasos/jax2onnx/issues/173",
    since="0.12.0",
    context="examples.nnx",
    children=["nnx.Conv", "nnx.List", "nnx.silu"],
    testcases=[
        {
            "testcase": "nested_residual_group_static",
            "callable": construct_and_call(
                _ResGroup,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, filters)],
            "expected_output_shapes": [(1, 8, 8, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Transpose:1x8x8x32 -> Add:1x8x8x32 -> "
                        "Transpose:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Transpose:1x8x8x32 -> Add:1x8x8x32 -> "
                        "Transpose:1x32x8x8 -> Conv:1x32x8x8 -> Transpose:1x8x8x32 -> "
                        "Add:1x8x8x32",
                        {
                            "counts": {
                                "Conv": 5,
                                "Add": 3,
                                "Mul": 2,
                                "Transpose": 6,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nested_residual_group_static_nchw",
            "callable": construct_and_call(
                _ResGroup,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, filters)],
            "expected_output_shapes": [(1, filters, 8, 8)],
            "inputs_as_nchw": [0],
            "outputs_as_nchw": [0],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Conv:1x32x8x8 -> Mul:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8 -> Conv:1x32x8x8 -> Mul:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Add:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8",
                        {
                            "counts": {
                                "Conv": 5,
                                "Add": 3,
                                "Mul": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nested_residual_group_with_lead_conv_static",
            "callable": construct_and_call(
                _ResGroupWithLeadConv,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, filters)],
            "expected_output_shapes": [(1, 8, 8, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x32x8x8 -> Conv:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Mul:1x32x8x8 -> Conv:1x32x8x8 -> Add:1x32x8x8 -> "
                        "Conv:1x32x8x8 -> Mul:1x32x8x8 -> Conv:1x32x8x8 -> "
                        "Add:1x32x8x8 -> Conv:1x32x8x8 -> Transpose:1x8x8x32 -> "
                        "Add:1x8x8x32",
                        {
                            "counts": {
                                "Conv": 6,
                                "Add": 3,
                                "Mul": 2,
                                "Transpose": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nested_residual_stack_static",
            "callable": construct_and_call(
                _ResNetWithNestedGroups,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 16, 16, filters)],
            "expected_output_shapes": [(1, 16, 16, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x32x16x16 -> Conv:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Mul:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Mul:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Mul:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Mul:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Mul:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Mul:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Transpose:1x16x16x32",
                        {
                            "counts": {
                                "Conv": 17,
                                "Add": 9,
                                "Mul": 6,
                                "Transpose": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nested_residual_stack_static_no_extra_transpose",
            "callable": construct_and_call(
                _ResNetWithNestedGroups,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 16, 16, filters)],
            "expected_output_shapes": [(1, 16, 16, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose -> Conv -> Conv -> Mul -> Conv -> Add",
                        {
                            "counts": {
                                "Conv": 17,
                                "Add": 9,
                                "Mul": 6,
                                "Transpose": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nested_residual_stack_with_lead_conv_static",
            "callable": construct_and_call(
                _ResNetWithNestedGroupsLeadConv,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 16, 16, filters)],
            "expected_output_shapes": [(1, 16, 16, filters)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x32x16x16 -> Conv:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Mul:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Mul:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Mul:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Mul:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Add:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Conv:1x32x16x16 -> Mul:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Mul:1x32x16x16 -> Conv:1x32x16x16 -> Add:1x32x16x16 -> "
                        "Conv:1x32x16x16 -> Add:1x32x16x16 -> Conv:1x32x16x16 -> "
                        "Transpose:1x16x16x32",
                        {
                            "counts": {
                                "Conv": 20,
                                "Add": 9,
                                "Mul": 6,
                                "Transpose": 2,
                            }
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
