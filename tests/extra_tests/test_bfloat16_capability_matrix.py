# tests/extra_tests/test_bfloat16_capability_matrix.py

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tests.extra_tests.capability_matrix import (
    CapabilityCase,
    ShapeVariant,
    assert_capability_case,
)

BF16_CAPABILITY_CASES = [
    CapabilityCase(
        id="jnp_add",
        source=("primitives.jnp", "add", "add"),
    ),
    CapabilityCase(
        id="jnp_minimum",
        source=("primitives.jnp", "minimum", "jnp_minimum_basic"),
    ),
    CapabilityCase(
        id="jnp_maximum",
        source=("primitives.jnp", "maximum", "jnp_maximum_basic"),
    ),
    CapabilityCase(
        id="jnp_tanh",
        source=("primitives.jnp", "tanh", "jnp_tanh_basic"),
    ),
    CapabilityCase(
        id="jnp_sqrt_reduce",
        source=("primitives.jnp", "sqrt", "jnp_sqrt_reduce_sum_square_axis1"),
    ),
    CapabilityCase(
        id="jnp_sum_axis",
        source=("primitives.jnp", "sum", "jnp_sum_axis"),
    ),
    CapabilityCase(
        id="jnp_max_axis",
        source=("primitives.jnp", "max", "jnp_max_axis"),
    ),
    CapabilityCase(
        id="jnp_transpose",
        source=("primitives.jnp", "transpose", "transpose_basic"),
    ),
    CapabilityCase(
        id="jnp_concatenate",
        source=("primitives.jnp", "concatenate", "concatenate_basic"),
    ),
    CapabilityCase(
        id="jnp_take",
        source=("primitives.jnp", "take", "take_data_dependent_indices"),
    ),
    CapabilityCase(
        id="jnp_matmul",
        source=("primitives.jnp", "matmul", "matmul_2d"),
    ),
    CapabilityCase(
        id="linen_dense",
        source=("primitives.linen", "dense", "dense_basic"),
    ),
    CapabilityCase(
        id="linen_conv",
        source=("primitives.linen", "conv", "conv_no_bias"),
    ),
    CapabilityCase(
        id="linen_conv_non_square_resolution",
        source=("primitives.linen", "conv", "conv_no_bias"),
        shape_variant=ShapeVariant(
            name="non_square_resolution",
            input_shapes={0: (1, 12, 16, 3)},
        ),
    ),
    CapabilityCase(
        id="linen_avg_pool",
        source=("primitives.linen", "avg_pool", "avg_pool_stride1"),
    ),
    CapabilityCase(
        id="linen_avg_pool_non_square_resolution",
        source=("primitives.linen", "avg_pool", "avg_pool_stride1"),
        shape_variant=ShapeVariant(
            name="non_square_resolution",
            input_shapes={0: (1, 6, 10, 3)},
        ),
    ),
    CapabilityCase(
        id="linen_max_pool",
        source=("primitives.linen", "max_pool", "max_pool_basic"),
    ),
    CapabilityCase(
        id="linen_pool_add",
        source=("primitives.linen", "pool", "pool_sum_basic"),
    ),
    CapabilityCase(
        id="linen_layer_norm",
        source=("primitives.linen", "layer_norm", "layer_norm"),
    ),
    CapabilityCase(
        id="linen_group_norm",
        source=("primitives.linen", "group_norm", "group_norm_rank2"),
        numeric_rtol=8e-2,
        numeric_atol=8e-2,
    ),
]


@pytest.mark.parametrize(
    "case",
    BF16_CAPABILITY_CASES,
    ids=[case.id for case in BF16_CAPABILITY_CASES],
)
def test_bfloat16_capability_matrix(case: CapabilityCase) -> None:
    assert case.dtype is jnp.bfloat16
    assert_capability_case(case)
