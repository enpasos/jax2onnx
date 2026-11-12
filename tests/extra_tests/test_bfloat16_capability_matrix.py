# tests/extra_tests/test_bfloat16_capability_matrix.py

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest
from onnx import TensorProto

from tests.extra_tests.capability_matrix import (
    CapabilityCase,
    ShapeVariant,
    assert_capability_case,
    assert_public_float_io_dtype,
    assert_reference_evaluator_matches_jax,
    export_capability_case,
)

# ONNX admitted bfloat16 operator by operator (Conv and pooling from opset 22,
# TopK from 24, Einsum from 28). Below that opset the export computes the
# operator in float32 between bfloat16 values, so these cases cannot require
# every float tensor to be bfloat16.
_FLOAT32_WORK_AT_DEFAULT_OPSET = {"require_all_float_tensors_dtype": False}

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
        id="jnp_where_weak_int_else",
        source=("primitives.jnp", "where", "jnp_where_float16_weak_int_else"),
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
        id="nn_dot_product_attention_mask",
        source=("primitives.nn", "dot_product_attention", "dpa_with_tensor_mask"),
        # Like JAX, the logits and softmax are computed in float32 while
        # bfloat16 is preserved at the public boundary.
        require_all_float_tensors_dtype=False,
        # The ONNX reference MatMul returns float32 for batched bfloat16
        # operands and rejects the result.
        numeric=False,
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
        id="linen_layer_norm_explicit",
        source=("primitives.linen", "layer_norm", "layer_norm_decomposed"),
        # Like Linen itself, the default explicit LayerNorm computes statistics
        # in float32 while preserving bfloat16 at the public boundary.
        require_all_float_tensors_dtype=False,
    ),
    CapabilityCase(
        id="linen_group_norm",
        source=("primitives.linen", "group_norm", "group_norm_rank2"),
        numeric_rtol=8e-2,
        numeric_atol=8e-2,
        # Linen promotes GroupNorm statistics to float32 by default while
        # preserving bfloat16 at the public boundary.
        require_all_float_tensors_dtype=False,
    ),
    CapabilityCase(
        id="lax_conv",
        source=("primitives.lax", "conv", "conv_nchw"),
    ),
    CapabilityCase(
        id="eqx_conv",
        source=("primitives.eqx", "conv", "eqx_conv2d_nchw"),
    ),
    CapabilityCase(
        id="eqx_max_pool",
        source=("primitives.eqx", "max_pool", "eqx_max_pool2d_basic"),
    ),
    CapabilityCase(
        id="eqx_avg_pool",
        source=("primitives.eqx", "avg_pool", "eqx_avg_pool2d_basic"),
    ),
    CapabilityCase(
        id="lax_sin",
        source=("primitives.lax", "sin", "sin"),
    ),
    CapabilityCase(
        id="nn_softplus",
        source=("primitives.nn", "softplus", "jaxnn_softplus"),
    ),
    CapabilityCase(
        id="nn_selu",
        source=("primitives.nn", "selu", "jaxnn_selu"),
    ),
    CapabilityCase(
        id="nn_hard_swish",
        source=("primitives.nn", "hard_swish", "jaxnn_hard_swish"),
    ),
    CapabilityCase(
        id="nn_celu",
        source=("primitives.nn", "celu", "jaxnn_celu"),
        **_FLOAT32_WORK_AT_DEFAULT_OPSET,
    ),
    CapabilityCase(
        id="jnp_einsum_matrix_matrix",
        source=("primitives.jnp", "einsum", "einsum_matrix_matrix"),
        **_FLOAT32_WORK_AT_DEFAULT_OPSET,
    ),
    CapabilityCase(
        id="linen_einsum",
        source=("primitives.linen", "einsum", "einsum_with_bias"),
        **_FLOAT32_WORK_AT_DEFAULT_OPSET,
    ),
    CapabilityCase(
        id="lax_top_k",
        source=("primitives.lax", "top_k", "top_k_last_axis"),
        expected_output_elem_types=(TensorProto.BFLOAT16, TensorProto.INT32),
        **_FLOAT32_WORK_AT_DEFAULT_OPSET,
    ),
]

# Opsets around the admission boundaries, and opset 14, which has neither
# CastLike (opset 15) nor LayerNormalization (opset 17). The default opset 23
# is covered by the matrix test itself.
_OTHER_OPSETS = (14, 17, 21, 22, 24)


@pytest.mark.parametrize(
    "case",
    BF16_CAPABILITY_CASES,
    ids=[case.id for case in BF16_CAPABILITY_CASES],
)
def test_bfloat16_capability_matrix(case: CapabilityCase) -> None:
    assert case.dtype is jnp.bfloat16
    assert_capability_case(case)


@pytest.mark.parametrize("opset", _OTHER_OPSETS)
@pytest.mark.parametrize(
    "case",
    BF16_CAPABILITY_CASES,
    ids=[case.id for case in BF16_CAPABILITY_CASES],
)
def test_bfloat16_capability_matrix_is_valid_at_other_opsets(
    case: CapabilityCase, opset: int
) -> None:
    # export_capability_case runs onnx.checker.check_model(..., full_check=True),
    # which rejects an operator that does not admit bfloat16 at this opset.
    result = export_capability_case(replace(case, opset=opset))
    assert_public_float_io_dtype(result)
    if case.numeric:
        assert_reference_evaluator_matches_jax(result)
