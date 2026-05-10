# tests/extra_tests/test_issue_221_bfloat16_pool.py

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

import jax
import jax.numpy as jnp
from flax.linen import pooling as linen_pooling

from jax2onnx.user_interface import to_onnx


def _pool_sum_bfloat16(x: jax.Array) -> jax.Array:
    return linen_pooling.pool(
        x,
        0.0,
        jax.lax.add,
        window_shape=(2, 2),
        strides=(1, 1),
        padding="VALID",
    )


def _elem_types_by_name(model: onnx.ModelProto) -> dict[str, int]:
    elem_types: dict[str, int] = {}
    for value_info in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type:
            elem_types[value_info.name] = tensor_type.elem_type
    return elem_types


def test_issue_221_linen_pool_add_bfloat16_exports_and_runs() -> None:
    model = to_onnx(
        _pool_sum_bfloat16,
        [jax.ShapeDtypeStruct((1, 4, 4, 1), jnp.bfloat16)],
        model_name="issue221_linen_pool_add_bfloat16",
        opset=23,
    )

    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    nodes = list(inferred.graph.node)
    assert [node.op_type for node in nodes] == [
        "Transpose",
        "AveragePool",
        "Mul",
        "Transpose",
    ]

    elem_types = _elem_types_by_name(inferred)
    assert elem_types[inferred.graph.input[0].name] == TensorProto.BFLOAT16
    assert elem_types[inferred.graph.output[0].name] == TensorProto.BFLOAT16
    assert elem_types[nodes[1].output[0]] == TensorProto.BFLOAT16
    assert elem_types[nodes[2].output[0]] == TensorProto.BFLOAT16

    (scale_initializer,) = [
        initializer
        for initializer in inferred.graph.initializer
        if initializer.name.startswith("pool_sum_scale")
    ]
    assert scale_initializer.data_type == TensorProto.BFLOAT16

    x_f32 = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1) / np.float32(10)
    x_bf16 = x_f32.astype(ml_dtypes.bfloat16)
    expected = np.asarray(_pool_sum_bfloat16(jnp.asarray(x_bf16)))

    evaluator = ReferenceEvaluator(inferred)
    input_name = inferred.graph.input[0].name
    (got,) = evaluator.run(None, {input_name: x_bf16})

    assert got.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(got.astype(np.float32), expected.astype(np.float32))
