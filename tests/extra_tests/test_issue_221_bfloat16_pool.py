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

    average_pool_nodes = [
        node for node in inferred.graph.node if node.op_type == "AveragePool"
    ]
    mul_nodes = [node for node in inferred.graph.node if node.op_type == "Mul"]
    assert average_pool_nodes
    assert mul_nodes

    elem_types = _elem_types_by_name(inferred)
    assert elem_types[inferred.graph.input[0].name] == TensorProto.BFLOAT16
    assert elem_types[inferred.graph.output[0].name] == TensorProto.BFLOAT16
    for node in average_pool_nodes + mul_nodes:
        for output in node.output:
            assert elem_types[output] == TensorProto.BFLOAT16

    (scale_initializer,) = [
        initializer
        for initializer in inferred.graph.initializer
        if initializer.name.startswith("pool_sum_scale")
    ]
    assert scale_initializer.data_type == TensorProto.BFLOAT16
    assert any(scale_initializer.name in node.input for node in mul_nodes)

    x_f32 = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1) / np.float32(10)
    x_bf16 = x_f32.astype(ml_dtypes.bfloat16)
    expected = np.asarray(_pool_sum_bfloat16(jnp.asarray(x_bf16)))

    evaluator = ReferenceEvaluator(inferred)
    input_name = inferred.graph.input[0].name
    (got,) = evaluator.run(None, {input_name: x_bf16})

    assert got.dtype == ml_dtypes.bfloat16
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
