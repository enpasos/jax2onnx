# tests/extra_tests/test_opset27_range_and_resize.py

from __future__ import annotations

import jax
import jax.image as jimage
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto

from jax2onnx.user_interface import to_onnx


def _node_types(model: onnx.ModelProto) -> list[str]:
    return [node.op_type for node in model.graph.node]


def test_iota_bfloat16_opset23_keeps_cast_fallback() -> None:
    model = to_onnx(lambda: jax.lax.iota(jnp.bfloat16, 5), [], opset=23)

    onnx.checker.check_model(model)
    ops = _node_types(model)
    assert "Range" in ops
    assert "Cast" in ops
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16


def test_iota_bfloat16_opset27_uses_native_range_dtype() -> None:
    model = to_onnx(lambda: jax.lax.iota(jnp.bfloat16, 5), [], opset=27)

    onnx.checker.check_model(model)
    ops = _node_types(model)
    assert "Range" in ops
    assert "Cast" not in ops
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16


def test_dynamic_arange_bfloat16_opset23_casts_after_range() -> None:
    model = to_onnx(
        lambda stop: jnp.arange(stop, dtype=jnp.bfloat16),
        [jax.ShapeDtypeStruct((), jnp.float32)],
        opset=23,
    )

    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node[-2:]] == ["Range", "Cast"]
    assert model.graph.output[0].name == model.graph.node[-1].output[0]
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16


def test_dynamic_arange_bfloat16_opset27_range_is_graph_output() -> None:
    model = to_onnx(
        lambda stop: jnp.arange(stop, dtype=jnp.bfloat16),
        [jax.ShapeDtypeStruct((), jnp.float32)],
        opset=27,
    )

    onnx.checker.check_model(model)
    range_nodes = [node for node in model.graph.node if node.op_type == "Range"]
    assert len(range_nodes) == 1
    assert model.graph.output[0].name == range_nodes[0].output[0]
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16


def test_resize_area_static_exact_matrix_path() -> None:
    model = to_onnx(
        lambda x: jimage.resize(x, (2, 2), method="area", antialias=False),
        [jax.ShapeDtypeStruct((4, 4), np.float32)],
        opset=23,
    )

    onnx.checker.check_model(model)
    assert _node_types(model) == ["Reshape", "MatMul", "Reshape"]
    assert tuple(
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ) == (2, 2)


def test_resize_cubic_pytorch_static_exact_matrix_path() -> None:
    model = to_onnx(
        lambda x: jimage.resize(x, (3, 5), method="cubic-pytorch", antialias=False),
        [jax.ShapeDtypeStruct((2, 2), np.float32)],
        opset=23,
    )

    onnx.checker.check_model(model)
    assert _node_types(model) == ["Reshape", "MatMul", "Reshape"]
    assert tuple(
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ) == (3, 5)
