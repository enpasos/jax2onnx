# tests/extra_tests/test_issue_223_nnx_silu_swish.py

from __future__ import annotations

import json

import jax
from flax import nnx

_CAPTURED_NNX_SILU = nnx.silu
_CAPTURED_NNX_SWISH = nnx.swish

from jax2onnx import to_onnx  # noqa: E402


def _op_types(model) -> list[str]:
    return [node.op_type for node in model.graph.node]


def test_direct_nnx_silu_binds_shared_silu_primitive(tmp_path) -> None:
    record_path = tmp_path / "primitive_calls.json"

    model = to_onnx(
        lambda x: nnx.silu(x),
        inputs=[(2, 5)],
        opset=24,
        record_primitive_calls_file=str(record_path),
    )

    assert _op_types(model) == ["Swish"]
    primitive_names = [
        entry["primitive_name"] for entry in json.loads(record_path.read_text())
    ]
    assert primitive_names == ["jax.nn.silu"]


def test_captured_nnx_silu_rewrites_expanded_pattern_to_swish_for_opset24() -> None:
    model = to_onnx(lambda x: _CAPTURED_NNX_SILU(x), inputs=[(2, 5)], opset=24)

    assert _op_types(model) == ["Swish"]


def test_captured_nnx_swish_rewrites_expanded_pattern_to_swish_for_opset24() -> None:
    model = to_onnx(lambda x: _CAPTURED_NNX_SWISH(x), inputs=[(2, 5)], opset=24)

    assert _op_types(model) == ["Swish"]


def test_rewrite_handles_reversed_sigmoid_mul_operands() -> None:
    model = to_onnx(lambda x: jax.lax.logistic(x) * x, inputs=[(2, 5)], opset=24)

    assert _op_types(model) == ["Swish"]


def test_rewrite_keeps_sigmoid_when_it_has_other_consumers() -> None:
    model = to_onnx(
        lambda x: (x * jax.lax.logistic(x), jax.lax.logistic(x)),
        inputs=[(2, 5)],
        opset=24,
    )

    assert _op_types(model) == ["Sigmoid", "Swish"]


def test_captured_nnx_silu_keeps_sigmoid_mul_before_opset24() -> None:
    model = to_onnx(lambda x: _CAPTURED_NNX_SILU(x), inputs=[(2, 5)], opset=23)

    assert _op_types(model) == ["Sigmoid", "Mul"]
