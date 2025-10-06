# tests/extra_tests/framework/test_ir_roundtrip.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import onnx
import onnx_ir as ir

from jax2onnx.user_interface import to_onnx


def _smoke_function(x: jax.Array) -> jax.Array:
    return jnp.cos(x) + jnp.sin(x)


def test_ir_model_serializes_to_proto() -> None:
    model_ir = to_onnx(
        _smoke_function,
        [(4,)],
        return_mode="ir",
        opset=18,
        model_name="ir_roundtrip_smoke",
    )

    model_proto = ir.to_proto(model_ir)
    onnx.checker.check_model(model_proto)

    assert model_proto.graph.name == "ir_roundtrip_smoke"
    assert any(node.op_type == "Add" for node in model_proto.graph.node)
