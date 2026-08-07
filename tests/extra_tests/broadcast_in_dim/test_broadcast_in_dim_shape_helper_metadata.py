# tests/extra_tests/broadcast_in_dim/test_broadcast_in_dim_shape_helper_metadata.py

from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import onnx
import onnx_ir as ir

from jax2onnx import to_onnx


_SHAPE_HELPER_PREFIXES = (
    "bcast_src_shape",
    "bcast_dim_dyn",
    "bcast_reshape_sym_shape",
    "bcast_reshape_sym_dim",
)


def test_symbolic_nested_group_norm_shape_helpers_serialize_as_int64(
    caplog,
) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(jax.vmap(norm))
    symbolic_input = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("B1,B2,4,2,2"),
        jnp.float32,
    )

    model = to_onnx(
        fn,
        [symbolic_input],
        opset=18,
        return_mode="ir",
    )

    helper_values = [
        value
        for node in model.graph
        for value in node.outputs
        if any(
            (value.name or "").startswith(prefix) for prefix in _SHAPE_HELPER_PREFIXES
        )
    ]
    helper_names = {value.name or "" for value in helper_values}
    for required_prefix in (
        "bcast_src_shape",
        "bcast_dim_dyn",
        "bcast_reshape_sym_dim",
    ):
        assert any(name.startswith(required_prefix) for name in helper_names)
    assert all(
        getattr(value.type, "dtype", None) == ir.DataType.INT64
        for value in helper_values
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="onnx_ir.serde"):
        model_proto = ir.serde.serialize_model(model)

    serde_warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "onnx_ir.serde" and record.levelno >= logging.WARNING
    ]
    assert not serde_warnings
    onnx.checker.check_model(model_proto, full_check=True)
