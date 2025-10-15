# jax2onnx/plugins/jax/lax/bitcast_convert_type.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="bitcast_convert_type",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitcast_convert_type.html",
    onnx=[
        {
            "component": "Bitcast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Bitcast.html",
        }
    ],
    since="v0.7.2",
    context="primitives.lax",
    component="bitcast_convert_type",
    testcases=[
        {
            "testcase": "bitcast_scalar_f32_to_i32",
            "callable": lambda x: jax.lax.bitcast_convert_type(x, new_dtype=jnp.int32),
            "input_values": [np.array(3.5, dtype=np.float32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Bitcast"],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
        },
        {
            "testcase": "bitcast_tensor_i32_to_f32",
            "callable": lambda x: jax.lax.bitcast_convert_type(
                x, new_dtype=jnp.float32
            ),
            "input_values": [np.array([0x3F800000, 0x40000000], dtype=np.uint32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Bitcast:2"],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
        },
    ],
)
class BitcastConvertTypePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.bitcast_convert_type`` via ONNX Bitcast."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        target_dtype = _dtype_to_ir(
            np.dtype(eqn.params.get("new_dtype", out_var.aval.dtype)),
            ctx.builder.enable_double_precision,
        )

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("bitcast_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("bitcast_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Bitcast")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Bitcast")

        result = ctx.builder.Bitcast(
            operand_val,
            _outputs=[desired_name],
            to=int(target_dtype.value),
        )
        result.type = ir.TensorType(target_dtype)
        result.shape = operand_val.shape
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
