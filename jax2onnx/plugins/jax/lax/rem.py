# jax2onnx/plugins/jax/lax/rem.py

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.rem_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rem.html",
    onnx=[
        {
            "component": "Mod",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html",
        },
        {
            "component": "Div",
            "doc": "https://onnx.ai/onnx/operators/onnx__Div.html",
        },
    ],
    since="v0.6.5",
    context="primitives.lax",
    component="rem",
    testcases=[
        {
            "testcase": "rem_int",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32),
                np.array([4, 4, 3, 3, 2, 2, 1, 1, 5, 5], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["Sub:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "rem_float",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                np.array(
                    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    dtype=np.float32,
                ),
                np.array(
                    [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 5.0, 5.0],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["Mod:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "rem_int_neg",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                np.array([-10, -9, -8, -7, 6, 5, 4, 3, -2, -1], dtype=np.int32),
                np.array([4, -4, 3, -3, 2, -2, 1, -1, 5, -5], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["Sub:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "rem_float_neg",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                np.array(
                    [-10.0, -9.0, -8.0, -7.0, 6.0, 5.0, 4.0, 3.0, -2.0, -1.0],
                    dtype=np.float32,
                ),
                np.array(
                    [4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 5.0, -5.0],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["Mod:10"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class RemPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.rem`` (truncated remainder) using Mod or Div/Mul/Sub."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rem_x"))
        y_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("rem_y"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rem_out"))

        aval = getattr(x_var, "aval", None)
        dtype = np.dtype(getattr(aval, "dtype", np.float32))
        out_shape = tuple(getattr(aval, "shape", ()))

        x_dtype_enum = getattr(getattr(x_val, "type", None), "dtype", ir.DataType.FLOAT)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("rem_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("rem_out")

        if np.issubdtype(dtype, np.floating):
            result = ctx.builder.Mod(
                x_val,
                y_val,
                fmod=1,
                _outputs=[desired_name],
            )
            result.type = ir.TensorType(x_dtype_enum)
            result.shape = ir.Shape(out_shape)
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

        div_val = ctx.builder.Div(
            x_val,
            y_val,
            _outputs=[ctx.fresh_name("rem_div")],
        )
        div_val.type = ir.TensorType(x_dtype_enum)
        div_val.shape = ir.Shape(out_shape)
        _stamp_type_and_shape(div_val, out_shape)
        _ensure_value_metadata(ctx, div_val)

        mul_val = ctx.builder.Mul(
            div_val,
            y_val,
            _outputs=[ctx.fresh_name("rem_mul")],
        )
        mul_val.type = ir.TensorType(x_dtype_enum)
        mul_val.shape = ir.Shape(out_shape)
        _stamp_type_and_shape(mul_val, out_shape)
        _ensure_value_metadata(ctx, mul_val)

        result = ctx.builder.Sub(
            x_val,
            mul_val,
            _outputs=[desired_name],
        )
        result.type = ir.TensorType(x_dtype_enum)
        result.shape = ir.Shape(out_shape)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
