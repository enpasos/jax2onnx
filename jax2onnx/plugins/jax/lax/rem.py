from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
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
        },
        {
            "testcase": "rem_int_neg",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                np.array([-10, -9, -8, -7, 6, 5, 4, 3, -2, -1], dtype=np.int32),
                np.array([4, -4, 3, -3, 2, -2, 1, -1, 5, -5], dtype=np.int32),
            ],
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
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rem_out"))

        aval = getattr(x_var, "aval", None)
        dtype = np.dtype(getattr(aval, "dtype", np.float32))
        out_shape = tuple(getattr(aval, "shape", ()))

        out_dtype_enum = getattr(getattr(out_val, "type", None), "dtype", None)
        if out_dtype_enum is None:
            out_dtype_enum = ir.DataType.FLOAT

        if np.issubdtype(dtype, np.floating):
            mod_tmp = ir.Value(
                name=ctx.fresh_name("rem_mod"),
                type=ir.TensorType(out_dtype_enum),
                shape=out_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Mod",
                    domain="",
                    inputs=[x_val, y_val],
                    outputs=[mod_tmp],
                    name=ctx.fresh_name("Mod"),
                    attributes=[IRAttr("fmod", IRAttrType.INT, 1)],
                )
            )
            _stamp_type_and_shape(mod_tmp, out_shape)
            _ensure_value_info(ctx, mod_tmp)

            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[mod_tmp],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )
        else:
            quot_val = ir.Value(
                name=ctx.fresh_name("rem_div"),
                type=ir.TensorType(out_dtype_enum),
                shape=out_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Div",
                    domain="",
                    inputs=[x_val, y_val],
                    outputs=[quot_val],
                    name=ctx.fresh_name("Div"),
                )
            )
            _stamp_type_and_shape(quot_val, out_shape)
            _ensure_value_info(ctx, quot_val)

            prod_val = ir.Value(
                name=ctx.fresh_name("rem_mul"),
                type=ir.TensorType(out_dtype_enum),
                shape=out_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Mul",
                    domain="",
                    inputs=[quot_val, y_val],
                    outputs=[prod_val],
                    name=ctx.fresh_name("Mul"),
                )
            )
            _stamp_type_and_shape(prod_val, out_shape)
            _ensure_value_info(ctx, prod_val)

            ctx.add_node(
                ir.Node(
                    op_type="Sub",
                    domain="",
                    inputs=[x_val, prod_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Sub"),
                )
            )

        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
