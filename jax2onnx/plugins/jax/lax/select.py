from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="select",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html",
    onnx=[
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        }
    ],
    since="v0.7.1",
    context="primitives.lax",
    component="select",
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda c, x, y: jax.lax.select(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [np.bool_, np.float32, np.float32],
        },
        {
            "testcase": "select_basic",
            "callable": lambda c, x, y: jax.lax.select(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [np.bool_, np.float32, np.float32],
        },
        {
            "testcase": "select_mask_scores_tensor_else_dynamic",
            "callable": lambda m, s, z: jax.lax.select(m, s, z),
            "input_values": [
                np.array(
                    np.arange(2 * 12 * 5 * 5).reshape(2, 12, 5, 5) % 2 == 0, dtype=bool
                ),
                np.linspace(0.0, 1.0, num=2 * 12 * 5 * 5, dtype=np.float32).reshape(
                    2, 12, 5, 5
                ),
                np.linspace(1.0, 2.0, num=2 * 12 * 5 * 5, dtype=np.float32).reshape(
                    2, 12, 5, 5
                ),
            ],
        },
        {
            "testcase": "select_mask_scores_tensor_else_dynamic_f64",
            "callable": lambda m, s, z: jax.lax.select(m, s, z),
            "input_values": [
                np.array(
                    np.arange(2 * 12 * 5 * 5).reshape(2, 12, 5, 5) % 3 == 0, dtype=bool
                ),
                np.linspace(0.0, 1.0, num=2 * 12 * 5 * 5, dtype=np.float64).reshape(
                    2, 12, 5, 5
                ),
                np.linspace(1.0, 2.0, num=2 * 12 * 5 * 5, dtype=np.float64).reshape(
                    2, 12, 5, 5
                ),
            ],
            "enable_double_precision": True,
        },
        {
            "testcase": "select_mask_scores_tensor_else",
            "callable": lambda m, s, z: jax.lax.select(m, s, z),
            "input_values": [
                np.array(
                    np.arange(3 * 12 * 4 * 4).reshape(3, 12, 4, 4) % 2 == 1, dtype=bool
                ),
                np.linspace(0.0, 1.0, num=3 * 12 * 4 * 4, dtype=np.float32).reshape(
                    3, 12, 4, 4
                ),
                np.linspace(1.0, 2.0, num=3 * 12 * 4 * 4, dtype=np.float32).reshape(
                    3, 12, 4, 4
                ),
            ],
        },
        {
            "testcase": "select_mask_scores_tensor_else_f64",
            "callable": lambda m, s, z: jax.lax.select(m, s, z),
            "input_values": [
                np.array(
                    np.arange(3 * 12 * 4 * 4).reshape(3, 12, 4, 4) % 4 == 0, dtype=bool
                ),
                np.linspace(0.0, 1.0, num=3 * 12 * 4 * 4, dtype=np.float64).reshape(
                    3, 12, 4, 4
                ),
                np.linspace(1.0, 2.0, num=3 * 12 * 4 * 4, dtype=np.float64).reshape(
                    3, 12, 4, 4
                ),
            ],
            "enable_double_precision": True,
        },
    ],
)
class SelectPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.select`` to ONNX ``Where``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        cond_var, x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        cond_val = ctx.get_value_for_var(
            cond_var, name_hint=ctx.fresh_name("select_cond")
        )
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("select_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("select_y"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("select_out"))

        # Cast condition to bool if required.
        if getattr(cond_var.aval, "dtype", np.bool_) != np.bool_:
            bool_cast = ir.Value(
                name=ctx.fresh_name("select_cond_bool"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=cond_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[cond_val],
                    outputs=[bool_cast],
                    name=ctx.fresh_name("Cast"),
                    attributes=[
                        IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))
                    ],
                )
            )
            _stamp_type_and_shape(bool_cast, tuple(getattr(cond_var.aval, "shape", ())))
            _ensure_value_info(ctx, bool_cast)
            cond_val = bool_cast

        node = ir.Node(
            op_type="Where",
            domain="",
            inputs=[cond_val, x_val, y_val],
            outputs=[out_val],
            name=ctx.fresh_name("Where"),
        )
        ctx.add_node(node)

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype_enum = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", np.float32)),
            ctx.builder.enable_double_precision,
        )
        out_val.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
