from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


def _cast_value(
    ctx: "IRContext", value: ir.Value, target: ir.DataType, shape: tuple, name: str
) -> ir.Value:
    current = getattr(getattr(value, "type", None), "dtype", None)
    if current == target:
        return value
    cast_val = ir.Value(
        name=ctx.fresh_name(name),
        type=ir.TensorType(target),
        shape=ir.Shape(tuple(shape)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[cast_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(target.value))],
        )
    )
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_info(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=jax.lax.clamp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.clamp.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="v0.8.0",
    context="primitives2.lax",
    component="clamp",
    testcases=[
        {
            "testcase": "clamp_i32_scalar_bounds",
            "callable": lambda x: jax.lax.clamp(
                jax.numpy.asarray(0, dtype=x.dtype),
                x,
                jax.numpy.asarray(4, dtype=x.dtype),
            ),
            "input_values": [np.array([-3, 1, 9, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "use_onnx_ir": True,
        },
        {
            "testcase": "clamp_scalar_float_bounds_match_x",
            "callable": lambda x: jax.lax.clamp(
                jax.numpy.asarray(-1.5, dtype=x.dtype),
                x,
                jax.numpy.asarray(2.5, dtype=x.dtype),
            ),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "use_onnx_ir": True,
        },
        {
            "testcase": "clamp_vector_bounds_match",
            "callable": lambda x, lo, hi: jax.lax.clamp(lo, x, hi),
            "input_values": [
                np.array([-5, -1, 0, 1, 5], dtype=np.float64),
                np.array([-1, -1, -1, -1, -1], dtype=np.float64),
                np.array([1, 1, 1, 1, 1], dtype=np.float64),
            ],
            "expected_output_shapes": [(5,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "clamp_pyint_bounds_promote_to_x_dtype",
            "callable": lambda x: jax.lax.clamp(
                jax.numpy.asarray(0, dtype=x.dtype),
                x,
                jax.numpy.asarray(1, dtype=x.dtype),
            ),
            "input_values": [np.array([-2.0, 0.25, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "use_onnx_ir": True,
        },
    ],
)
class ClampPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        min_var, x_var, max_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("clamp_x"))
        min_val = ctx.get_value_for_var(
            min_var,
            name_hint=ctx.fresh_name("clamp_min"),
            prefer_np_dtype=np.dtype(getattr(x_var.aval, "dtype", np.float32)),
        )
        max_val = ctx.get_value_for_var(
            max_var,
            name_hint=ctx.fresh_name("clamp_max"),
            prefer_np_dtype=np.dtype(getattr(x_var.aval, "dtype", np.float32)),
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("clamp_out"))

        target_dtype = _dtype_to_ir(
            np.dtype(getattr(x_var.aval, "dtype", np.float32)),
            ctx.builder.enable_double_precision,
        )

        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        min_cast = _cast_value(ctx, min_val, target_dtype, x_shape, "ClampMinCast")
        max_cast = _cast_value(ctx, max_val, target_dtype, x_shape, "ClampMaxCast")

        max_out = ir.Value(
            name=ctx.fresh_name("clamp_max_out"),
            type=ir.TensorType(target_dtype),
            shape=ir.Shape(x_shape),
        )
        ctx.add_node(
            ir.Node(
                op_type="Max",
                domain="",
                inputs=[x_val, min_cast],
                outputs=[max_out],
                name=ctx.fresh_name("Max"),
            )
        )
        _stamp_type_and_shape(max_out, x_shape)
        _ensure_value_info(ctx, max_out)

        ctx.add_node(
            ir.Node(
                op_type="Min",
                domain="",
                inputs=[max_out, max_cast],
                outputs=[out_val],
                name=ctx.fresh_name("Min"),
            )
        )
        _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, out_val)
