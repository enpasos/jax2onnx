# jax2onnx/plugins/jax/lax/atan2.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _cast_to(
    ctx: "IRContext",
    value: ir.Value,
    *,
    target_dtype: ir.DataType,
    output_shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    if value.dtype == target_dtype:
        return value
    cast_val = ctx.builder.Cast(
        value,
        to=int(target_dtype.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    cast_val.type = ir.TensorType(target_dtype)
    _stamp_type_and_shape(cast_val, output_shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=jax.lax.atan2_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.atan2.html",
    onnx=[
        {"component": "Atan", "doc": "https://onnx.ai/onnx/operators/onnx__Atan.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        },
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="atan2",
    testcases=[
        {
            "testcase": "atan2_quadrants_and_zero",
            "callable": lambda x, y: jax.lax.atan2(x, y),
            "input_values": [
                np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Div:7 -> Atan:7"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "atan2_broadcast",
            "callable": lambda x, y: jax.lax.atan2(x, y),
            "input_shapes": [(2, 1), (1, 3)],
            "expected_output_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class Atan2Plugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var, y_var = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("atan2_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("atan2_y"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("atan2_out"))

        out_np_dtype = np.dtype(getattr(out_var.aval, "dtype", np.float32))
        out_dtype = _dtype_to_ir(out_np_dtype, ctx.builder.enable_double_precision)
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        y_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        x_ready = _cast_to(
            ctx,
            x_val,
            target_dtype=out_dtype,
            output_shape=x_shape,
            name_hint="atan2_x_cast",
        )
        y_ready = _cast_to(
            ctx,
            y_val,
            target_dtype=out_dtype,
            output_shape=y_shape,
            name_hint="atan2_y_cast",
        )

        const_np_dtype = np.float64 if out_dtype == ir.DataType.DOUBLE else np.float32

        def _const(value: float) -> ir.Value:
            return ctx.bind_const_for_var(
                object(), np.asarray(value, dtype=const_np_dtype)
            )

        zero = _const(0.0)
        pi = _const(np.pi)
        pi_half = _const(np.pi / 2.0)
        neg_pi_half = _const(-np.pi / 2.0)

        x_over_y = ctx.builder.Div(
            x_ready,
            y_ready,
            _outputs=[ctx.fresh_name("atan2_div")],
        )
        x_over_y.type = ir.TensorType(out_dtype)
        _stamp_type_and_shape(x_over_y, out_shape)
        _ensure_value_metadata(ctx, x_over_y)
        base = ctx.builder.Atan(x_over_y, _outputs=[ctx.fresh_name("atan2_base")])
        base.type = ir.TensorType(out_dtype)
        _stamp_type_and_shape(base, out_shape)
        _ensure_value_metadata(ctx, base)

        y_gt0 = ctx.builder.Greater(y_ready, zero, _outputs=[ctx.fresh_name("y_gt0")])
        y_lt0 = ctx.builder.Less(y_ready, zero, _outputs=[ctx.fresh_name("y_lt0")])
        x_gt0 = ctx.builder.Greater(x_ready, zero, _outputs=[ctx.fresh_name("x_gt0")])
        x_lt0 = ctx.builder.Less(x_ready, zero, _outputs=[ctx.fresh_name("x_lt0")])
        x_eq0 = ctx.builder.Equal(x_ready, zero, _outputs=[ctx.fresh_name("x_eq0")])
        x_ge0 = ctx.builder.Or(x_gt0, x_eq0, _outputs=[ctx.fresh_name("x_ge0")])

        base_plus_pi = ctx.builder.Add(
            base,
            pi,
            _outputs=[ctx.fresh_name("atan2_base_plus_pi")],
        )
        base_minus_pi = ctx.builder.Sub(
            base,
            pi,
            _outputs=[ctx.fresh_name("atan2_base_minus_pi")],
        )
        res_y_lt0 = ctx.builder.Where(
            x_ge0,
            base_plus_pi,
            base_minus_pi,
            _outputs=[ctx.fresh_name("atan2_res_y_lt0")],
        )

        res_y_eq0_inner = ctx.builder.Where(
            x_lt0,
            neg_pi_half,
            zero,
            _outputs=[ctx.fresh_name("atan2_res_y_eq0_inner")],
        )
        res_y_eq0 = ctx.builder.Where(
            x_gt0,
            pi_half,
            res_y_eq0_inner,
            _outputs=[ctx.fresh_name("atan2_res_y_eq0")],
        )

        res_not_y_gt0 = ctx.builder.Where(
            y_lt0,
            res_y_lt0,
            res_y_eq0,
            _outputs=[ctx.fresh_name("atan2_res_not_y_gt0")],
        )

        output_name = getattr(out_spec, "name", None) or ctx.fresh_name("atan2_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            output_name = ctx.fresh_name("atan2_out")

        result = ctx.builder.Where(
            y_gt0,
            base,
            res_not_y_gt0,
            _outputs=[output_name],
        )
        result.type = ir.TensorType(out_dtype)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
