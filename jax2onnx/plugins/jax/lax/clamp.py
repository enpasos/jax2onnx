# jax2onnx/plugins/jax/lax/clamp.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _cast_value(
    ctx: "IRContext", value: ir.Value, target: ir.DataType, shape: tuple, name: str
) -> ir.Value:
    current = getattr(getattr(value, "type", None), "dtype", None)
    if current == target:
        return value
    cast_name = ctx.fresh_name(name)
    cast_val = ctx.builder.Cast(
        value,
        _outputs=[cast_name],
        to=int(target.value),
    )
    cast_val.type = ir.TensorType(target)
    cast_val.shape = ir.Shape(tuple(shape))
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=jax.lax.clamp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.clamp.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="v0.7.5",
    context="primitives.lax",
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
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Max:4 -> Min:4",
                        "inputs": {1: {"const": 4.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Max:3 -> Min:3",
                        "inputs": {1: {"const": 2.5}},
                    }
                ],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Max:5 -> Min:5"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Max:3 -> Min:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
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
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("clamp_out"))

        target_dtype = _dtype_to_ir(
            np.dtype(getattr(x_var.aval, "dtype", np.float32)),
            ctx.builder.enable_double_precision,
        )

        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        min_cast = _cast_value(ctx, min_val, target_dtype, x_shape, "ClampMinCast")
        max_cast = _cast_value(ctx, max_val, target_dtype, x_shape, "ClampMaxCast")

        max_name = ctx.fresh_name("clamp_max_out")
        max_out = ctx.builder.Max(x_val, min_cast, _outputs=[max_name])
        max_out.type = ir.TensorType(target_dtype)
        max_out.shape = ir.Shape(x_shape)
        _stamp_type_and_shape(max_out, x_shape)
        _ensure_value_metadata(ctx, max_out)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("clamp_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("clamp_out")

        result = ctx.builder.Min(max_out, max_cast, _outputs=[desired_name])
        result.type = ir.TensorType(target_dtype)
        result.shape = ir.Shape(tuple(getattr(out_var.aval, "shape", x_shape)))
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
