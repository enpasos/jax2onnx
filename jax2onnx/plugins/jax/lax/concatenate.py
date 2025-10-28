# jax2onnx/plugins/jax/lax/concatenate.py

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        return 0
    ax = int(axis)
    return ax % rank if ax < 0 else ax


def _promote_dtype(dtypes: Sequence[np.dtype]) -> np.dtype:
    result = dtypes[0]
    for dt in dtypes[1:]:
        result = np.promote_types(result, dt)
    return result


def _cast_value(
    ctx: "IRContext", value: ir.Value, target: ir.DataType, shape: tuple[int | str, ...]
) -> ir.Value:
    current = getattr(getattr(value, "type", None), "dtype", None)
    if current == target:
        return value
    cast_val = ctx.builder.Cast(
        value,
        _outputs=[ctx.fresh_name("concat_cast")],
        to=int(target.value),
    )
    cast_val.type = ir.TensorType(target)
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=jax.lax.concatenate_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Concat:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_axis1",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [("B", 3), ("B", 4)],
            "post_check_onnx_graph": EG(
                ["Concat:Bx7"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_axis0",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(7, 3), (4, 3)],
            "post_check_onnx_graph": EG(
                ["Concat:11x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_3d",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
            "post_check_onnx_graph": EG(
                ["Concat:2x8x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_internal_int32_then_cast_to_f32_zeroarg",
            "callable": (
                lambda: jax.lax.concatenate(
                    (
                        jax.numpy.array([1], dtype=jax.numpy.int32),
                        jax.numpy.array([2], dtype=jax.numpy.int32),
                    ),
                    dimension=0,
                ).astype(jax.numpy.float32)
            ),
            "expected_output_shapes": [(2,)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Concat:2 -> Cast:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ConcatenatePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        axis = int(params.get("dimension", 0))

        in_vars: Iterable = eqn.invars
        out_var = eqn.outvars[0]

        shapes = [tuple(getattr(v.aval, "shape", ())) for v in in_vars]
        dtypes = [np.dtype(getattr(v.aval, "dtype", np.float32)) for v in in_vars]
        target_dtype = np.dtype(getattr(out_var.aval, "dtype", np.float32))
        if not target_dtype:
            target_dtype = _promote_dtype(dtypes)

        rank = len(shapes[0]) if shapes else 0
        norm_axis = _normalize_axis(axis, rank)
        target_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        inputs: list[ir.Value] = []
        for var, shape, dtype in zip(in_vars, shapes, dtypes):
            val = ctx.get_value_for_var(var, name_hint=ctx.fresh_name("concat_in"))
            if np.dtype(dtype) != target_dtype:
                val = _cast_value(ctx, val, target_enum, shape)
            inputs.append(val)

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("concat_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Concat")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Concat")

        result = ctx.builder.Concat(
            *inputs,
            axis=int(norm_axis),
            _outputs=[desired_name],
        )
        result.type = ir.TensorType(target_enum)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        for inp in inputs:
            propagate_axis0_override(inp, result)
        if norm_axis == 0:
            override = next(
                (
                    int(val)
                    for val in (
                        get_axis0_override(inp) for inp in inputs  # type: ignore[arg-type]
                    )
                    if isinstance(val, (int, np.integer)) and int(val) > 1
                ),
                None,
            )
            if override is not None:
                set_axis0_override(result, override)
        ctx.bind_value_for_var(out_var, result)
