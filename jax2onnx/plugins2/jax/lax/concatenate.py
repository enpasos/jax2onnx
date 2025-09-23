from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


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
    cast_val = ir.Value(
        name=ctx.fresh_name("concat_cast"),
        type=ir.TensorType(target),
        shape=ir.Shape(shape),
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
    context="primitives2.lax",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "concatenate_axis1",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [("B", 3), ("B", 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "concatenate_axis0",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(7, 3), (4, 3)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "concatenate_3d",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("concat_out"))
        node = ir.Node(
            op_type="Concat",
            domain="",
            inputs=inputs,
            outputs=[out_val],
            name=ctx.fresh_name("Concat"),
            attributes=[IRAttr("axis", IRAttrType.INT, int(norm_axis))],
        )
        ctx.add_node(node)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_val.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
