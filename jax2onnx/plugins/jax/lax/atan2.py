# jax2onnx/plugins/jax/lax/atan2.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
    with_requested_dtype,
)


def _atan2_graph_check(model: Any) -> bool:
    double = model.graph.output[0].type.tensor_type.elem_type == int(
        ir.DataType.DOUBLE.value
    )
    if double:
        valid = bool(
            EG(
                [("Sqrt:7 -> Add:7 -> Div:7", {"counts": {"Sqrt": 2, "Mul": 17}})],
                must_absent=["Atan", "Cast", "CastLike"],
                no_unused_inputs=True,
            )(model)
        )
        floating = {
            int(t.value)
            for t in (
                ir.DataType.FLOAT16,
                ir.DataType.BFLOAT16,
                ir.DataType.FLOAT,
                ir.DataType.DOUBLE,
            )
        }
        tensors = [*model.graph.input, *model.graph.value_info, *model.graph.output]
        return (
            valid
            and all(
                value.type.tensor_type.elem_type not in floating
                or value.type.tensor_type.elem_type == int(ir.DataType.DOUBLE.value)
                for value in tensors
            )
            and all(
                t.data_type == int(ir.DataType.DOUBLE.value)
                for t in model.graph.initializer
            )
        )
    return bool(
        EG(
            [
                (
                    "Where:7 -> Atan:7 -> Sub:7 -> Where:7 -> Sub:7 -> Where:7 -> Mul:7",
                    {"counts": {"Atan": 1, "Sqrt": 0}},
                )
            ],
            no_unused_inputs=True,
        )(model)
    )


@register_native_primitive(
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
        *[
            {"component": op, "doc": f"https://onnx.ai/onnx/operators/onnx__{op}.html"}
            for op in ("Abs", "IsInf", "And", "Sqrt", "Mul", "Cast")
        ],
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
            "expected_output_dtypes": [with_requested_dtype()],
            "post_check_onnx_graph": _atan2_graph_check,
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
    def lower(self, ctx: NativeLoweringContext, eqn: Any) -> None:
        vertical_var, horizontal_var = eqn.invars
        (out_var,) = eqn.outvars
        vertical = ctx.require_value_for_var(vertical_var)
        horizontal = ctx.require_value_for_var(horizontal_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("atan2_out")
        )
        target_dtype = numpy_dtype_to_ir(out_var.aval.dtype)
        work_dtype = (
            ir.DataType.DOUBLE
            if target_dtype == ir.DataType.DOUBLE
            else ir.DataType.FLOAT
        )
        y = cast_value_to_dtype(ctx, vertical, work_dtype, name_hint="atan2_y_work")
        x = cast_value_to_dtype(ctx, horizontal, work_dtype, name_hint="atan2_x_work")
        out_shape: tuple[DimInput, ...] = tuple(out_var.aval.shape)

        def emit(
            op: str,
            *inputs: ir.Value,
            dtype: ir.DataType = work_dtype,
            like: ir.Value | None = None,
        ) -> ir.Value:
            value = ir.Value(
                name=ctx.fresh_name(f"atan2_{op.lower()}"), type=ir.TensorType(dtype)
            )
            if like is not None:
                value.shape = like.shape
            else:
                _stamp_type_and_shape(value, out_shape)
            return ctx.builder.op(op, list(inputs), output=value)

        def scalar(number: float) -> ir.Value:
            return ctx.bind_const_for_var(
                object(), np.asarray(number, dtype=work_dtype.numpy())
            )

        zero, one, negative_one = (scalar(v) for v in (0.0, 1.0, -1.0))
        pi, half_pi = scalar(np.pi), scalar(np.pi / 2)
        ax, ay = emit("Abs", x, like=x), emit("Abs", y, like=y)
        swap = emit("Greater", ay, ax, dtype=ir.DataType.BOOL)
        small = emit("Where", swap, ax, ay)
        large = emit("Where", swap, ay, ax)
        ratio = emit("Div", small, large)
        x_zero = emit("Equal", x, zero, dtype=ir.DataType.BOOL, like=x)
        y_zero = emit("Equal", y, zero, dtype=ir.DataType.BOOL, like=y)
        both_zero = emit("And", x_zero, y_zero, dtype=ir.DataType.BOOL)
        both_inf = emit(
            "And",
            emit("IsInf", ax, dtype=ir.DataType.BOOL, like=ax),
            emit("IsInf", ay, dtype=ir.DataType.BOOL, like=ay),
            dtype=ir.DataType.BOOL,
        )
        ratio = emit("Where", both_zero, zero, ratio)
        ratio = emit("Where", both_inf, one, ratio)

        if work_dtype == ir.DataType.DOUBLE:
            # Two half-angle reductions bound r by tan(pi/16). The degree-21
            # alternating polynomial then has truncation error below 1.3e-17.
            d1 = emit(
                "Add", one, emit("Sqrt", emit("Add", one, emit("Mul", ratio, ratio)))
            )
            reduced = emit("Div", ratio, d1)
            d2 = emit(
                "Add",
                one,
                emit("Sqrt", emit("Add", one, emit("Mul", reduced, reduced))),
            )
            r = emit("Div", reduced, d2)
            squared = emit("Mul", r, r)
            polynomial = scalar(1 / 21)
            for index in range(9, -1, -1):
                polynomial = emit(
                    "Add",
                    scalar((-1.0) ** index / (2 * index + 1)),
                    emit("Mul", squared, polynomial),
                )
            # Reconstruct from the original ratio, preserving tiny angles even
            # when an intermediate reduced argument underflows.
            scale = emit("Div", scalar(4), emit("Mul", d1, d2))
            base = emit("Mul", emit("Mul", ratio, scale), polynomial)
        else:
            base = emit("Atan", ratio)

        angle = emit("Where", swap, emit("Sub", half_pi, base), base)

        def negative(value: ir.Value, is_zero: ir.Value) -> ir.Value:
            reciprocal = emit("Div", one, value, like=value)
            signed = emit("Where", is_zero, reciprocal, value, like=value)
            return emit("Less", signed, zero, dtype=ir.DataType.BOOL, like=value)

        angle = emit("Where", negative(x, x_zero), emit("Sub", pi, angle), angle)
        factor = emit("Where", negative(y, y_zero), negative_one, one, like=y)
        result = emit("Mul", angle, factor)
        result = cast_value_to_dtype(
            ctx, result, target_dtype, name_hint="atan2_result"
        )
        result.name = out_spec.name or ctx.fresh_name("atan2_out")
        ctx.bind_value_for_var(out_var, result)
