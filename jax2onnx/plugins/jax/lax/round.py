# jax2onnx/plugins/jax/lax/round.py

from __future__ import annotations

from jax2onnx._compat.jax import JaxprEqn
import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.round_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.round.html",
    onnx=[
        {
            "component": "Round",
            "doc": "https://onnx.ai/onnx/operators/onnx__Round.html",
        },
        *[
            {"component": op, "doc": f"https://onnx.ai/onnx/operators/onnx__{op}.html"}
            for op in (
                "Abs",
                "Sub",
                "Equal",
                "Cast",
                "Add",
                "Div",
                "Where",
                "Less",
                "Mul",
            )
        ],
    ],
    since="0.12.1",
    context="primitives.lax",
    component="round",
    testcases=[
        {
            "testcase": "round",
            "callable": lambda x: jax.lax.round(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Round:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class RoundPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("round_out")
        )

        method = eqn.params["rounding_method"]
        if method not in (
            jax.lax.RoundingMethod.TO_NEAREST_EVEN,
            jax.lax.RoundingMethod.AWAY_FROM_ZERO,
        ):
            raise NotImplementedError(
                f"Round does not implement rounding_method={method!r}"
            )
        target_dtype = numpy_dtype_to_ir(out_var.aval.dtype)
        work_dtype = (
            ir.DataType.FLOAT
            if target_dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
            else target_dtype
        )
        work = cast_value_to_dtype(ctx, x_val, work_dtype, name_hint="round_work")

        def emit(
            op: str, *inputs: ir.Value, dtype: ir.DataType = work_dtype
        ) -> ir.Value:
            return ctx.builder.op(
                op,
                list(inputs),
                output=ir.Value(
                    name=ctx.fresh_name(f"round_{op.lower()}"),
                    type=ir.TensorType(dtype),
                    shape=work.shape,
                ),
            )

        if method == jax.lax.RoundingMethod.TO_NEAREST_EVEN:
            rounded = emit("Round", work)
        else:

            def scalar(value: float) -> ir.Value:
                return ctx.bind_const_for_var(
                    object(), np.asarray(value, dtype=work_dtype.numpy())
                )

            zero, half, one, negative_one = (scalar(v) for v in (0.0, 0.5, 1.0, -1.0))
            absolute = emit("Abs", work)
            even = emit("Round", absolute)
            # Only even-down ties need an increment. Adding 0.5 before Floor
            # would incorrectly round the representable value just below 0.5.
            tie = emit(
                "Equal", emit("Sub", absolute, even), half, dtype=ir.DataType.BOOL
            )
            increment = ctx.builder.op(
                "Cast",
                [tie],
                {"to": int(work_dtype.value)},
                output=ir.Value(
                    name=ctx.fresh_name("round_increment"),
                    type=ir.TensorType(work_dtype),
                    shape=work.shape,
                ),
            )
            magnitude = emit("Add", even, increment)
            is_zero = emit("Equal", work, zero, dtype=ir.DataType.BOOL)
            sign_source = emit("Where", is_zero, emit("Div", one, work), work)
            negative = emit("Less", sign_source, zero, dtype=ir.DataType.BOOL)
            factor = emit("Where", negative, negative_one, one)
            # Where need not transport -0: the final multiplication creates it.
            rounded = emit("Mul", magnitude, factor)

        result = cast_value_to_dtype(
            ctx, rounded, target_dtype, name_hint="round_result"
        )
        result.name = out_spec.name or ctx.fresh_name("round_out")
        ctx.bind_value_for_var(out_var, result)
