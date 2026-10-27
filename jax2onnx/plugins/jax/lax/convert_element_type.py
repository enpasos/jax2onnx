# jax2onnx/plugins/jax/lax/convert_element_type.py

from __future__ import annotations

from typing import Any, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="convert_element_type",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html",
    onnx=[
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"}
    ],
    since="0.2.0",
    context="primitives.lax",
    component="convert_element_type",
    testcases=[
        {
            "testcase": "convert_element_type",
            "callable": lambda x: x.astype(np.int16),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Cast:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class ConvertElementTypePlugin(PrimitiveLeafPlugin):
    """Preserve explicit conversion boundaries and native integer saturation."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("convert_in")
        )
        source_dtype = np.dtype(operand_var.aval.dtype)
        result_dtype = np.dtype(out_var.aval.dtype)
        if (
            ctx.precision_policy is PrecisionPolicy.SOURCE
            and source_dtype == np.dtype(np.float64)
            and result_dtype == np.dtype(np.float16)
        ):
            from jax2onnx.converter._source_cast_authority import (
                _take_selected_cast,
                _record_selected_cast_result,
            )

            selected = _take_selected_cast(ctx, eqn)
            if selected is not None:
                from jax2onnx.plugins.jax._f64_to_f16_rne import emit_f64_to_f16_rne
                from jax2onnx.plugins.jax._f64_to_f16_double_round import (
                    emit_f64_to_f16_double_round,
                )

                if selected.algorithm == "DIRECT_RNE_53_TO_11":
                    result = emit_f64_to_f16_rne(ctx, operand_val)
                elif selected.algorithm == "DOUBLE_ROUND_RNE_53_TO_24_TO_11":
                    result = emit_f64_to_f16_double_round(ctx, operand_val)
                else:
                    raise RuntimeError("selected cast algorithm unavailable")
                _stamp_type_and_shape(result, tuple(out_var.aval.shape))
                _ensure_value_metadata(ctx, result)
                ctx.bind_value_for_var(out_var, result)
                _record_selected_cast_result(ctx, result)
                return
        if (
            ctx.precision_policy is PrecisionPolicy.SOURCE
            and source_dtype.name in {"float16", "bfloat16", "float32", "float64"}
            and np.issubdtype(result_dtype, np.integer)
        ):
            self._saturating_integer(
                ctx, operand_val, out_var, source_dtype, result_dtype
            )
            return
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("convert_out")
        )

        target_dtype = numpy_dtype_to_ir(np.dtype(out_var.aval.dtype))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("convert_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("convert_out")

        result = cast(
            ir.Value,
            ctx.builder.Cast(
                operand_val,
                _outputs=[desired_name],
                to=int(target_dtype.value),
            ),
        )
        result.type = ir.TensorType(target_dtype)
        result.shape = operand_val.shape
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @staticmethod
    def _saturating_integer(
        ctx: LoweringContextProtocol,
        operand: ir.Value,
        out_var: Any,
        source_dtype: np.dtype[Any],
        target_dtype: np.dtype[Any],
    ) -> None:
        """Cast only a finite in-range value; select exact integer endpoints.

        Native conversion truncates toward zero, saturates, and maps NaN to
        zero. A float32/64 upper bound can round INT64/UINT64 max upward; its
        previous representable float is safe to cast, while an integer Where
        supplies the exact endpoint. FP16/BF16 widen exactly on this consumer
        only. Canonical inputs and their sibling consumers remain untouched.
        """
        working_dtype = np.dtype(
            np.float64 if source_dtype == np.dtype(np.float64) else np.float32
        )
        working_ir = numpy_dtype_to_ir(working_dtype)
        target_ir = numpy_dtype_to_ir(target_dtype)
        shape = tuple(out_var.aval.shape)

        def stamp(value: ir.Value, dtype: ir.DataType) -> ir.Value:
            value.type = ir.TensorType(dtype)
            _stamp_type_and_shape(value, shape)
            _ensure_value_metadata(ctx, value)
            return value

        def constant(value: Any, dtype: np.dtype[Any]) -> ir.Value:
            return ctx.bind_const_for_var(object(), np.asarray(value, dtype=dtype))

        value = operand
        if source_dtype != working_dtype:
            value = stamp(
                ctx.builder.op("Cast", [value], {"to": int(working_ir.value)}),
                working_ir,
            )
        limits = np.iinfo(target_dtype)
        upper = np.asarray(limits.max, dtype=working_dtype)
        # int() inspects the constant exactly; no integer data travels through
        # a floating point adapter or a floating initializer.
        safe_upper = (
            np.nextafter(upper, np.asarray(-np.inf, dtype=working_dtype))
            if int(upper) > limits.max
            else upper
        )
        lower_value = constant(limits.min, working_dtype)
        upper_value = constant(upper, working_dtype)
        safe_upper_value = constant(safe_upper, working_dtype)
        zero = constant(0, working_dtype)
        isnan = stamp(ctx.builder.op("IsNaN", [value]), ir.DataType.BOOL)
        clean = stamp(ctx.builder.op("Where", [isnan, zero, value]), working_ir)
        clipped_low = stamp(ctx.builder.op("Max", [clean, lower_value]), working_ir)
        clipped = stamp(
            ctx.builder.op("Min", [clipped_low, safe_upper_value]), working_ir
        )
        # CPU Where has integer kernel gaps, including INT16. Select through
        # INT64 without any integer-to-float round trip. All narrower integer
        # values fit exactly; UINT64 uses the exact modulo-2**64 Cast mapping.
        carrier_ir = ir.DataType.INT64
        cast_ir = target_ir if target_dtype == np.dtype(np.uint64) else carrier_ir
        converted = stamp(
            ctx.builder.op("Cast", [clipped], {"to": int(cast_ir.value)}), cast_ir
        )
        if cast_ir != carrier_ir:
            converted = stamp(
                ctx.builder.op("Cast", [converted], {"to": int(carrier_ir.value)}),
                carrier_ir,
            )
        overflow = stamp(
            ctx.builder.op("GreaterOrEqual", [clean, upper_value]), ir.DataType.BOOL
        )
        carrier_max = limits.max if limits.max <= np.iinfo(np.int64).max else -1
        exact_max = constant(carrier_max, np.dtype(np.int64))
        output = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("convert_out")
        )
        if target_ir == carrier_ir:
            result = ctx.builder.op(
                "Where", [overflow, exact_max, converted], output=output
            )
        else:
            selected = stamp(
                ctx.builder.op("Where", [overflow, exact_max, converted]), carrier_ir
            )
            result = ctx.builder.op(
                "Cast", [selected], {"to": int(target_ir.value)}, output=output
            )
        result = stamp(result, target_ir)
        ctx.bind_value_for_var(out_var, result)
