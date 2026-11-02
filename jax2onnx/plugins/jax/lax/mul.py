# jax2onnx/plugins/jax/lax/mul.py

from __future__ import annotations

from typing import cast
import jax
import numpy as np
import onnx_ir as ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    split_packed_real_imag,
    pack_known_real,
    pack_real_imag_pair,
    coerce_dim_values,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import emit_float16_arithmetic_with_rounding
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax.lax.convert_element_type import ConvertElementTypePlugin
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def emit_packed_complex_mul(
    ctx: LoweringContextProtocol,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    target_dtype: ir.DataType,
    result_shape: tuple[object, ...],
    output_name: str | None = None,
) -> ir.Value:
    dim_values = coerce_dim_values(result_shape)

    lhs_real, lhs_imag = split_packed_real_imag(
        ctx, lhs, target_dtype, prefix="mul_lhs"
    )
    rhs_real, rhs_imag = split_packed_real_imag(
        ctx, rhs, target_dtype, prefix="mul_rhs"
    )

    def _binary(op_name: str, a: ir.Value, b: ir.Value, name: str) -> ir.Value:
        op = getattr(ctx.builder, op_name)
        result = cast(
            ir.Value,
            op(a, b, _outputs=[ctx.fresh_name(name)]),
        )
        result.type = ir.TensorType(target_dtype)
        result.dtype = target_dtype
        _stamp_type_and_shape(result, dim_values)
        _ensure_value_metadata(ctx, result)
        return result

    ar_br = _binary("Mul", lhs_real, rhs_real, "mul_ar_br")
    ai_bi = _binary("Mul", lhs_imag, rhs_imag, "mul_ai_bi")
    real_part = _binary("Sub", ar_br, ai_bi, "mul_real_part")

    ar_bi = _binary("Mul", lhs_real, rhs_imag, "mul_ar_bi")
    ai_br = _binary("Mul", lhs_imag, rhs_real, "mul_ai_br")
    imag_part = _binary("Add", ar_bi, ai_br, "mul_imag_part")

    packed = pack_real_imag_pair(
        ctx,
        real_part,
        imag_part,
        target_dtype,
        name_hint="mul_output",
        output_name=output_name,
    )
    return packed


@register_native_primitive(
    jaxpr_primitive=jax.lax.mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.mul.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="0.1.0",
    context="primitives.lax",
    component="mul",
    testcases=[
        {
            "testcase": "mul_test1",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mul_test2",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(2, 2), (2, 2)],
            "post_check_onnx_graph": EG(
                ["Mul:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mul_pyfloat_promotes_to_array_dtype_f64",
            "callable": lambda x: x * 1.5,
            "input_values": [np.array([1.0, 2.0], dtype=np.float64)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Mul:2",
                        "inputs": {1: {"const": 1.5}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mul_scalar_broadcast_promote_to_f64",
            "callable": lambda x: x.astype(np.float64) * 1.5,
            "input_values": [np.array([1.0, 2.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Mul:2",
                        "inputs": {1: {"const": 1.5}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mul_complex128",
            "callable": lambda x, y: x * y,
            "input_values": [
                np.array([1.0 + 2.0j, -3.5 + 0.25j], dtype=np.complex128),
                np.array([0.5 - 1.0j, 2.0 + 3.0j], dtype=np.complex128),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Mul", "counts": {"Mul": 4}},
                    {"path": "Sub", "counts": {"Sub": 1}},
                    {"path": "Add", "counts": {"Add": 1}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mul_complex64",
            "callable": lambda x, y: x * y,
            "input_values": [
                np.array([1.0 + 0.5j, -0.75 + 1.25j], dtype=np.complex64),
                np.array([0.5 - 2.0j, 1.5 + 0.25j], dtype=np.complex64),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Mul", "counts": {"Mul": 4}},
                    {"path": "Sub", "counts": {"Sub": 1}},
                    {"path": "Add", "counts": {"Add": 1}},
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class MulPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.require_value_for_var(x_var)
        b_val = ctx.require_value_for_var(y_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("mul_out")
        )
        a_val, b_val, override = maybe_expand_binary_axis0(
            ctx, a_val, b_val, out_spec, out_var
        )

        def _is_complex_var(var: object) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        complex_var_hint = (
            _is_complex_var(out_var) or _is_complex_var(x_var) or _is_complex_var(y_var)
        )
        complex_dtype_hint = (
            a_val.dtype in COMPLEX_DTYPES or b_val.dtype in COMPLEX_DTYPES
        )
        packed_hint = False
        if complex_var_hint or complex_dtype_hint:
            packed_hint = is_packed_complex_tensor(a_val) or is_packed_complex_tensor(
                b_val
            )
        complex_route = complex_var_hint or complex_dtype_hint or packed_hint

        if complex_route and not _is_complex_var(out_var):
            # Complex-to-real conversion drops each imaginary component before
            # multiplication, rather than projecting the complex product.
            if _is_complex_var(x_var):
                packed, component_dtype = ensure_packed_real_pair(
                    ctx, a_val, name_hint="mul_lhs"
                )
                a_val, _ = split_packed_real_imag(
                    ctx, packed, component_dtype, prefix="mul_lhs"
                )
            if _is_complex_var(y_var):
                packed, component_dtype = ensure_packed_real_pair(
                    ctx, b_val, name_hint="mul_rhs"
                )
                b_val, _ = split_packed_real_imag(
                    ctx, packed, component_dtype, prefix="mul_rhs"
                )
            complex_route = False

        if complex_route:
            # out_dtype converts both complex operands before multiplication.
            result_dtype = {
                np.dtype(np.complex64): ir.DataType.FLOAT,
                np.dtype(np.complex128): ir.DataType.DOUBLE,
            }[np.dtype(out_var.aval.dtype)]

            def prepare_complex_operand(
                value: ir.Value, var: object, prefix: str
            ) -> ir.Value:
                if _is_complex_var(var):
                    packed, _ = ensure_packed_real_pair(ctx, value, name_hint=prefix)
                    return cast_value_to_dtype(
                        ctx, packed, result_dtype, name_hint=f"{prefix}_cast"
                    )
                # A known-real trailing axis of two contains two real values.
                # Cast directly to the requested component width before packing.
                return pack_known_real(ctx, value, result_dtype, name_hint=prefix)

            result = emit_packed_complex_mul(
                ctx,
                prepare_complex_operand(a_val, x_var, "mul_lhs"),
                prepare_complex_operand(b_val, y_var, "mul_rhs"),
                target_dtype=result_dtype,
                result_shape=tuple(out_var.aval.shape),
            )
            if getattr(out_spec, "type", None) is not None:
                out_spec.type = ir.TensorType(result_dtype)
                out_spec.dtype = result_dtype
            if (
                getattr(out_spec, "shape", None) is not None
                and getattr(result, "shape", None) is not None
            ):
                out_spec.shape = result.shape
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(a_val, result)
            propagate_axis0_override(b_val, result)
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            ctx.bind_value_for_var(out_var, result)
            try:
                outputs = ctx.builder.outputs
            except AttributeError:
                outputs = []
            result_name = getattr(result, "name", None)
            for idx, out_val in enumerate(outputs):
                if out_val is result or getattr(out_val, "name", None) == result_name:
                    out_val.type = ir.TensorType(result_dtype)
                    out_val.dtype = result_dtype
                    if getattr(result, "shape", None) is not None:
                        out_val.shape = result.shape
                    _ensure_value_metadata(ctx, out_val)
                    break
        else:
            # JAX's out_dtype controls operand conversion, not just the result.
            target_dtype = numpy_dtype_to_ir(out_var.aval.dtype)

            def prepare_operand(value: ir.Value, name_hint: str) -> ir.Value:
                if (
                    target_dtype in {ir.DataType.INT8, ir.DataType.UINT8}
                    and value.dtype is not None
                    and value.dtype.is_floating_point()
                ):
                    saturated = ConvertElementTypePlugin._saturate_float_operand(
                        ctx, value, target_dtype
                    )
                    propagate_axis0_override(value, saturated)
                    value = saturated
                return cast_value_to_dtype(
                    ctx, value, target_dtype, name_hint=name_hint
                )

            a_val = prepare_operand(a_val, "mul_lhs_cast")
            b_val = prepare_operand(b_val, "mul_rhs_cast")
            if all(
                value.dtype == ir.DataType.FLOAT16 for value in (a_val, b_val, out_spec)
            ):
                result = emit_float16_arithmetic_with_rounding(
                    ctx, "Mul", a_val, b_val, output=out_spec
                )
            else:
                result = ctx.builder.op("Mul", [a_val, b_val], output=out_spec)
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(a_val, result)
            propagate_axis0_override(b_val, result)
            ctx.bind_value_for_var(out_var, result)
