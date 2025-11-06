# jax2onnx/plugins/jax/lax/mul.py

from typing import TYPE_CHECKING, Optional, cast
import jax
import numpy as np
import onnx_ir as ir
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._complex_utils import (
    cast_real_tensor,
    ensure_packed_real_pair,
    resolve_common_real_dtype,
    split_packed_real_imag,
    pack_real_imag_pair,
    _shape_tuple,
    coerce_dim_values,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.mul.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="v0.1.0",
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
    def lower(self, ctx, eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )
        a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("mul_lhs"))
        b_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("mul_rhs"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("mul_out"))
        a_val, b_val, override = maybe_expand_binary_axis0(
            ctx, a_val, b_val, out_spec, out_var
        )

        complex_types = {
            ir.DataType.COMPLEX64,
            ir.DataType.COMPLEX128,
        }
        if a_val.dtype in complex_types and b_val.dtype in complex_types:
            result, result_dtype = self._lower_complex_mul(ctx, a_val, b_val)
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
            result = ctx.builder.Mul(a_val, b_val, _outputs=[out_spec.name])
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(a_val, result)
            propagate_axis0_override(b_val, result)
            ctx.bind_value_for_var(out_var, result)

    def _lower_complex_mul(
        self,
        ctx: "IRContext",
        lhs: ir.Value,
        rhs: ir.Value,
    ) -> tuple[ir.Value, ir.DataType]:
        lhs_packed, lhs_dtype = ensure_packed_real_pair(ctx, lhs, name_hint="mul_lhs")
        rhs_packed, rhs_dtype = ensure_packed_real_pair(ctx, rhs, name_hint="mul_rhs")

        target_dtype = resolve_common_real_dtype(lhs_dtype, rhs_dtype)
        if (
            getattr(ctx.builder, "enable_double_precision", False)
            and target_dtype == ir.DataType.FLOAT
        ):
            target_dtype = ir.DataType.DOUBLE

        lhs_ready = (
            lhs_packed
            if lhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, lhs_packed, target_dtype, name_hint="mul_lhs_cast"
            )
        )
        rhs_ready = (
            rhs_packed
            if rhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, rhs_packed, target_dtype, name_hint="mul_rhs_cast"
            )
        )

        base_dims = _shape_tuple(lhs_ready)[:-1]
        dim_values = coerce_dim_values(base_dims)

        lhs_real, lhs_imag = split_packed_real_imag(
            ctx, lhs_ready, target_dtype, prefix="mul_lhs"
        )
        rhs_real, rhs_imag = split_packed_real_imag(
            ctx, rhs_ready, target_dtype, prefix="mul_rhs"
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
        )
        return packed, target_dtype
