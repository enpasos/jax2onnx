# jax2onnx/plugins/jax/lax/mul.py

from typing import TYPE_CHECKING, Optional
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
    pack_native_complex,
    _base_dtype_for_complex,
    _shape_tuple,
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
        base_dtype = _base_dtype_for_complex(lhs.dtype)
        lhs_packed = pack_native_complex(ctx, lhs, name_hint="mul_lhs")
        rhs_packed = pack_native_complex(ctx, rhs, name_hint="mul_rhs")
        inferred_dtype = getattr(lhs_packed, "dtype", None)
        if inferred_dtype is not None:
            base_dtype = inferred_dtype

        if (
            getattr(ctx.builder, "enable_double_precision", False)
            and base_dtype == ir.DataType.FLOAT
        ):
            target_dtype = ir.DataType.DOUBLE
            lhs_packed = self._cast_real_tensor(
                ctx,
                lhs_packed,
                target_dtype,
                name_hint="mul_lhs_cast",
            )
            rhs_packed = self._cast_real_tensor(
                ctx,
                rhs_packed,
                target_dtype,
                name_hint="mul_rhs_cast",
            )
            base_dtype = target_dtype
        dims = _shape_tuple(lhs_packed)
        axis = len(dims) - 1
        base_dims = dims[:-1]

        idx0 = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("mul_idx0"),
            value=np.asarray(0, dtype=np.int64),
        )
        idx1 = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("mul_idx1"),
            value=np.asarray(1, dtype=np.int64),
        )
        for idx in (idx0, idx1):
            _stamp_type_and_shape(idx, ())

        def _extract_parts(value: ir.Value, prefix: str) -> tuple[ir.Value, ir.Value]:
            real = ctx.builder.Gather(
                value,
                idx0,
                axis=axis,
                _outputs=[ctx.fresh_name(f"{prefix}_real")],
            )
            imag = ctx.builder.Gather(
                value,
                idx1,
                axis=axis,
                _outputs=[ctx.fresh_name(f"{prefix}_imag")],
            )
            for part in (real, imag):
                part.type = ir.TensorType(base_dtype)
                part.dtype = base_dtype
                _stamp_type_and_shape(part, base_dims)
                _ensure_value_metadata(ctx, part)
            return real, imag

        a_real, a_imag = _extract_parts(lhs_packed, "mul_lhs")
        b_real, b_imag = _extract_parts(rhs_packed, "mul_rhs")

        ar_br = ctx.builder.Mul(
            a_real,
            b_real,
            _outputs=[ctx.fresh_name("mul_ar_br")],
        )
        ai_bi = ctx.builder.Mul(
            a_imag,
            b_imag,
            _outputs=[ctx.fresh_name("mul_ai_bi")],
        )
        ar_br.type = ir.TensorType(base_dtype)
        ai_bi.type = ir.TensorType(base_dtype)
        ar_br.dtype = base_dtype
        ai_bi.dtype = base_dtype
        _stamp_type_and_shape(ar_br, base_dims)
        _stamp_type_and_shape(ai_bi, base_dims)
        _ensure_value_metadata(ctx, ar_br)
        _ensure_value_metadata(ctx, ai_bi)

        real_part = ctx.builder.Sub(
            ar_br,
            ai_bi,
            _outputs=[ctx.fresh_name("mul_real")],
        )
        real_part.type = ir.TensorType(base_dtype)
        real_part.dtype = base_dtype
        _stamp_type_and_shape(real_part, base_dims)
        _ensure_value_metadata(ctx, real_part)

        ar_bi = ctx.builder.Mul(
            a_real,
            b_imag,
            _outputs=[ctx.fresh_name("mul_ar_bi")],
        )
        ai_br = ctx.builder.Mul(
            a_imag,
            b_real,
            _outputs=[ctx.fresh_name("mul_ai_br")],
        )
        for part in (ar_bi, ai_br):
            part.type = ir.TensorType(base_dtype)
            part.dtype = base_dtype
            _stamp_type_and_shape(part, base_dims)
            _ensure_value_metadata(ctx, part)

        imag_part = ctx.builder.Add(
            ar_bi,
            ai_br,
            _outputs=[ctx.fresh_name("mul_imag")],
        )
        imag_part.type = ir.TensorType(base_dtype)
        imag_part.dtype = base_dtype
        _stamp_type_and_shape(imag_part, base_dims)
        _ensure_value_metadata(ctx, imag_part)

        axis_new = len(base_dims)

        def _add_channel(value: ir.Value, prefix: str) -> ir.Value:
            target = base_dims + (1,)
            shape_arr = np.asarray(target, dtype=np.int64)
            shape_init = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name(f"{prefix}_shape"),
                array=shape_arr,
            )
            shape_init.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(shape_init, (len(target),))
            _ensure_value_metadata(ctx, shape_init)
            reshaped = ctx.builder.Reshape(
                value,
                shape_init,
                _outputs=[ctx.fresh_name(prefix)],
            )
            reshaped.type = ir.TensorType(base_dtype)
            reshaped.dtype = base_dtype
            _stamp_type_and_shape(reshaped, target)
            _ensure_value_metadata(ctx, reshaped)
            return reshaped

        real_unsq = _add_channel(real_part, "mul_real_unsq")
        imag_unsq = _add_channel(imag_part, "mul_imag_unsq")

        packed = ctx.builder.Concat(
            real_unsq,
            imag_unsq,
            axis=axis_new,
            _outputs=[ctx.fresh_name("mul_output")],
        )
        packed.type = ir.TensorType(base_dtype)
        packed.dtype = base_dtype
        _stamp_type_and_shape(packed, base_dims + (2,))
        _ensure_value_metadata(ctx, packed)
        if (
            getattr(ctx.builder, "enable_double_precision", False)
            and base_dtype == ir.DataType.FLOAT
        ):
            packed = self._cast_real_tensor(
                ctx,
                packed,
                ir.DataType.DOUBLE,
                name_hint="mul_output_cast",
            )
            base_dtype = ir.DataType.DOUBLE
        return packed, base_dtype

    def _cast_real_tensor(
        self,
        ctx: "IRContext",
        value: ir.Value,
        target_dtype: ir.DataType,
        *,
        name_hint: str,
    ) -> ir.Value:
        cast = ctx.builder.Cast(
            value,
            to=int(target_dtype.value),
            _outputs=[ctx.fresh_name(name_hint)],
        )
        cast.type = ir.TensorType(target_dtype)
        cast.dtype = target_dtype
        dims = _shape_tuple(value)
        _stamp_type_and_shape(cast, dims)
        _ensure_value_metadata(ctx, cast)
        return cast
