# jax2onnx/plugins/jax/lax/shift_right_arithmetic.py

from typing import Any

import jax
import numpy as np
import onnx_ir as ir
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SIGNED_TO_UNSIGNED: dict[ir.DataType, tuple[ir.DataType, np.dtype[Any], int]] = {
    ir.DataType.INT8: (ir.DataType.UINT8, np.dtype(np.uint8), 8),
    ir.DataType.INT16: (ir.DataType.UINT16, np.dtype(np.uint16), 16),
    ir.DataType.INT32: (ir.DataType.UINT32, np.dtype(np.uint32), 32),
    ir.DataType.INT64: (ir.DataType.UINT64, np.dtype(np.uint64), 64),
}

_INTEGER_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    {
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
    }
)


@register_primitive(
    jaxpr_primitive=jax.lax.shift_right_arithmetic_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.shift_right_arithmetic.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="shift_right_arithmetic",
    testcases=[
        {
            "testcase": "shift_right_arithmetic_vec",
            "callable": lambda x, s: jax.lax.shift_right_arithmetic(x, s),
            "input_values": [
                np.array([-16, -1, 16, 31], dtype=np.int32),
                np.array([1, 2, 2, 1], dtype=np.int32),
            ],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "shift_right_arithmetic_scalar",
            "callable": lambda x, s: jax.lax.shift_right_arithmetic(x, s),
            "input_values": [
                np.array(-33, dtype=np.int32),
                np.array(4, dtype=np.int32),
            ],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ShiftRightArithmeticPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.shift_right_arithmetic`` to ONNX BitShift(direction="RIGHT")."""

    @staticmethod
    def _infer_signed_dtype_enum(
        lhs_var: Any,
        lhs_val: Any,
    ) -> ir.DataType:
        dtype_enum = getattr(getattr(lhs_val, "type", None), "dtype", None)
        if isinstance(dtype_enum, ir.DataType):
            return dtype_enum
        if isinstance(dtype_enum, (int, np.integer)):
            return ir.DataType(int(dtype_enum))

        aval = getattr(lhs_var, "aval", None)
        aval_dtype: np.dtype[Any] = np.dtype(getattr(aval, "dtype", np.int32))
        if aval_dtype in _INTEGER_DTYPES:
            return numpy_dtype_to_ir(aval_dtype)
        raise NotImplementedError(
            f"shift_right_arithmetic unsupported dtype '{aval_dtype}'"
        )

    @staticmethod
    def _stamp_like(value: Any, ref: Any, *, dtype: ir.DataType | None = None) -> None:
        if dtype is not None:
            value.type = ir.TensorType(dtype)
        elif getattr(ref, "type", None) is not None:
            value.type = ref.type
        if getattr(ref, "shape", None) is not None:
            value.shape = ref.shape

    def lower(self, ctx: LoweringContextProtocol, eqn: "core.JaxprEqn") -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dtype: np.dtype[Any] = np.dtype(getattr(lhs_var.aval, "dtype", np.int32))

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("sra_input"))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("sra_shift"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sra_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("sra_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("sra_out")

        signed_enum = self._infer_signed_dtype_enum(lhs_var, lhs_val)

        # Unsigned inputs can use direct logical shift.
        if signed_enum in {
            ir.DataType.UINT8,
            ir.DataType.UINT16,
            ir.DataType.UINT32,
            ir.DataType.UINT64,
        }:
            result = ctx.builder.BitShift(
                lhs_val,
                rhs_val,
                direction="RIGHT",
                _outputs=[desired_name],
            )
            self._stamp_like(result, out_spec)
            ctx.bind_value_for_var(out_var, result)
            return

        if signed_enum not in _SIGNED_TO_UNSIGNED:
            raise NotImplementedError(
                f"shift_right_arithmetic unsupported dtype '{signed_enum}'"
            )

        unsigned_enum, unsigned_np_dtype, bit_width = _SIGNED_TO_UNSIGNED[signed_enum]
        signed_np_dtype = np.dtype(prefer_dtype)

        zero_signed = ctx.bind_const_for_var(
            object(), np.asarray(0, dtype=signed_np_dtype)
        )
        zero_unsigned = ctx.bind_const_for_var(
            object(), np.asarray(0, dtype=unsigned_np_dtype)
        )
        ones_unsigned = ctx.bind_const_for_var(
            object(),
            np.asarray(np.iinfo(unsigned_np_dtype).max, dtype=unsigned_np_dtype),
        )
        bit_width_unsigned = ctx.bind_const_for_var(
            object(), np.asarray(bit_width, dtype=unsigned_np_dtype)
        )

        s_nonneg = ctx.builder.Max(
            rhs_val,
            zero_signed,
            _outputs=[ctx.fresh_name("sra_nonneg_shift")],
        )
        self._stamp_like(s_nonneg, rhs_val, dtype=signed_enum)

        s_unsigned = ctx.builder.Cast(
            s_nonneg,
            to=int(unsigned_enum.value),
            _outputs=[ctx.fresh_name("sra_shift_u")],
        )
        self._stamp_like(s_unsigned, rhs_val, dtype=unsigned_enum)

        x_unsigned = ctx.builder.Cast(
            lhs_val,
            to=int(unsigned_enum.value),
            _outputs=[ctx.fresh_name("sra_input_u")],
        )
        self._stamp_like(x_unsigned, lhs_val, dtype=unsigned_enum)

        s_clamped = ctx.builder.Min(
            s_unsigned,
            bit_width_unsigned,
            _outputs=[ctx.fresh_name("sra_shift_clamped")],
        )
        self._stamp_like(s_clamped, s_unsigned, dtype=unsigned_enum)

        shifted = ctx.builder.BitShift(
            x_unsigned,
            s_clamped,
            direction="RIGHT",
            _outputs=[ctx.fresh_name("sra_shifted")],
        )
        self._stamp_like(shifted, x_unsigned, dtype=unsigned_enum)

        n_minus_s = ctx.builder.Sub(
            bit_width_unsigned,
            s_clamped,
            _outputs=[ctx.fresh_name("sra_n_minus_s")],
        )
        self._stamp_like(n_minus_s, s_unsigned, dtype=unsigned_enum)

        mask_raw = ctx.builder.BitShift(
            ones_unsigned,
            n_minus_s,
            direction="LEFT",
            _outputs=[ctx.fresh_name("sra_sign_mask_raw")],
        )
        self._stamp_like(mask_raw, shifted, dtype=unsigned_enum)

        is_zero_shift = ctx.builder.Equal(
            s_clamped,
            zero_unsigned,
            _outputs=[ctx.fresh_name("sra_is_zero_shift")],
        )
        self._stamp_like(is_zero_shift, shifted, dtype=ir.DataType.BOOL)

        nonzero_shift = ctx.builder.Not(
            is_zero_shift,
            _outputs=[ctx.fresh_name("sra_nonzero_shift")],
        )
        self._stamp_like(nonzero_shift, shifted, dtype=ir.DataType.BOOL)

        nonzero_shift_u = ctx.builder.Cast(
            nonzero_shift,
            to=int(unsigned_enum.value),
            _outputs=[ctx.fresh_name("sra_nonzero_shift_u")],
        )
        self._stamp_like(nonzero_shift_u, shifted, dtype=unsigned_enum)

        sign_mask = ctx.builder.Mul(
            mask_raw,
            nonzero_shift_u,
            _outputs=[ctx.fresh_name("sra_sign_mask")],
        )
        self._stamp_like(sign_mask, shifted, dtype=unsigned_enum)

        neg_shifted = ctx.builder.BitwiseOr(
            shifted,
            sign_mask,
            _outputs=[ctx.fresh_name("sra_neg_shifted")],
        )
        self._stamp_like(neg_shifted, shifted, dtype=unsigned_enum)

        is_negative = ctx.builder.Less(
            lhs_val,
            zero_signed,
            _outputs=[ctx.fresh_name("sra_is_negative")],
        )
        self._stamp_like(is_negative, lhs_val, dtype=ir.DataType.BOOL)

        is_negative_u = ctx.builder.Cast(
            is_negative,
            to=int(unsigned_enum.value),
            _outputs=[ctx.fresh_name("sra_is_negative_u")],
        )
        self._stamp_like(is_negative_u, shifted, dtype=unsigned_enum)

        delta = ctx.builder.Sub(
            neg_shifted,
            shifted,
            _outputs=[ctx.fresh_name("sra_delta")],
        )
        self._stamp_like(delta, shifted, dtype=unsigned_enum)

        correction = ctx.builder.Mul(
            delta,
            is_negative_u,
            _outputs=[ctx.fresh_name("sra_correction")],
        )
        self._stamp_like(correction, shifted, dtype=unsigned_enum)

        selected = ctx.builder.Add(
            shifted,
            correction,
            _outputs=[ctx.fresh_name("sra_selected_u")],
        )
        self._stamp_like(selected, shifted, dtype=unsigned_enum)

        result = ctx.builder.Cast(
            selected,
            to=int(signed_enum.value),
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
