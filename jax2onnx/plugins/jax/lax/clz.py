# jax2onnx/plugins/jax/lax/clz.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import onnx_ir as ir
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)

_DTYPE_INFO: dict[
    np.dtype[Any], tuple[ir.DataType, ir.DataType, np.dtype[Any], int]
] = {
    np.dtype(np.int8): (ir.DataType.INT8, ir.DataType.UINT8, np.dtype(np.uint8), 8),
    np.dtype(np.int16): (
        ir.DataType.INT16,
        ir.DataType.UINT16,
        np.dtype(np.uint16),
        16,
    ),
    np.dtype(np.int32): (
        ir.DataType.INT32,
        ir.DataType.UINT32,
        np.dtype(np.uint32),
        32,
    ),
    np.dtype(np.int64): (
        ir.DataType.INT64,
        ir.DataType.UINT64,
        np.dtype(np.uint64),
        64,
    ),
    np.dtype(np.uint8): (
        ir.DataType.UINT8,
        ir.DataType.UINT8,
        np.dtype(np.uint8),
        8,
    ),
    np.dtype(np.uint16): (
        ir.DataType.UINT16,
        ir.DataType.UINT16,
        np.dtype(np.uint16),
        16,
    ),
    np.dtype(np.uint32): (
        ir.DataType.UINT32,
        ir.DataType.UINT32,
        np.dtype(np.uint32),
        32,
    ),
    np.dtype(np.uint64): (
        ir.DataType.UINT64,
        ir.DataType.UINT64,
        np.dtype(np.uint64),
        64,
    ),
}


@register_primitive(
    jaxpr_primitive=jax.lax.clz_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.clz.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        },
        {
            "component": "BitwiseAnd",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="clz",
    testcases=[
        {
            "testcase": "clz_i32",
            "callable": lambda x: jax.lax.clz(x),
            "input_values": [np.asarray([0, 1, 2, 3, 8, -1], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
        {
            "testcase": "clz_u8",
            "callable": lambda x: jax.lax.clz(x),
            "input_values": [np.asarray([0, 1, 2, 3, 8, 255], dtype=np.uint8)],
            "expected_output_dtypes": [np.uint8],
        },
    ],
)
class ClzPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.clz`` via explicit leading-bit scan in ONNX."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        in_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        in_dtype = np.dtype(getattr(getattr(in_var, "aval", None), "dtype", np.int32))
        if in_dtype not in _DTYPE_INFO:
            raise NotImplementedError(f"clz unsupported dtype '{in_dtype}'")
        out_ir, unsigned_ir, unsigned_np, bit_width = _DTYPE_INFO[in_dtype]

        x_val = ctx.get_value_for_var(in_var, name_hint=ctx.fresh_name("clz_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("clz_out"))
        out_shape = getattr(out_spec, "shape", None) or getattr(x_val, "shape", None)

        x_unsigned = x_val
        if out_ir != unsigned_ir:
            x_unsigned = ctx.builder.Cast(
                x_val,
                to=int(unsigned_ir.value),
                _outputs=[ctx.fresh_name("clz_uin")],
            )
            if out_shape is not None:
                x_unsigned.shape = out_shape
            x_unsigned.type = ir.TensorType(unsigned_ir)

        zero_u = ctx.bind_const_for_var(object(), np.asarray(0, dtype=unsigned_np))
        one_u = ctx.bind_const_for_var(object(), np.asarray(1, dtype=unsigned_np))

        zero_like = ctx.builder.Mul(
            x_unsigned,
            zero_u,
            _outputs=[ctx.fresh_name("clz_zero_like")],
        )
        if out_shape is not None:
            zero_like.shape = out_shape
        zero_like.type = ir.TensorType(unsigned_ir)

        seen = zero_like
        count = zero_like

        for i in reversed(range(bit_width)):
            shift_i = ctx.bind_const_for_var(object(), np.asarray(i, dtype=unsigned_np))
            shifted = ctx.builder.BitShift(
                x_unsigned,
                shift_i,
                direction="RIGHT",
                _outputs=[ctx.fresh_name("clz_shift")],
            )
            if out_shape is not None:
                shifted.shape = out_shape
            shifted.type = ir.TensorType(unsigned_ir)

            bit = ctx.builder.BitwiseAnd(
                shifted,
                one_u,
                _outputs=[ctx.fresh_name("clz_bit")],
            )
            if out_shape is not None:
                bit.shape = out_shape
            bit.type = ir.TensorType(unsigned_ir)

            bit_is_zero = ctx.builder.Equal(
                bit,
                zero_u,
                _outputs=[ctx.fresh_name("clz_bit_is_zero")],
            )
            seen_is_zero = ctx.builder.Equal(
                seen,
                zero_u,
                _outputs=[ctx.fresh_name("clz_seen_is_zero")],
            )
            should_inc = ctx.builder.And(
                bit_is_zero,
                seen_is_zero,
                _outputs=[ctx.fresh_name("clz_should_inc")],
            )

            inc_u = ctx.builder.Cast(
                should_inc,
                to=int(unsigned_ir.value),
                _outputs=[ctx.fresh_name("clz_inc_u")],
            )
            if out_shape is not None:
                inc_u.shape = out_shape
            inc_u.type = ir.TensorType(unsigned_ir)

            count = ctx.builder.Add(
                count,
                inc_u,
                _outputs=[ctx.fresh_name("clz_count")],
            )
            if out_shape is not None:
                count.shape = out_shape
            count.type = ir.TensorType(unsigned_ir)

            seen = ctx.builder.BitwiseOr(
                seen,
                bit,
                _outputs=[ctx.fresh_name("clz_seen")],
            )
            if out_shape is not None:
                seen.shape = out_shape
            seen.type = ir.TensorType(unsigned_ir)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("clz_out")
        result = count
        if out_ir != unsigned_ir:
            result = ctx.builder.Cast(
                count,
                to=int(out_ir.value),
                _outputs=[desired_name],
            )
            if out_shape is not None:
                result.shape = out_shape
            result.type = ir.TensorType(out_ir)
        elif getattr(result, "name", None) != desired_name:
            result = ctx.builder.Identity(result, _outputs=[desired_name])
            if out_shape is not None:
                result.shape = out_shape
            result.type = ir.TensorType(out_ir)

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
